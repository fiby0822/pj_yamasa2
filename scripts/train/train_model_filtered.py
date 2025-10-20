#!/usr/bin/env python3
"""
ヤマサ確定注文需要予測の学習スクリプト（Material Keyフィルタリング対応版）
"""
import sys
import argparse
from pathlib import Path
from datetime import datetime
import time
sys.path.append(str(Path(__file__).resolve().parents[2]))

import pandas as pd
import numpy as np
import boto3
from io import BytesIO
from modules.models.train_predict import TimeSeriesPredictor
from modules.evaluation.metrics import ModelEvaluator, display_evaluation_summary
import gc

def filter_material_keys(df, min_positive_count=96, period_start='2021-01-01', period_end='2024-12-31',
                        test_start='2024-01-01', test_end='2024-01-31', min_test_positive=4,
                        train_end_date='2024-12-31'):
    """
    Material Keyをactual_value > 0のレコード数でフィルタリング
    条件: 96個以上 OR テストデータに含まれる

    Parameters:
    -----------
    df : pd.DataFrame
        入力データ
    min_positive_count : int
        最小positive数の閾値
    period_start : str
        集計期間開始日
    period_end : str
        集計期間終了日
    test_start : str
        テスト期間開始日
    test_end : str
        テスト期間終了日
    min_test_positive : int
        テスト期間内の最小positive数

    Returns:
    --------
    tuple
        (フィルタリング後のデータ, positive_counts辞書, over_threshold_mks, train_positive_counts辞書)
    """
    print(f"\nMaterial Keyフィルタリング中...")
    print(f"  集計期間: {period_start} ~ {period_end}")
    print(f"  テスト期間: {test_start} ~ {test_end}")
    print(f"  条件1: actual_value > 0のレコード数 >= {min_positive_count}")
    print(f"  条件2: テストデータでactual_value > 0が1回以上")
    print(f"  最終条件: 条件1 OR 条件2")

    # 元のデータサイズ
    original_size = len(df)
    original_mk_count = df['material_key'].nunique()

    # 期間でフィルタ
    df['file_date'] = pd.to_datetime(df['file_date'])
    df_period = df[(df['file_date'] >= period_start) & (df['file_date'] <= period_end)]
    df_test = df[(df['file_date'] >= test_start) & (df['file_date'] <= test_end)]

    # actual_value > 0のレコードをカウント（学習期間：2021-01-01～train_end_date）
    mk_positive_counts = df_period[df_period['actual_value'] > 0].groupby('material_key').size()

    # 学習期間内（～train_end_date）の実績発生数を計算
    df_train = df[(df['file_date'] >= period_start) & (df['file_date'] <= train_end_date)]
    train_positive_counts = df_train[df_train['actual_value'] > 0].groupby('material_key').size()

    # 条件2を「テストデータで実績値>0が1回以上」に変更
    test_mks_with_actual = set(df_test[df_test['actual_value'] > 0]['material_key'].unique())

    # 閾値以上のMaterial Key（96個以上）
    over_threshold_mks = set(mk_positive_counts[mk_positive_counts >= min_positive_count].index)

    # 条件を満たすMaterial Key（96個以上 OR テストデータで実績値>0）
    valid_mks = over_threshold_mks | test_mks_with_actual

    # フィルタリング
    df_filtered = df[df['material_key'].isin(valid_mks)]

    # positive_counts辞書を作成（後で使用）
    positive_counts_dict = mk_positive_counts.to_dict()
    train_positive_counts_dict = train_positive_counts.to_dict()

    # 結果を表示
    filtered_size = len(df_filtered)
    filtered_mk_count = df_filtered['material_key'].nunique()
    reduction_rate = (1 - filtered_size / original_size) * 100

    print(f"\nフィルタリング結果:")
    print(f"  条件1を満たす（{min_positive_count}個以上）: {len(over_threshold_mks):,} Material Keys")
    print(f"  条件2を満たす（テストデータで実績>0）: {len(test_mks_with_actual):,} Material Keys")
    print(f"  両条件の重複: {len(over_threshold_mks & test_mks_with_actual):,} Material Keys")
    print(f"  Material Key: {original_mk_count:,} → {filtered_mk_count:,} ({filtered_mk_count/original_mk_count*100:.1f}%)")
    print(f"  レコード数: {original_size:,} → {filtered_size:,} ({100-reduction_rate:.1f}%)")
    print(f"  削減率: {reduction_rate:.1f}%")

    return df_filtered, positive_counts_dict, over_threshold_mks, train_positive_counts_dict

def main():
    """メイン処理"""
    parser = argparse.ArgumentParser(description='ヤマサ確定注文需要予測の学習（フィルタリング版）')
    parser.add_argument('--train-end-date', type=str, default='2024-12-31',
                        help='学習データの終了日 (YYYY-MM-DD)')
    parser.add_argument('--step-count', type=int, default=1,
                        help='予測ステップ数')
    parser.add_argument('--features-path', type=str, default=None,
                        help='特徴量ファイルのS3パス')
    parser.add_argument('--save-dir', type=str, default=None,
                        help='モデル保存先のS3ディレクトリ')
    parser.add_argument('--use-optuna', action='store_true',
                        help='ハイパーパラメータ最適化を有効化')
    parser.add_argument('--n-trials', type=int, default=50,
                        help='Optunaの試行回数')
    parser.add_argument('--no-outlier-handling', action='store_true',
                        help='外れ値処理を無効化')
    parser.add_argument('--use-gpu', action='store_true',
                        help='GPU使用を有効化')
    parser.add_argument('--input-path', type=str, default=None,
                        help='入力データのS3パス')
    parser.add_argument('--min-positive-count', type=int, default=48,
                        help='Material Keyの最小positive数')
    parser.add_argument('--min-test-positive', type=int, default=4,
                        help='テスト期間内の最小positive数')

    args = parser.parse_args()

    print("="*60)
    print(" ヤマサ確定注文需要予測 - モデル学習（フィルタリング版）")
    print("="*60)
    print()

    # 開始時刻を記録
    start_time = time.time()

    # S3クライアント
    s3_client = boto3.client('s3')

    # データ読み込み
    print("データを読み込み中...")

    try:
        # 特徴量データがある場合
        if args.features_path:
            features_key = args.features_path
        else:
            features_key = "output/features/confirmed_order_demand_yamasa_features_latest.parquet"

        response = s3_client.get_object(
            Bucket="fiby-yamasa-prediction",
            Key=features_key
        )
        df = pd.read_parquet(BytesIO(response['Body'].read()))
        print(f"特徴量データを読み込みました: {len(df)} レコード")

    except s3_client.exceptions.NoSuchKey:
        print(f"Warning: 特徴量データが見つかりません: {features_key}")
        print("入力データから特徴量を生成します...")

        # 入力データを読み込み
        if args.input_path:
            input_key = args.input_path
        else:
            input_key = "output/df_confirmed_order_input_yamasa_fill_zero.parquet"

        response = s3_client.get_object(
            Bucket="fiby-yamasa-prediction",
            Key=input_key
        )
        df = pd.read_parquet(BytesIO(response['Body'].read()))
        print(f"入力データを読み込みました: {len(df)} レコード")

        # 特徴量生成が必要な場合
        from modules.features.timeseries_features import add_timeseries_features
        print("特徴量を生成中...")
        df = add_timeseries_features(
            df,
            train_end_date=args.train_end_date,
            model_type="confirmed_order_demand_yamasa"
        )

    # Material Keyフィルタリング
    # テスト期間を動的に設定（train_end_dateの翌月）
    from dateutil.relativedelta import relativedelta
    train_end = pd.to_datetime(args.train_end_date)
    test_start = (train_end + relativedelta(days=1)).strftime('%Y-%m-%d')
    test_end = (train_end + relativedelta(months=1)).strftime('%Y-%m-%d')

    df_filtered, positive_counts_dict, over_threshold_mks, train_positive_counts_dict = filter_material_keys(
        df,
        min_positive_count=args.min_positive_count,
        period_start='2021-01-01',
        period_end='2024-12-31',
        test_start=test_start,
        test_end=test_end,
        min_test_positive=args.min_test_positive,
        train_end_date=args.train_end_date
    )

    # メモリ解放
    del df
    gc.collect()

    # モデル学習の実行
    print("\nモデル学習を開始します...")
    print(f"  学習データ終了日: {args.train_end_date}")
    print(f"  予測月数: {args.step_count}ヶ月")
    print(f"  ハイパーパラメータ最適化: {'有効' if args.use_optuna else '無効'}")
    print(f"  外れ値処理: {'無効' if args.no_outlier_handling else '有効'}")
    print()

    # TimeSeriesPredictorのインスタンス化
    predictor = TimeSeriesPredictor(
        bucket_name="fiby-yamasa-prediction",
        model_type="confirmed_order_demand_yamasa"
    )

    # 学習実行時刻を記録
    train_start_time = time.time()

    # train_test_predict_time_split 関数の新しいシグネチャに合わせて呼び出し
    df_pred_all, bykey_df, imp_last, best_params, model_last, metrics = predictor.train_test_predict_time_split(
        _df_features=df_filtered,
        train_end_date=args.train_end_date,
        step_count=args.step_count,
        target_col='actual_value',  # ヤマサデータのターゲット列
        use_optuna=args.use_optuna,
        n_trials=args.n_trials,
        apply_winsorization=not args.no_outlier_handling,
        apply_hampel=not args.no_outlier_handling,
        use_gpu=args.use_gpu,
        save_dir=args.save_dir,
        verbose=True
    )

    # 学習時間を計算
    train_time = time.time() - train_start_time

    # 評価の実行
    if model_last is not None:
        print("\n評価指標を計算中...")
        evaluator = ModelEvaluator(
            bucket_name="fiby-yamasa-prediction",
            model_type="confirmed_order_demand_yamasa"
        )

        # 予測結果に追加カラムを付与
        if not df_pred_all.empty:
            # is_over_48_threカラムを追加（48個以上の条件を満たすMaterial Keyか）
            df_pred_all['is_over_48_thre'] = df_pred_all['material_key'].apply(
                lambda x: 1 if x in over_threshold_mks else 0
            )

            # actual_value_countカラムを追加（該当Material Keyのpositive数）
            df_pred_all['actual_value_count'] = df_pred_all['material_key'].apply(
                lambda x: positive_counts_dict.get(x, 0)
            )

        # 評価結果の整形
        results = {
            'predictions': df_pred_all['predicted'].values if not df_pred_all.empty else [],
            'actuals': df_pred_all['actual'].values if not df_pred_all.empty else [],
            'dates': df_pred_all['date'].values if not df_pred_all.empty else [],
            'material_keys': df_pred_all['material_key'].values if not df_pred_all.empty else [],
            'metrics': metrics,
            'is_over_48_thre': df_pred_all['is_over_48_thre'].values if not df_pred_all.empty else [],
            'actual_value_count': df_pred_all['actual_value_count'].values if not df_pred_all.empty else []
        }

        evaluation = evaluator.evaluate_predictions(
            results=results,
            save_results=True,
            generate_plots=True,
            verbose=True,
            train_actual_counts=train_positive_counts_dict
        )

        # 評価サマリーの表示
        display_evaluation_summary(evaluation)

    # 実行時間の計算
    total_time = time.time() - start_time

    # 特徴量重要度の表示
    if not imp_last.empty:
        print("\n===== 特徴量重要度 Top 10 =====")
        print(imp_last.head(10))

    # Material Key別ワースト5
    if not bykey_df.empty:
        print("\n===== Material Key別ワースト5（RMSE） =====")
        worst5 = bykey_df.nlargest(5, 'RMSE')
        for _, row in worst5.iterrows():
            print(f"  {row['material_key']}: RMSE={row['RMSE']:.2f}, MAE={row['MAE']:.2f}")

    # 実行時間の表示
    print("\n" + "="*60)
    print(" 実行時間サマリー")
    print("="*60)
    print(f"学習時間: {train_time:.2f}秒 ({train_time/60:.2f}分)")
    print(f"総実行時間: {total_time:.2f}秒 ({total_time/60:.2f}分)")

    # 前回実行時間との比較（7時間20分 = 26400秒）
    previous_time = 26400
    speedup = previous_time / total_time
    print(f"\n前回実行時間: 26400秒 (440分)")
    print(f"今回実行時間: {total_time:.2f}秒 ({total_time/60:.2f}分)")
    print(f"高速化率: {speedup:.2f}倍")
    print(f"削減時間: {(previous_time - total_time):.2f}秒 ({(previous_time - total_time)/60:.2f}分)")

    print("\n学習が完了しました！")
    print("モデルと評価結果はS3に保存されました。")

if __name__ == "__main__":
    main()