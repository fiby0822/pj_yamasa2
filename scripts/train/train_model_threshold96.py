#!/usr/bin/env python3
"""
モデル学習実行スクリプト（閾値96版）
テスト期間での実績発生数が96件以上のmaterial_keyを対象とする
"""
import sys
import argparse
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[2]))

import pandas as pd
from modules.models.train_predict import TimeSeriesPredictor
from modules.evaluation.metrics import ModelEvaluator, display_evaluation_summary

def main():
    """メイン処理"""
    parser = argparse.ArgumentParser(description='ヤマサ確定注文需要予測モデルの学習（閾値96版）')
    parser.add_argument('--train-end-date', type=str, default='2024-12-31',
                        help='学習データの終了日 (YYYY-MM-DD)')
    parser.add_argument('--step-count', type=int, default=1,
                        help='予測する月数')
    parser.add_argument('--use-optuna', action='store_true',
                        help='Optunaでハイパーパラメータ最適化を実行')
    parser.add_argument('--n-trials', type=int, default=50,
                        help='Optunaの試行回数')
    parser.add_argument('--enable-outlier-handling', action='store_true',
                        help='外れ値処理を有効化（デフォルト: 無効）')
    parser.add_argument('--use-gpu', action='store_true',
                        help='GPU使用')
    parser.add_argument('--features-path', type=str, default=None,
                        help='特徴量ファイルのS3パス')
    parser.add_argument('--save-dir', type=str, default=None,
                        help='モデル保存先のS3パス')

    args = parser.parse_args()

    print("="*60)
    print(" ヤマサ確定注文需要予測 - モデル学習（閾値96版）")
    print("="*60)
    print()

    # 予測器の初期化
    predictor = TimeSeriesPredictor(
        bucket_name="fiby-yamasa-prediction",
        model_type="confirmed_order_demand_yamasa"
    )

    # データの読み込み
    print("データを読み込み中...")

    if args.features_path:
        # 特徴量生成済みデータを使用
        features_key = args.features_path
    else:
        # デフォルトの最新特徴量データ
        features_key = "output/features/confirmed_order_demand_yamasa_features_latest.parquet"

    try:
        # S3からデータを読み込み
        import boto3
        from io import BytesIO

        s3_client = boto3.client('s3')
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

    # モデル学習の実行
    print("\nモデル学習を開始します...")
    print(f"  学習データ終了日: {args.train_end_date}")
    print(f"  予測月数: {args.step_count}ヶ月")
    print(f"  実績発生数の閾値: 96")
    print(f"  ハイパーパラメータ最適化: {'有効' if args.use_optuna else '無効'}")
    print(f"  外れ値処理: {'有効' if args.enable_outlier_handling else '無効'}")
    print()

    # train_test_predict_time_split 関数の新しいシグネチャに合わせて呼び出し
    df_pred_all, bykey_df, imp_last, best_params, model_last, metrics = predictor.train_test_predict_time_split(
        _df_features=df,
        train_end_date=args.train_end_date,
        step_count=args.step_count,
        target_col='actual_value',  # ヤマサデータのターゲット列
        use_optuna=args.use_optuna,
        n_trials=args.n_trials,
        apply_winsorization=args.enable_outlier_handling,
        apply_hampel=args.enable_outlier_handling,
        use_gpu=args.use_gpu,
        save_dir=args.save_dir if args.save_dir else "output/models/threshold96",
        verbose=True,
        min_test_active_records=96  # 閾値を96に設定
    )

    # 評価の実行
    if len(df_pred_all) > 0:
        print("\n評価指標を計算中...")

        # 学習期間内の実績発生数を計算（actual_value_count_in_train_periodのため）
        df_train = df[(df['file_date'] <= args.train_end_date)]
        train_positive_counts = df_train[df_train['actual_value'] > 0].groupby('material_key').size()
        train_positive_counts_dict = train_positive_counts.to_dict()

        # 結果を評価用の形式に変換
        results_for_eval = {
            'predictions': df_pred_all['predicted'].tolist(),
            'actuals': df_pred_all['actual'].tolist(),
            'dates': df_pred_all['date'].tolist(),
            'material_keys': df_pred_all['material_key'].tolist(),
            'metrics': metrics.get('overall', {})
        }

        evaluator = ModelEvaluator(
            bucket_name="fiby-yamasa-prediction",
            model_type="confirmed_order_demand_yamasa"
        )

        evaluation = evaluator.evaluate_predictions(
            results=results_for_eval,
            save_results=True,
            generate_plots=True,
            verbose=True,
            train_actual_counts=train_positive_counts_dict,
            feature_importance=imp_last,
            output_prefix="threshold96"  # 出力ファイルのプレフィックス
        )

        # サマリー表示
        display_evaluation_summary(evaluation)

        # 評価結果をファイルに保存
        import json
        evaluation_path = "output/evaluation/threshold96_metrics.json"
        with open(f"/tmp/{evaluation_path.split('/')[-1]}", 'w') as f:
            json.dump(evaluation['metrics'], f, indent=2)

        # S3にアップロード
        s3_client.put_object(
            Bucket="fiby-yamasa-prediction",
            Key=evaluation_path,
            Body=json.dumps(evaluation['metrics'], indent=2)
        )
        print(f"\n評価結果を保存しました: {evaluation_path}")

        # 特徴量重要度の表示
        if imp_last is not None:
            print("\n===== 特徴量重要度 Top 10 =====")
            print(imp_last.head(10))

        # Material Key別のワースト表示
        if not bykey_df.empty:
            print("\n===== Material Key別ワースト5（RMSE） =====")
            worst = bykey_df.nlargest(5, 'RMSE')
            for _, row in worst.iterrows():
                print(f"  {row['material_key']}: RMSE={row['RMSE']:.2f}, MAE={row['MAE']:.2f}")

        print("\n学習が完了しました！（閾値96版）")
        print("モデルと評価結果はS3に保存されました。")

    else:
        print("\nWarning: 予測結果がないため評価をスキップしました")

    return 0

if __name__ == "__main__":
    exit(main())