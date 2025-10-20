#!/usr/bin/env python3
"""
ローカルの特徴量ファイルを使用したモデル学習実行スクリプト
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
    parser = argparse.ArgumentParser(description='ヤマサ確定注文需要予測モデルの学習')
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
    parser.add_argument('--features-path', type=str, required=True,
                        help='ローカル特徴量ファイルパス')
    parser.add_argument('--save-dir', type=str, default=None,
                        help='モデル保存先のS3パス')

    args = parser.parse_args()

    print("="*60)
    print(" ヤマサ確定注文需要予測 - モデル学習（ローカル特徴量使用）")
    print("="*60)
    print()

    # 予測器の初期化
    predictor = TimeSeriesPredictor(
        bucket_name="fiby-yamasa-prediction-2",
        model_type="confirmed_order_demand_yamasa"
    )

    # データの読み込み
    print("データを読み込み中...")

    # ローカルファイルから読み込み
    features_path = Path(args.features_path)
    if not features_path.exists():
        print(f"Error: 特徴量ファイルが見つかりません: {features_path}")
        return 1

    df = pd.read_parquet(features_path)
    print(f"特徴量データを読み込みました: {len(df):,} レコード")
    print(f"カラム数: {len(df.columns)}")
    print(f"Material Key数: {df['material_key'].nunique():,}")

    # モデル学習の実行
    print("\nモデル学習を開始します...")
    print(f"  学習データ終了日: {args.train_end_date}")
    print(f"  予測月数: {args.step_count}ヶ月")
    print(f"  ハイパーパラメータ最適化: {'有効' if args.use_optuna else '無効'}")
    print(f"  外れ値処理: {'有効' if args.enable_outlier_handling else '無効'}")
    if args.use_optuna:
        print(f"  Optuna試行回数: {args.n_trials}")
    print()

    # モデル学習と予測
    predictions_df, metrics_by_material_key, feature_importance, best_params, model_last, metrics = predictor.train_test_predict_time_split(
        _df_features=df,
        train_end_date=args.train_end_date,
        step_count=args.step_count,
        use_optuna=args.use_optuna,
        n_trials=args.n_trials,
        apply_winsorization=args.enable_outlier_handling,
        apply_hampel=args.enable_outlier_handling,
        use_gpu=args.use_gpu
    )

    # 結果の保存
    print("\n結果を保存中...")

    # ローカルに保存
    output_dir = Path("data/output")
    output_dir.mkdir(parents=True, exist_ok=True)

    # 予測結果を保存
    predictions_path = output_dir / "confirmed_order_demand_yamasa_predictions_latest.parquet"
    predictions_df.to_parquet(predictions_path)
    print(f"  予測結果: {predictions_path}")

    # Material Key別のメトリクスを保存
    if metrics_by_material_key is not None:
        metrics_path = output_dir / "confirmed_order_demand_yamasa_material_summary_latest.parquet"
        metrics_by_material_key.to_parquet(metrics_path)
        print(f"  Material Key別メトリクス: {metrics_path}")

    # 特徴量重要度を保存
    if feature_importance is not None:
        importance_path = output_dir / "feature_importance_latest.parquet"
        feature_importance.to_parquet(importance_path)
        print(f"  特徴量重要度: {importance_path}")

    # S3にも保存（オプション）
    if args.save_dir:
        print(f"\nS3にアップロード中: {args.save_dir}")
        predictor.save_results(
            predictions_df=predictions_df,
            metrics_by_material_key=metrics_by_material_key,
            feature_importance=feature_importance,
            save_key_prefix=args.save_dir
        )

    # 評価サマリーの表示
    print("\n" + "="*60)
    print(" モデル評価サマリー")
    print("="*60)

    evaluator = ModelEvaluator()
    metrics = evaluator.evaluate_model(predictions_df)
    display_evaluation_summary(metrics, predictions_df, metrics_by_material_key)

    print("\n学習・予測が完了しました。")
    return 0

if __name__ == "__main__":
    exit(main())