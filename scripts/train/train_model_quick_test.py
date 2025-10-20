#!/usr/bin/env python3
"""
モデル学習実行スクリプト（クイックテスト版）
"""
import sys
import argparse
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[2]))

import pandas as pd
from modules.models.train_predict import TimeSeriesPredictor
from modules.models.predictor import DemandPredictor
from modules.evaluation.metrics import ModelEvaluator, display_evaluation_summary

def main():
    """メイン処理"""
    print("="*60)
    print(" ヤマサ確定注文需要予測 - モデル学習（クイックテスト）")
    print("="*60)
    print()

    # 予測器の初期化（学習用と読み込み用を分ける）
    train_predictor = TimeSeriesPredictor(
        bucket_name="fiby-yamasa-prediction",
        model_type="confirmed_order_demand_yamasa"
    )

    # データ読み込み用
    predictor = DemandPredictor(
        bucket_name="fiby-yamasa-prediction",
        model_type="confirmed_order_demand_yamasa"
    )

    # データの読み込み
    print("データを読み込み中...")

    # 特徴量の読み込み（最新の特徴量を使用）
    features_path = "output/features/confirmed_order_demand_yamasa_features_latest.parquet"
    df = predictor.load_features_from_s3(features_path)

    # データをサンプリング（高速化のため）
    print("データをサンプリング中...")
    # material_keyごとに最新の3ヶ月分のデータのみを使用
    df['file_date'] = pd.to_datetime(df['file_date'])
    df = df[df['file_date'] >= '2024-10-01']

    # 学習パラメータ
    train_end_date = "2024-12-31"
    step_count = 1

    print(f"\n学習パラメータ:")
    print(f"  学習データ終了日: {train_end_date}")
    print(f"  予測月数: {step_count}")
    print(f"  データ期間: 2024-10-01 ~ ")
    print(f"  レコード数: {len(df)}")

    # モデル学習・予測
    print("\nモデル学習を開始...")
    df_pred_all, bykey_df, imp, best_params, model, metrics = train_predictor.train_test_predict_time_split(
        _df_features=df,
        train_end_date=train_end_date,
        step_count=step_count,
        use_optuna=False,  # 高速化のため無効化
        n_trials=10,
        apply_winsorization=True,
        apply_hampel=True,
        use_gpu=False,
        save_dir=None,
        verbose=True
    )

    print("\n" + "="*60)
    print(" 学習結果サマリー")
    print("="*60)

    # 評価結果の表示
    if len(df_pred_all) > 0:
        display_evaluation_summary(df_pred_all, metrics)

        # 簡易的な精度表示
        print("\n予測精度（2025年1月）:")
        print(f"  MAPE: {metrics.get('overall', {}).get('mape', 0):.2f}%")
        print(f"  RMSE: {metrics.get('overall', {}).get('rmse', 0):.2f}")
        print(f"  R²: {metrics.get('overall', {}).get('r2', 0):.4f}")

    print("\n学習完了！")
    return 0

if __name__ == '__main__':
    sys.exit(main())