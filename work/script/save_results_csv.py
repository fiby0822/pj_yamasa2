#!/usr/bin/env python3
"""
Product_keyレベルの結果をyamasaプロジェクトと同じ形式のCSVで保存
"""
import os
from datetime import datetime

import numpy as np
import pandas as pd


EPSILON = 1e-6


def _ensure_output_dir(base_dir: str) -> str:
    output_dir = os.path.join(base_dir, "output")
    os.makedirs(output_dir, exist_ok=True)
    return output_dir


def create_predictions_csv(base_dir: str = "/home/ubuntu/yamasa2/work/data") -> str | None:
    """予測詳細から集計済みCSVを作成する。"""
    pred_path = os.path.join(base_dir, "predictions", "product_level_predictions_latest.parquet")
    if not os.path.exists(pred_path):
        print(f"Prediction file not found: {pred_path}")
        return None

    df = pd.read_parquet(pred_path)
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"])

    grouped = []
    for material_key, group in df.groupby("material_key"):
        actual_sum = float(group["actual"].sum())
        predicted_sum = float(group["predicted"].sum())
        mae = float(np.mean(np.abs(group["predicted"] - group["actual"])))
        rmse = float(np.sqrt(np.mean((group["predicted"] - group["actual"]) ** 2)))
        smape = float(
            (
                2 * np.abs(group["predicted"] - group["actual"])
                / (np.abs(group["predicted"]) + np.abs(group["actual"]) + EPSILON)
            ).mean()
            * 100
        )
        accuracy = float(100 - smape)
        wape = float(np.sum(np.abs(group["predicted"] - group["actual"])) / (np.sum(np.abs(group["actual"])) + EPSILON))
        grouped.append(
            {
                "material_key": material_key,
                "actual_sum": actual_sum,
                "predicted_sum": predicted_sum,
                "mae": mae,
                "rmse": rmse,
                "mape": smape,
                "accuracy": accuracy,
                "wape": wape,
                "n_samples": int(len(group)),
            }
        )

    df_pred = pd.DataFrame(grouped).sort_values("accuracy", ascending=False)
    output_dir = _ensure_output_dir(base_dir)
    csv_path = os.path.join(output_dir, "confirmed_order_demand_yamasa_predictions_latest.csv")
    df_pred.to_csv(csv_path, index=False)
    print(f"Saved predictions CSV: {csv_path}")
    return csv_path


def create_material_summary_csv(base_dir: str = "/home/ubuntu/yamasa2/work/data") -> str | None:
    """既存のサマリーパケットをCSVとして保存する。"""
    summary_path = os.path.join(base_dir, "predictions", "product_level_summary_latest.parquet")
    if not os.path.exists(summary_path):
        print(f"Summary file not found: {summary_path}")
        return None

    df_summary = pd.read_parquet(summary_path)
    output_dir = _ensure_output_dir(base_dir)
    csv_path = os.path.join(output_dir, "confirmed_order_demand_yamasa_material_summary_latest.csv")
    df_summary.to_csv(csv_path, index=False)
    print(f"Saved material summary CSV: {csv_path}")
    return csv_path


def create_feature_importance_csv(base_dir: str = "/home/ubuntu/yamasa2/work/data") -> str | None:
    """特徴量重要度をコピー出力する。"""
    fi_path = os.path.join(base_dir, "models", "feature_importance_latest.csv")
    if not os.path.exists(fi_path):
        print(f"Feature importance file not found: {fi_path}")
        return None

    df_fi = pd.read_csv(fi_path)
    output_dir = _ensure_output_dir(base_dir)
    csv_path = os.path.join(output_dir, "confirmed_order_demand_yamasa_feature_importance_latest.csv")
    df_fi.to_csv(csv_path, index=False)
    print(f"Saved feature importance CSV: {csv_path}")
    return csv_path


def main(base_dir: str = "/home/ubuntu/yamasa2/work/data", train_end_date="2024-12-31", step_count=1):
    """メイン処理"""
    print("="*60)
    print("Creating CSV files for Product-level results")
    print(f"Timestamp: {datetime.now()}")
    print(f"Parameters: train_end_date={train_end_date}, step_count={step_count}")
    print("="*60)

    # 各CSVファイルを作成
    print("\n1. Creating predictions CSV...")
    pred_csv = create_predictions_csv(base_dir)

    print("\n2. Creating material summary CSV...")
    summary_csv = create_material_summary_csv(base_dir)

    print("\n3. Creating feature importance CSV...")
    fi_csv = create_feature_importance_csv(base_dir)

    print("\n✅ All CSV files have been created!")
    print("\nLocal files:")
    output_dir = os.path.join(base_dir, "output")
    print(f"  {os.path.join(output_dir, 'confirmed_order_demand_yamasa_predictions_latest.csv')}")
    print(f"  {os.path.join(output_dir, 'confirmed_order_demand_yamasa_material_summary_latest.csv')}")
    print(f"  {os.path.join(output_dir, 'confirmed_order_demand_yamasa_feature_importance_latest.csv')}")

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Creating CSV files for Product-level results')
    parser.add_argument(
        '--base_dir',
        type=str,
        default='/home/ubuntu/yamasa2/work/data',
        help='データディレクトリのパス (デフォルト: /home/ubuntu/yamasa2/work/data)'
    )
    parser.add_argument(
        '--train_end_date',
        type=str,
        default='2024-12-31',
        help='学習データの終了日 (YYYY-MM-DD形式、デフォルト: 2024-12-31)'
    )
    parser.add_argument(
        '--step_count',
        type=int,
        default=1,
        help='予測月数 (デフォルト: 1)'
    )
    args = parser.parse_args()

    main(base_dir=args.base_dir, train_end_date=args.train_end_date, step_count=args.step_count)
