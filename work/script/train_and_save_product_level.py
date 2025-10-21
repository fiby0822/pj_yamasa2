#!/usr/bin/env python3
"""
逐次更新対応パイプライン一括実行スクリプト

1. 特徴量生成 (`generate_features_product_level.py`)
2. モデル学習・逐次予測 (`train_model_product_level.py`)
3. 結果CSV整備 (`save_results_csv.py`)
"""

from __future__ import annotations

import argparse
from datetime import datetime

from generate_features_product_level import main as generate_features_main
from train_model_product_level import main as train_model_main
from save_results_csv import main as save_results_main


DEFAULT_BASE_DIR = "/home/ubuntu/yamasa2/work/data"
DEFAULT_VIS_DIR = "/home/ubuntu/yamasa2/work/vis_2024-12-31_3"


def main(
    base_dir: str,
    train_end_date: str,
    forecast_month_start: str,
    visualization_dir: str,
    skip_features: bool,
) -> None:
    print("=" * 60)
    print("Product-level Pipeline Orchestrator")
    print(f"Timestamp        : {datetime.now()}")
    print(f"Base directory   : {base_dir}")
    print(f"Train end date   : {train_end_date}")
    print(f"Visualization dir: {visualization_dir}")
    print(f"Forecast month start: {forecast_month_start if forecast_month_start else 'auto (next month)'}")
    print("=" * 60)

    if not skip_features:
        print("\n[Step 1/3] Generating features...")
        generate_features_main(base_dir=base_dir, train_end_date=train_end_date)
    else:
        print("\n[Step 1/3] Skipping feature generation (requested).")

    print("\n[Step 2/3] Training model & generating forecasts...")
    train_model_main(
        base_dir=base_dir,
        train_end_date=train_end_date,
        forecast_month_start=forecast_month_start,
        visualization_dir=visualization_dir,
        disable_visualization=False,
    )

    print("\n[Step 3/3] Creating CSV exports...")
    save_results_main(base_dir=base_dir, train_end_date=train_end_date, step_count=0)

    print("\nPipeline completed successfully.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the full product-level training pipeline.")
    parser.add_argument(
        "--base-dir",
        type=str,
        default=DEFAULT_BASE_DIR,
        help=f"データベースディレクトリ (default: {DEFAULT_BASE_DIR})",
    )
    parser.add_argument(
        "--train-end-date",
        type=str,
        default="2024-12-31",
        help="学習データの終了日 (YYYY-MM-DD)",
    )
    parser.add_argument(
        "--visualization-dir",
        type=str,
        default=DEFAULT_VIS_DIR,
        help="可視化出力先ディレクトリ",
    )
    parser.add_argument(
        "--forecast-month-start",
        type=str,
        default=None,
        help="予測対象月の開始 (YYYY-MM)。未指定の場合は train_end_date の翌月",
    )
    parser.add_argument(
        "--skip-features",
        action="store_true",
        help="既存の特徴量ファイルを再利用する場合に指定",
    )

    args = parser.parse_args()
    main(
        base_dir=args.base_dir,
        train_end_date=args.train_end_date,
        forecast_month_start=args.forecast_month_start,
        visualization_dir=args.visualization_dir,
        skip_features=args.skip_features,
    )
