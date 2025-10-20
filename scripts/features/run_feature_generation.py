#!/usr/bin/env python3
"""
特徴量生成の実行スクリプト（S3統合版）
"""
import pandas as pd
import argparse
from datetime import datetime
from modules.features.feature_generator_with_s3 import FeatureGeneratorWithS3
from modules.config.feature_window_config import WINDOW_SIZE_CONFIG


def main():
    """メイン処理"""
    parser = argparse.ArgumentParser(description='Generate features and save to S3')
    parser.add_argument('--input-key', type=str, required=True,
                        help='S3 input file key (e.g., data/input/raw_data.parquet)')
    parser.add_argument('--output-key', type=str,
                        help='S3 output file key (default: auto-generated)')
    parser.add_argument('--model-type', type=str,
                        default='confirmed_order_demand_yamasa',
                        choices=['confirmed_order_demand_yamasa', 'unofficial', 'use_actual_value_by_category'],
                        help='Model type for feature generation')
    parser.add_argument('--start-year', type=int, default=2021,
                        help='Start year for output data')
    parser.add_argument('--end-year', type=int, default=2025,
                        help='End year for output data')
    parser.add_argument('--train-end-date', type=str, default='2024-12-31',
                        help='Training end date (format: YYYY-MM-DD)')
    parser.add_argument('--save-format', type=str, default='parquet',
                        choices=['parquet', 'csv'],
                        help='Output file format')
    parser.add_argument('--batch-process', action='store_true',
                        help='Process all model types at once')

    args = parser.parse_args()

    # 特徴量生成器の初期化
    generator = FeatureGeneratorWithS3()

    print(f"Loading data from S3: {args.input_key}")

    # S3からデータを読み込み
    try:
        if args.input_key.endswith('.parquet'):
            df_input = generator.load_data_from_s3(args.input_key, file_type='parquet')
        elif args.input_key.endswith('.xlsx'):
            df_input = generator.load_data_from_s3(args.input_key, file_type='excel')
        else:
            raise ValueError(f"Unsupported input file format: {args.input_key}")

        print(f"Loaded data shape: {df_input.shape}")

    except Exception as e:
        print(f"Error loading data: {e}")
        return 1

    # バッチ処理の場合
    if args.batch_process:
        print("\nRunning batch processing for all model types...")

        results = generator.process_batch_models(
            df_input=df_input,
            start_year=args.start_year,
            end_year=args.end_year,
            train_end_date=args.train_end_date
        )

        print("\n" + "="*50)
        print("Batch processing completed!")
        print("="*50)
        for model_type, df in results.items():
            print(f"{model_type}: {df.shape}")

    # 単一モデル処理の場合
    else:
        # 出力キーの生成（指定された場合のみ使用）
        output_key = args.output_key if args.output_key else None

        print(f"\nGenerating features for model: {args.model_type}")
        print(f"Parameters:")
        print(f"  - Start year: {args.start_year}")
        print(f"  - End year: {args.end_year}")
        print(f"  - Train end date: {args.train_end_date}")
        print(f"  - Output format: {args.save_format}")

        # 特徴量生成と保存
        df_features = generator.generate_and_save_features(
            df_input=df_input,
            output_key=output_key,
            model_type=args.model_type,
            start_year=args.start_year,
            end_year=args.end_year,
            train_end_date=args.train_end_date,
            save_format=args.save_format
        )

        print(f"\nProcessing completed!")
        print(f"Output shape: {df_features.shape}")

        # 特徴量のサマリー表示
        feature_cols = [col for col in df_features.columns if col.endswith('_f')]
        print(f"\nGenerated {len(feature_cols)} feature columns")
        print("Sample features:")
        for col in feature_cols[:10]:
            null_pct = df_features[col].isna().sum() / len(df_features) * 100
            print(f"  - {col}: {null_pct:.1f}% null")

    return 0


if __name__ == "__main__":
    exit(main())