#!/usr/bin/env python3
"""
usage_type別の特徴量生成スクリプト
- businessとhouseholdで別々に特徴量を生成
- category_lvl1関連の特徴量は生成しない
- 各usage_type用の特徴量ファイルを出力
"""
import pandas as pd
import numpy as np
from datetime import datetime
import boto3
import sys
import os
import argparse
from typing import Optional, Tuple

# プロジェクトルートをPythonパスに追加
sys.path.append('/home/ubuntu/yamasa')

from modules.features.feature_generator_with_s3 import FeatureGeneratorWithS3
from modules.features.timeseries_features import add_timeseries_features
from modules.config.feature_window_config import WINDOW_SIZE_CONFIG


def load_input_data(s3_bucket: str = "fiby-yamasa-prediction-2") -> pd.DataFrame:
    """
    S3から入力データを読み込み

    Returns:
        pd.DataFrame: 入力データ
    """
    print("Loading input data from S3...")
    s3 = boto3.client('s3')

    # 最新の処理済みデータを読み込み
    input_key = "output/processed/confirmed_order_demand_yamasa_processed_latest.parquet"

    try:
        response = s3.get_object(Bucket=s3_bucket, Key=input_key)
        df = pd.read_parquet(response['Body'])
        print(f"Loaded {len(df):,} records from S3")
        return df
    except Exception as e:
        print(f"Error loading data from S3: {e}")
        # ローカルファイルから読み込み（フォールバック）
        local_path = "/home/ubuntu/yamasa/data/features/confirmed_order_demand_yamasa_features_20251016_020011.parquet"
        if os.path.exists(local_path):
            print(f"Loading from local file: {local_path}")
            df = pd.read_parquet(local_path)
            print(f"Loaded {len(df):,} records from local file")
            return df
        else:
            raise Exception("Could not load data from S3 or local file")


def generate_features_for_usage_type(
    df_input: pd.DataFrame,
    usage_type: str,
    start_year: int = 2021,
    end_year: int = 2025,
    train_end_date: str = "2024-12-31",
    save_local: bool = True
) -> Tuple[pd.DataFrame, str]:
    """
    特定のusage_typeに対して特徴量を生成

    Args:
        df_input: 入力データ
        usage_type: 'business' or 'household'
        start_year: 開始年
        end_year: 終了年
        train_end_date: 学習データの終了日
        save_local: ローカルにも保存するか

    Returns:
        (特徴量DataFrame, 出力ファイルパス)
    """
    print(f"\n{'='*70}")
    print(f"Generating features for {usage_type.upper()}")
    print('='*70)

    # usage_typeでフィルタリング
    df_usage = df_input[df_input['usage_type'] == usage_type].copy()
    print(f"Filtered data: {len(df_usage):,} records for {usage_type}")

    # category_lvl1関連の列を除外（混在を避けるため）
    if 'category_lvl1' in df_usage.columns:
        print("Dropping category_lvl1 column to avoid mixing")
        df_usage = df_usage.drop(columns=['category_lvl1'])
    if 'category_lvl2' in df_usage.columns:
        df_usage = df_usage.drop(columns=['category_lvl2'])
    if 'category_lvl3' in df_usage.columns:
        df_usage = df_usage.drop(columns=['category_lvl3'])

    # 特徴量生成
    print(f"Generating features for {usage_type}...")
    df_features = add_timeseries_features(
        df_usage,
        window_size_config=WINDOW_SIZE_CONFIG,
        start_year=start_year,
        end_year=end_year,
        model_type="confirmed_order_demand_yamasa",
        train_end_date=train_end_date
    )

    # category_lvl1関連の特徴量が生成されていないことを確認
    cat_features = [col for col in df_features.columns if 'category_lvl1' in col.lower()]
    if cat_features:
        print(f"Removing {len(cat_features)} category_lvl1 related features: {cat_features}")
        df_features = df_features.drop(columns=cat_features)

    # usage_typeカラムを確実に保持
    df_features['usage_type'] = usage_type

    print(f"Generated features shape: {df_features.shape}")
    print(f"Memory usage: {df_features.memory_usage(deep=True).sum() / (1024**3):.2f} GB")

    # タイムスタンプ
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # S3に保存
    s3_client = boto3.client('s3')
    bucket_name = "fiby-yamasa-prediction-2"

    # タイムスタンプ付きファイル
    s3_key = f"output/features/confirmed_order_demand_yamasa_features_{usage_type}_{timestamp}.parquet"

    # parquetとして保存
    from io import BytesIO
    buffer = BytesIO()
    df_features.to_parquet(buffer, index=False)
    buffer.seek(0)

    s3_client.put_object(
        Bucket=bucket_name,
        Key=s3_key,
        Body=buffer.getvalue()
    )
    print(f"Saved to S3: s3://{bucket_name}/{s3_key}")

    # latestファイルも作成
    latest_key = f"output/features/confirmed_order_demand_yamasa_features_{usage_type}_latest.parquet"
    s3_client.put_object(
        Bucket=bucket_name,
        Key=latest_key,
        Body=buffer.getvalue()
    )
    print(f"Saved latest to S3: s3://{bucket_name}/{latest_key}")

    # ローカルにも保存
    if save_local:
        local_dir = f"/home/ubuntu/yamasa/data/features"
        os.makedirs(local_dir, exist_ok=True)

        local_path = f"{local_dir}/confirmed_order_demand_yamasa_features_{usage_type}_{timestamp}.parquet"
        df_features.to_parquet(local_path, index=False)
        print(f"Saved to local: {local_path}")

        # latestのシンボリックリンク
        latest_local = f"{local_dir}/confirmed_order_demand_yamasa_features_{usage_type}_latest.parquet"
        if os.path.exists(latest_local):
            os.remove(latest_local)
        df_features.to_parquet(latest_local, index=False)
        print(f"Saved latest to local: {latest_local}")

    return df_features, s3_key


def main():
    """メイン処理"""
    parser = argparse.ArgumentParser(description='Generate features by usage_type')
    parser.add_argument('--start-year', type=int, default=2021,
                        help='Start year for output data')
    parser.add_argument('--end-year', type=int, default=2025,
                        help='End year for output data')
    parser.add_argument('--train-end-date', type=str, default="2024-12-31",
                        help='Training data end date')
    parser.add_argument('--usage-types', nargs='+', default=['business', 'household'],
                        help='Usage types to process')
    parser.add_argument('--save-local', action='store_true', default=True,
                        help='Save to local file system as well')

    args = parser.parse_args()

    print(f"""
    ============================================================
    usage_type別特徴量生成
    ============================================================
    開始年: {args.start_year}
    終了年: {args.end_year}
    学習終了日: {args.train_end_date}
    対象usage_type: {args.usage_types}
    ローカル保存: {args.save_local}
    """)

    # 入力データの読み込み
    df_input = load_input_data()

    # usage_typeの確認
    print(f"\nAvailable usage_types in data: {df_input['usage_type'].unique().tolist()}")

    # 各usage_typeで特徴量生成
    results = {}
    for usage_type in args.usage_types:
        if usage_type not in df_input['usage_type'].unique():
            print(f"Warning: {usage_type} not found in data, skipping...")
            continue

        df_features, s3_key = generate_features_for_usage_type(
            df_input=df_input,
            usage_type=usage_type,
            start_year=args.start_year,
            end_year=args.end_year,
            train_end_date=args.train_end_date,
            save_local=args.save_local
        )

        results[usage_type] = {
            'shape': df_features.shape,
            's3_key': s3_key,
            'feature_columns': [col for col in df_features.columns if col.endswith('_f')][:10]
        }

    # サマリー表示
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)

    for usage_type, info in results.items():
        print(f"\n{usage_type.upper()}:")
        print(f"  Shape: {info['shape']}")
        print(f"  S3 Key: {info['s3_key']}")
        print(f"  Sample features: {info['feature_columns'][:5]}")

    print("\n✅ Feature generation completed successfully!")

    return results


if __name__ == "__main__":
    main()