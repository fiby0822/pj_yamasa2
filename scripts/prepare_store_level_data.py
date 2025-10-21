#!/usr/bin/env python3
"""
Store_codeレベルでデータを準備
元のyamasaプロジェクトのデータを読み込み、store_codeで集約
"""
import pandas as pd
import numpy as np
import boto3
from datetime import datetime
from io import BytesIO

def main():
    print("="*60)
    print("Store-level Data Preparation")
    print(f"Timestamp: {datetime.now()}")
    print("="*60)

    # S3クライアント
    s3_client = boto3.client('s3', region_name='ap-northeast-1')

    # 元のyamasaプロジェクトからデータを読み込み
    source_bucket = 'fiby-yamasa-prediction'
    source_key = 'output/df_confirmed_order_input_yamasa_fill_zero.parquet'

    print(f"\n1. Reading data from original yamasa project...")
    print(f"   Source: s3://{source_bucket}/{source_key}")

    try:
        response = s3_client.get_object(Bucket=source_bucket, Key=source_key)
        df = pd.read_parquet(BytesIO(response['Body'].read()))
        print(f"   Loaded: {df.shape[0]:,} rows, {df.shape[1]} columns")
        print(f"   Date range: {df['file_date'].min()} to {df['file_date'].max()}")
        print(f"   Unique material_keys (original): {df['material_key'].nunique():,}")
        print(f"   Unique store_codes: {df['store_code'].nunique():,}")

        # store_codeレベルで集約
        print("\n2. Aggregating at store_code level...")

        # store_code × file_date でグループ化
        agg_df = df.groupby(['store_code', 'file_date']).agg({
            'actual_value': 'sum',  # 実績値の合計
            'usage_type': 'first',   # usage_typeを保持（店舗ごとに一定のはず）
        }).reset_index()

        # material_key = store_code とする
        agg_df['material_key'] = agg_df['store_code']

        print(f"   Aggregated: {agg_df.shape[0]:,} rows")
        print(f"   Unique material_keys (store-level): {agg_df['material_key'].nunique():,}")

        # 統計情報
        print("\n3. Statistics:")
        print(f"   Total actual_value sum: {agg_df['actual_value'].sum():,.0f}")
        print(f"   Average daily value per store: {agg_df.groupby('material_key')['actual_value'].mean().mean():.2f}")
        print(f"   Zero value ratio: {(agg_df['actual_value'] == 0).sum() / len(agg_df) * 100:.1f}%")

        # usage_typeの分布
        usage_dist = agg_df.groupby('usage_type')['material_key'].nunique()
        print("\n   Usage type distribution:")
        for usage, count in usage_dist.items():
            print(f"     {usage}: {count} stores")

        # 新しいS3バケットに保存
        target_bucket = 'fiby-yamasa-prediction-2'

        # 1. 欠損なしバージョン（集約のみ）
        print("\n4. Saving aggregated data...")
        target_key1 = 'output/df_confirmed_order_input_yamasa.parquet'

        buffer = BytesIO()
        agg_df.to_parquet(buffer, index=False)
        buffer.seek(0)

        s3_client.put_object(
            Bucket=target_bucket,
            Key=target_key1,
            Body=buffer.getvalue()
        )
        print(f"   Saved to: s3://{target_bucket}/{target_key1}")

        # 2. fill_zeroバージョン（すでに元データが0埋め済みなので同じ）
        target_key2 = 'output/df_confirmed_order_input_yamasa_fill_zero.parquet'

        buffer = BytesIO()
        agg_df.to_parquet(buffer, index=False)
        buffer.seek(0)

        s3_client.put_object(
            Bucket=target_bucket,
            Key=target_key2,
            Body=buffer.getvalue()
        )
        print(f"   Saved to: s3://{target_bucket}/{target_key2}")

        # サンプルデータ表示
        print("\n5. Sample data (first 5 rows):")
        print(agg_df.head())

        print("\n" + "="*60)
        print("Data preparation completed successfully!")
        print("="*60)

        return agg_df

    except Exception as e:
        print(f"\nError: {e}")
        print("\nNote: Make sure you have access to both S3 buckets:")
        print("  - Source: fiby-yamasa-prediction (read access)")
        print("  - Target: fiby-yamasa-prediction-2 (write access)")
        raise

if __name__ == "__main__":
    df = main()