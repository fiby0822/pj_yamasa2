#!/usr/bin/env python3
"""
Product_keyレベルでデータを準備（ローカル保存版）
work/data/inputからデータを読み込み、product_keyで集約
"""
import pandas as pd
import numpy as np
from datetime import datetime
import os
import glob

def main():
    print("="*60)
    print("Product-level Data Preparation (Local Save)")
    print(f"Timestamp: {datetime.now()}")
    print("="*60)

    # ローカルinputディレクトリからデータを読み込み
    input_dir = '/home/ubuntu/yamasa2/work/data/input'

    print(f"\n1. Reading data from local input directory...")
    print(f"   Source: {input_dir}")

    try:
        # inputディレクトリから最初のparquetファイルを探す
        parquet_files = glob.glob(f"{input_dir}/*.parquet")
        if parquet_files:
            input_file = parquet_files[0]
            print(f"   Found file: {os.path.basename(input_file)}")
            df = pd.read_parquet(input_file)
        else:
            # parquetファイルが無い場合はCSVを探す
            csv_files = glob.glob(f"{input_dir}/*.csv")
            if csv_files:
                input_file = csv_files[0]
                print(f"   Found file: {os.path.basename(input_file)}")
                df = pd.read_csv(input_file)
                # file_dateをdatetime型に変換
                if 'file_date' in df.columns:
                    df['file_date'] = pd.to_datetime(df['file_date'])
            else:
                raise FileNotFoundError(f"No input data found in {input_dir}. Please place .parquet or .csv file in the input directory.")
        print(f"   Loaded: {df.shape[0]:,} rows, {df.shape[1]} columns")
        print(f"   Date range: {df['file_date'].min()} to {df['file_date'].max()}")
        print(f"   Unique material_keys (original): {df['material_key'].nunique():,}")
        print(f"   Unique product_keys: {df['product_key'].nunique():,}")

        # product_keyレベルで集約
        print("\n2. Aggregating at product_key level...")

        # product_key × file_date でグループ化
        agg_df = df.groupby(['product_key', 'file_date']).agg({
            'actual_value': 'sum',  # 実績値の合計
        }).reset_index()

        # material_key = product_key とする
        agg_df['material_key'] = agg_df['product_key']

        # カラムの順序を整理
        agg_df = agg_df[['material_key', 'product_key', 'file_date', 'actual_value']]

        print(f"   Aggregated: {agg_df.shape[0]:,} rows")
        print(f"   Unique material_keys (product-level): {agg_df['material_key'].nunique():,}")

        # 統計情報
        print("\n3. Statistics:")
        print(f"   Total actual_value sum: {agg_df['actual_value'].sum():,.0f}")
        print(f"   Average daily value per product: {agg_df.groupby('material_key')['actual_value'].mean().mean():.2f}")
        print(f"   Zero value ratio: {(agg_df['actual_value'] == 0).sum() / len(agg_df) * 100:.1f}%")

        # product_keyの統計
        product_stats = agg_df.groupby('material_key')['actual_value'].agg(['mean', 'std', 'sum'])
        print("\n   Product statistics:")
        print(f"     Total products: {len(product_stats)}")
        print(f"     Top 10 products by volume: {product_stats['sum'].nlargest(10).sum() / product_stats['sum'].sum() * 100:.1f}% of total")

        # work/data/preparedディレクトリの作成
        output_dir = '/home/ubuntu/yamasa2/work/data/prepared'
        os.makedirs(output_dir, exist_ok=True)

        # 1. 欠損なしバージョン（集約のみ）
        print("\n4. Saving aggregated data locally...")
        local_path1 = f'{output_dir}/df_confirmed_order_input_yamasa.parquet'
        agg_df.to_parquet(local_path1, index=False)
        print(f"   Saved to: {local_path1}")
        print(f"   File size: {os.path.getsize(local_path1) / (1024*1024):.2f} MB")

        # 2. fill_zeroバージョン（すでに元データが0埋め済みなので同じ）
        local_path2 = f'{output_dir}/df_confirmed_order_input_yamasa_fill_zero.parquet'
        agg_df.to_parquet(local_path2, index=False)
        print(f"   Saved to: {local_path2}")
        print(f"   File size: {os.path.getsize(local_path2) / (1024*1024):.2f} MB")

        # サンプルデータ表示
        print("\n5. Sample data (first 5 rows):")
        print(agg_df.head())

        print("\n6. Data shape by year:")
        agg_df['year'] = pd.to_datetime(agg_df['file_date']).dt.year
        year_stats = agg_df.groupby('year').agg({
            'material_key': 'nunique',
            'actual_value': ['sum', 'mean']
        })
        print(year_stats)

        print("\n" + "="*60)
        print("Data preparation completed successfully!")
        print("="*60)
        print("\nNext steps:")
        print("1. Generate features using the prepared data")
        print("2. Train model with product-level aggregation")
        print("3. Make predictions and evaluate")

        return agg_df

    except Exception as e:
        print(f"\nError: {e}")
        raise

if __name__ == "__main__":
    df = main()