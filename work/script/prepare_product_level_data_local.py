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


def _create_full_calendar(df: pd.DataFrame) -> pd.DataFrame:
    """material_keyごとに日次カレンダーを生成し、欠損日を0で補完する。"""
    calendar_frames = []

    for material_key, group in df.groupby('material_key'):
        group = group.sort_values('file_date')
        start_date = group['file_date'].min()
        end_date = group['file_date'].max()
        full_dates = pd.date_range(start=start_date, end=end_date, freq='D')

        calendar = (
            pd.DataFrame({'file_date': full_dates})
            .merge(group[['file_date', 'actual_value']], on='file_date', how='left')
        )
        calendar['material_key'] = material_key
        calendar['product_key'] = group['product_key'].iloc[0]
        calendar['actual_value'] = calendar['actual_value'].fillna(0).astype(np.float32)

        calendar_frames.append(calendar[['material_key', 'product_key', 'file_date', 'actual_value']])

    full_calendar_df = pd.concat(calendar_frames, ignore_index=True)
    full_calendar_df.sort_values(['material_key', 'file_date'], inplace=True)
    return full_calendar_df

def main(train_end_date="2024-12-31", step_count=1):
    print("="*60)
    print("Product-level Data Preparation (Local Save)")
    print(f"Timestamp: {datetime.now()}")
    print(f"Parameters: train_end_date={train_end_date}, step_count={step_count}")
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

        agg_df = (
            df.groupby(['product_key', 'file_date'])['actual_value']
            .sum()
            .reset_index()
        )
        agg_df['material_key'] = agg_df['product_key']
        agg_df = agg_df[['material_key', 'product_key', 'file_date', 'actual_value']]
        print(f"   Aggregated (observed days only): {agg_df.shape[0]:,} rows")

        # 日次カレンダーを生成し欠損日を0で補完
        print("\n3. Creating full daily calendar per product (fill zeros)...")
        calendar_df = _create_full_calendar(agg_df)
        print(f"   Calendarized rows: {len(calendar_df):,}")
        print(f"   Date range after calendar: {calendar_df['file_date'].min()} to {calendar_df['file_date'].max()}")
        print(f"   Zero value ratio: {(calendar_df['actual_value'] == 0).sum() / len(calendar_df) * 100:.1f}%")

        # 統計情報
        print("\n4. Statistics (calendarized):")
        print(f"   Total actual_value sum: {calendar_df['actual_value'].sum():,.0f}")
        print(f"   Average daily value per product: {calendar_df.groupby('material_key')['actual_value'].mean().mean():.2f}")

        product_stats = calendar_df.groupby('material_key')['actual_value'].agg(['mean', 'std', 'sum'])
        print("\n   Product statistics:")
        print(f"     Total products: {len(product_stats)}")
        print(f"     Top 10 products by volume: {product_stats['sum'].nlargest(10).sum() / product_stats['sum'].sum() * 100:.1f}% of total")

        # work/data/preparedディレクトリの作成
        output_dir = '/home/ubuntu/yamasa2/work/data/prepared'
        os.makedirs(output_dir, exist_ok=True)

        # 1. 欠損なしバージョン（集約のみ）
        print("\n5. Saving aggregated data locally...")
        local_path1 = f'{output_dir}/df_confirmed_order_input_yamasa.parquet'
        agg_df.to_parquet(local_path1, index=False)
        print(f"   Saved to: {local_path1}")
        print(f"   File size: {os.path.getsize(local_path1) / (1024*1024):.2f} MB")

        # 2. fill_zeroバージョン（カレンダー補完結果）
        local_path2 = f'{output_dir}/df_confirmed_order_input_yamasa_fill_zero.parquet'
        calendar_df.to_parquet(local_path2, index=False)
        print(f"   Saved to: {local_path2}")
        print(f"   File size: {os.path.getsize(local_path2) / (1024*1024):.2f} MB")

        # サンプルデータ表示
        print("\n6. Sample data (first 5 rows):")
        print(calendar_df.head())

        print("\n7. Data shape by year:")
        calendar_df['year'] = pd.to_datetime(calendar_df['file_date']).dt.year
        year_stats = calendar_df.groupby('year').agg({
            'material_key': 'nunique',
            'actual_value': ['sum', 'mean']
        })
        print(year_stats)
        calendar_df = calendar_df.drop(columns=['year'])

        print("\n" + "="*60)
        print("Data preparation completed successfully!")
        print("="*60)
        print("\nNext steps:")
        print("1. Generate features using the prepared data")
        print("2. Train model with product-level aggregation")
        print("3. Make predictions and evaluate")

        return calendar_df

    except Exception as e:
        print(f"\nError: {e}")
        raise

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Product-level Data Preparation (Local Save)')
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

    df = main(train_end_date=args.train_end_date, step_count=args.step_count)
