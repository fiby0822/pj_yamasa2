#!/usr/bin/env python3
"""
データ準備処理（store_codeレベル）
material_key = store_code として処理
"""
import pandas as pd
import numpy as np
import boto3
from datetime import datetime, timedelta
from io import BytesIO
import sys

class DataPreparation:
    def __init__(self, s3_bucket: str = "fiby-yamasa-prediction-2"):
        """
        初期化

        Args:
            s3_bucket: S3バケット名（必ずfiby-yamasa-prediction-2を使用）
        """
        self.s3_bucket = s3_bucket
        self.s3_client = boto3.client('s3', region_name='ap-northeast-1')
        print(f"S3バケット: {self.s3_bucket}")

    def load_raw_data(self, key: str = "input_data/df_confirmed_order_raw_yamasa.parquet"):
        """
        S3から生データを読み込み
        """
        print(f"Loading data from s3://{self.s3_bucket}/{key}")

        response = self.s3_client.get_object(Bucket=self.s3_bucket, Key=key)
        df = pd.read_parquet(BytesIO(response['Body'].read()))

        print(f"Loaded data shape: {df.shape}")
        print(f"Columns: {df.columns.tolist()}")

        return df

    def prepare_store_level_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        store_codeレベルでデータを準備

        Args:
            df: 生データ

        Returns:
            store_codeレベルで集約されたデータ
        """
        print("\nPreparing store-level data...")

        # 必要なカラムの確認
        required_cols = ['store_code', 'file_date', 'actual_value']
        for col in required_cols:
            if col not in df.columns:
                print(f"Warning: {col} not found in data")

        # store_codeをmaterial_keyとして使用
        df['material_key'] = df['store_code']

        # store_code × file_date でグループ化して集約
        print("Aggregating by store_code and file_date...")
        agg_df = df.groupby(['material_key', 'file_date']).agg({
            'actual_value': 'sum',  # 実績値の合計
            'store_code': 'first'    # store_codeを保持
        }).reset_index()

        # usage_typeの追加（store_codeから推定）
        # 例：A で始まる店舗は household、B で始まる店舗は business など
        # 実際のロジックはデータに応じて調整が必要
        agg_df['usage_type'] = agg_df['store_code'].apply(
            lambda x: 'household' if x.startswith('A') else 'business'
        )

        print(f"Aggregated data shape: {agg_df.shape}")
        print(f"Unique stores: {agg_df['material_key'].nunique()}")
        print(f"Date range: {agg_df['file_date'].min()} to {agg_df['file_date'].max()}")

        return agg_df

    def fill_missing_dates(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        欠損日付を0で埋める
        """
        print("\nFilling missing dates with zero...")

        # 日付範囲の取得
        date_min = pd.to_datetime(df['file_date'].min())
        date_max = pd.to_datetime(df['file_date'].max())

        # 全日付のリスト作成
        date_range = pd.date_range(start=date_min, end=date_max, freq='D')

        # 全store × 全日付の組み合わせを作成
        all_stores = df['material_key'].unique()

        # MultiIndexを作成
        idx = pd.MultiIndex.from_product(
            [all_stores, date_range],
            names=['material_key', 'file_date']
        )

        # 新しいDataFrameを作成
        df_full = pd.DataFrame(index=idx).reset_index()
        df_full['file_date'] = df_full['file_date'].dt.strftime('%Y-%m-%d')

        # 元のデータとマージ
        df_merged = pd.merge(
            df_full,
            df,
            on=['material_key', 'file_date'],
            how='left'
        )

        # 欠損値を0で埋める
        df_merged['actual_value'] = df_merged['actual_value'].fillna(0)

        # store_codeとusage_typeを埋める
        for col in ['store_code', 'usage_type']:
            if col in df_merged.columns:
                df_merged[col] = df_merged.groupby('material_key')[col].transform(
                    lambda x: x.fillna(method='ffill').fillna(method='bfill')
                )

        print(f"Filled data shape: {df_merged.shape}")
        print(f"Zero values: {(df_merged['actual_value'] == 0).sum()}/{len(df_merged)} "
              f"({(df_merged['actual_value'] == 0).sum()/len(df_merged)*100:.1f}%)")

        return df_merged

    def save_to_s3(self, df: pd.DataFrame, key: str):
        """
        データをS3に保存
        """
        print(f"\nSaving to s3://{self.s3_bucket}/{key}")

        buffer = BytesIO()
        df.to_parquet(buffer, index=False)
        buffer.seek(0)

        self.s3_client.put_object(
            Bucket=self.s3_bucket,
            Key=key,
            Body=buffer.getvalue()
        )

        print(f"Saved successfully: {len(df)} rows")

    def run(self):
        """
        メイン処理
        """
        print("="*60)
        print("Data Preparation for Store-level Prediction")
        print(f"Timestamp: {datetime.now()}")
        print("="*60)

        # 1. データ読み込み
        df_raw = self.load_raw_data()

        # 2. store_codeレベルで集約
        df_store = self.prepare_store_level_data(df_raw)

        # 3. 欠損日付を埋める前のデータを保存
        self.save_to_s3(
            df_store,
            "output/df_confirmed_order_input_yamasa.parquet"
        )

        # 4. 欠損日付を0で埋める
        df_filled = self.fill_missing_dates(df_store)

        # 5. 埋めた後のデータを保存
        self.save_to_s3(
            df_filled,
            "output/df_confirmed_order_input_yamasa_fill_zero.parquet"
        )

        print("\n" + "="*60)
        print("Data preparation completed successfully!")
        print("="*60)

        return df_filled


def main():
    """メイン処理"""
    prep = DataPreparation(s3_bucket="fiby-yamasa-prediction-2")
    df = prep.run()

    # サマリー表示
    print("\nSummary Statistics:")
    print(f"- Total records: {len(df):,}")
    print(f"- Unique stores: {df['material_key'].nunique():,}")
    print(f"- Date range: {df['file_date'].min()} to {df['file_date'].max()}")
    print(f"- Average daily value per store: {df.groupby('material_key')['actual_value'].mean().mean():.2f}")


if __name__ == "__main__":
    main()