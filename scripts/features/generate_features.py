#!/usr/bin/env python3
"""
特徴量生成のメインスクリプト（正規版）
- 通常の全データ版とusage_type別版を統合
- デフォルトはusage_type別での生成
"""
import pandas as pd
import numpy as np
from datetime import datetime
import boto3
import sys
import os
import argparse
from typing import Optional, Tuple, List

# プロジェクトルートをPythonパスに追加
sys.path.append('/home/ubuntu/yamasa')

from modules.features.feature_generator_with_s3 import FeatureGeneratorWithS3
from modules.features.timeseries_features import add_timeseries_features
from modules.config.feature_window_config import WINDOW_SIZE_CONFIG


class FeatureGenerator:
    """特徴量生成の統合クラス"""

    def __init__(self, s3_bucket: str = "fiby-yamasa-prediction"):
        self.s3_bucket = s3_bucket
        self.s3_client = boto3.client('s3', region_name='ap-northeast-1')

    def load_input_data(self) -> pd.DataFrame:
        """
        S3から入力データを読み込み

        Returns:
            pd.DataFrame: 入力データ
        """
        print("Loading input data from S3...")

        # ゼロ埋め済みのデータを読み込み
        input_key = "output/df_confirmed_order_input_yamasa_fill_zero.parquet"

        try:
            local_path = f"/tmp/{input_key.split('/')[-1]}"
            self.s3_client.download_file(self.s3_bucket, input_key, local_path)
            df = pd.read_parquet(local_path)
            print(f"Loaded {len(df):,} records from S3")
            return df
        except Exception as e:
            print(f"Error loading data from S3: {e}")
            raise

    def generate_features_for_usage_type(
        self,
        df: pd.DataFrame,
        usage_type: str,
        train_end_date: str
    ) -> pd.DataFrame:
        """
        特定のusage_typeに対して特徴量を生成

        Args:
            df: 入力データ
            usage_type: 'business' or 'household'
            train_end_date: 学習データの終了日

        Returns:
            pd.DataFrame: 特徴量付きデータ
        """
        print(f"\n{'='*60}")
        print(f"Generating features for {usage_type.upper()}")
        print(f"{'='*60}")

        # usage_typeでフィルタリング
        df_usage = df[df['usage_type'] == usage_type].copy()
        print(f"Records for {usage_type}: {len(df_usage):,}")

        if len(df_usage) == 0:
            print(f"No data for {usage_type}")
            return pd.DataFrame()

        # file_dateをdatetime型に変換
        df_usage['file_date'] = pd.to_datetime(df_usage['file_date'])

        # 基本的な時系列特徴量を追加
        print("Adding time series features...")
        df_with_features = add_timeseries_features(
            df_usage,
            target_column='actual_value',
            date_column='file_date',
            group_columns=['material_key'],
            train_end_date=train_end_date
        )

        # 追加の特徴量生成
        print("Adding additional features...")

        # 曜日特徴量
        df_with_features['day_of_week_f'] = df_with_features['file_date'].dt.dayofweek
        df_with_features['is_weekend_f'] = (df_with_features['day_of_week_f'] >= 5).astype(int)
        df_with_features['is_business_day_f'] = (~df_with_features['is_weekend_f']).astype(int)

        # 月関連の特徴量
        df_with_features['month_f'] = df_with_features['file_date'].dt.month
        df_with_features['quarter_f'] = df_with_features['file_date'].dt.quarter
        df_with_features['year_f'] = df_with_features['file_date'].dt.year
        df_with_features['day_of_month_f'] = df_with_features['file_date'].dt.day
        df_with_features['week_of_year_f'] = df_with_features['file_date'].dt.isocalendar().week

        # container特徴量（usage_typeに基づく）
        if usage_type == 'business':
            df_with_features['container_f'] = 1  # businessは全て1
        else:
            df_with_features['container_f'] = 0  # householdは全て0

        # 曜日のone-hot encoding
        for i in range(7):
            df_with_features[f'day_of_week_{i}_f'] = (df_with_features['day_of_week_f'] == i).astype(int)

        # グループ統計量の追加
        for group in ['material_key', 'store_code', 'product_key']:
            if group in df_with_features.columns:
                print(f"  Adding {group} statistics...")

                # 週平均
                weekly_mean = df_with_features.groupby([group, df_with_features['file_date'].dt.to_period('W')])['actual_value'].mean()
                weekly_mean = weekly_mean.reset_index(name=f'{group}_weekly_mean')
                weekly_mean['file_date'] = weekly_mean['file_date'].dt.to_timestamp()

                # 月平均
                monthly_mean = df_with_features.groupby([group, df_with_features['file_date'].dt.to_period('M')])['actual_value'].mean()
                monthly_mean = monthly_mean.reset_index(name=f'{group}_monthly_mean')
                monthly_mean['file_date'] = monthly_mean['file_date'].dt.to_timestamp()

        print(f"Feature generation complete. Total features: {len([c for c in df_with_features.columns if c.endswith('_f')])}")

        return df_with_features

    def save_features_to_s3(
        self,
        df: pd.DataFrame,
        usage_type: Optional[str] = None
    ) -> str:
        """
        特徴量データをS3に保存

        Args:
            df: 特徴量付きデータ
            usage_type: usage_typeの名前（Noneの場合は全体）

        Returns:
            str: 保存したS3パス
        """
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

        if usage_type:
            # usage_type別のファイル名
            filename = f"confirmed_order_demand_yamasa_features_{usage_type}_{timestamp}.parquet"
            latest_filename = f"confirmed_order_demand_yamasa_features_{usage_type}_latest.parquet"
            s3_key = f"output/features/{filename}"
            latest_s3_key = f"output/features/{latest_filename}"

            # ローカル保存も行う
            local_dir = "/home/ubuntu/yamasa/data/features"
            os.makedirs(local_dir, exist_ok=True)
            local_path = f"{local_dir}/{filename}"
            latest_local_path = f"{local_dir}/{latest_filename}"
        else:
            # 全体のファイル名
            filename = f"confirmed_order_demand_yamasa_features_{timestamp}.parquet"
            latest_filename = f"confirmed_order_demand_yamasa_features_latest.parquet"
            s3_key = f"output/features/{filename}"
            latest_s3_key = f"output/features/{latest_filename}"
            local_path = f"/tmp/{filename}"
            latest_local_path = f"/tmp/{latest_filename}"

        # ローカルに保存
        df.to_parquet(local_path, index=False, compression='snappy')

        # S3にアップロード（タイムスタンプ付き）
        self.s3_client.upload_file(local_path, self.s3_bucket, s3_key)
        print(f"Saved to S3: s3://{self.s3_bucket}/{s3_key}")

        # latest版もアップロード
        self.s3_client.upload_file(local_path, self.s3_bucket, latest_s3_key)
        print(f"Saved to S3: s3://{self.s3_bucket}/{latest_s3_key}")

        # usage_type別の場合はローカルにもlatest版を保存
        if usage_type:
            df.to_parquet(latest_local_path, index=False, compression='snappy')
            print(f"Saved locally: {latest_local_path}")

        return s3_key

    def generate_features(
        self,
        mode: str = "by_usage_type",
        train_end_date: str = "2024-12-31"
    ):
        """
        特徴量生成のメイン処理

        Args:
            mode: "by_usage_type" or "all"
            train_end_date: 学習データの終了日
        """
        print(f"\n{'='*80}")
        print(f"Feature Generation - Mode: {mode}")
        print(f"Train end date: {train_end_date}")
        print(f"{'='*80}")

        # データ読み込み
        df = self.load_input_data()

        if mode == "by_usage_type":
            # usage_type別に処理
            usage_types = df['usage_type'].unique()
            print(f"Found usage types: {usage_types}")

            for usage_type in usage_types:
                # 特徴量生成
                df_features = self.generate_features_for_usage_type(
                    df, usage_type, train_end_date
                )

                if not df_features.empty:
                    # S3に保存
                    self.save_features_to_s3(df_features, usage_type)

                    # メモリ解放
                    del df_features

        else:
            # 全データで処理
            print("Generating features for all data...")
            df['file_date'] = pd.to_datetime(df['file_date'])

            df_features = add_timeseries_features(
                df,
                target_column='actual_value',
                date_column='file_date',
                group_columns=['material_key'],
                train_end_date=train_end_date
            )

            # 基本特徴量追加
            df_features['day_of_week_f'] = df_features['file_date'].dt.dayofweek
            df_features['month_f'] = df_features['file_date'].dt.month
            df_features['year_f'] = df_features['file_date'].dt.year

            # S3に保存
            self.save_features_to_s3(df_features, None)

        print("\n" + "="*80)
        print("Feature generation completed successfully!")
        print("="*80)


def main():
    """メイン処理"""
    parser = argparse.ArgumentParser(description='特徴量生成スクリプト')
    parser.add_argument(
        '--mode',
        type=str,
        choices=['by_usage_type', 'all'],
        default='by_usage_type',
        help='生成モード: by_usage_type（デフォルト）またはall'
    )
    parser.add_argument(
        '--train-end-date',
        type=str,
        default='2024-12-31',
        help='学習データの終了日（YYYY-MM-DD）'
    )
    args = parser.parse_args()

    # 特徴量生成実行
    generator = FeatureGenerator()
    generator.generate_features(
        mode=args.mode,
        train_end_date=args.train_end_date
    )


if __name__ == "__main__":
    main()