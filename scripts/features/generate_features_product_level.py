#!/usr/bin/env python3
"""
商品レベルの特徴量生成
material_key = product_key として処理
"""
import pandas as pd
import numpy as np
import boto3
from datetime import datetime, timedelta
from io import BytesIO
import gc
import os
from typing import List, Dict, Tuple

class ProductFeatureGenerator:
    def __init__(self, s3_bucket: str = "fiby-yamasa-prediction-2"):
        """
        初期化

        Args:
            s3_bucket: S3バケット名（必ずfiby-yamasa-prediction-2を使用）
        """
        self.s3_bucket = s3_bucket
        self.s3_client = boto3.client('s3', region_name='ap-northeast-1')
        print(f"S3バケット: {self.s3_bucket}")

    def load_data(self, key: str = "output/df_confirmed_order_input_yamasa_fill_zero.parquet"):
        """
        ローカルまたはS3からデータを読み込み
        """
        # まずローカルファイルをチェック
        local_path = "/home/ubuntu/yamasa2/data/prepared/df_confirmed_order_input_yamasa_fill_zero.parquet"
        if os.path.exists(local_path):
            print(f"Loading data from local: {local_path}")
            df = pd.read_parquet(local_path)
        else:
            print(f"Loading data from s3://{self.s3_bucket}/{key}")
            response = self.s3_client.get_object(Bucket=self.s3_bucket, Key=key)
            df = pd.read_parquet(BytesIO(response['Body'].read()))

        print(f"Loaded data shape: {df.shape}")

        # 日付型に変換
        df['file_date'] = pd.to_datetime(df['file_date'])

        return df

    def generate_basic_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        基本的な特徴量を生成

        Args:
            df: 入力データ

        Returns:
            特徴量を追加したデータ
        """
        print("\nGenerating basic features...")

        # 曜日特徴量
        df['dow'] = df['file_date'].dt.dayofweek
        df['is_weekend'] = (df['dow'] >= 5).astype(int)

        # 月・四半期特徴量
        df['month'] = df['file_date'].dt.month
        df['quarter'] = df['file_date'].dt.quarter
        df['year'] = df['file_date'].dt.year

        # 月初・月末フラグ
        df['is_month_start'] = (df['file_date'].dt.day <= 7).astype(int)
        df['is_month_end'] = (df['file_date'].dt.day >= 24).astype(int)

        print(f"Added basic features: dow, is_weekend, month, quarter, year, is_month_start, is_month_end")

        return df

    def generate_lag_features(self, df: pd.DataFrame, lags: List[int] = None) -> pd.DataFrame:
        """
        ラグ特徴量を生成

        Args:
            df: 入力データ
            lags: ラグのリスト

        Returns:
            ラグ特徴量を追加したデータ
        """
        if lags is None:
            lags = [1, 2, 3, 7, 14, 21, 28]

        print(f"\nGenerating lag features: {lags}")

        # material_key（product_key）でグループ化
        for lag in lags:
            print(f"  Creating lag_{lag}...")
            df[f'lag_{lag}_f'] = df.groupby('material_key')['actual_value'].shift(lag)

        return df

    def generate_rolling_features(self, df: pd.DataFrame, windows: List[int] = None) -> pd.DataFrame:
        """
        移動平均・移動標準偏差特徴量を生成

        Args:
            df: 入力データ
            windows: ウィンドウサイズのリスト

        Returns:
            移動統計特徴量を追加したデータ
        """
        if windows is None:
            windows = [7, 14, 28]

        print(f"\nGenerating rolling features: {windows}")

        for window in windows:
            print(f"  Creating rolling_{window} features...")

            # 移動平均
            df[f'rolling_mean_{window}_f'] = df.groupby('material_key')['actual_value'].transform(
                lambda x: x.shift(1).rolling(window=window, min_periods=1).mean()
            )

            # 移動標準偏差
            df[f'rolling_std_{window}_f'] = df.groupby('material_key')['actual_value'].transform(
                lambda x: x.shift(1).rolling(window=window, min_periods=1).std()
            )

            # 移動最大値
            df[f'rolling_max_{window}_f'] = df.groupby('material_key')['actual_value'].transform(
                lambda x: x.shift(1).rolling(window=window, min_periods=1).max()
            )

            # 移動最小値
            df[f'rolling_min_{window}_f'] = df.groupby('material_key')['actual_value'].transform(
                lambda x: x.shift(1).rolling(window=window, min_periods=1).min()
            )

        return df

    def generate_product_profile_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        商品プロファイル特徴量を生成

        Args:
            df: 入力データ

        Returns:
            商品プロファイル特徴量を追加したデータ
        """
        print("\nGenerating product profile features...")

        # 各商品の全期間統計（学習データのみから計算すべき）
        product_stats = df.groupby('material_key')['actual_value'].agg([
            'mean', 'std', 'median', 'min', 'max'
        ]).reset_index()

        product_stats.columns = ['material_key'] + [f'product_{col}_f' for col in product_stats.columns[1:]]

        # マージ
        df = pd.merge(df, product_stats, on='material_key', how='left')

        print(f"Added product profile features")

        return df

    def generate_trend_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        トレンド特徴量を生成

        Args:
            df: 入力データ

        Returns:
            トレンド特徴量を追加したデータ
        """
        print("\nGenerating trend features...")

        # 7日前からの変化率
        df['trend_7d_f'] = df.groupby('material_key').apply(
            lambda x: (x['actual_value'] - x['actual_value'].shift(7)) / (x['actual_value'].shift(7) + 1)
        ).reset_index(level=0, drop=True)

        # 28日前からの変化率
        df['trend_28d_f'] = df.groupby('material_key').apply(
            lambda x: (x['actual_value'] - x['actual_value'].shift(28)) / (x['actual_value'].shift(28) + 1)
        ).reset_index(level=0, drop=True)

        return df

    def split_train_test(self, df: pd.DataFrame, train_end_date: str) -> pd.DataFrame:
        """
        学習データとテストデータを分割

        Args:
            df: 入力データ
            train_end_date: 学習データの終了日

        Returns:
            data_typeカラムを追加したデータ
        """
        print(f"\nSplitting train/test data (train_end_date: {train_end_date})")

        train_end = pd.to_datetime(train_end_date)
        df['data_type'] = df['file_date'].apply(
            lambda x: 'train' if x <= train_end else 'test'
        )

        train_size = (df['data_type'] == 'train').sum()
        test_size = (df['data_type'] == 'test').sum()

        print(f"Train size: {train_size:,} ({train_size/len(df)*100:.1f}%)")
        print(f"Test size: {test_size:,} ({test_size/len(df)*100:.1f}%)")

        return df

    def save_features(self, df: pd.DataFrame):
        """
        特徴量をローカルおよびS3に保存
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # ローカルに保存
        os.makedirs('/home/ubuntu/yamasa2/data/features', exist_ok=True)

        local_path_with_timestamp = f"/home/ubuntu/yamasa2/data/features/product_level_features_{timestamp}.parquet"
        local_path_latest = f"/home/ubuntu/yamasa2/data/features/product_level_features_latest.parquet"

        for path in [local_path_with_timestamp, local_path_latest]:
            print(f"\nSaving to local: {path}")
            df.to_parquet(path, index=False)
            print(f"  File size: {os.path.getsize(path) / (1024*1024):.2f} MB")

        print(f"\nSaved successfully: {len(df)} rows, {df.shape[1]} columns")

        # S3にアップロードするためのスクリプトを別途作成予定
        print("\nNote: S3へのアップロードは別途upload_to_s3.pyスクリプトを使用してください")

    def run(self, train_end_date: str = "2024-12-31"):
        """
        メイン処理

        Args:
            train_end_date: 学習データの終了日
        """
        print("="*60)
        print("Product-level Feature Generation")
        print(f"Timestamp: {datetime.now()}")
        print("="*60)

        # 1. データ読み込み
        df = self.load_data()

        # 2. 基本特徴量
        df = self.generate_basic_features(df)

        # 3. ラグ特徴量
        df = self.generate_lag_features(df)

        # 4. 移動統計特徴量
        df = self.generate_rolling_features(df)

        # 5. 商品プロファイル特徴量
        df = self.generate_product_profile_features(df)

        # 6. トレンド特徴量
        df = self.generate_trend_features(df)

        # 7. 学習/テスト分割
        df = self.split_train_test(df, train_end_date)

        # 8. 保存
        self.save_features(df)

        # サマリー
        print("\n" + "="*60)
        print("Feature Generation Summary")
        print("="*60)

        feature_cols = [col for col in df.columns if col.endswith('_f')]
        print(f"Total features: {len(feature_cols)}")
        print(f"Feature columns: {feature_cols[:10]}...")

        return df


def main():
    """メイン処理"""
    generator = ProductFeatureGenerator(s3_bucket="fiby-yamasa-prediction-2")
    df = generator.run(train_end_date="2024-12-31")

    # 特徴量の統計情報
    print("\nFeature statistics:")
    feature_cols = [col for col in df.columns if col.endswith('_f')]
    print(df[feature_cols].describe().T.head(10))


if __name__ == "__main__":
    main()