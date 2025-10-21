#!/usr/bin/env python3
"""
商品レベルの特徴量生成
material_key = product_key として処理
"""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import gc
import os
from typing import List, Dict, Tuple
import jpholiday

class ProductFeatureGenerator:
    def __init__(self, base_dir: str = "/home/ubuntu/yamasa2/work/data"):
        """
        初期化

        Args:
            base_dir: データの基本ディレクトリ
        """
        self.base_dir = base_dir
        print(f"Base directory: {self.base_dir}")

    def load_data(self, filename: str = "df_confirmed_order_input_yamasa_fill_zero.parquet"):
        """
        ローカルからデータを読み込み
        """
        # work/data/preparedディレクトリから読み込み
        local_path = os.path.join(self.base_dir, "prepared", filename)
        if os.path.exists(local_path):
            print(f"Loading data from local: {local_path}")
            df = pd.read_parquet(local_path)
        else:
            raise FileNotFoundError(f"Data file not found: {local_path}. Please run prepare_product_level_data_local.py first.")

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

        # day_of_week_f: カテゴリカル変数として曜日（1=月曜日、土日はnull）
        def get_weekday_business(date):
            """営業日の曜日を取得（1=月曜日、土日はnull）"""
            if pd.isna(date):
                return np.nan
            weekday = date.weekday()  # 0=月曜日
            if weekday >= 5:  # 土日
                return np.nan
            return weekday + 1  # 1=月曜日に変換

        df["day_of_week_f"] = df['file_date'].apply(get_weekday_business).astype("float32")

        # is_business_day_f: 営業日フラグ（拡張祝日判定）
        def is_business_day(date):
            """営業日判定（土日祝日以外）"""
            if pd.isna(date):
                return np.nan
            # 土日判定
            if date.weekday() >= 5:  # 5=土曜, 6=日曜
                return 0
            # 祝日判定
            if jpholiday.is_holiday(date):
                return 0
            # 年末年始判定（1/1~1/5と12/30~12/31）
            if (date.month == 1 and date.day <= 5) or (date.month == 12 and date.day >= 30):
                return 0
            return 1

        df['is_business_day_f'] = df['file_date'].apply(is_business_day).astype("int8")

        # 月・四半期特徴量
        df['month'] = df['file_date'].dt.month
        df['quarter'] = df['file_date'].dt.quarter
        df['year'] = df['file_date'].dt.year

        # 月初・月末フラグ
        df['is_month_start'] = (df['file_date'].dt.day <= 7).astype(int)
        df['is_month_end'] = (df['file_date'].dt.day >= 24).astype(int)

        print(f"Added basic features: dow, is_weekend, day_of_week_f, is_business_day_f, month, quarter, year, is_month_start, is_month_end")

        return df

    def generate_lag_features(self, df: pd.DataFrame, lags: List[int] = None) -> pd.DataFrame:
        """
        ラグ特徴量を生成（データリーク防止版）
        テストデータには学習期間のデータのみを使用

        Args:
            df: 入力データ（data_typeカラムが必要）
            lags: ラグのリスト

        Returns:
            ラグ特徴量を追加したデータ
        """
        if lags is None:
            lags = [1, 2, 3, 7, 14, 21, 28]

        print(f"\nGenerating lag features (NO LEAK): {lags}")

        # data_typeが存在しない場合はエラー
        if 'data_type' not in df.columns:
            raise ValueError("data_type column must exist. Run split_train_test first.")

        # material_key（product_key）でグループ化
        for lag in lags:
            print(f"  Creating lag_{lag} (leak-free)...")

            # 新しいカラムを初期化
            df[f'lag_{lag}_f'] = np.nan

            # 各商品ごとに処理
            for material_key in df['material_key'].unique():
                material_mask = df['material_key'] == material_key
                material_df = df[material_mask].copy()
                material_df = material_df.sort_values('file_date')

                # 学習データのラグ特徴量（通常のshift）
                train_mask = (material_df['data_type'] == 'train')
                if train_mask.any():
                    train_indices = material_df[train_mask].index
                    df.loc[train_indices, f'lag_{lag}_f'] = material_df[train_mask]['actual_value'].shift(lag).values

                # テストデータのラグ特徴量（学習期間の最後からのみ取得）
                test_mask = (material_df['data_type'] == 'test')
                if test_mask.any() and train_mask.any():
                    test_indices = material_df[test_mask].index
                    train_values = material_df[train_mask]['actual_value'].values
                    test_dates = material_df[test_mask]['file_date'].values

                    # テスト期間の各行について、学習期間の最後からlag分前の値を設定
                    for i, idx in enumerate(test_indices):
                        # 学習期間の最後からlag分遡った位置
                        source_position = len(train_values) - lag + i
                        if source_position >= 0 and source_position < len(train_values):
                            df.loc[idx, f'lag_{lag}_f'] = train_values[source_position]

        return df

    def generate_rolling_features(self, df: pd.DataFrame, windows: List[int] = None) -> pd.DataFrame:
        """
        移動平均・移動標準偏差特徴量を生成（データリーク防止版）
        テストデータには学習期間のデータのみを使用

        Args:
            df: 入力データ（data_typeカラムが必要）
            windows: ウィンドウサイズのリスト

        Returns:
            移動統計特徴量を追加したデータ
        """
        if windows is None:
            windows = [7, 14, 28]

        print(f"\nGenerating rolling features (NO LEAK): {windows}")

        # data_typeが存在しない場合はエラー
        if 'data_type' not in df.columns:
            raise ValueError("data_type column must exist. Run split_train_test first.")

        for window in windows:
            print(f"  Creating rolling_{window} features (leak-free)...")

            # 新しいカラムを初期化
            df[f'rolling_mean_{window}_f'] = np.nan
            df[f'rolling_std_{window}_f'] = np.nan
            df[f'rolling_max_{window}_f'] = np.nan
            df[f'rolling_min_{window}_f'] = np.nan

            # 各商品ごとに処理
            for material_key in df['material_key'].unique():
                material_mask = df['material_key'] == material_key
                material_df = df[material_mask].copy()
                material_df = material_df.sort_values('file_date')

                # 学習データの移動統計特徴量（通常のrolling）
                train_mask = (material_df['data_type'] == 'train')
                if train_mask.any():
                    train_indices = material_df[train_mask].index
                    train_values = material_df[train_mask]['actual_value'].values

                    # shift(1)を適用してからrollingを計算
                    shifted_values = pd.Series(train_values).shift(1)
                    rolling_mean = shifted_values.rolling(window=window, min_periods=1).mean()
                    rolling_std = shifted_values.rolling(window=window, min_periods=1).std()
                    rolling_max = shifted_values.rolling(window=window, min_periods=1).max()
                    rolling_min = shifted_values.rolling(window=window, min_periods=1).min()

                    df.loc[train_indices, f'rolling_mean_{window}_f'] = rolling_mean.values
                    df.loc[train_indices, f'rolling_std_{window}_f'] = rolling_std.values
                    df.loc[train_indices, f'rolling_max_{window}_f'] = rolling_max.values
                    df.loc[train_indices, f'rolling_min_{window}_f'] = rolling_min.values

                # テストデータの移動統計特徴量（学習期間の最後のwindow分のデータを使用）
                test_mask = (material_df['data_type'] == 'test')
                if test_mask.any() and train_mask.any():
                    test_indices = material_df[test_mask].index
                    train_values = material_df[train_mask]['actual_value'].values

                    # 学習期間最後のwindow分のデータで統計量を計算
                    if len(train_values) >= window:
                        # 学習期間最後のwindow分のデータ
                        last_window_values = train_values[-window:]

                        # テスト期間のすべての行に同じ値を設定
                        # (テスト期間内では更新されない）
                        test_mean = np.mean(last_window_values)
                        test_std = np.std(last_window_values)
                        test_max = np.max(last_window_values)
                        test_min = np.min(last_window_values)

                        df.loc[test_indices, f'rolling_mean_{window}_f'] = test_mean
                        df.loc[test_indices, f'rolling_std_{window}_f'] = test_std
                        df.loc[test_indices, f'rolling_max_{window}_f'] = test_max
                        df.loc[test_indices, f'rolling_min_{window}_f'] = test_min
                    else:
                        # ウィンドウサイズに満たない場合は、利用可能なデータで計算
                        if len(train_values) > 0:
                            test_mean = np.mean(train_values)
                            test_std = np.std(train_values) if len(train_values) > 1 else 0
                            test_max = np.max(train_values)
                            test_min = np.min(train_values)

                            df.loc[test_indices, f'rolling_mean_{window}_f'] = test_mean
                            df.loc[test_indices, f'rolling_std_{window}_f'] = test_std
                            df.loc[test_indices, f'rolling_max_{window}_f'] = test_max
                            df.loc[test_indices, f'rolling_min_{window}_f'] = test_min

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

        # 学習データのみから統計量を計算（データリーク防止）
        train_df = df[df['data_type'] == 'train']
        product_stats = train_df.groupby('material_key')['actual_value'].agg([
            'mean', 'std', 'median', 'min', 'max'
        ]).reset_index()

        product_stats.columns = ['material_key'] + [f'product_{col}_f' for col in product_stats.columns[1:]]

        # 全データ（train+test）にマージ
        df = pd.merge(df, product_stats, on='material_key', how='left')

        print(f"Added product profile features (calculated from {len(train_df)} train records)")

        return df

    def generate_trend_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        トレンド特徴量を生成（データリーク防止版）

        Args:
            df: 入力データ（data_typeカラムが必要）

        Returns:
            トレンド特徴量を追加したデータ
        """
        print("\nGenerating trend features (NO LEAK)...")

        # data_typeが存在しない場合はエラー
        if 'data_type' not in df.columns:
            raise ValueError("data_type column must exist. Run split_train_test first.")

        # 新しいカラムを初期化
        df['trend_7d_f'] = np.nan
        df['trend_28d_f'] = np.nan

        # 各商品ごとに処理
        for material_key in df['material_key'].unique():
            material_mask = df['material_key'] == material_key
            material_df = df[material_mask].copy()
            material_df = material_df.sort_values('file_date')

            # 学習データのトレンド特徴量
            train_mask = (material_df['data_type'] == 'train')
            if train_mask.any():
                train_indices = material_df[train_mask].index
                train_values = material_df[train_mask]['actual_value'].values

                # 7日前からの変化率
                trend_7d = pd.Series(train_values).pct_change(7).fillna(0).values
                df.loc[train_indices, 'trend_7d_f'] = trend_7d

                # 28日前からの変化率
                trend_28d = pd.Series(train_values).pct_change(28).fillna(0).values
                df.loc[train_indices, 'trend_28d_f'] = trend_28d

            # テストデータのトレンド特徴量（学習期間最後の値から計算）
            test_mask = (material_df['data_type'] == 'test')
            if test_mask.any() and train_mask.any():
                test_indices = material_df[test_mask].index
                train_values = material_df[train_mask]['actual_value'].values

                # 学習期間最後の7日間と28日間のトレンドを計算
                if len(train_values) >= 28:
                    # 7日前からの変化率
                    trend_7d_value = (train_values[-1] - train_values[-8]) / (train_values[-8] + 1) if len(train_values) >= 8 else 0
                    # 28日前からの変化率
                    trend_28d_value = (train_values[-1] - train_values[-29]) / (train_values[-29] + 1)

                    # テスト期間のすべての行に同じ値を設定
                    df.loc[test_indices, 'trend_7d_f'] = trend_7d_value
                    df.loc[test_indices, 'trend_28d_f'] = trend_28d_value
                elif len(train_values) > 0:
                    # データが不足している場合は0を設定
                    df.loc[test_indices, 'trend_7d_f'] = 0
                    df.loc[test_indices, 'trend_28d_f'] = 0

        return df

    def generate_dow_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        曜日別の特徴量を生成
        """
        print("\nGenerating day-of-week features...")

        # material_dow_mean_f: Material Key×曜日の過去平均
        df = self._create_entity_dow_mean(df, 'material_key', 'material_dow_mean_f')

        # store_code_dow_mean_f: Store Code×曜日の過去平均
        if 'store_code' in df.columns:
            df = self._create_entity_dow_mean(df, 'store_code', 'store_code_dow_mean_f')
        else:
            print("Warning: store_code column not found, skipping store_code_dow_mean_f")

        print("Day-of-week features completed")
        return df

    def _create_entity_dow_mean(self, df: pd.DataFrame, entity_col: str, feature_name: str) -> pd.DataFrame:
        """エンティティ×曜日の過去平均を計算（学習データのみ使用）"""
        print(f"  Creating {feature_name}...")

        # 学習データから統計を計算
        train_df = df[df['data_type'] == 'train']
        dow_stats = train_df.groupby([entity_col, 'day_of_week_f'])['actual_value'].mean().reset_index()
        dow_stats.columns = [entity_col, 'day_of_week_f', feature_name]

        # 全データにマージ
        df = pd.merge(df, dow_stats, on=[entity_col, 'day_of_week_f'], how='left')

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
        特徴量をローカルに保存
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # work/data/featuresに保存
        features_dir = os.path.join(self.base_dir, 'features')
        os.makedirs(features_dir, exist_ok=True)

        local_path_with_timestamp = os.path.join(features_dir, f"product_level_features_{timestamp}.parquet")
        local_path_latest = os.path.join(features_dir, "product_level_features_latest.parquet")

        for path in [local_path_with_timestamp, local_path_latest]:
            print(f"\nSaving to local: {path}")
            df.to_parquet(path, index=False)
            print(f"  File size: {os.path.getsize(path) / (1024*1024):.2f} MB")

        print(f"\nSaved successfully: {len(df)} rows, {df.shape[1]} columns")

    def run(self, train_end_date: str = "2024-12-31", step_count: int = 1):
        """
        メイン処理（データリーク防止版）

        Args:
            train_end_date: 学習データの終了日
            step_count: 予測月数
        """
        print("="*60)
        print("Product-level Feature Generation (No Data Leakage Version)")
        print(f"Timestamp: {datetime.now()}")
        print(f"Parameters: train_end_date={train_end_date}, step_count={step_count}")
        print("="*60)

        # 1. データ読み込み
        df = self.load_data()

        # 2. データ分割（最初に実行：データリーク防止の核心）
        df = self.split_train_test(df, train_end_date)

        # 統計情報の表示
        train_size = (df['data_type'] == 'train').sum()
        test_size = (df['data_type'] == 'test').sum()
        print(f"\nData split completed:")
        print(f"  Train: {train_size:,} records (~{train_end_date})")
        print(f"  Test:  {test_size:,} records ({train_end_date}~)")

        # 3. 基本特徴量
        df = self.generate_basic_features(df)

        # 4. ラグ特徴量
        df = self.generate_lag_features(df)

        # 5. 移動統計特徴量
        df = self.generate_rolling_features(df)

        # 6. 商品プロファイル特徴量（学習データのみから統計を計算）
        df = self.generate_product_profile_features(df)

        # 7. トレンド特徴量
        df = self.generate_trend_features(df)

        # 8. 曜日別特徴量（学習データのみから統計を計算）
        df = self.generate_dow_features(df)

        # 9. 保存
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
    import argparse

    parser = argparse.ArgumentParser(description='商品レベル特徴量生成')
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

    generator = ProductFeatureGenerator()
    df = generator.run(train_end_date=args.train_end_date, step_count=args.step_count)

    # 特徴量の統計情報
    print("\nFeature statistics:")
    feature_cols = [col for col in df.columns if col.endswith('_f')]
    print(df[feature_cols].describe().T.head(10))


if __name__ == "__main__":
    main()