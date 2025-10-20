#!/usr/bin/env python3
"""
モデル学習・予測のメインスクリプト（正規版）
- 通常版とusage_type別版を統合
- デフォルトはusage_type別での学習・予測
"""
import sys
sys.path.append('/home/ubuntu/yamasa')

import pandas as pd
import numpy as np
import argparse
from datetime import datetime
import boto3
import json
import os
import gc
import warnings
warnings.filterwarnings('ignore')

from modules.models.train_predict import ModelTrainer
from modules.evaluation.metrics import ModelEvaluator


class UnifiedModelTrainer:
    """統合モデル学習クラス"""

    def __init__(self, s3_bucket: str = "fiby-yamasa-prediction-2"):
        self.s3_bucket = s3_bucket
        self.s3_client = boto3.client('s3', region_name='ap-northeast-1')

    def load_features(self, usage_type: str = None) -> pd.DataFrame:
        """
        特徴量データを読み込み

        Args:
            usage_type: 'business', 'household', またはNone（全体）

        Returns:
            pd.DataFrame: 特徴量データ
        """
        if usage_type:
            # usage_type別のファイル
            local_path = f"/home/ubuntu/yamasa/data/features/confirmed_order_demand_yamasa_features_{usage_type}_latest.parquet"
            if os.path.exists(local_path):
                print(f"Loading features for {usage_type} from: {local_path}")
                df = pd.read_parquet(local_path)
                print(f"  Loaded {len(df):,} records, {df['material_key'].nunique():,} material keys")
                return df

            # S3から読み込み
            s3_path = f"output/features/confirmed_order_demand_yamasa_features_{usage_type}_latest.parquet"
            local_temp = f"/tmp/features_{usage_type}.parquet"
            try:
                self.s3_client.download_file(self.s3_bucket, s3_path, local_temp)
                df = pd.read_parquet(local_temp)
                print(f"  Loaded {len(df):,} records from S3")
                return df
            except:
                raise FileNotFoundError(f"Features not found for {usage_type}")

        else:
            # 全体のファイル
            s3_path = "output/features/confirmed_order_demand_yamasa_features_latest.parquet"
            local_temp = "/tmp/features_all.parquet"
            try:
                self.s3_client.download_file(self.s3_bucket, s3_path, local_temp)
                df = pd.read_parquet(local_temp)
                print(f"Loaded {len(df):,} records (all data)")
                return df
            except:
                raise FileNotFoundError("Features not found")

    def filter_material_keys(
        self,
        df: pd.DataFrame,
        train_end_date: str,
        step_count: int,
        usage_type: str = None
    ) -> pd.DataFrame:
        """
        Material Keyのフィルタリング

        Args:
            df: 特徴量データ
            train_end_date: 学習終了日
            step_count: 予測月数
            usage_type: usage_type名

        Returns:
            pd.DataFrame: フィルタリング後のデータ
        """
        print(f"\n{'='*5} Material Keyフィルタリング {'='*5}")

        df['file_date'] = pd.to_datetime(df['file_date'])
        train_end = pd.to_datetime(train_end_date)

        # 学習期間とテスト期間を分離
        df_train = df[df['file_date'] <= train_end]
        df_test = df[df['file_date'] > train_end]

        print(f"フィルタリング前: {len(df):,}行, {df['material_key'].nunique():,} Material Keys")

        # 学習期間での実績発生数でTop N個を選択
        if usage_type == 'business':
            top_n = 3000
            min_test_count = step_count * 2  # businessは最低要件を緩める
        elif usage_type == 'household':
            top_n = 3500
            min_test_count = step_count * 4
        else:
            top_n = 7000
            min_test_count = step_count * 4

        # 学習期間での実績発生数を計算
        train_counts = df_train[df_train['actual_value'] > 0].groupby('material_key').size()
        top_keys_train = set(train_counts.nlargest(top_n).index)

        print(f"  実績発生数上位{top_n:,}個のMaterial Key: {len(top_keys_train):,}個")

        # 統計情報
        total_actual = df_train[df_train['actual_value'] > 0].shape[0]
        top_actual = df_train[(df_train['material_key'].isin(top_keys_train)) &
                              (df_train['actual_value'] > 0)].shape[0]
        print(f"    - 全体の実績発生数: {total_actual:,}レコード")
        print(f"    - 上位{top_n}個の実績発生数: {top_actual:,}レコード")
        print(f"    - カバー率: {top_actual/total_actual*100:.1f}%")

        # テスト期間でアクティブなMaterial Key
        test_counts = df_test[df_test['actual_value'] > 0].groupby('material_key').size()
        active_keys_test = set(test_counts[test_counts >= min_test_count].index)
        print(f"  テスト期間でアクティブなMaterial Key: {len(active_keys_test):,}個")
        print(f"    （actual_value>0が{min_test_count}レコード以上）")

        # 結合
        selected_keys = top_keys_train | active_keys_test
        print(f"  選択されたMaterial Key数: {len(selected_keys):,}個")

        # フィルタリング実行
        df_filtered = df[df['material_key'].isin(selected_keys)].copy()
        print(f"\nフィルタリング後: {len(df_filtered):,}行, {df_filtered['material_key'].nunique():,} Material Keys")
        print(f"データ削減率: {(1 - len(df_filtered)/len(df))*100:.1f}%")

        # メモリ最適化
        original_memory = df_filtered.memory_usage().sum() / 1024 / 1024
        for col in df_filtered.columns:
            if df_filtered[col].dtype == 'float64':
                df_filtered[col] = df_filtered[col].astype('float32')
        new_memory = df_filtered.memory_usage().sum() / 1024 / 1024
        print(f"メモリ削減: {original_memory:.1f}MB → {new_memory:.1f}MB ({(1-new_memory/original_memory)*100:.1f}%削減)")

        return df_filtered

    def train_model_for_usage_type(
        self,
        usage_type: str,
        args
    ) -> dict:
        """
        特定のusage_typeでモデルを学習

        Args:
            usage_type: 'business' or 'household'
            args: コマンドライン引数

        Returns:
            dict: 評価結果
        """
        print(f"\n{'='*70}")
        print(f"Training model for {usage_type.upper()}")
        print(f"{'='*70}")

        # 特徴量を読み込み
        df = self.load_features(usage_type)

        # フィルタリング設定
        print(f"\n{usage_type.upper()} フィルタリング設定:")
        if usage_type == 'business':
            print(f"  学習期間: Top 3000 material keys")
            print(f"  テスト期間: 最低 {args.step_count * 2} 件以上")
        else:
            print(f"  学習期間: Top 3500 material keys")
            print(f"  テスト期間: 最低 {args.step_count * 4} 件以上")

        # フィルタリング実行
        df_filtered = self.filter_material_keys(
            df,
            args.train_end_date,
            args.step_count,
            usage_type
        )

        # モデル学習
        trainer = ModelTrainer(
            model_type=f"confirmed_order_demand_yamasa_{usage_type}",
            use_gpu=args.use_gpu
        )

        results = trainer.train_and_predict(
            df_filtered,
            train_end_date=args.train_end_date,
            step_count=args.step_count,
            use_optuna=args.use_optuna,
            n_trials=args.n_trials,
            enable_outlier_handling=args.enable_outlier_handling
        )

        # 結果を保存
        if results:
            # モデル保存
            model_path = f"output/models/confirmed_order_demand_yamasa_model_{usage_type}_latest.pkl"
            trainer.save_model_to_s3(model_path)
            print(f"Model saved: s3://{self.s3_bucket}/{model_path}")

            # 予測結果を保存
            if 'predictions' in results:
                pred_df = results['predictions']
                pred_df['usage_type'] = usage_type

                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                pred_path = f"output/predictions/confirmed_order_demand_yamasa_predictions_{usage_type}_{timestamp}.parquet"
                pred_df.to_parquet(f"/tmp/{usage_type}_predictions.parquet")
                self.s3_client.upload_file(
                    f"/tmp/{usage_type}_predictions.parquet",
                    self.s3_bucket,
                    pred_path
                )

            # メトリクスを表示
            print(f"\n{usage_type.upper()} Model Performance:")
            print(f"  RMSE: {results.get('rmse', 'N/A'):.4f}")
            print(f"  MAE: {results.get('mae', 'N/A'):.4f}")
            print(f"  Correlation: {results.get('correlation', 'N/A'):.4f}")
            print(f"  Samples: {results.get('test_samples', 0):,}")

        return results

    def train_all(self, args):
        """
        全体での学習（通常版）

        Args:
            args: コマンドライン引数
        """
        print("\n" + "="*70)
        print("Training model on ALL DATA")
        print("="*70)

        # 特徴量を読み込み
        df = self.load_features(None)

        # フィルタリング
        df_filtered = self.filter_material_keys(
            df,
            args.train_end_date,
            args.step_count,
            None
        )

        # モデル学習
        trainer = ModelTrainer(
            model_type="confirmed_order_demand_yamasa",
            use_gpu=args.use_gpu
        )

        results = trainer.train_and_predict(
            df_filtered,
            train_end_date=args.train_end_date,
            step_count=args.step_count,
            use_optuna=args.use_optuna,
            n_trials=args.n_trials,
            enable_outlier_handling=args.enable_outlier_handling
        )

        print("\n学習が完了しました！")
        print("モデルと評価結果はS3に保存されました。")

    def train_by_usage_type(self, args):
        """
        usage_type別での学習

        Args:
            args: コマンドライン引数
        """
        print("\n" + "="*60)
        print("    usage_type別モデル学習・予測")
        print("="*60)
        print(f"学習終了日: {args.train_end_date}")
        print(f"予測期間: {args.step_count}ヶ月")

        # 利用可能なusage_typeを確認
        usage_types = ['business', 'household']
        print(f"対象usage_type: {usage_types}")

        all_results = {}
        all_predictions = []

        # 各usage_typeで学習
        for usage_type in usage_types:
            try:
                results = self.train_model_for_usage_type(usage_type, args)
                all_results[usage_type] = results

                if results and 'predictions' in results:
                    pred_df = results['predictions']
                    pred_df['usage_type'] = usage_type
                    all_predictions.append(pred_df)

            except Exception as e:
                print(f"Error training {usage_type}: {e}")
                continue

        # 結果を結合
        if all_predictions:
            print("\n" + "="*70)
            print("Combining results from all usage types")
            print("="*70)

            combined_df = pd.concat(all_predictions, ignore_index=True)

            # 全体のメトリクスを計算
            evaluator = ModelEvaluator(model_type="confirmed_order_demand_yamasa_combined")
            combined_results = evaluator.evaluate(
                combined_df[combined_df['data_type'] == 'test'],
                y_col='actual_value',
                pred_col='predicted_value'
            )

            print("\nCombined Metrics:")
            print(f"  Total RMSE: {combined_results['rmse']:.4f}")
            print(f"  Total MAE: {combined_results['mae']:.4f}")
            print(f"  Total MAPE: {combined_results.get('mape', 0):.2f}%")
            print(f"  Total Correlation: {combined_results['correlation']:.4f}")
            print(f"  Total R²: {combined_results.get('r2_score', 0):.4f}")
            print(f"  Total Samples: {len(combined_df[combined_df['data_type'] == 'test']):,}")

            # 結合結果を保存
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

            # 予測結果
            pred_path = f"output/predictions/confirmed_order_demand_yamasa_predictions_by_usage_type_{timestamp}.parquet"
            combined_df.to_parquet(f"/tmp/combined_predictions.parquet")
            self.s3_client.upload_file(
                f"/tmp/combined_predictions.parquet",
                self.s3_bucket,
                pred_path
            )

            # latest版も更新
            latest_path = "output/predictions/confirmed_order_demand_yamasa_predictions_by_usage_type_latest.parquet"
            self.s3_client.upload_file(
                f"/tmp/combined_predictions.parquet",
                self.s3_bucket,
                latest_path
            )

            print(f"\nSaved combined predictions: s3://{self.s3_bucket}/{pred_path}")

        print("\n" + "="*70)
        print("✅ Model training and prediction completed successfully!")
        print("="*70)


def main():
    """メイン処理"""
    parser = argparse.ArgumentParser(description='モデル学習・予測スクリプト')

    parser.add_argument(
        '--mode',
        type=str,
        choices=['by_usage_type', 'all'],
        default='by_usage_type',
        help='学習モード: by_usage_type（デフォルト）またはall'
    )
    parser.add_argument(
        '--train-end-date',
        type=str,
        default='2024-12-31',
        help='学習データの終了日（YYYY-MM-DD）'
    )
    parser.add_argument(
        '--step-count',
        type=int,
        default=1,
        help='予測月数（デフォルト: 1）'
    )
    parser.add_argument(
        '--use-optuna',
        action='store_true',
        help='Optunaでハイパーパラメータ最適化'
    )
    parser.add_argument(
        '--n-trials',
        type=int,
        default=50,
        help='Optunaの試行回数'
    )
    parser.add_argument(
        '--enable-outlier-handling',
        action='store_true',
        help='外れ値処理を有効化'
    )
    parser.add_argument(
        '--use-gpu',
        action='store_true',
        help='GPU使用（利用可能な場合）'
    )

    args = parser.parse_args()

    # モデル学習実行
    trainer = UnifiedModelTrainer()

    if args.mode == 'by_usage_type':
        trainer.train_by_usage_type(args)
    else:
        trainer.train_all(args)


if __name__ == "__main__":
    main()