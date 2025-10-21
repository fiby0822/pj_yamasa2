#!/usr/bin/env python3
"""
Product_keyレベルでのモデル学習・予測スクリプト
material_key = product_keyとして処理
"""
import sys
import os
sys.path.append('/home/ubuntu/yamasa')

import pandas as pd
import numpy as np
import argparse
from datetime import datetime
import json
import gc
import warnings
warnings.filterwarnings('ignore')
from typing import Optional

from modules.models.train_predict import TimeSeriesPredictor
from modules.evaluation.metrics import ModelEvaluator


def apply_leak_safe_predictor_patch():
    """
    TimeSeriesPredictor の Material Key フィルタリングをリーク防止版に差し替える。
    未来実績を参照しないよう、train_end_date 以前のデータのみから選別する。
    """
    def _filter_no_leak(
        self,
        df: pd.DataFrame,
        train_end_date: str,
        target_col: str,
        step_count: int,
        verbose: bool = True,
        min_test_active_records: Optional[int] = None
    ) -> tuple[pd.DataFrame, set]:
        train_end = pd.to_datetime(train_end_date)
        df_train = df[df['date'] <= train_end].copy()

        if verbose:
            print("\n===== Material Keyフィルタリング（リーク対策版） =====")
            print(f"フィルタリング対象: {len(df):,}行, Material Keys: {df['material_key'].nunique():,}")
            print(f"使用データ範囲: ~{train_end.strftime('%Y-%m-%d')} (train)")

        if df_train.empty:
            if verbose:
                print("  学習データが空のため、フィルタリングをスキップします。")
            return df, set(df['material_key'].unique())

        # 学習期間における実績発生数を算出
        active_train = df_train[df_train[target_col] > 0]
        history_counts = active_train.groupby('material_key').size()

        top_limit = 7000
        top_keys = set(history_counts.nlargest(top_limit).index)

        # train_end_date 以前の情報のみで最低活動回数を算出
        required_history = min_test_active_records if min_test_active_records is not None else step_count * 4
        if required_history <= 0:
            required_history = 1
        history_active_keys = set(history_counts[history_counts >= required_history].index)

        selected_keys = top_keys | history_active_keys
        if not selected_keys:
            selected_keys = set(df_train['material_key'].unique())

        df_filtered = df[df['material_key'].isin(selected_keys)].copy()

        if verbose:
            print(f"  学習期間アクティブ製品: {len(history_counts):,}")
            print(f"  上位{len(top_keys):,}製品（train実績数ベース）を確保")
            print(f"  最低活動回数 {required_history} 件以上: {len(history_active_keys):,} 製品")
            print(f"  選択後: {len(df_filtered):,}行, Material Keys: {df_filtered['material_key'].nunique():,}")

        return df_filtered, set(selected_keys)

    TimeSeriesPredictor._filter_important_material_keys = _filter_no_leak


class ProductLevelModelTrainer:
    """Product_keyレベルでのモデル学習クラス"""

    def __init__(self, base_dir: str = "/home/ubuntu/yamasa2/work/data"):
        self.base_dir = base_dir
        self.prediction_targets: set[str] = set()

    def load_features(self) -> pd.DataFrame:
        """
        Product_keyレベルの特徴量データを読み込み

        Returns:
            pd.DataFrame: 特徴量データ
        """
        # work/data/featuresから読み込み
        local_path = os.path.join(self.base_dir, "features", "product_level_features_latest.parquet")
        if os.path.exists(local_path):
            print(f"Loading features from local: {local_path}")
            df = pd.read_parquet(local_path)
            print(f"  Loaded {len(df):,} records, {df['material_key'].nunique():,} product keys")
            return df
        else:
            raise FileNotFoundError(f"Product-level features not found: {local_path}")

    def filter_products(
        self,
        df: pd.DataFrame,
        train_end_date: str,
        step_count: int
    ) -> pd.DataFrame:
        """
        Product_keyのフィルタリング
        製品レベルなので閾値を調整

        Args:
            df: 特徴量データ
            train_end_date: 学習終了日
            step_count: 予測月数

        Returns:
            pd.DataFrame: フィルタリング後のデータ
        """
        print(f"\n{'='*5} Product Keyフィルタリング {'='*5}")

        df['file_date'] = pd.to_datetime(df['file_date'])
        train_end = pd.to_datetime(train_end_date)

        # 学習期間のデータのみを利用したフィルタリング
        df_train = df[df['file_date'] <= train_end].copy()
        if df_train.empty:
            print("  学習期間データが存在しないため、フィルタリングをスキップします。")
            self.prediction_targets = set(df['material_key'].unique())
            return df

        top_n = 50  # 上位50製品（全98製品の約半分）
        min_train_count = max(step_count * 10, 1)  # 未来実績を参照せず、train側で最低活動回数を設定

        train_positive = df_train[df_train['actual_value'] > 0]
        train_counts = train_positive.groupby('material_key').size()
        top_products_train = set(train_counts.nlargest(top_n).index)
        active_products_train = set(train_counts[train_counts >= min_train_count].index)

        train_products = top_products_train | active_products_train
        if not train_products:
            train_products = set(df_train['material_key'].unique())

        df_filtered = df[df['material_key'].isin(train_products)]

        print(f"\n[フィルタリング結果（trainベース）]")
        print(f"  製品総数: {df['material_key'].nunique()} → {df_filtered['material_key'].nunique()}")
        print(f"  - 学習期間Top{top_n}: {len(top_products_train)} products")
        print(f"  - 学習期間で{min_train_count}件以上の実績: {len(active_products_train)} products")
        print(f"  - 両条件の和集合: {len(train_products)} products")
        print(f"  レコード数: {len(df):,} → {len(df_filtered):,}")

        # 予測対象のproduct_keyを記録（学習期間の活動に基づく）
        self.prediction_targets = train_products

        return df_filtered

    def train_and_evaluate(
        self,
        df: pd.DataFrame,
        train_end_date: str,
        step_count: int,
        use_optuna: bool = False,
        n_trials: int = 30
    ):
        """
        モデル学習と評価

        Args:
            df: フィルタリング済み特徴量データ
            train_end_date: 学習終了日
            step_count: 予測月数
            use_optuna: Optuna使用フラグ
            n_trials: Optunaの試行回数
        """
        print(f"\n{'='*60}")
        print("Product-level Model Training & Prediction")
        print(f"{'='*60}")

        # TimeSeriesPredictor のフィルタリングをリーク対策版に差し替え
        apply_leak_safe_predictor_patch()

        # TimeSeriesPredictorインスタンス作成 (ローカルモード)
        predictor = TimeSeriesPredictor(
            bucket_name="dummy-bucket",
            model_type="product_level"
        )

        # 特徴量カラムの特定
        feature_cols = [col for col in df.columns if col.endswith('_f')]
        print(f"特徴量数: {len(feature_cols)}")

        # 学習・予測実行
        df_train, df_test, df_pred, metrics_dict, model, feature_importance_dict = predictor.train_test_predict_time_split(
            _df_features=df,
            train_end_date=train_end_date,
            step_count=step_count,
            target_col='actual_value',
            feature_cols=feature_cols,
            use_optuna=use_optuna,
            n_trials=n_trials,
            use_gpu=False,
            apply_winsorization=False,
            apply_hampel=False
        )

        # 結果を辞書形式に整形
        # df_testはメトリクス、df_predは特徴量重要度が返される
        results = {
            'df_train': df_train,
            'df_test': df_test,  # メトリクス
            'df_pred': df_pred,  # 特徴量重要度
            'model': model,
            'feature_importance': df_pred,  # df_predが実際には特徴量重要度
            'metrics': metrics_dict,
            'best_params': None
        }

        # 結果の取得
        df_train = results['df_train']
        df_test = results['df_test']
        df_pred = results['df_pred']
        model = results['model']
        feature_importance = results['feature_importance']
        best_params = results.get('best_params')

        predicted_keys = df_train['material_key'].nunique() if isinstance(df_train, pd.DataFrame) and 'material_key' in df_train.columns else 0
        total_keys = len(self.prediction_targets) if self.prediction_targets else df['material_key'].nunique()
        print(f"\nPredictions generated for {predicted_keys} / {total_keys} material_keys.")

        # df_testには実際にはメトリクスが入っている
        # metrics_dictに全体のメトリクスが含まれている
        print(f"\n予測対象製品数: {len(self.prediction_targets)}")

        # 評価メトリクスはmetrics_dictに既に含まれている
        metrics = metrics_dict

        print("\n" + "="*60)
        print("Product-level Prediction Metrics")
        print("="*60)
        for metric_name, value in metrics.items():
            if isinstance(value, (int, float)):
                print(f"  {metric_name}: {value:.4f}")

        # Material Key別のサマリーはdf_testにmaterial_key別メトリクスとして含まれている
        if 'material_key' in df_test.columns:
            summary = df_test[df_test['material_key'].isin(self.prediction_targets)]
            print(f"\n製品別サマリー作成完了: {len(summary)} products")
        else:
            # material_key別メトリクスがない場合は空のDataFrameを作成
            summary = pd.DataFrame()
            print("\n製品別サマリーは利用できません")

        # 結果の保存
        self.save_results(
            df_pred=df_train,  # 予測結果がないためdf_trainを代用
            summary=summary,
            metrics=metrics,
            feature_importance=results['feature_importance'],
            best_params=results.get('best_params'),
            train_end_date=train_end_date,
            step_count=step_count
        )

        return {
            'metrics': metrics,
            'summary': summary,
            'df_pred': df_train,  # 実際の予測データは保存されていないためdf_trainを代用,
            'model': model,
            'feature_importance': feature_importance
        }

    def save_results(
        self,
        df_pred: pd.DataFrame,
        summary: pd.DataFrame,
        metrics: dict,
        feature_importance: pd.DataFrame,
        best_params: dict,
        train_end_date: str,
        step_count: int
    ):
        """結果をローカルとS3に保存"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # ローカル保存ディレクトリ
        os.makedirs('/home/ubuntu/yamasa2/data/predictions', exist_ok=True)
        os.makedirs('/home/ubuntu/yamasa2/data/models', exist_ok=True)

        # 1. 予測結果の保存
        pred_path = f"/home/ubuntu/yamasa2/data/predictions/product_level_predictions_{timestamp}.parquet"
        pred_path_latest = f"/home/ubuntu/yamasa2/data/predictions/product_level_predictions_latest.parquet"

        for path in [pred_path, pred_path_latest]:
            df_pred.to_parquet(path, index=False)
            print(f"Saved predictions: {path}")

        # 2. サマリーの保存
        summary_path = f"/home/ubuntu/yamasa2/data/predictions/product_level_summary_{timestamp}.parquet"
        summary_path_latest = f"/home/ubuntu/yamasa2/data/predictions/product_level_summary_latest.parquet"

        for path in [summary_path, summary_path_latest]:
            summary.to_parquet(path, index=False)
            print(f"Saved summary: {path}")

        # 3. メトリクスの保存
        metrics_with_info = {
            'timestamp': timestamp,
            'train_end_date': train_end_date,
            'step_count': step_count,
            'metrics': metrics,
            'best_params': best_params
        }

        metrics_path = f"/home/ubuntu/yamasa2/data/models/product_level_metrics_{timestamp}.json"
        metrics_path_latest = f"/home/ubuntu/yamasa2/data/models/product_level_metrics_latest.json"

        for path in [metrics_path, metrics_path_latest]:
            with open(path, 'w') as f:
                json.dump(metrics_with_info, f, indent=2, default=str)
            print(f"Saved metrics: {path}")

        # 4. 特徴量重要度の保存
        fi_path = f"/home/ubuntu/yamasa2/data/models/feature_importance_{timestamp}.csv"
        fi_path_latest = f"/home/ubuntu/yamasa2/data/models/feature_importance_latest.csv"

        for path in [fi_path, fi_path_latest]:
            feature_importance.to_csv(path, index=False)
            print(f"Saved feature importance: {path}")

        print("\n" + "="*60)
        print("Results saved successfully!")
        print("="*60)

        # S3へのアップロード案内
        print("\nTo upload to S3, run:")
        print("python /home/ubuntu/yamasa2/scripts/upload_results_to_s3.py")


def main():
    """メイン処理"""
    parser = argparse.ArgumentParser(description='Product-level モデル学習・予測')

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
        default=30,
        help='Optunaの試行回数'
    )

    args = parser.parse_args()

    print("="*60)
    print("Product-level Model Training")
    print("="*60)
    print(f"Train end date: {args.train_end_date}")
    print(f"Step count: {args.step_count}")
    print(f"Use Optuna: {args.use_optuna}")
    if args.use_optuna:
        print(f"N trials: {args.n_trials}")
    print("="*60)

    # トレーナーのインスタンス化
    trainer = ProductLevelModelTrainer()

    # 特徴量の読み込み
    print("\nLoading features...")
    df = trainer.load_features()

    # フィルタリング
    df_filtered = trainer.filter_products(
        df,
        train_end_date=args.train_end_date,
        step_count=args.step_count
    )

    # 学習・評価
    results = trainer.train_and_evaluate(
        df_filtered,
        train_end_date=args.train_end_date,
        step_count=args.step_count,
        use_optuna=args.use_optuna,
        n_trials=args.n_trials
    )

    print("\n" + "="*60)
    print("Training completed successfully!")
    print("="*60)
    print(f"\nFinal Metrics:")
    for metric_name, value in results['metrics'].items():
        print(f"  {metric_name}: {value:.4f}")


if __name__ == "__main__":
    main()
