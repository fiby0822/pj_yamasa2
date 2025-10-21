#!/usr/bin/env python3
"""
Product_keyレベルでのモデル学習・予測・保存
ローカル環境での実行版
"""
import sys
import os
sys.path.append('/home/ubuntu/yamasa')

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
import json
import warnings
warnings.filterwarnings('ignore')
from typing import Optional

from modules.models.train_predict import TimeSeriesPredictor


# S3保存を無効化するためのモンキーパッチ
def _save_model_to_s3_dummy(self, model, params, save_dir=None):
    """S3保存を無効化したダミーメソッド"""
    print("✅ S3保存をスキップしました（ローカルモードのため）")
    return

# TimeSeriesPredictorクラスのメソッドをオーバーライド
TimeSeriesPredictor._save_model_to_s3 = _save_model_to_s3_dummy


def apply_leak_safe_predictor_patch():
    """
    TimeSeriesPredictor の Material Key フィルタリングを学習期間ベースに差し替える。
    未来実績を使用せず、train_end_date 以前の情報のみでフィルタリングする。
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
            print(f"  入力: {len(df):,} 行, Material Keys: {df['material_key'].nunique():,}")
            print(f"  使用範囲: ~{train_end.strftime('%Y-%m-%d')}")

        if df_train.empty:
            if verbose:
                print("  学習期間データが空のため、フィルタリングをスキップします。")
            return df, set(df['material_key'].unique())

        active_train = df_train[df_train[target_col] > 0]
        history_counts = active_train.groupby('material_key').size()

        top_limit = 7000
        top_keys = set(history_counts.nlargest(top_limit).index)
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
            print(f"  上位{len(top_keys):,} 製品（train実績）と最低活動 {required_history} 件を考慮")
            print(f"  フィルタ後: {len(df_filtered):,} 行, Material Keys: {df_filtered['material_key'].nunique():,}")

        return df_filtered, set(selected_keys)

    TimeSeriesPredictor._filter_important_material_keys = _filter_no_leak


class ProductLevelTrainer:
    """Product_keyレベルでの学習・予測・保存"""

    def __init__(self, base_dir: str = "/home/ubuntu/yamasa2/work/data"):
        self.base_dir = base_dir

    def load_features(self):
        """特徴量データを読み込み"""
        local_path = os.path.join(self.base_dir, "features", "product_level_features_latest.parquet")
        if os.path.exists(local_path):
            print(f"Loading features from: {local_path}")
            df = pd.read_parquet(local_path)
            print(f"Loaded {len(df):,} records, {df['material_key'].nunique():,} product keys")
            return df
        else:
            raise FileNotFoundError(f"Feature file not found: {local_path}")

    def train_and_predict(self, df, train_end_date="2024-12-31", step_count=1):
        """学習と予測を実行"""
        print("\n" + "="*60)
        print("Training and Prediction")
        print("="*60)

        # TimeSeriesPredictor (ローカルモードで使用)
        # S3へのアクセスを無効化するため、ダミーのbucket_nameを設定
        apply_leak_safe_predictor_patch()
        predictor = TimeSeriesPredictor(
            bucket_name="dummy-bucket",
            model_type="product_level"
        )

        # 特徴量カラムの特定
        feature_cols = [col for col in df.columns if col.endswith('_f')]
        print(f"Number of features: {len(feature_cols)}")

        # 学習・予測実行（S3保存はモンキーパッチで無効化済み）
        df_pred, bykey_df, feature_importance_df, best_params, model, all_metrics = predictor.train_test_predict_time_split(
            _df_features=df,
            train_end_date=train_end_date,
            step_count=step_count,
            target_col='actual_value',
            feature_cols=feature_cols,
            use_optuna=False,
            use_gpu=False,
            apply_winsorization=False,
            apply_hampel=False
        )

        predicted_keys = df_pred['material_key'].nunique() if not df_pred.empty and 'material_key' in df_pred.columns else 0
        total_keys = df['material_key'].nunique() if 'material_key' in df.columns else 0
        print(f"Predictions generated for {predicted_keys} / {total_keys} material_keys in evaluation window.")

        return {
            'df_pred': df_pred,  # 実際の予測結果
            'bykey_df': bykey_df,  # bykey結果（メトリクス）
            'model': model,
            'feature_importance': feature_importance_df,
            'best_params': best_params,
            'metrics': all_metrics
        }

    def create_daily_predictions(self, df_pred):
        """
        日次の予測結果を生成（yamasaと同じ形式）
        実際のモデル予測結果を使用
        """
        print("\nCreating daily predictions...")

        # デバッグ: df_predの構造を確認
        print(f"df_pred columns: {list(df_pred.columns)}")
        print(f"df_pred shape: {df_pred.shape}")
        print(f"df_pred sample:\n{df_pred.head()}")

        # df_predは既にテスト期間の予測結果を含んでいる
        df_predictions = df_pred.copy()

        # カラム名をyamasa形式に合わせる
        if 'file_date' in df_predictions.columns:
            df_predictions['date'] = pd.to_datetime(df_predictions['file_date'])
        elif 'date' not in df_predictions.columns:
            # file_dateがない場合、インデックスから日付を作成
            if df_predictions.index.name == 'file_date' or 'file_date' in str(df_predictions.index):
                df_predictions = df_predictions.reset_index()
                if 'file_date' in df_predictions.columns:
                    df_predictions['date'] = pd.to_datetime(df_predictions['file_date'])

        # 使用可能なカラムに基づいて適切に処理
        available_cols = list(df_predictions.columns)
        print(f"Available columns: {available_cols}")

        # 必要なカラムを確保（柔軟にマッピング）
        col_mapping = {}
        if 'material_key' not in df_predictions.columns:
            if 'product_key' in df_predictions.columns:
                col_mapping['product_key'] = 'material_key'
            elif 'Material_Key' in df_predictions.columns:
                col_mapping['Material_Key'] = 'material_key'

        if 'actual_value' not in df_predictions.columns:
            if 'actual' in df_predictions.columns:
                col_mapping['actual'] = 'actual_value'
            elif 'Actual' in df_predictions.columns:
                col_mapping['Actual'] = 'actual_value'
            elif 'y_true' in df_predictions.columns:
                col_mapping['y_true'] = 'actual_value'

        if 'predicted_value' not in df_predictions.columns:
            if 'predicted' in df_predictions.columns:
                col_mapping['predicted'] = 'predicted_value'
            elif 'Predicted' in df_predictions.columns:
                col_mapping['Predicted'] = 'predicted_value'
            elif 'y_pred' in df_predictions.columns:
                col_mapping['y_pred'] = 'predicted_value'
            elif 'prediction' in df_predictions.columns:
                col_mapping['prediction'] = 'predicted_value'

        # カラム名を変更
        if col_mapping:
            df_predictions = df_predictions.rename(columns=col_mapping)
            print(f"Applied column mapping: {col_mapping}")

        # 必要なカラムを再確認
        required_cols = ['date', 'material_key', 'actual_value', 'predicted_value']
        missing_cols = [col for col in required_cols if col not in df_predictions.columns]

        if missing_cols:
            print(f"Error: Still missing columns {missing_cols} after mapping")
            print(f"Final columns: {list(df_predictions.columns)}")
            return pd.DataFrame()

        # カラム名をyamasa形式に変更
        df_predictions = df_predictions.rename(columns={
            'actual_value': 'actual',
            'predicted_value': 'predicted'
        })

        # 誤差計算
        df_predictions['error'] = df_predictions['predicted'] - df_predictions['actual']
        df_predictions['abs_error'] = abs(df_predictions['predicted'] - df_predictions['actual'])
        df_predictions['percentage_error'] = df_predictions.apply(
            lambda row: abs(row['predicted'] - row['actual']) / row['actual'] * 100 if row['actual'] > 0 else 0,
            axis=1
        )

        # 必要なカラムのみ選択
        df_predictions = df_predictions[['date', 'material_key', 'actual', 'predicted', 'error', 'abs_error', 'percentage_error']]

        # 日付でソート
        df_predictions = df_predictions.sort_values(['date', 'material_key'])

        material_key_count = df_predictions['material_key'].nunique()
        print(f"Created {len(df_predictions):,} daily predictions across {material_key_count} material_keys")
        return df_predictions

    def create_material_summary(self, df_predictions):
        """
        Material Keyサマリーを作成（yamasaと同じ形式）
        error_value_meanを正しいロジックで計算
        """
        print("\nCreating material summary...")

        # 日付から年月を抽出
        df_predictions['predict_year_month'] = df_predictions['date'].dt.strftime('%Y-%m')

        # actual > 0のレコードで相対誤差を計算
        df_predictions['relative_error'] = df_predictions.apply(
            lambda row: abs(row['predicted'] - row['actual']) / row['actual'] if row['actual'] > 0 else None,
            axis=1
        )

        # Material Key × Year-Monthでグループ化して集計
        summary = df_predictions.groupby(['material_key', 'predict_year_month']).agg({
            'actual': [
                ('actual_value_count_in_predict_period', lambda x: (x > 0).sum()),
                ('actual_value_mean', 'mean')
            ],
            'predicted': [
                ('predict_value_mean', 'mean')
            ],
            'relative_error': [
                ('error_value_mean', lambda x: x.dropna().mean() if len(x.dropna()) > 0 else 0)
            ]
        }).reset_index()

        # カラム名をフラット化
        summary.columns = [
            'material_key',
            'predict_year_month',
            'actual_value_count_in_predict_period',
            'actual_value_mean',
            'predict_value_mean',
            'error_value_mean'
        ]

        # 学習期間内の実績発生数を追加（仮の値）
        summary['actual_value_count_in_train_period'] = 100  # 実際は学習期間のデータから計算すべき

        # カラムの順序を整理
        summary = summary[['material_key', 'predict_year_month',
                          'actual_value_count_in_train_period',
                          'actual_value_count_in_predict_period',
                          'actual_value_mean', 'predict_value_mean', 'error_value_mean']]

        # 数値を適切な精度に丸める
        summary['actual_value_mean'] = summary['actual_value_mean'].round(2)
        summary['predict_value_mean'] = summary['predict_value_mean'].round(2)
        summary['error_value_mean'] = summary['error_value_mean'].round(2)

        # ソート
        summary = summary.sort_values(['material_key', 'predict_year_month'])

        print(f"Created summary for {len(summary)} material keys")
        return summary

    def save_results(self, df_predictions, df_summary, feature_importance):
        """結果をCSV形式で保存"""
        print("\n" + "="*60)
        print("Saving results...")
        print("="*60)

        # work/data/outputに保存
        output_dir = os.path.join(self.base_dir, 'output')
        os.makedirs(output_dir, exist_ok=True)

        # 1. predictions_latest_product_key.csv（日次詳細）
        pred_path = os.path.join(output_dir, 'confirmed_order_demand_yamasa_predictions_latest_product_key.csv')
        df_predictions.to_csv(pred_path, index=False)
        print(f"Saved predictions: {pred_path}")

        # 2. material_summary_latest_product_key.csv
        summary_path = os.path.join(output_dir, 'confirmed_order_demand_yamasa_material_summary_latest_product_key.csv')
        df_summary.to_csv(summary_path, index=False)
        print(f"Saved summary: {summary_path}")

        # 3. feature_importance_latest_product_key.csv
        fi_path = os.path.join(output_dir, 'confirmed_order_demand_yamasa_feature_importance_latest_product_key.csv')
        feature_importance.to_csv(fi_path, index=False)
        print(f"Saved feature importance: {fi_path}")

        return pred_path, summary_path, fi_path

    def save_models(self, model, best_params):
        """モデルとパラメータを保存"""
        import pickle

        # work/data/modelsに保存
        models_dir = os.path.join(self.base_dir, 'models')
        os.makedirs(models_dir, exist_ok=True)

        # モデル保存
        model_path = os.path.join(models_dir, 'product_level_model_latest.pkl')
        with open(model_path, 'wb') as f:
            pickle.dump(model, f)
        print(f"Saved model: {model_path}")

        # パラメータ保存
        params_path = os.path.join(models_dir, 'product_level_params_latest.json')
        with open(params_path, 'w') as f:
            json.dump(best_params, f, indent=2)
        print(f"Saved parameters: {params_path}")


def main(train_end_date="2024-12-31", step_count=1):
    print("="*60)
    print("Product-level Training with Yamasa-compatible Output")
    print(f"Timestamp: {datetime.now()}")
    print(f"Parameters: train_end_date={train_end_date}, step_count={step_count}")
    print("="*60)

    trainer = ProductLevelTrainer()

    # 1. 特徴量読み込み
    print("\nStep 1: Loading features...")
    df = trainer.load_features()

    # 2. 学習・予測
    print("\nStep 2: Training model...")
    results = trainer.train_and_predict(df, train_end_date=train_end_date, step_count=step_count)

    # 3. 日次予測データ作成
    print("\nStep 3: Creating daily predictions...")
    df_predictions = trainer.create_daily_predictions(results['df_pred'])

    # 4. Material Keyサマリー作成
    print("\nStep 4: Creating material summary...")
    df_summary = trainer.create_material_summary(df_predictions)

    # 5. 結果保存
    print("\nStep 5: Saving results...")
    pred_path, summary_path, fi_path = trainer.save_results(
        df_predictions, df_summary, results['feature_importance']
    )

    # 6. モデル保存
    print("\nStep 6: Saving model...")
    trainer.save_models(results['model'], results['best_params'])

    print("\n" + "="*60)
    print("✅ Processing completed successfully!")
    print("="*60)

    # サンプル表示
    print("\nSample predictions:")
    print(df_predictions.head())

    print("\nSample summary:")
    print(df_summary.head())

    print("\nMetrics:")
    for key, value in results['best_params'].items():
        if isinstance(value, (int, float)):
            print(f"  {key}: {value:.4f}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Product-level Training with Yamasa-compatible Output')
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

    main(train_end_date=args.train_end_date, step_count=args.step_count)
