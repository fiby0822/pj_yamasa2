#!/usr/bin/env python3
"""
usage_type別モデル学習・予測・評価スクリプト
- businessとhouseholdで別々にモデルを学習
- 結果を統合して出力
"""
import pandas as pd
import numpy as np
import lightgbm as lgb
from datetime import datetime
import boto3
import json
import sys
import os
import argparse
import pickle
import warnings
warnings.filterwarnings('ignore')

# プロジェクトルートをPythonパスに追加
sys.path.append('/home/ubuntu/yamasa')

from modules.evaluation.metrics import ModelEvaluator
from modules.models.train_predict import TimeSeriesPredictor
from modules.data_io.s3_handler import S3Handler


def load_features_for_usage_type(usage_type: str, local_path: str = None) -> pd.DataFrame:
    """
    usage_type用の特徴量を読み込み
    """
    if local_path:
        file_path = local_path
    else:
        file_path = f"/home/ubuntu/yamasa/data/features/confirmed_order_demand_yamasa_features_{usage_type}_latest.parquet"

    print(f"Loading features for {usage_type} from: {file_path}")
    df = pd.read_parquet(file_path)
    print(f"  Loaded {len(df):,} records, {df['material_key'].nunique():,} material keys")

    return df


def filter_material_keys_by_usage_type(df: pd.DataFrame, usage_type: str,
                                       train_end_date: str, step_count: int) -> pd.DataFrame:
    """
    usage_type別のフィルタリング設定でMaterial Keyを絞り込み
    """
    train_end_date = pd.to_datetime(train_end_date)

    # 学習期間とテスト期間でデータを分割
    train_df = df[df['file_date'] <= train_end_date]
    test_df = df[df['file_date'] > train_end_date]

    # usage_type別の設定
    if usage_type == 'business':
        train_top_n = 3000
        test_min_count = step_count * 2  # business は閾値を低めに
    else:  # household
        train_top_n = 3500
        test_min_count = step_count * 4  # household は閾値を標準に

    print(f"\n{usage_type.upper()} フィルタリング設定:")
    print(f"  学習期間: Top {train_top_n} material keys")
    print(f"  テスト期間: 最低 {test_min_count} 件以上")

    # 学習期間での実績発生数上位N個
    train_mk_counts = train_df[train_df['actual_value'] > 0].groupby('material_key').size()
    top_mks = set(train_mk_counts.nlargest(train_top_n).index)

    # テスト期間でアクティブなMaterial Key
    test_mk_counts = test_df[test_df['actual_value'] > 0].groupby('material_key').size()
    active_test_mks = set(test_mk_counts[test_mk_counts >= test_min_count].index)

    # 結合（学習用）
    selected_mks_for_train = top_mks | active_test_mks

    print(f"\nフィルタリング結果:")
    print(f"  学習期間Top: {len(top_mks):,} keys")
    print(f"  テスト期間アクティブ: {len(active_test_mks):,} keys")
    print(f"  学習用選択: {len(selected_mks_for_train):,} keys")
    print(f"  予測対象（テストのみ）: {len(active_test_mks):,} keys")

    # フィルタリング適用
    filtered_df = df[df['material_key'].isin(selected_mks_for_train)]

    print(f"  フィルタリング後: {len(filtered_df):,} records (削減率: {(1-len(filtered_df)/len(df))*100:.1f}%)")

    return filtered_df, active_test_mks


def train_model_for_usage_type(df_features: pd.DataFrame, usage_type: str,
                               train_end_date: str, step_count: int) -> dict:
    """
    usage_type用のモデルを学習・予測
    """
    print(f"\n{'='*70}")
    print(f"Training model for {usage_type.upper()}")
    print('='*70)

    # Material Keyフィルタリング
    df_filtered, test_material_keys = filter_material_keys_by_usage_type(
        df_features, usage_type, train_end_date, step_count
    )

    # モデル学習の準備
    predictor = TimeSeriesPredictor()

    # 特徴量カラムの選択（_fで終わるカラム）
    feature_cols = [col for col in df_filtered.columns if col.endswith('_f')]
    print(f"Using {len(feature_cols)} features")

    # 学習・予測実行
    df_pred_all, bykey_df, imp_last, best_params, model_last, metrics = predictor.train_test_predict_time_split(
        _df_features=df_filtered,
        train_end_date=train_end_date,
        step_count=step_count,
        target_col='actual_value',
        use_optuna=False,
        n_trials=50
    )

    # 結果を整理（カラム名を修正）
    results = {
        'predictions': df_pred_all['predicted'].values if not df_pred_all.empty else [],
        'actuals': df_pred_all['actual'].values if not df_pred_all.empty else [],
        'dates': df_pred_all['date'].values if not df_pred_all.empty else [],
        'material_keys': df_pred_all['material_key'].values if not df_pred_all.empty else [],
        'model': model_last,
        'feature_importance': imp_last,
        'metrics': metrics
    }

    # テスト期間のMaterial Keyでフィルタリング（予測対象のみ）
    if 'material_keys' in results:
        mask = [mk in test_material_keys for mk in results['material_keys']]
        for key in ['predictions', 'actuals', 'dates', 'material_keys']:
            if key in results:
                results[key] = [val for val, m in zip(results[key], mask) if m]

    # usage_typeを追加
    results['usage_type'] = usage_type

    # メトリクスを再計算
    if len(results.get('predictions', [])) > 0:
        from sklearn.metrics import mean_squared_error, mean_absolute_error
        rmse = np.sqrt(mean_squared_error(results['actuals'], results['predictions']))
        mae = mean_absolute_error(results['actuals'], results['predictions'])

        # 相関係数
        if len(results['actuals']) > 1:
            correlation = np.corrcoef(results['actuals'], results['predictions'])[0, 1]
        else:
            correlation = 0

        results['metrics'] = {
            'RMSE': rmse,
            'MAE': mae,
            'Correlation': correlation,
            'Total_Samples': len(results['predictions'])
        }

        print(f"\n{usage_type.upper()} Model Performance:")
        print(f"  RMSE: {rmse:.4f}")
        print(f"  MAE: {mae:.4f}")
        print(f"  Correlation: {correlation:.4f}")
        print(f"  Samples: {len(results['predictions']):,}")

    # モデルと結果を返す
    return {
        'model': results.get('model'),
        'results': results,
        'feature_importance': results.get('feature_importance'),
        'test_material_keys': test_material_keys
    }


def combine_results(business_data: dict, household_data: dict) -> dict:
    """
    business とhousehold の結果を統合
    """
    print("\n" + "="*70)
    print("Combining results from both models")
    print("="*70)

    # 予測結果の統合
    combined_predictions = []
    combined_actuals = []
    combined_dates = []
    combined_material_keys = []
    combined_usage_types = []

    for data, usage_type in [(business_data, 'business'), (household_data, 'household')]:
        results = data['results']
        if 'predictions' in results:
            combined_predictions.extend(results['predictions'])
            combined_actuals.extend(results['actuals'])
            combined_dates.extend(results['dates'])
            combined_material_keys.extend(results['material_keys'])
            combined_usage_types.extend([usage_type] * len(results['predictions']))

    # 統合メトリクスの計算
    if len(combined_predictions) > 0:
        from sklearn.metrics import mean_squared_error, mean_absolute_error

        combined_rmse = np.sqrt(mean_squared_error(combined_actuals, combined_predictions))
        combined_mae = mean_absolute_error(combined_actuals, combined_predictions)

        # 相関係数
        if len(combined_actuals) > 1:
            combined_correlation = np.corrcoef(combined_actuals, combined_predictions)[0, 1]
        else:
            combined_correlation = 0

        # R2スコア
        ss_res = np.sum((np.array(combined_actuals) - np.array(combined_predictions)) ** 2)
        ss_tot = np.sum((np.array(combined_actuals) - np.mean(combined_actuals)) ** 2)
        r2 = 1 - (ss_res / (ss_tot + 1e-10)) if ss_tot > 0 else 0

        # MAPE（ゼロ値を避ける）
        mape_values = []
        for actual, pred in zip(combined_actuals, combined_predictions):
            if actual > 0:
                mape_values.append(abs(actual - pred) / actual * 100)
        mape = np.mean(mape_values) if mape_values else 0

        combined_metrics = {
            'RMSE': combined_rmse,
            'MAE': combined_mae,
            'MAPE': mape,
            'Correlation': combined_correlation,
            'R2': r2,
            'Total_Samples': len(combined_predictions),
            'Business_Samples': len(business_data['results'].get('predictions', [])),
            'Household_Samples': len(household_data['results'].get('predictions', []))
        }

        print("\nCombined Metrics:")
        print(f"  Total RMSE: {combined_rmse:.4f}")
        print(f"  Total MAE: {combined_mae:.4f}")
        print(f"  Total MAPE: {mape:.2f}%")
        print(f"  Total Correlation: {combined_correlation:.4f}")
        print(f"  Total R²: {r2:.4f}")
        print(f"  Total Samples: {len(combined_predictions):,}")
    else:
        combined_metrics = {}

    return {
        'predictions': combined_predictions,
        'actuals': combined_actuals,
        'dates': combined_dates,
        'material_keys': combined_material_keys,
        'usage_types': combined_usage_types,
        'metrics': combined_metrics,
        'business_metrics': business_data['results'].get('metrics', {}),
        'household_metrics': household_data['results'].get('metrics', {}),
        'business_feature_importance': business_data.get('feature_importance'),
        'household_feature_importance': household_data.get('feature_importance')
    }


def save_combined_results(combined_results: dict, s3_handler: S3Handler):
    """
    統合結果をS3に保存
    """
    print("\n" + "="*70)
    print("Saving combined results")
    print("="*70)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # 1. predictions_latest.csv
    pred_df = pd.DataFrame({
        'material_key': combined_results['material_keys'],
        'date': combined_results['dates'],
        'actual': combined_results['actuals'],
        'predicted': combined_results['predictions'],
        'usage_type': combined_results['usage_types']
    })

    # エラー計算
    pred_df['error'] = pred_df['predicted'] - pred_df['actual']
    pred_df['abs_error'] = np.abs(pred_df['error'])

    # S3に保存
    csv_key = f"output/evaluation/confirmed_order_demand_yamasa_predictions_{timestamp}.csv"
    latest_csv_key = f"output/evaluation/confirmed_order_demand_yamasa_predictions_latest.csv"

    s3_handler.write_csv(pred_df, csv_key)
    s3_handler.write_csv(pred_df, latest_csv_key)

    print(f"  Predictions saved: s3://{s3_handler.bucket_name}/{latest_csv_key}")

    # 2. material_summary_latest.csv
    pred_df['date'] = pd.to_datetime(pred_df['date'])
    pred_df['predict_year_month'] = pred_df['date'].dt.strftime('%Y-%m')

    # material_key × year_month × usage_type で集計
    summary_df = pred_df.groupby(['material_key', 'predict_year_month', 'usage_type']).agg({
        'actual': [
            ('actual_value_count', lambda x: (x > 0).sum()),
            ('actual_value_mean', 'mean')
        ],
        'predicted': [
            ('predict_value_mean', 'mean')
        ]
    }).reset_index()

    # カラム名をフラット化
    summary_df.columns = ['material_key', 'predict_year_month', 'usage_type',
                          'actual_value_count', 'actual_value_mean', 'predict_value_mean']

    # error_value_meanを計算（actual > 0のレコードのみ）
    error_means = []
    for _, group in pred_df[pred_df['actual'] > 0].groupby(['material_key', 'predict_year_month', 'usage_type']):
        relative_errors = np.abs(group['predicted'] - group['actual']) / group['actual']
        error_means.append({
            'material_key': group['material_key'].iloc[0],
            'predict_year_month': group['predict_year_month'].iloc[0],
            'usage_type': group['usage_type'].iloc[0],
            'error_value_mean': relative_errors.mean()
        })

    if error_means:
        error_df = pd.DataFrame(error_means)
        summary_df = summary_df.merge(
            error_df,
            on=['material_key', 'predict_year_month', 'usage_type'],
            how='left'
        )
    else:
        summary_df['error_value_mean'] = 0

    # 数値を丸める
    summary_df['actual_value_mean'] = summary_df['actual_value_mean'].round(2)
    summary_df['predict_value_mean'] = summary_df['predict_value_mean'].round(2)
    summary_df['error_value_mean'] = summary_df['error_value_mean'].round(4)

    # S3に保存
    summary_key = f"output/evaluation/confirmed_order_demand_yamasa_material_summary_{timestamp}.csv"
    latest_summary_key = f"output/evaluation/confirmed_order_demand_yamasa_material_summary_latest.csv"

    s3_handler.write_csv(summary_df, summary_key)
    s3_handler.write_csv(summary_df, latest_summary_key)

    print(f"  Material summary saved: s3://{s3_handler.bucket_name}/{latest_summary_key}")

    # 3. feature_importance_latest.csv
    importance_dfs = []

    if combined_results.get('business_feature_importance') is not None:
        business_fi = combined_results['business_feature_importance'].copy()
        business_fi['usage_type'] = 'business'
        importance_dfs.append(business_fi)

    if combined_results.get('household_feature_importance') is not None:
        household_fi = combined_results['household_feature_importance'].copy()
        household_fi['usage_type'] = 'household'
        importance_dfs.append(household_fi)

    if importance_dfs:
        combined_importance = pd.concat(importance_dfs, ignore_index=True)
        combined_importance = combined_importance.sort_values(
            ['usage_type', 'importance'], ascending=[True, False]
        )

        # S3に保存
        importance_key = f"output/evaluation/confirmed_order_demand_yamasa_feature_importance_{timestamp}.csv"
        latest_importance_key = f"output/evaluation/confirmed_order_demand_yamasa_feature_importance_latest.csv"

        s3_handler.write_csv(combined_importance, importance_key)
        s3_handler.write_csv(combined_importance, latest_importance_key)

        print(f"  Feature importance saved: s3://{s3_handler.bucket_name}/{latest_importance_key}")

    # 4. メトリクスのJSON保存
    metrics_json = {
        'execution_date': datetime.now().isoformat(),
        'combined_metrics': combined_results['metrics'],
        'business_metrics': combined_results['business_metrics'],
        'household_metrics': combined_results['household_metrics']
    }

    # エラー分析を追加
    abs_errors = pred_df['abs_error'].values
    for threshold in [5, 10, 20, 30, 50]:
        ratio = np.mean(abs_errors <= threshold) * 100
        metrics_json['combined_metrics'][f'within_{threshold}_ratio'] = ratio

    metrics_key = f"output/evaluation/confirmed_order_demand_yamasa_metrics_{timestamp}.json"
    latest_metrics_key = f"output/evaluation/confirmed_order_demand_yamasa_metrics_latest.json"

    s3_handler.s3_client.put_object(
        Bucket=s3_handler.bucket_name,
        Key=metrics_key,
        Body=json.dumps(metrics_json, indent=2, default=str),
        ContentType='application/json'
    )

    s3_handler.s3_client.put_object(
        Bucket=s3_handler.bucket_name,
        Key=latest_metrics_key,
        Body=json.dumps(metrics_json, indent=2, default=str),
        ContentType='application/json'
    )

    print(f"  Metrics saved: s3://{s3_handler.bucket_name}/{latest_metrics_key}")

    return {
        'predictions_path': latest_csv_key,
        'summary_path': latest_summary_key,
        'importance_path': latest_importance_key,
        'metrics_path': latest_metrics_key
    }


def main():
    """メイン処理"""
    parser = argparse.ArgumentParser(description='Train models by usage_type')
    parser.add_argument('--train-end-date', type=str, default="2024-12-31",
                        help='Training data end date')
    parser.add_argument('--step-count', type=int, default=1,
                        help='Number of months to predict')
    parser.add_argument('--usage-types', nargs='+', default=['business', 'household'],
                        help='Usage types to process')

    args = parser.parse_args()

    print(f"""
    ============================================================
    usage_type別モデル学習・予測
    ============================================================
    学習終了日: {args.train_end_date}
    予測期間: {args.step_count}ヶ月
    対象usage_type: {args.usage_types}
    """)

    s3_handler = S3Handler("fiby-yamasa-prediction-2")

    # 各usage_typeでモデル学習
    results_by_type = {}

    for usage_type in args.usage_types:
        # 特徴量読み込み
        df_features = load_features_for_usage_type(usage_type)

        # モデル学習・予測
        model_data = train_model_for_usage_type(
            df_features=df_features,
            usage_type=usage_type,
            train_end_date=args.train_end_date,
            step_count=args.step_count
        )

        results_by_type[usage_type] = model_data

        # モデルを保存
        if model_data.get('model'):
            model_key = f"output/models/confirmed_order_demand_yamasa_model_{usage_type}_latest.pkl"
            buffer = BytesIO()
            pickle.dump(model_data['model'], buffer)
            buffer.seek(0)
            s3_handler.s3_client.put_object(
                Bucket=s3_handler.bucket_name,
                Key=model_key,
                Body=buffer.getvalue()
            )
            print(f"Model saved: s3://{s3_handler.bucket_name}/{model_key}")

    # 結果を統合
    if 'business' in results_by_type and 'household' in results_by_type:
        combined_results = combine_results(
            business_data=results_by_type['business'],
            household_data=results_by_type['household']
        )

        # 統合結果を保存
        saved_paths = save_combined_results(combined_results, s3_handler)

        print("\n" + "="*70)
        print("✅ Model training and prediction completed successfully!")
        print("="*70)

        return combined_results
    else:
        print("Error: Both business and household models are required")
        return None


if __name__ == "__main__":
    from io import BytesIO
    main()