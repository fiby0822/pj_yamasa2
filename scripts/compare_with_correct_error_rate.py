#!/usr/bin/env python3
"""
誤差率平均の正しい計算方法で使用タイプ分離前・分離後の比較
誤差率平均の定義：各material_keyに対して、実績値>0の日の誤差率の平均値
"""
import pandas as pd
import numpy as np
import json
from pathlib import Path
import boto3
from datetime import datetime

# S3クライアント
s3_client = boto3.client('s3', region_name='ap-northeast-1')
BUCKET_NAME = 'fiby-yamasa-prediction-2'

def download_from_s3(s3_path):
    """S3からファイルをダウンロード"""
    local_path = f"/tmp/{Path(s3_path).name}"
    s3_client.download_file(BUCKET_NAME, s3_path, local_path)
    return local_path

def calculate_error_metrics_corrected(predictions_path):
    """
    誤差率平均を正しく計算
    - 各material_keyに対して、実績値>0の日の誤差率の平均を計算
    - 1つのmaterial_keyに対して1つの誤差率平均値
    """
    # データ読み込み
    if predictions_path.startswith('output/'):
        local_path = download_from_s3(predictions_path)
        df = pd.read_parquet(local_path)
    else:
        df = pd.read_parquet(predictions_path)

    # テストデータのみ抽出
    test_df = df[df['data_type'] == 'test'].copy()

    if len(test_df) == 0:
        return None

    # 実績値が正の場合のみ誤差率を計算
    test_df['error_rate'] = np.where(
        test_df['actual_value'] > 0,
        np.abs(test_df['predicted_value'] - test_df['actual_value']) / test_df['actual_value'],
        np.nan
    )

    # material_key毎の誤差率平均を計算（実績値>0の日のみ）
    mk_error_rates = test_df.groupby('material_key')['error_rate'].apply(
        lambda x: x.dropna().mean()
    ).dropna()

    # 全体の基本指標
    mae = np.abs(test_df['predicted_value'] - test_df['actual_value']).mean()
    rmse = np.sqrt(((test_df['predicted_value'] - test_df['actual_value']) ** 2).mean())

    # 誤差率平均の統計
    error_rate_mean = mk_error_rates.mean()
    error_rate_median = mk_error_rates.median()

    # 誤差率平均の閾値別カウント
    within_20 = (mk_error_rates <= 0.2).sum()
    within_30 = (mk_error_rates <= 0.3).sum()
    within_50 = (mk_error_rates <= 0.5).sum()
    total_mks = len(mk_error_rates)

    # MAPE（実績値>0の場合のみ）
    mape_df = test_df[test_df['actual_value'] > 0].copy()
    mape = np.abs((mape_df['predicted_value'] - mape_df['actual_value']) / mape_df['actual_value']).mean()

    # 相関係数とR2スコア
    correlation = test_df['predicted_value'].corr(test_df['actual_value'])

    # R2スコア
    ss_res = ((test_df['actual_value'] - test_df['predicted_value']) ** 2).sum()
    ss_tot = ((test_df['actual_value'] - test_df['actual_value'].mean()) ** 2).sum()
    r2_score = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0

    metrics = {
        'total_test_records': len(test_df),
        'total_material_keys': total_mks,
        'mae': float(mae),
        'rmse': float(rmse),
        'error_rate_mean': float(error_rate_mean),
        'error_rate_median': float(error_rate_median),
        'within_20_percent': int(within_20),
        'within_20_percent_ratio': float(within_20 / total_mks),
        'within_30_percent': int(within_30),
        'within_30_percent_ratio': float(within_30 / total_mks),
        'within_50_percent': int(within_50),
        'within_50_percent_ratio': float(within_50 / total_mks),
        'mape': float(mape),
        'correlation': float(correlation),
        'r2_score': float(r2_score)
    }

    return metrics

def compare_models():
    """分離前と分離後のモデルを比較"""

    print("="*80)
    print("誤差率平均の正しい計算方法での比較分析")
    print("="*80)
    print("\n誤差率平均の定義:")
    print("- 各material_keyに対して、実績値>0の日の誤差率の平均を計算")
    print("- 1つのmaterial_keyに対して1つの誤差率平均値")
    print("="*80)

    # 分離前のモデル結果（フィルタリング48個版）
    print("\n1. 使用タイプ分離前のモデル（フィルタリング48個版）の結果を計算...")
    before_path = "output/predictions/confirmed_order_demand_yamasa_predictions_threshold48_20241020_135044.parquet"
    before_metrics = calculate_error_metrics_corrected(before_path)

    if before_metrics:
        print(f"  - テストレコード数: {before_metrics['total_test_records']:,}")
        print(f"  - Material Key数: {before_metrics['total_material_keys']:,}")
        print(f"  - 誤差率平均の平均値: {before_metrics['error_rate_mean']:.2%}")
        print(f"  - 誤差率平均の中央値: {before_metrics['error_rate_median']:.2%}")

    # 分離後のモデル結果
    print("\n2. 使用タイプ分離後のモデルの結果を計算...")
    after_path = "output/predictions/confirmed_order_demand_yamasa_predictions_by_usage_type_20241020_211537.parquet"
    after_metrics = calculate_error_metrics_corrected(after_path)

    if after_metrics:
        print(f"  - テストレコード数: {after_metrics['total_test_records']:,}")
        print(f"  - Material Key数: {after_metrics['total_material_keys']:,}")
        print(f"  - 誤差率平均の平均値: {after_metrics['error_rate_mean']:.2%}")
        print(f"  - 誤差率平均の中央値: {after_metrics['error_rate_median']:.2%}")

    # 比較結果の表示
    if before_metrics and after_metrics:
        print("\n" + "="*80)
        print("比較結果サマリー")
        print("="*80)

        # 表形式で比較
        comparison_data = []
        metrics_names = [
            ('MAE', 'mae'),
            ('RMSE', 'rmse'),
            ('誤差率平均の平均値', 'error_rate_mean'),
            ('誤差率平均の中央値', 'error_rate_median'),
            ('誤差率平均≤20%', 'within_20_percent_ratio'),
            ('誤差率平均≤30%', 'within_30_percent_ratio'),
            ('誤差率平均≤50%', 'within_50_percent_ratio'),
            ('MAPE', 'mape'),
            ('相関係数', 'correlation'),
            ('R²スコア', 'r2_score')
        ]

        print(f"\n{'指標':<25} {'分離前':<15} {'分離後':<15} {'改善率':<15}")
        print("-"*70)

        for name, key in metrics_names:
            before_val = before_metrics[key]
            after_val = after_metrics[key]

            # 改善率の計算（低い方が良い指標と高い方が良い指標で分ける）
            if key in ['mae', 'rmse', 'error_rate_mean', 'error_rate_median', 'mape']:
                # 低い方が良い指標
                improvement = (before_val - after_val) / before_val * 100
                before_str = f"{before_val:.4f}" if before_val < 1 else f"{before_val:.2f}"
                after_str = f"{after_val:.4f}" if after_val < 1 else f"{after_val:.2f}"
            else:
                # 高い方が良い指標
                improvement = (after_val - before_val) / before_val * 100
                if key.endswith('_ratio'):
                    before_str = f"{before_val:.1%}"
                    after_str = f"{after_val:.1%}"
                else:
                    before_str = f"{before_val:.4f}"
                    after_str = f"{after_val:.4f}"

            improvement_str = f"{improvement:+.1f}%"
            if improvement > 0:
                improvement_str = f"✓ {improvement_str}"
            elif improvement < 0:
                improvement_str = f"✗ {improvement_str}"

            print(f"{name:<25} {before_str:<15} {after_str:<15} {improvement_str:<15}")

            comparison_data.append({
                'metric': name,
                'before': before_val,
                'after': after_val,
                'improvement_percent': improvement
            })

        # 改善のサマリー
        print("\n" + "="*80)
        print("改善サマリー")
        print("="*80)

        improvements = {
            'error_rate_mean': (before_metrics['error_rate_mean'] - after_metrics['error_rate_mean']) / before_metrics['error_rate_mean'] * 100,
            'error_rate_median': (before_metrics['error_rate_median'] - after_metrics['error_rate_median']) / before_metrics['error_rate_median'] * 100,
            'mae': (before_metrics['mae'] - after_metrics['mae']) / before_metrics['mae'] * 100,
            'rmse': (before_metrics['rmse'] - after_metrics['rmse']) / before_metrics['rmse'] * 100,
        }

        for metric, improvement in improvements.items():
            if metric == 'error_rate_mean':
                metric_name = '誤差率平均の平均値'
            elif metric == 'error_rate_median':
                metric_name = '誤差率平均の中央値'
            else:
                metric_name = metric.upper()

            if improvement > 0:
                print(f"✓ {metric_name}: {improvement:.1f}% 改善")
            else:
                print(f"✗ {metric_name}: {abs(improvement):.1f}% 悪化")

        # 結果をJSON保存
        result = {
            'execution_date': datetime.now().isoformat(),
            'definition': '誤差率平均 = 各material_keyに対して実績値>0の日の誤差率の平均',
            'before_separation': before_metrics,
            'after_separation': after_metrics,
            'comparison': comparison_data,
            'improvements': improvements
        }

        output_path = 'corrected_error_rate_comparison.json'
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(result, f, ensure_ascii=False, indent=2)

        print(f"\n結果を {output_path} に保存しました。")

        return result

    return None

if __name__ == "__main__":
    result = compare_models()