#!/usr/bin/env python3
"""
利用可能な予測結果を分析して比較
"""
import pandas as pd
import numpy as np
import json
from datetime import datetime
import boto3
from pathlib import Path

s3_client = boto3.client('s3', region_name='ap-northeast-1')
BUCKET_NAME = 'fiby-yamasa-prediction-2'

def download_from_s3(s3_path):
    """S3からファイルをダウンロード"""
    local_path = f"/tmp/{Path(s3_path).name}"
    try:
        s3_client.download_file(BUCKET_NAME, s3_path, local_path)
        return local_path
    except:
        return None

def calculate_error_metrics(df, model_name=""):
    """
    誤差率平均を正しく計算
    """
    print(f"\n{model_name}の評価...")

    # テストデータのみ抽出
    test_df = df[df['data_type'] == 'test'].copy()

    if len(test_df) == 0:
        print("  警告: テストデータが見つかりません")
        return None

    print(f"  - テストレコード数: {len(test_df):,}")

    # 実績値が正の場合のみ誤差率を計算
    test_df['error_rate'] = np.where(
        test_df['actual_value'] > 0,
        np.abs(test_df['predicted_value'] - test_df['actual_value']) / test_df['actual_value'],
        np.nan
    )

    # material_key毎の誤差率平均を計算
    mk_error_rates = test_df.groupby('material_key')['error_rate'].apply(
        lambda x: x.dropna().mean()
    ).dropna()

    print(f"  - Material Key数: {len(mk_error_rates):,}")
    print(f"  - 実績値>0のレコード数: {test_df[test_df['actual_value'] > 0].shape[0]:,}")

    # 基本指標
    mae = np.abs(test_df['predicted_value'] - test_df['actual_value']).mean()
    rmse = np.sqrt(((test_df['predicted_value'] - test_df['actual_value']) ** 2).mean())

    # 誤差率平均の統計
    error_rate_mean = mk_error_rates.mean()
    error_rate_median = mk_error_rates.median()

    # 閾値別カウント
    within_20 = (mk_error_rates <= 0.2).sum()
    within_30 = (mk_error_rates <= 0.3).sum()
    within_50 = (mk_error_rates <= 0.5).sum()
    total_mks = len(mk_error_rates)

    print(f"  - 誤差率平均の平均値: {error_rate_mean:.2%}")
    print(f"  - 誤差率平均の中央値: {error_rate_median:.2%}")
    print(f"  - 誤差率平均≤20%: {within_20}/{total_mks} ({within_20/total_mks*100:.1f}%)")

    # 誤差率の分布を表示
    percentiles = [10, 25, 50, 75, 90]
    pcts = np.percentile(mk_error_rates, percentiles)
    print(f"\n  誤差率平均の分布:")
    for p, v in zip(percentiles, pcts):
        print(f"    {p}パーセンタイル: {v:.2%}")

    return {
        'total_test_records': len(test_df),
        'total_material_keys': total_mks,
        'mae': float(mae),
        'rmse': float(rmse),
        'error_rate_mean': float(error_rate_mean),
        'error_rate_median': float(error_rate_median),
        'within_20_percent': int(within_20),
        'within_20_percent_ratio': float(within_20 / total_mks) if total_mks > 0 else 0,
        'within_30_percent': int(within_30),
        'within_30_percent_ratio': float(within_30 / total_mks) if total_mks > 0 else 0,
        'within_50_percent': int(within_50),
        'within_50_percent_ratio': float(within_50 / total_mks) if total_mks > 0 else 0,
    }

def main():
    """メイン処理"""
    print("="*80)
    print("誤差率平均の正しい計算方法での分析")
    print("="*80)
    print("\n誤差率平均の定義:")
    print("- 各material_keyの各日: 誤差率 = |予測値 - 実績値| / 実績値（実績値>0の日のみ）")
    print("- 各material_keyの誤差率平均 = そのmaterial_keyの全誤差率の平均")
    print("- 1material_keyに対して1つの誤差率平均値")

    results = {}

    # 分離前モデル（最新）の確認
    print("\n1. 分離前モデル（最新）を確認...")
    pred_path = "output/predictions/confirmed_order_demand_yamasa_predictions_latest.parquet"
    local_path = download_from_s3(pred_path)

    if local_path:
        df_before = pd.read_parquet(local_path)
        results['before'] = calculate_error_metrics(df_before, "分離前モデル")
    else:
        print("  分離前モデルの予測結果が見つかりません")

    # 分離後モデルが完了しているか確認
    print("\n2. 分離後モデルを確認...")
    pred_path_after = "output/predictions/confirmed_order_demand_yamasa_predictions_by_usage_type_latest.parquet"
    local_path_after = download_from_s3(pred_path_after)

    if local_path_after:
        df_after = pd.read_parquet(local_path_after)
        results['after'] = calculate_error_metrics(df_after, "分離後モデル")
    else:
        print("  分離後モデルはまだ実行中のようです")

    # 結果の表示
    if 'before' in results:
        print("\n" + "="*80)
        print("分離前モデルの結果サマリー")
        print("="*80)

        metrics = results['before']
        print(f"\n基本指標:")
        print(f"  - MAE: {metrics['mae']:.2f}")
        print(f"  - RMSE: {metrics['rmse']:.2f}")
        print(f"  - 誤差率平均の平均値: {metrics['error_rate_mean']:.2%}")
        print(f"  - 誤差率平均の中央値: {metrics['error_rate_median']:.2%}")

        print(f"\n精度分布:")
        print(f"  - 誤差率平均≤20%: {metrics['within_20_percent_ratio']:.1%}")
        print(f"  - 誤差率平均≤30%: {metrics['within_30_percent_ratio']:.1%}")
        print(f"  - 誤差率平均≤50%: {metrics['within_50_percent_ratio']:.1%}")

    # 両方の結果がある場合は比較
    if 'before' in results and 'after' in results:
        print("\n" + "="*80)
        print("比較結果")
        print("="*80)

        before_metrics = results['before']
        after_metrics = results['after']

        metrics_names = [
            ('誤差率平均の平均値', 'error_rate_mean', True),
            ('誤差率平均の中央値', 'error_rate_median', True),
            ('誤差率平均≤20%', 'within_20_percent_ratio', False),
            ('誤差率平均≤30%', 'within_30_percent_ratio', False),
            ('誤差率平均≤50%', 'within_50_percent_ratio', False),
            ('MAE', 'mae', True),
            ('RMSE', 'rmse', True),
        ]

        print(f"\n{'指標':<25} {'分離前':<15} {'分離後':<15} {'改善':<15}")
        print("-"*70)

        for name, key, lower_is_better in metrics_names:
            before_val = before_metrics[key]
            after_val = after_metrics[key]

            # フォーマット
            if key in ['error_rate_mean', 'error_rate_median']:
                before_str = f"{before_val:.2%}"
                after_str = f"{after_val:.2%}"
            elif key.endswith('_ratio'):
                before_str = f"{before_val:.1%}"
                after_str = f"{after_val:.1%}"
            else:
                before_str = f"{before_val:.2f}"
                after_str = f"{after_val:.2f}"

            # 改善率計算
            if lower_is_better:
                improvement = (before_val - after_val) / before_val * 100 if before_val != 0 else 0
            else:
                improvement = (after_val - before_val) / before_val * 100 if before_val != 0 else 0

            if abs(improvement) < 0.1:
                improvement_str = "→ 変化なし"
            elif improvement > 0:
                improvement_str = f"✓ {improvement:.1f}%改善"
            else:
                improvement_str = f"✗ {abs(improvement):.1f}%悪化"

            print(f"{name:<25} {before_str:<15} {after_str:<15} {improvement_str:<15}")

if __name__ == "__main__":
    main()