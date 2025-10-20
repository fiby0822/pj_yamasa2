#!/usr/bin/env python3
"""
誤差率平均の正しい計算方法で使用タイプ分離前・分離後の比較
最新のファイルを使用して比較
"""
import pandas as pd
import numpy as np
import json
from pathlib import Path
import boto3
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# S3クライアント
s3_client = boto3.client('s3', region_name='ap-northeast-1')
BUCKET_NAME = 'fiby-yamasa-prediction'

def download_from_s3(s3_path):
    """S3からファイルをダウンロード"""
    local_path = f"/tmp/{Path(s3_path).name}"
    s3_client.download_file(BUCKET_NAME, s3_path, local_path)
    return local_path

def calculate_error_metrics_corrected(predictions_path, model_name=""):
    """
    誤差率平均を正しく計算
    - 各material_keyに対して、実績値>0の日の誤差率の平均を計算
    - 1つのmaterial_keyに対して1つの誤差率平均値
    """
    print(f"\n{model_name}のデータを処理中...")

    # データ読み込み
    try:
        if predictions_path.startswith('s3://'):
            # S3パスから読み込み
            s3_path = predictions_path.replace(f's3://{BUCKET_NAME}/', '')
            local_path = download_from_s3(s3_path)
            df = pd.read_parquet(local_path)
        else:
            df = pd.read_parquet(predictions_path)

        print(f"  - データ読み込み完了: {len(df):,}レコード")
    except Exception as e:
        print(f"  - エラー: {e}")
        return None

    # テストデータのみ抽出
    test_df = df[df['data_type'] == 'test'].copy()
    print(f"  - テストデータ: {len(test_df):,}レコード")

    if len(test_df) == 0:
        print("  - 警告: テストデータが見つかりません")
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

    print(f"  - Material Key数: {len(mk_error_rates):,}")
    print(f"  - 実績値>0のレコード数: {test_df[test_df['actual_value'] > 0].shape[0]:,}")

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
    if len(mape_df) > 0:
        mape = np.abs((mape_df['predicted_value'] - mape_df['actual_value']) / mape_df['actual_value']).mean()
    else:
        mape = np.nan

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
        'within_20_percent_ratio': float(within_20 / total_mks) if total_mks > 0 else 0,
        'within_30_percent': int(within_30),
        'within_30_percent_ratio': float(within_30 / total_mks) if total_mks > 0 else 0,
        'within_50_percent': int(within_50),
        'within_50_percent_ratio': float(within_50 / total_mks) if total_mks > 0 else 0,
        'mape': float(mape) if not np.isnan(mape) else None,
        'correlation': float(correlation),
        'r2_score': float(r2_score)
    }

    return metrics

def find_latest_predictions():
    """S3から最新の予測ファイルを探す"""
    print("\nS3から最新の予測ファイルを検索中...")

    try:
        # 最新のlatestファイルを確認
        response = s3_client.list_objects_v2(
            Bucket=BUCKET_NAME,
            Prefix='output/predictions/confirmed_order_demand_yamasa_predictions_latest'
        )

        latest_file = None
        if 'Contents' in response:
            latest_file = 'output/predictions/confirmed_order_demand_yamasa_predictions_latest.parquet'
            print(f"  - 最新ファイル（分離前）: {latest_file}")

        # usage_type分離版を確認
        response_ut = s3_client.list_objects_v2(
            Bucket=BUCKET_NAME,
            Prefix='output/predictions/confirmed_order_demand_yamasa_predictions_by_usage_type_latest'
        )

        usage_type_file = None
        if 'Contents' in response_ut:
            usage_type_file = 'output/predictions/confirmed_order_demand_yamasa_predictions_by_usage_type_latest.parquet'
            print(f"  - 最新ファイル（分離後）: {usage_type_file}")

        return latest_file, usage_type_file

    except Exception as e:
        print(f"  - エラー: {e}")
        return None, None

def compare_models():
    """分離前と分離後のモデルを比較"""

    print("="*80)
    print("誤差率平均の正しい計算方法での比較分析")
    print("="*80)
    print("\n誤差率平均の定義:")
    print("- 各material_keyの各日について: 誤差率 = |予測値 - 実績値| / 実績値（実績値>0の日のみ）")
    print("- 各material_keyの誤差率平均 = そのmaterial_keyの全誤差率の平均")
    print("- 1つのmaterial_keyに対して1つの誤差率平均値")
    print("="*80)

    # 最新ファイルを探す
    before_file, after_file = find_latest_predictions()

    if not before_file or not after_file:
        print("\n警告: 予測ファイルが見つかりません")
        return None

    # 分離前のモデル結果
    print("\n1. 使用タイプ分離前のモデル")
    before_path = f"s3://{BUCKET_NAME}/{before_file}"
    before_metrics = calculate_error_metrics_corrected(before_path, "分離前モデル")

    if before_metrics:
        print(f"\n  結果サマリー:")
        print(f"    - 誤差率平均の平均値: {before_metrics['error_rate_mean']:.2%}")
        print(f"    - 誤差率平均の中央値: {before_metrics['error_rate_median']:.2%}")
        print(f"    - MAE: {before_metrics['mae']:.2f}")
        print(f"    - RMSE: {before_metrics['rmse']:.2f}")

    # 分離後のモデル結果
    print("\n2. 使用タイプ分離後のモデル")
    after_path = f"s3://{BUCKET_NAME}/{after_file}"
    after_metrics = calculate_error_metrics_corrected(after_path, "分離後モデル")

    if after_metrics:
        print(f"\n  結果サマリー:")
        print(f"    - 誤差率平均の平均値: {after_metrics['error_rate_mean']:.2%}")
        print(f"    - 誤差率平均の中央値: {after_metrics['error_rate_median']:.2%}")
        print(f"    - MAE: {after_metrics['mae']:.2f}")
        print(f"    - RMSE: {after_metrics['rmse']:.2f}")

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
            before_val = before_metrics.get(key)
            after_val = after_metrics.get(key)

            if before_val is None or after_val is None:
                continue

            # 改善率の計算（低い方が良い指標と高い方が良い指標で分ける）
            if key in ['mae', 'rmse', 'error_rate_mean', 'error_rate_median', 'mape']:
                # 低い方が良い指標
                improvement = (before_val - after_val) / before_val * 100 if before_val != 0 else 0
                if key in ['error_rate_mean', 'error_rate_median']:
                    before_str = f"{before_val:.2%}"
                    after_str = f"{after_val:.2%}"
                elif key == 'mape':
                    before_str = f"{before_val:.2%}"
                    after_str = f"{after_val:.2%}"
                else:
                    before_str = f"{before_val:.2f}"
                    after_str = f"{after_val:.2f}"
            else:
                # 高い方が良い指標
                improvement = (after_val - before_val) / abs(before_val) * 100 if before_val != 0 else 0
                if key.endswith('_ratio'):
                    before_str = f"{before_val:.1%}"
                    after_str = f"{after_val:.1%}"
                else:
                    before_str = f"{before_val:.4f}"
                    after_str = f"{after_val:.4f}"

            improvement_str = f"{improvement:+.1f}%"
            if abs(improvement) < 0.1:
                improvement_str = "→ 変化なし"
            elif improvement > 0 and key in ['mae', 'rmse', 'error_rate_mean', 'error_rate_median', 'mape']:
                improvement_str = f"✓ {improvement:.1f}%改善"
            elif improvement > 0:
                improvement_str = f"✓ {improvement:.1f}%向上"
            else:
                improvement_str = f"✗ {abs(improvement):.1f}%悪化"

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

        improvements = {}
        for key in ['error_rate_mean', 'error_rate_median', 'mae', 'rmse']:
            if before_metrics.get(key) and after_metrics.get(key):
                improvements[key] = (before_metrics[key] - after_metrics[key]) / before_metrics[key] * 100

        for metric, improvement in improvements.items():
            if metric == 'error_rate_mean':
                metric_name = '誤差率平均の平均値'
            elif metric == 'error_rate_median':
                metric_name = '誤差率平均の中央値'
            else:
                metric_name = metric.upper()

            if abs(improvement) < 0.1:
                print(f"→ {metric_name}: ほぼ変化なし")
            elif improvement > 0:
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