#!/usr/bin/env python3
"""
分離前と分離後のモデルを学習し、誤差率平均の正しい計算方法で比較
"""
import subprocess
import pandas as pd
import numpy as np
import json
from datetime import datetime
import boto3
from pathlib import Path

s3_client = boto3.client('s3', region_name='ap-northeast-1')
BUCKET_NAME = 'fiby-yamasa-prediction'

def download_from_s3(s3_path):
    """S3からファイルをダウンロード"""
    local_path = f"/tmp/{Path(s3_path).name}"
    s3_client.download_file(BUCKET_NAME, s3_path, local_path)
    return local_path

def calculate_error_metrics(predictions_path, model_name=""):
    """
    誤差率平均を正しく計算
    - 各material_keyに対して、実績値>0の日の誤差率の平均を計算
    """
    print(f"\n{model_name}の評価...")

    # データ読み込み
    try:
        if predictions_path.startswith('output/'):
            local_path = download_from_s3(predictions_path)
            df = pd.read_parquet(local_path)
        else:
            df = pd.read_parquet(predictions_path)
    except Exception as e:
        print(f"  エラー: {e}")
        return None

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
    print(f"  - MAE: {mae:.2f}, RMSE: {rmse:.2f}")

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

def run_training(model_type):
    """モデル学習を実行"""
    print(f"\n{'='*60}")
    print(f"{model_type}モデルの学習を開始...")
    print(f"{'='*60}")

    if model_type == "分離前":
        # 通常のモデル学習
        cmd = [
            'python', 'scripts/train/train_model.py',
            '--train-end-date', '2024-12-31',
            '--step-count', '1'
        ]
    else:
        # 使用タイプ分離版の学習
        cmd = [
            'python', 'scripts/train/train_model_by_usage_type.py',
            '--train-end-date', '2024-12-31',
            '--step-count', '1'
        ]

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        print("学習完了！")
        return True
    except subprocess.CalledProcessError as e:
        print(f"学習エラー: {e}")
        print(f"エラー出力: {e.stderr}")
        return False

def main():
    """メイン処理"""
    print("="*80)
    print("誤差率平均の正しい計算方法での比較分析")
    print("="*80)
    print("\n誤差率平均の定義:")
    print("- 各material_keyの各日: 誤差率 = |予測値 - 実績値| / 実績値（実績値>0の日のみ）")
    print("- 各material_keyの誤差率平均 = そのmaterial_keyの全誤差率の平均")
    print("- 1material_keyに対して1つの誤差率平均値")

    results = {}

    # 1. 分離前モデルの学習と評価
    if run_training("分離前"):
        # 予測結果の評価
        pred_path = "output/predictions/confirmed_order_demand_yamasa_predictions_latest.parquet"
        results['before'] = calculate_error_metrics(pred_path, "分離前モデル")

    # 2. 分離後モデルの学習と評価
    if run_training("分離後"):
        # 予測結果の評価
        pred_path = "output/predictions/confirmed_order_demand_yamasa_predictions_by_usage_type_latest.parquet"
        results['after'] = calculate_error_metrics(pred_path, "分離後モデル")

    # 3. 比較結果の表示
    if 'before' in results and 'after' in results:
        print("\n" + "="*80)
        print("比較結果サマリー")
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

        # 結果を保存
        result = {
            'execution_date': datetime.now().isoformat(),
            'definition': '誤差率平均 = 各material_keyに対して実績値>0の日の誤差率の平均',
            'before_separation': before_metrics,
            'after_separation': after_metrics,
        }

        with open('final_comparison_results.json', 'w', encoding='utf-8') as f:
            json.dump(result, f, ensure_ascii=False, indent=2)

        print(f"\n結果を final_comparison_results.json に保存しました。")

if __name__ == "__main__":
    main()