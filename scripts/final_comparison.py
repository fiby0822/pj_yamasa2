#!/usr/bin/env python3
"""
最終的な誤差率平均比較（正しい計算方法）
"""
import boto3
import pandas as pd
import numpy as np
from io import StringIO
import json
from datetime import datetime

s3_client = boto3.client('s3', region_name='ap-northeast-1')
BUCKET_NAME = 'fiby-yamasa-prediction'

def read_csv_from_s3(path):
    """S3からCSVファイルを読み込み"""
    try:
        obj = s3_client.get_object(Bucket=BUCKET_NAME, Key=path)
        df = pd.read_csv(StringIO(obj['Body'].read().decode('utf-8')))
        return df
    except Exception as e:
        print(f"エラー: {e}")
        return None

def calculate_error_metrics_from_csv(df, model_name=""):
    """
    CSVファイルから誤差率平均を正しく計算
    """
    print(f"\n{model_name}の評価:")
    print(f"  総レコード数: {len(df):,}")

    # material_key毎にグループ化
    grouped = df.groupby('material_key')

    error_rates = []
    for mk, group in grouped:
        # 実績値>0のレコードのみ
        positive_actual = group[group['actual'] > 0]
        if len(positive_actual) > 0:
            # 各レコードの誤差率を計算
            mk_error_rates = np.abs(positive_actual['predicted'] - positive_actual['actual']) / positive_actual['actual']
            # material_keyの誤差率平均
            error_rate_mean = mk_error_rates.mean()
            error_rates.append(error_rate_mean)

    error_rates = np.array(error_rates)
    print(f"  Material Key数: {len(error_rates):,}")
    print(f"  実績値>0のMaterial Key数: {len(error_rates):,}")

    # 統計
    error_rate_mean = error_rates.mean()
    error_rate_median = np.median(error_rates)

    # 閾値別カウント
    within_20 = (error_rates <= 0.2).sum()
    within_30 = (error_rates <= 0.3).sum()
    within_50 = (error_rates <= 0.5).sum()
    total_mks = len(error_rates)

    print(f"  誤差率平均の平均値: {error_rate_mean:.2%}")
    print(f"  誤差率平均の中央値: {error_rate_median:.2%}")
    print(f"  誤差率平均≤20%: {within_20}/{total_mks} ({within_20/total_mks*100:.1f}%)")
    print(f"  誤差率平均≤30%: {within_30}/{total_mks} ({within_30/total_mks*100:.1f}%)")
    print(f"  誤差率平均≤50%: {within_50}/{total_mks} ({within_50/total_mks*100:.1f}%)")

    # 基本指標（全レコード）
    mae = np.abs(df['predicted'] - df['actual']).mean()
    rmse = np.sqrt(((df['predicted'] - df['actual']) ** 2).mean())
    print(f"  MAE: {mae:.2f}, RMSE: {rmse:.2f}")

    # 誤差率の分布
    print(f"\n  誤差率平均の分布:")
    percentiles = [10, 25, 50, 75, 90]
    pcts = np.percentile(error_rates, percentiles)
    for p, v in zip(percentiles, pcts):
        print(f"    {p}パーセンタイル: {v:.2%}")

    return {
        'total_records': len(df),
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
    print("誤差率平均の正しい計算方法での比較分析（最終版）")
    print("="*80)
    print("\n誤差率平均の定義:")
    print("- 各material_keyの各日: 誤差率 = |予測値 - 実績値| / 実績値（実績値>0の日のみ）")
    print("- 各material_keyの誤差率平均 = そのmaterial_keyの全誤差率の平均")
    print("- 1material_keyに対して1つの誤差率平均値")
    print("="*80)

    results = {}

    # 分離後モデル（最新のCSV）
    print("\n1. 分離後モデル（usage_type別）の結果")
    df_after = read_csv_from_s3('output/evaluation/confirmed_order_demand_yamasa_predictions_latest.csv')

    if df_after is not None:
        # usage_type列があるか確認
        if 'usage_type' in df_after.columns:
            results['after'] = calculate_error_metrics_from_csv(df_after, "分離後モデル（全体）")

            # usage_type別の評価も実施
            for usage_type in df_after['usage_type'].unique():
                df_type = df_after[df_after['usage_type'] == usage_type]
                print(f"\n  {usage_type.upper()}のみ:")
                type_metrics = calculate_error_metrics_from_csv(df_type, f"  {usage_type}")
                results[f'after_{usage_type}'] = type_metrics
        else:
            print("  注意: usage_type列が見つかりません（これは分離前モデルの可能性があります）")
            results['before'] = calculate_error_metrics_from_csv(df_after, "モデル")

    # 過去の分離前モデルの結果も探す（もしあれば）
    print("\n2. 分離前モデル（通常版）を探しています...")

    # 最新の通常モデルの予測を探す
    try:
        # タイムスタンプ付きファイルをリスト
        response = s3_client.list_objects_v2(
            Bucket=BUCKET_NAME,
            Prefix='output/evaluation/confirmed_order_demand_yamasa_predictions_2025',
            MaxKeys=100
        )

        if 'Contents' in response:
            # usage_typeを含まないファイルを探す
            for obj in sorted(response['Contents'], key=lambda x: x['LastModified'], reverse=True):
                if 'usage_type' not in obj['Key'] and obj['Key'].endswith('.csv'):
                    print(f"  見つかりました: {obj['Key']}")
                    df_before = read_csv_from_s3(obj['Key'])
                    if df_before is not None and 'usage_type' not in df_before.columns:
                        results['before'] = calculate_error_metrics_from_csv(df_before, "分離前モデル")
                        break
    except Exception as e:
        print(f"  探索中にエラー: {e}")

    # 比較結果の表示
    if 'before' in results and 'after' in results:
        print("\n" + "="*80)
        print("分離前 vs 分離後 比較結果")
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

        improvements = {}
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

            improvements[key] = improvement

            if abs(improvement) < 0.1:
                improvement_str = "→ 変化なし"
            elif improvement > 0:
                improvement_str = f"✓ {improvement:.1f}%改善"
            else:
                improvement_str = f"✗ {abs(improvement):.1f}%悪化"

            print(f"{name:<25} {before_str:<15} {after_str:<15} {improvement_str:<15}")

        # 改善のサマリー
        print("\n" + "="*80)
        print("改善サマリー")
        print("="*80)

        improved = []
        worsened = []

        for name, key, lower_is_better in metrics_names:
            imp = improvements[key]
            if abs(imp) > 0.1:
                if imp > 0:
                    improved.append(f"  ✓ {name}: {imp:.1f}%改善")
                else:
                    worsened.append(f"  ✗ {name}: {abs(imp):.1f}%悪化")

        if improved:
            print("改善した指標:")
            for item in improved:
                print(item)

        if worsened:
            print("\n悪化した指標:")
            for item in worsened:
                print(item)

        # 結果をJSON保存
        result = {
            'execution_date': datetime.now().isoformat(),
            'definition': '誤差率平均 = 各material_keyに対して実績値>0の日の誤差率の平均',
            'before_separation': before_metrics,
            'after_separation': after_metrics,
            'improvements': improvements
        }

        with open('final_comparison_results.json', 'w', encoding='utf-8') as f:
            json.dump(result, f, ensure_ascii=False, indent=2)

        print(f"\n結果を final_comparison_results.json に保存しました。")

    elif 'after' in results:
        print("\n" + "="*80)
        print("分離後モデルの結果サマリー")
        print("="*80)
        print("（分離前モデルの結果は見つかりませんでした）")

        # 結果をJSON保存
        result = {
            'execution_date': datetime.now().isoformat(),
            'definition': '誤差率平均 = 各material_keyに対して実績値>0の日の誤差率の平均',
            'after_separation': results.get('after'),
            'after_business': results.get('after_business'),
            'after_household': results.get('after_household')
        }

        with open('after_model_results.json', 'w', encoding='utf-8') as f:
            json.dump(result, f, ensure_ascii=False, indent=2)

        print(f"\n結果を after_model_results.json に保存しました。")

if __name__ == "__main__":
    main()