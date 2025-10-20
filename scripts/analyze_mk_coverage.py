#!/usr/bin/env python3
"""
Material Key数とカバー率の関係を分析
"""
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))

import pandas as pd
import boto3
from io import BytesIO
import numpy as np

def analyze_coverage():
    """Material Keyのカバー率を詳細分析"""
    print("Material Keyカバー率分析")
    print("="*60)

    # S3からデータ読み込み
    s3_client = boto3.client('s3')
    response = s3_client.get_object(
        Bucket="fiby-yamasa-prediction-2",
        Key="output/features/confirmed_order_demand_yamasa_features_latest.parquet"
    )
    df = pd.read_parquet(BytesIO(response['Body'].read()))

    # Material Key別の統計
    mk_stats = df.groupby('material_key')['actual_value'].agg(['sum', 'count', 'mean']).reset_index()
    mk_stats.columns = ['material_key', 'total_value', 'count', 'mean_value']
    mk_stats = mk_stats.sort_values('total_value', ascending=False)

    total_value = mk_stats['total_value'].sum()
    total_keys = len(mk_stats)

    print(f"総Material Key数: {total_keys:,}個")
    print(f"総取引量: {total_value:,.0f}")
    print()

    # 各閾値でのカバー率を計算
    thresholds = [100, 500, 1000, 1500, 2000, 2500, 3000, 5000, 10000]

    print("Material Key数と取引量カバー率:")
    print("-" * 50)
    print(f"{'上位N個':>10} | {'カバー率':>10} | {'累積取引量':>15} | {'データ削減率':>12}")
    print("-" * 50)

    for n in thresholds:
        if n <= total_keys:
            top_n_value = mk_stats.head(n)['total_value'].sum()
            coverage = top_n_value / total_value * 100

            # データ削減率を推定（Material Key数の削減率として）
            reduction = (1 - n/total_keys) * 100

            print(f"{n:10,} | {coverage:9.1f}% | {top_n_value:15,.0f} | {reduction:11.1f}%")

    print("-" * 50)

    # 2000個の詳細分析
    print(f"\n上位2000個の詳細分析:")
    print("="*60)

    top_2000 = mk_stats.head(2000)
    top_2000_value = top_2000['total_value'].sum()
    coverage_2000 = top_2000_value / total_value * 100

    print(f"カバー率: {coverage_2000:.2f}%")
    print(f"総取引量: {top_2000_value:,.0f}")
    print(f"平均取引量: {top_2000['total_value'].mean():,.0f}")
    print(f"中央値取引量: {top_2000['total_value'].median():,.0f}")

    # データサイズの推定
    df_2000 = df[df['material_key'].isin(top_2000['material_key'])]
    data_reduction = (1 - len(df_2000)/len(df)) * 100
    memory_reduction = (1 - df_2000.memory_usage(deep=True).sum()/df.memory_usage(deep=True).sum()) * 100

    print(f"\nデータ削減効果:")
    print(f"行数: {len(df):,} → {len(df_2000):,} ({data_reduction:.1f}%削減)")
    print(f"メモリ: {df.memory_usage(deep=True).sum()/1024**2:.1f}MB → {df_2000.memory_usage(deep=True).sum()/1024**2:.1f}MB ({memory_reduction:.1f}%削減)")

    # カバー率の推移をグラフ的に表示
    print(f"\nカバー率の推移（視覚化）:")
    print("-" * 50)

    for n in [100, 500, 1000, 1500, 2000, 3000, 5000]:
        if n <= total_keys:
            coverage = mk_stats.head(n)['total_value'].sum() / total_value * 100
            bar_length = int(coverage / 2)  # 50文字で100%を表現
            bar = '█' * bar_length
            print(f"{n:5,}個: {bar:<50} {coverage:.1f}%")

    # 推奨事項
    print(f"\n推奨事項:")
    print("="*60)
    if coverage_2000 >= 65:
        print(f"✓ 上位2000個で{coverage_2000:.1f}%をカバー - 十分な精度が期待できます")
        print(f"✓ データ量を{data_reduction:.1f}%削減でき、処理速度が大幅に向上します")
    else:
        print(f"△ 上位2000個で{coverage_2000:.1f}%のカバー率")
        print(f"  より多くのMaterial Key（3000-5000個）の使用を検討してください")

    return mk_stats

if __name__ == "__main__":
    analyze_coverage()