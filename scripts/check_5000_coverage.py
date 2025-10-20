#!/usr/bin/env python3
"""
上位5000個のMaterial Keyのカバー率を確認
"""
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))

import pandas as pd
import boto3
from io import BytesIO

def check_coverage():
    """上位5000個のカバー率を確認"""
    print("="*70)
    print(" 上位5000個のMaterial Keyカバー率確認")
    print("="*70)

    # S3からデータを読み込み
    print("\nデータを読み込み中...")
    s3_client = boto3.client('s3')
    response = s3_client.get_object(
        Bucket="fiby-yamasa-prediction-2",
        Key="output/features/confirmed_order_demand_yamasa_features_latest.parquet"
    )
    df = pd.read_parquet(BytesIO(response['Body'].read()))

    print(f"全データ: {len(df):,}行, {df['material_key'].nunique():,} Material Keys")

    # 実績発生数の分析
    total_active_records = df[df['actual_value'] > 0].shape[0]
    print(f"\n全体の実績発生数: {total_active_records:,}レコード")

    # Material Key別の実績発生数
    mk_active_counts = df[df['actual_value'] > 0].groupby('material_key').size().reset_index(name='active_count')
    mk_active_counts = mk_active_counts.sort_values('active_count', ascending=False)

    # カバー率の計算
    thresholds = [1000, 2000, 3000, 4000, 5000, 7000, 10000]

    print("\n実績発生数上位N個のカバー率:")
    print("-" * 60)
    print(f"{'上位N個':>10} | {'実績発生数':>15} | {'カバー率':>10} | {'累積カバー率':>12}")
    print("-" * 60)

    for n in thresholds:
        if n <= len(mk_active_counts):
            top_n = mk_active_counts.head(n)
            active_count = top_n['active_count'].sum()
            coverage = active_count / total_active_records * 100
            bar = '█' * int(coverage / 2)  # 50文字で100%
            print(f"{n:10,} | {active_count:15,} | {coverage:9.1f}% | {bar}")

    # 上位5000個の詳細
    print("\n" + "="*70)
    print(" 上位5000個の詳細分析")
    print("="*70)

    top_5000 = mk_active_counts.head(5000)
    top_5000_active = top_5000['active_count'].sum()
    coverage_5000 = top_5000_active / total_active_records * 100

    print(f"\n上位5000個のMaterial Key:")
    print(f"  実績発生数合計: {top_5000_active:,}レコード")
    print(f"  カバー率: {coverage_5000:.2f}%")
    print(f"  平均実績発生数: {top_5000['active_count'].mean():.1f}レコード/MK")
    print(f"  最大: {top_5000['active_count'].max():,}レコード")
    print(f"  最小: {top_5000['active_count'].min():,}レコード")
    print(f"  中央値: {top_5000['active_count'].median():.0f}レコード")

    # 3000個との比較
    top_3000 = mk_active_counts.head(3000)
    top_3000_active = top_3000['active_count'].sum()
    coverage_3000 = top_3000_active / total_active_records * 100

    print(f"\n【比較】上位3000個との差:")
    print(f"  3000個のカバー率: {coverage_3000:.2f}%")
    print(f"  5000個のカバー率: {coverage_5000:.2f}%")
    print(f"  カバー率の向上: +{coverage_5000 - coverage_3000:.2f}%ポイント")
    print(f"  追加Material Key数: 2,000個")
    print(f"  追加実績発生数: {top_5000_active - top_3000_active:,}レコード")

    # テスト期間での影響
    test_df = df[(df['file_date'] >= '2025-01-01') & (df['file_date'] <= '2025-01-31')]
    if len(test_df) > 0:
        test_active = test_df[test_df['actual_value'] > 0].groupby('material_key').size()
        test_active_keys = test_active[test_active >= 4].index

        # 上位5000個に含まれるテストMK
        test_in_5000 = set(test_active_keys) & set(top_5000['material_key'].values)
        test_in_3000 = set(test_active_keys) & set(top_3000['material_key'].values)

        print(f"\nテスト期間（2025年1月）のアクティブなMaterial Key:")
        print(f"  上位3000個に含まれる: {len(test_in_3000):,}個")
        print(f"  上位5000個に含まれる: {len(test_in_5000):,}個")
        print(f"  追加でカバー: {len(test_in_5000 - test_in_3000):,}個")

    print("\n" + "="*70)
    print(" まとめ")
    print("="*70)
    print(f"\n✓ 上位5000個への拡張により:")
    print(f"  - カバー率が{coverage_3000:.1f}%から{coverage_5000:.1f}%に向上（+{coverage_5000-coverage_3000:.1f}%）")
    print(f"  - 全実績発生の{coverage_5000:.1f}%をカバー")
    print(f"  - より多くの取引パターンを学習可能")

if __name__ == "__main__":
    check_coverage()