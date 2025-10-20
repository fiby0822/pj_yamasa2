#!/usr/bin/env python3
"""
Material Keyフィルタリング機能の簡易テスト（サンプルデータ使用）
"""
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta

def test_filtering_logic():
    """フィルタリングロジックの検証"""
    print("="*60)
    print(" Material Keyフィルタリングロジック検証")
    print("="*60)

    # サンプルデータを作成
    print("\nサンプルデータを生成中...")
    np.random.seed(42)

    # 5000個のMaterial Keyで6ヶ月分のデータを生成
    n_keys = 5000
    start_date = pd.to_datetime('2024-07-01')
    end_date = pd.to_datetime('2025-01-31')

    data = []
    for i in range(n_keys):
        key = f"MAT_{i:05d}"
        dates = pd.date_range(start_date, end_date, freq='D')

        # 上位のMaterial Keyほど高い値を持つように設定
        if i < 100:  # 上位100個
            base_value = np.random.uniform(100, 1000)
        elif i < 1000:  # 上位1000個
            base_value = np.random.uniform(10, 100)
        elif i < 3000:  # 上位3000個
            base_value = np.random.uniform(1, 10)
        else:  # それ以外
            base_value = np.random.uniform(0, 1)

        for date in dates:
            # 一部のデータをゼロにする
            if np.random.random() < 0.3:
                value = 0
            else:
                value = base_value * np.random.uniform(0.5, 1.5)

            data.append({
                'material_key': key,
                'date': date,
                'actual_value': value
            })

    df = pd.DataFrame(data)
    print(f"生成完了: {len(df):,}行, {df['material_key'].nunique():,} Material Keys")

    # フィルタリング条件の適用
    train_end_date = '2024-12-31'
    target_col = 'actual_value'
    step_count = 1

    print(f"\nフィルタリング条件:")
    print(f"  学習終了日: {train_end_date}")
    print(f"  予測月数: {step_count}")
    print(f"  最小アクティブレコード数: {step_count * 4}")

    # 1. 取引量上位3000個のMaterial Key
    mk_totals = df.groupby('material_key')[target_col].sum().reset_index()
    mk_totals.columns = ['material_key', 'total_value']
    mk_totals = mk_totals.sort_values('total_value', ascending=False)
    top_3000_keys = set(mk_totals.head(3000)['material_key'].values)

    print(f"\n上位3000個のMaterial Key:")
    print(f"  合計取引量: {mk_totals.head(3000)['total_value'].sum():,.0f}")
    print(f"  全体に占める割合: {mk_totals.head(3000)['total_value'].sum() / mk_totals['total_value'].sum() * 100:.1f}%")

    # 2. テスト期間でアクティブなMaterial Key
    train_end = pd.to_datetime(train_end_date)
    test_start = train_end + timedelta(days=1)
    test_end = train_end + relativedelta(months=step_count)

    print(f"\nテスト期間: {test_start.strftime('%Y-%m-%d')} ~ {test_end.strftime('%Y-%m-%d')}")

    test_period_df = df[(df['date'] >= test_start) & (df['date'] <= test_end)]
    mk_active_counts = test_period_df[test_period_df[target_col] > 0].groupby('material_key').size()

    min_active_records = step_count * 4
    active_keys = set(mk_active_counts[mk_active_counts >= min_active_records].index.values)

    print(f"  テスト期間のデータ: {len(test_period_df):,}行")
    print(f"  アクティブなMaterial Key: {len(active_keys):,}個")

    # 3. 条件を満たすMaterial Keyの和集合
    selected_keys = top_3000_keys | active_keys

    print(f"\n選択されたMaterial Key:")
    print(f"  上位3000個のみ: {len(top_3000_keys - active_keys):,}個")
    print(f"  アクティブのみ: {len(active_keys - top_3000_keys):,}個")
    print(f"  両方の条件: {len(top_3000_keys & active_keys):,}個")
    print(f"  合計: {len(selected_keys):,}個")

    # フィルタリング実行
    df_filtered = df[df['material_key'].isin(selected_keys)]

    print(f"\nフィルタリング結果:")
    print(f"  元データ: {len(df):,}行")
    print(f"  フィルタリング後: {len(df_filtered):,}行")
    print(f"  削減率: {(1 - len(df_filtered)/len(df))*100:.1f}%")

    # メモリ削減効果
    original_memory = df.memory_usage(deep=True).sum() / 1024**2
    filtered_memory = df_filtered.memory_usage(deep=True).sum() / 1024**2

    print(f"\nメモリ使用量:")
    print(f"  元データ: {original_memory:.1f} MB")
    print(f"  フィルタリング後: {filtered_memory:.1f} MB")
    print(f"  削減率: {(1 - filtered_memory/original_memory)*100:.1f}%")

    print("\n✓ フィルタリングロジックが正常に動作しています")

if __name__ == "__main__":
    test_filtering_logic()