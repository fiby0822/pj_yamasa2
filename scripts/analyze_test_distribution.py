#!/usr/bin/env python3
"""
テスト期間でのmaterial_keyごとの実績発生数分布を分析
"""
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))

import pandas as pd
import boto3
from io import BytesIO
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta

def analyze_test_distribution():
    """テスト期間の実績発生数分布を分析"""

    # S3から特徴量データを読み込み
    s3_client = boto3.client('s3')
    features_key = "output/features/confirmed_order_demand_yamasa_features_latest.parquet"

    response = s3_client.get_object(
        Bucket="fiby-yamasa-prediction-2",
        Key=features_key
    )
    df = pd.read_parquet(BytesIO(response['Body'].read()))

    # 日付列の準備
    if 'file_date' in df.columns:
        df['date'] = pd.to_datetime(df['file_date'])
    else:
        df['date'] = pd.to_datetime(df['date'])

    # テスト期間のデータを抽出（2025年1月）
    train_end = pd.to_datetime("2024-12-31")
    test_start = train_end + timedelta(days=1)
    test_end = train_end + relativedelta(months=1)

    test_df = df[(df['date'] >= test_start) & (df['date'] <= test_end)]

    print("="*60)
    print("テスト期間（2025年1月）の実績発生数分布分析")
    print("="*60)
    print()

    # actual_value > 0のレコード数をmaterial_keyごとに集計
    mk_counts = test_df[test_df['actual_value'] > 0].groupby('material_key').size().reset_index(name='count')

    # 分布を表示
    print("【実績発生数の分布】")
    print("-"*40)

    thresholds = [1, 2, 3, 4, 5, 10, 12, 15, 20, 24, 30, 48, 96]

    for threshold in thresholds:
        n_keys = len(mk_counts[mk_counts['count'] >= threshold])
        if len(mk_counts) > 0:
            percentage = (n_keys / len(mk_counts)) * 100
        else:
            percentage = 0
        print(f"{threshold:3}件以上: {n_keys:5,d} material_keys ({percentage:5.1f}%)")

    print()
    print("【統計情報】")
    print("-"*40)

    if len(mk_counts) > 0:
        print(f"合計Material Key数: {len(mk_counts):,d}")
        print(f"平均実績発生数: {mk_counts['count'].mean():.1f}件")
        print(f"中央値: {mk_counts['count'].median():.0f}件")
        print(f"最大値: {mk_counts['count'].max()}件")
        print(f"最小値: {mk_counts['count'].min()}件")

        # 四分位数
        q25 = mk_counts['count'].quantile(0.25)
        q50 = mk_counts['count'].quantile(0.50)
        q75 = mk_counts['count'].quantile(0.75)
        print(f"\n第1四分位数: {q25:.0f}件")
        print(f"第2四分位数（中央値）: {q50:.0f}件")
        print(f"第3四分位数: {q75:.0f}件")
    else:
        print("テスト期間に実績があるmaterial_keyが存在しません")

    print()
    print("【推奨閾値】")
    print("-"*40)

    # 推奨閾値の計算（上位20-30%程度をカバー）
    if len(mk_counts) > 0:
        target_coverage = 0.25  # 25%カバー
        target_n_keys = int(len(mk_counts) * target_coverage)

        # 閾値ごとのカバー率を計算
        for threshold in [4, 5, 6, 8, 10, 12, 15, 20]:
            n_covered = len(mk_counts[mk_counts['count'] >= threshold])
            coverage = n_covered / len(mk_counts)
            print(f"閾値{threshold:2}件: {n_covered:4,d} keys ({coverage*100:4.1f}%カバー)")

    return mk_counts

if __name__ == "__main__":
    mk_counts = analyze_test_distribution()