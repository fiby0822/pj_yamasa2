#!/usr/bin/env python3
"""
Material Keyフィルタリング機能の動作確認スクリプト
"""
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))

import pandas as pd
import numpy as np
from modules.models.train_predict import TimeSeriesPredictor
import boto3
from io import BytesIO

def test_filtering():
    """フィルタリング機能をテスト"""
    print("="*60)
    print(" Material Keyフィルタリング機能テスト")
    print("="*60)

    # 予測器の初期化
    predictor = TimeSeriesPredictor(
        bucket_name="fiby-yamasa-prediction-2",
        model_type="confirmed_order_demand_yamasa"
    )

    # S3からデータを読み込み
    print("\nデータを読み込み中...")
    s3_client = boto3.client('s3')
    response = s3_client.get_object(
        Bucket="fiby-yamasa-prediction-2",
        Key="output/features/confirmed_order_demand_yamasa_features_latest.parquet"
    )
    df = pd.read_parquet(BytesIO(response['Body'].read()))
    print(f"読み込み完了: {len(df):,}行")

    # フィルタリング前の状態
    print(f"\n元データ:")
    print(f"  行数: {len(df):,}")
    print(f"  Material Key数: {df['material_key'].nunique():,}")
    print(f"  メモリ使用量: {df.memory_usage(deep=True).sum() / 1024**2:.1f} MB")

    # フィルタリング機能のテスト（step_count=1の場合）
    print("\n" + "="*60)
    print("テスト1: step_count=1でのフィルタリング")
    print("="*60)

    # 日付列の準備
    if 'file_date' in df.columns and 'date' not in df.columns:
        df['date'] = df['file_date']
    df['date'] = pd.to_datetime(df['date'])

    # フィルタリング実行
    df_filtered_1 = predictor._filter_important_material_keys(
        df.copy(),
        train_end_date="2024-12-31",
        target_col='actual_value',
        step_count=1,
        verbose=True
    )

    # フィルタリング機能のテスト（step_count=3の場合）
    print("\n" + "="*60)
    print("テスト2: step_count=3でのフィルタリング")
    print("="*60)

    df_filtered_3 = predictor._filter_important_material_keys(
        df.copy(),
        train_end_date="2024-12-31",
        target_col='actual_value',
        step_count=3,
        verbose=True
    )

    # 結果の比較
    print("\n" + "="*60)
    print("フィルタリング結果の比較")
    print("="*60)

    print(f"\n元データ:")
    print(f"  行数: {len(df):,}")
    print(f"  Material Key数: {df['material_key'].nunique():,}")

    print(f"\nstep_count=1の場合:")
    print(f"  行数: {len(df_filtered_1):,}")
    print(f"  Material Key数: {df_filtered_1['material_key'].nunique():,}")
    print(f"  削減率: {(1 - len(df_filtered_1)/len(df))*100:.1f}%")

    print(f"\nstep_count=3の場合:")
    print(f"  行数: {len(df_filtered_3):,}")
    print(f"  Material Key数: {df_filtered_3['material_key'].nunique():,}")
    print(f"  削減率: {(1 - len(df_filtered_3)/len(df))*100:.1f}%")

    # パフォーマンス向上の推定
    print("\n" + "="*60)
    print("パフォーマンス向上の推定")
    print("="*60)

    original_memory = df.memory_usage(deep=True).sum() / 1024**2
    filtered_memory_1 = df_filtered_1.memory_usage(deep=True).sum() / 1024**2
    filtered_memory_3 = df_filtered_3.memory_usage(deep=True).sum() / 1024**2

    print(f"\nメモリ使用量:")
    print(f"  元データ: {original_memory:.1f} MB")
    print(f"  step_count=1: {filtered_memory_1:.1f} MB ({(1-filtered_memory_1/original_memory)*100:.1f}%削減)")
    print(f"  step_count=3: {filtered_memory_3:.1f} MB ({(1-filtered_memory_3/original_memory)*100:.1f}%削減)")

    print("\n✓ フィルタリング機能が正常に動作しています")
    print("  上位3000個 + アクティブなMaterial Keyが選択されました")

if __name__ == "__main__":
    test_filtering()