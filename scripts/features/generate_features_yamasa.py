#!/usr/bin/env python3
"""
ヤマサ確定注文需要予測用の特徴量生成実行スクリプト
"""
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[2]))

import pandas as pd
from datetime import datetime
from modules.features.feature_generator_with_s3 import FeatureGeneratorWithS3
from modules.config.feature_window_config import WINDOW_SIZE_CONFIG

def main():
    print("="*60)
    print("ヤマサ確定注文需要予測 - 特徴量生成")
    print("="*60)

    # 特徴量生成器の初期化
    generator = FeatureGeneratorWithS3(bucket_name="fiby-yamasa-prediction-2")

    # 入力データのパス
    input_key = "output/df_confirmed_order_input_yamasa_fill_zero.parquet"

    print(f"\n1. S3からデータを読み込み中...")
    print(f"   入力ファイル: s3://fiby-yamasa-prediction-2/{input_key}")

    try:
        # データ読み込み
        df_input = generator.load_data_from_s3(input_key, file_type="parquet")
        print(f"   データ読み込み完了: {df_input.shape}")
        print(f"   期間: {df_input['file_date'].min()} ~ {df_input['file_date'].max()}")
        print(f"   material_key数: {df_input['material_key'].nunique()}")

    except Exception as e:
        print(f"エラー: データ読み込みに失敗しました - {e}")
        return 1

    print(f"\n2. 特徴量生成を開始...")
    print(f"   モデルタイプ: confirmed_order_demand_yamasa")
    print(f"   期間: 2021-2025")
    print(f"   train_end_date: 2024-12-31")
    print(f"   ※処理には時間がかかる可能性があります...")

    try:
        # 特徴量生成と保存
        start_time = datetime.now()

        df_features = generator.generate_and_save_features(
            df_input=df_input,
            model_type="confirmed_order_demand_yamasa",
            start_year=2021,
            end_year=2025,
            train_end_date="2024-12-31",
            window_size_config=WINDOW_SIZE_CONFIG,
            save_format="both",  # ParquetとCSVの両方で保存
            create_latest=True
        )

        end_time = datetime.now()
        elapsed_time = (end_time - start_time).total_seconds()

        print(f"\n3. 特徴量生成完了!")
        print(f"   処理時間: {elapsed_time:.2f}秒")
        print(f"   出力shape: {df_features.shape}")

        # 特徴量のサマリー
        feature_cols = [col for col in df_features.columns if col.endswith('_f')]
        print(f"   生成された特徴量数: {len(feature_cols)}")

        # NaN率の確認
        print(f"\n4. 特徴量のNaN率（サンプル）:")
        for col in feature_cols[:5]:
            nan_rate = df_features[col].isna().sum() / len(df_features) * 100
            print(f"   - {col}: {nan_rate:.2f}%")

        print(f"\n5. S3への保存完了")
        print(f"   タイムスタンプ付き:")
        print(f"     - output/features/confirmed_order_demand_yamasa_features_*.parquet")
        print(f"     - output/features/confirmed_order_demand_yamasa_features_*.csv")
        print(f"   最新版:")
        print(f"     - output/features/confirmed_order_demand_yamasa_features_latest.parquet")
        print(f"     - output/features/confirmed_order_demand_yamasa_features_latest.csv")

    except Exception as e:
        print(f"エラー: 特徴量生成に失敗しました - {e}")
        import traceback
        traceback.print_exc()
        return 1

    print("\n" + "="*60)
    print("処理が正常に完了しました")
    print("="*60)

    return 0

if __name__ == "__main__":
    exit(main())