#!/usr/bin/env python3
"""
バッチ処理による特徴量生成（メモリ効率化版）
material_keyを分割して処理
"""
import pandas as pd
import numpy as np
import gc
from datetime import datetime
from modules.features.timeseries_features import add_timeseries_features
from modules.data_io.s3_handler import S3Handler
from modules.config.feature_window_config import WINDOW_SIZE_CONFIG

def process_batch(df_batch, model_type, start_year, end_year, train_end_date):
    """バッチごとに特徴量生成"""
    try:
        df_features = add_timeseries_features(
            df_batch,
            window_size_config=WINDOW_SIZE_CONFIG,
            start_year=start_year,
            end_year=end_year,
            model_type=model_type,
            train_end_date=train_end_date
        )
        return df_features
    except Exception as e:
        print(f"バッチ処理エラー: {e}")
        return None

def main():
    print("="*60)
    print("ヤマサ確定注文需要予測 - 特徴量生成（バッチ処理版）")
    print("="*60)

    # S3ハンドラの初期化
    s3_handler = S3Handler(bucket_name="fiby-yamasa-prediction-2")

    # パラメータ設定
    input_key = "output/df_confirmed_order_input_yamasa_fill_zero.parquet"
    model_type = "confirmed_order_demand_yamasa"
    start_year = 2021
    end_year = 2025
    train_end_date = "2024-12-31"

    print(f"\n1. データ読み込み中...")
    print(f"   入力: s3://fiby-yamasa-prediction-2/{input_key}")

    try:
        # データ読み込み
        df_input = s3_handler.read_parquet(input_key)
        print(f"   shape: {df_input.shape}")

        # material_keyのユニーク値を取得
        unique_materials = df_input['material_key'].unique()
        n_materials = len(unique_materials)
        print(f"   material_key数: {n_materials:,}")

        # バッチサイズの設定（メモリに応じて調整）
        batch_size = 1000  # 一度に処理するmaterial_key数（メモリ制限のため小さく設定）
        n_batches = (n_materials + batch_size - 1) // batch_size

        print(f"\n2. バッチ処理設定:")
        print(f"   バッチサイズ: {batch_size} material_keys")
        print(f"   バッチ数: {n_batches}")

        # 結果を格納するリスト
        all_features = []

        print(f"\n3. 特徴量生成開始...")
        start_time = datetime.now()

        for batch_idx in range(n_batches):
            # バッチのmaterial_keyを取得
            start_idx = batch_idx * batch_size
            end_idx = min((batch_idx + 1) * batch_size, n_materials)
            batch_materials = unique_materials[start_idx:end_idx]

            print(f"\n   バッチ {batch_idx + 1}/{n_batches} 処理中...")
            print(f"   material_keys: {start_idx:,} ~ {end_idx:,}")

            # バッチデータの抽出
            df_batch = df_input[df_input['material_key'].isin(batch_materials)].copy()

            # 特徴量生成
            df_features_batch = process_batch(
                df_batch,
                model_type,
                start_year,
                end_year,
                train_end_date
            )

            if df_features_batch is not None:
                all_features.append(df_features_batch)
                print(f"   処理完了: {df_features_batch.shape}")
            else:
                print(f"   スキップ（エラー）")

            # メモリ解放
            del df_batch
            if df_features_batch is not None:
                del df_features_batch
            gc.collect()

            # 進捗表示
            if (batch_idx + 1) % 5 == 0:
                elapsed = (datetime.now() - start_time).total_seconds()
                avg_time = elapsed / (batch_idx + 1)
                remaining = avg_time * (n_batches - batch_idx - 1)
                print(f"   経過時間: {elapsed:.1f}秒, 推定残り時間: {remaining:.1f}秒")

        print(f"\n4. 結果の結合...")

        # 全バッチの結果を結合
        if all_features:
            df_final = pd.concat(all_features, ignore_index=True)
            print(f"   最終shape: {df_final.shape}")

            # S3に保存
            print(f"\n5. S3への保存...")
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

            # タイムスタンプ付きファイル
            output_key = f"output/features/{model_type}_features_{timestamp}.parquet"
            s3_handler.write_parquet(df_final, output_key)
            print(f"   保存完了: s3://fiby-yamasa-prediction-2/{output_key}")

            # 最新版ファイル
            latest_key = f"output/features/{model_type}_features_latest.parquet"
            s3_handler.write_parquet(df_final, latest_key)
            print(f"   最新版: s3://fiby-yamasa-prediction-2/{latest_key}")

            # 処理時間
            total_time = (datetime.now() - start_time).total_seconds()
            print(f"\n処理完了!")
            print(f"総処理時間: {total_time:.2f}秒 ({total_time/60:.1f}分)")

            # 特徴量のサマリー
            feature_cols = [col for col in df_final.columns if col.endswith('_f')]
            print(f"生成された特徴量数: {len(feature_cols)}")

        else:
            print("エラー: 特徴量が生成されませんでした")
            return 1

    except Exception as e:
        print(f"エラー: {e}")
        import traceback
        traceback.print_exc()
        return 1

    print("\n" + "="*60)
    print("処理が正常に完了しました")
    print("="*60)

    return 0

if __name__ == "__main__":
    exit(main())