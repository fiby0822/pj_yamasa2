#!/usr/bin/env python3
"""
バッチ処理による特徴量生成（S3中間保存版）
各バッチの結果を即座にS3に保存してメモリを解放
"""
import pandas as pd
import numpy as np
import gc
import os
from datetime import datetime
from modules.features.timeseries_features import add_timeseries_features
from modules.data_io.s3_handler import S3Handler
from modules.config.feature_window_config import WINDOW_SIZE_CONFIG

def process_and_save_batch(df_batch, batch_idx, model_type, start_year, end_year, train_end_date, s3_handler):
    """バッチごとに特徴量生成してS3に保存"""
    try:
        # 特徴量生成
        df_features = add_timeseries_features(
            df_batch,
            window_size_config=WINDOW_SIZE_CONFIG,
            start_year=start_year,
            end_year=end_year,
            model_type=model_type,
            train_end_date=train_end_date
        )

        # S3に中間結果を保存
        batch_key = f"output/features/temp_batches/batch_{batch_idx:04d}.parquet"
        s3_handler.write_parquet(df_features, batch_key)

        shape = df_features.shape

        # メモリ解放
        del df_features
        gc.collect()

        return batch_key, shape
    except Exception as e:
        print(f"バッチ{batch_idx}処理エラー: {e}")
        return None, None

def main():
    print("="*60)
    print("ヤマサ確定注文需要予測 - 特徴量生成（S3中間保存版）")
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

        # バッチサイズの設定
        batch_size = 500  # さらに小さくして安全にする
        n_batches = (n_materials + batch_size - 1) // batch_size

        print(f"\n2. バッチ処理設定:")
        print(f"   バッチサイズ: {batch_size} material_keys")
        print(f"   バッチ数: {n_batches}")
        print(f"   中間結果保存先: output/features/temp_batches/")

        # 処理済みバッチのキーを記録
        batch_keys = []
        start_time = datetime.now()

        print(f"\n3. 特徴量生成開始（中間結果をS3に保存）...")

        for batch_idx in range(n_batches):
            # バッチのmaterial_keyを取得
            start_idx = batch_idx * batch_size
            end_idx = min((batch_idx + 1) * batch_size, n_materials)
            batch_materials = unique_materials[start_idx:end_idx]

            print(f"\n   バッチ {batch_idx + 1}/{n_batches} 処理中...")
            print(f"   material_keys: {start_idx:,} ~ {end_idx:,}")

            # バッチデータの抽出
            df_batch = df_input[df_input['material_key'].isin(batch_materials)].copy()

            # 特徴量生成とS3保存
            batch_key, shape = process_and_save_batch(
                df_batch, batch_idx, model_type,
                start_year, end_year, train_end_date, s3_handler
            )

            if batch_key:
                batch_keys.append(batch_key)
                print(f"   保存完了: {batch_key} (shape: {shape})")
            else:
                print(f"   スキップ（エラー）")

            # メモリ解放
            del df_batch
            gc.collect()

            # 進捗表示
            if (batch_idx + 1) % 10 == 0:
                elapsed = (datetime.now() - start_time).total_seconds()
                avg_time = elapsed / (batch_idx + 1)
                remaining = avg_time * (n_batches - batch_idx - 1)
                print(f"   経過時間: {elapsed:.1f}秒, 推定残り時間: {remaining:.1f}秒")

        # 入力データのメモリを解放
        del df_input
        gc.collect()

        print(f"\n4. 中間結果の結合...")
        print(f"   保存されたバッチ数: {len(batch_keys)}")

        # バッチ結果を順次読み込んで結合
        all_features = []
        for i, batch_key in enumerate(batch_keys):
            if (i + 1) % 10 == 0:
                print(f"   結合中: {i + 1}/{len(batch_keys)}")

            df_batch = s3_handler.read_parquet(batch_key)
            all_features.append(df_batch)

            # 10バッチごとに結合してメモリを整理
            if (i + 1) % 10 == 0 or i == len(batch_keys) - 1:
                if len(all_features) > 1:
                    all_features = [pd.concat(all_features, ignore_index=True)]
                    gc.collect()

        # 最終結合
        df_final = all_features[0] if all_features else pd.DataFrame()
        print(f"   最終shape: {df_final.shape}")

        if not df_final.empty:
            # S3に最終結果を保存
            print(f"\n5. 最終結果をS3に保存...")
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

            # タイムスタンプ付きファイル
            output_key = f"output/features/{model_type}_features_{timestamp}.parquet"
            s3_handler.write_parquet(df_final, output_key)
            print(f"   保存完了: s3://fiby-yamasa-prediction-2/{output_key}")

            # 最新版ファイル
            latest_key = f"output/features/{model_type}_features_latest.parquet"
            s3_handler.write_parquet(df_final, latest_key)
            print(f"   最新版: s3://fiby-yamasa-prediction-2/{latest_key}")

            # 中間ファイルの削除（オプション）
            print(f"\n6. 中間ファイルをクリーンアップ...")
            for batch_key in batch_keys[:5]:  # 最初の5個だけ削除（テスト）
                try:
                    # S3から削除
                    s3_handler.s3_client.delete_object(
                        Bucket=s3_handler.bucket_name,
                        Key=batch_key
                    )
                    print(f"   削除: {batch_key}")
                except:
                    pass

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