#!/usr/bin/env python3
"""
最新の特徴量ファイルを_latestにコピーするスクリプト
"""
import boto3
from datetime import datetime

def copy_to_latest():
    """最新の特徴量ファイルを_latestにコピー"""
    s3 = boto3.client('s3')
    bucket_name = 'fiby-yamasa-prediction'

    # コピー元とコピー先
    source_key = 'output/features/confirmed_order_demand_yamasa_features_20251016_020011.parquet'
    dest_key = 'output/features/confirmed_order_demand_yamasa_features_latest.parquet'

    try:
        # ファイルの存在確認
        s3.head_object(Bucket=bucket_name, Key=source_key)
        print(f"ソースファイル確認: {source_key}")

        # コピー実行
        copy_source = {'Bucket': bucket_name, 'Key': source_key}
        s3.copy_object(
            CopySource=copy_source,
            Bucket=bucket_name,
            Key=dest_key
        )
        print(f"コピー完了: {source_key} -> {dest_key}")

        # ファイルサイズ確認
        response = s3.head_object(Bucket=bucket_name, Key=dest_key)
        size_mb = response['ContentLength'] / (1024 * 1024)
        print(f"ファイルサイズ: {size_mb:.1f} MB")
        print(f"最終更新: {response['LastModified']}")

        return True

    except s3.exceptions.NoSuchKey:
        print(f"エラー: ソースファイルが見つかりません: {source_key}")
        return False
    except Exception as e:
        print(f"エラー: {e}")
        return False

if __name__ == "__main__":
    success = copy_to_latest()
    if success:
        print("\n_latestファイルの更新が完了しました。")
    else:
        print("\n_latestファイルの更新に失敗しました。")