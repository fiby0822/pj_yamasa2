#!/usr/bin/env python3
"""
Product_keyレベルのデータと特徴量をS3にアップロード
fiby-yamasa-prediction-2バケットへアップロード
"""
import boto3
import os
from datetime import datetime

def upload_to_s3():
    """S3へのアップロード処理"""

    print("="*60)
    print("Upload Product-level Data to S3")
    print(f"Timestamp: {datetime.now()}")
    print("="*60)

    # S3クライアント初期化
    s3_client = boto3.client('s3', region_name='ap-northeast-1')
    bucket_name = 'fiby-yamasa-prediction'  # 元のバケットにアップロード

    # アップロードするファイルのマッピング
    files_to_upload = [
        {
            'local_path': '/home/ubuntu/yamasa2/data/prepared/df_confirmed_order_input_yamasa.parquet',
            's3_key': 'output/product_level/df_confirmed_order_input_yamasa.parquet',
            'description': 'Product-level aggregated data'
        },
        {
            'local_path': '/home/ubuntu/yamasa2/data/prepared/df_confirmed_order_input_yamasa_fill_zero.parquet',
            's3_key': 'output/product_level/df_confirmed_order_input_yamasa_fill_zero.parquet',
            'description': 'Product-level aggregated data (zero-filled)'
        },
        {
            'local_path': '/home/ubuntu/yamasa2/data/features/product_level_features_latest.parquet',
            's3_key': 'output/features/product_level/product_level_features_latest.parquet',
            'description': 'Product-level features'
        }
    ]

    # 各ファイルをアップロード
    for file_info in files_to_upload:
        local_path = file_info['local_path']
        s3_key = file_info['s3_key']
        description = file_info['description']

        if not os.path.exists(local_path):
            print(f"\n❌ File not found: {local_path}")
            continue

        try:
            print(f"\nUploading {description}...")
            print(f"  From: {local_path}")
            print(f"  To: s3://{bucket_name}/{s3_key}")

            # ファイルサイズ確認
            file_size = os.path.getsize(local_path) / (1024*1024)
            print(f"  File size: {file_size:.2f} MB")

            # S3にアップロード
            with open(local_path, 'rb') as f:
                s3_client.put_object(
                    Bucket=bucket_name,
                    Key=s3_key,
                    Body=f
                )

            print(f"  ✅ Successfully uploaded!")

        except Exception as e:
            print(f"  ❌ Upload failed: {str(e)}")

    print("\n" + "="*60)
    print("Upload process completed!")
    print("="*60)

    # 注意事項を表示
    print("\n⚠️ Important Notes:")
    print("- This data is aggregated at product_key level (material_key = product_key)")
    print("- Original material_key data is in fiby-yamasa-prediction bucket")
    print("- Use this data for product-level predictions only")
    print("\nNext steps:")
    print("1. Update train_model.py to use product-level data")
    print("2. Adjust filtering thresholds for product-level prediction")
    print("3. Run training and evaluation")

if __name__ == "__main__":
    upload_to_s3()