#!/usr/bin/env python3
"""
Product_keyレベルの学習結果をS3にアップロード
"""
import boto3
import os
from datetime import datetime

def upload_results():
    """結果をS3へアップロード"""

    print("="*60)
    print("Upload Product-level Results to S3")
    print(f"Timestamp: {datetime.now()}")
    print("="*60)

    # S3クライアント初期化
    s3_client = boto3.client('s3', region_name='ap-northeast-1')
    bucket_name = 'fiby-yamasa-prediction'  # 元のバケットを使用

    # アップロードするファイルのマッピング
    files_to_upload = [
        {
            'local_path': '/home/ubuntu/yamasa2/data/predictions/product_level_predictions_latest.parquet',
            's3_key': 'output/predictions/product_level/product_level_predictions_latest.parquet',
            'description': 'Product-level predictions'
        },
        {
            'local_path': '/home/ubuntu/yamasa2/data/predictions/product_level_summary_latest.parquet',
            's3_key': 'output/predictions/product_level/product_level_summary_latest.parquet',
            'description': 'Product-level summary'
        },
        {
            'local_path': '/home/ubuntu/yamasa2/data/models/product_level_metrics_latest.json',
            's3_key': 'output/models/product_level/product_level_metrics_latest.json',
            'description': 'Product-level metrics'
        },
        {
            'local_path': '/home/ubuntu/yamasa2/data/models/feature_importance_latest.csv',
            's3_key': 'output/models/product_level/feature_importance_latest.csv',
            'description': 'Feature importance'
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

    # 結果の要約
    print("\n📊 Training Results Summary:")

    # メトリクスの読み込みと表示
    try:
        import json
        with open('/home/ubuntu/yamasa2/data/models/product_level_metrics_latest.json', 'r') as f:
            metrics = json.load(f)

        print(f"\nTraining Configuration:")
        print(f"  Train end date: {metrics['train_end_date']}")
        print(f"  Step count: {metrics['step_count']}")

        print(f"\nModel Parameters:")
        for param, value in metrics['metrics'].items():
            print(f"  {param}: {value}")
    except Exception as e:
        print(f"Could not load metrics: {e}")

    print("\n✅ All results have been uploaded to S3!")
    print("You can access them at:")
    print(f"  s3://{bucket_name}/output/predictions/product_level/")
    print(f"  s3://{bucket_name}/output/models/product_level/")

if __name__ == "__main__":
    upload_results()