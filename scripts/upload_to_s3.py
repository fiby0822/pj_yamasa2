#!/usr/bin/env python3
"""
Store_codeレベルでデータを準備し、S3にアップロード
別のIAMロールまたは認証情報を使用
"""
import pandas as pd
import numpy as np
import boto3
from datetime import datetime
from io import BytesIO
import os

def main():
    print("="*60)
    print("Store-level Data Preparation and S3 Upload")
    print(f"Timestamp: {datetime.now()}")
    print("="*60)

    # 環境変数から認証情報を取得するか、IAMロールを使用
    try:
        # EC2インスタンスのIAMロールを使用
        session = boto3.Session()
        s3_client = session.client('s3', region_name='ap-northeast-1')
    except:
        # デフォルトの認証情報を使用
        s3_client = boto3.client('s3', region_name='ap-northeast-1')

    # 元のyamasaプロジェクトからデータを読み込み
    source_bucket = 'fiby-yamasa-prediction'
    source_key = 'output/df_confirmed_order_input_yamasa_fill_zero.parquet'

    print(f"\n1. Reading data from original yamasa project...")
    print(f"   Source: s3://{source_bucket}/{source_key}")

    try:
        response = s3_client.get_object(Bucket=source_bucket, Key=source_key)
        df = pd.read_parquet(BytesIO(response['Body'].read()))
        print(f"   Loaded: {df.shape[0]:,} rows, {df.shape[1]} columns")
        print(f"   Unique material_keys (original): {df['material_key'].nunique():,}")
        print(f"   Unique store_codes: {df['store_code'].nunique():,}")

        # store_codeレベルで集約
        print("\n2. Aggregating at store_code level...")
        agg_df = df.groupby(['store_code', 'file_date']).agg({
            'actual_value': 'sum',
            'usage_type': 'first',
        }).reset_index()

        # material_key = store_code
        agg_df['material_key'] = agg_df['store_code']
        agg_df = agg_df[['material_key', 'store_code', 'file_date', 'actual_value', 'usage_type']]

        print(f"   Aggregated: {agg_df.shape[0]:,} rows")
        print(f"   Unique stores: {agg_df['material_key'].nunique():,}")

        # S3にアップロード
        target_bucket = 'fiby-yamasa-prediction-2'
        target_key = 'output/df_confirmed_order_input_yamasa_fill_zero.parquet'

        print(f"\n3. Uploading to S3...")
        print(f"   Target: s3://{target_bucket}/{target_key}")

        # バッファに書き込み
        buffer = BytesIO()
        agg_df.to_parquet(buffer, index=False)
        buffer.seek(0)

        # 異なる方法でアップロードを試す
        try:
            # 方法1: put_object
            s3_client.put_object(
                Bucket=target_bucket,
                Key=target_key,
                Body=buffer.getvalue()
            )
            print(f"   ✓ Successfully uploaded using put_object")
        except Exception as e1:
            print(f"   ✗ put_object failed: {e1}")

            # 方法2: 別のセッションで試す
            try:
                import subprocess
                import tempfile

                # 一時ファイルとして保存
                with tempfile.NamedTemporaryFile(suffix='.parquet', delete=False) as tmp:
                    agg_df.to_parquet(tmp.name, index=False)
                    temp_path = tmp.name

                # AWS CLIが利用可能か確認
                result = subprocess.run(['which', 'aws'], capture_output=True, text=True)
                if result.returncode == 0:
                    # AWS CLIでアップロード
                    cmd = f'aws s3 cp {temp_path} s3://{target_bucket}/{target_key}'
                    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
                    if result.returncode == 0:
                        print(f"   ✓ Successfully uploaded using AWS CLI")
                    else:
                        print(f"   ✗ AWS CLI failed: {result.stderr}")

                # 一時ファイルを削除
                os.unlink(temp_path)
            except Exception as e2:
                print(f"   ✗ Alternative upload failed: {e2}")

        print("\n" + "="*60)
        print("Process completed")
        print("="*60)

        return agg_df

    except Exception as e:
        print(f"\nError: {e}")
        print("\nTroubleshooting:")
        print("1. Check S3 bucket permissions for fiby-yamasa-prediction-2")
        print("2. Ensure IAM user/role has s3:PutObject permission")
        print("3. Verify bucket exists and is in ap-northeast-1 region")
        raise

if __name__ == "__main__":
    df = main()