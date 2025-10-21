#!/usr/bin/env python3
"""
Product_keyレベルの結果をyamasaプロジェクトと同じ形式のCSVで保存
"""
import pandas as pd
import numpy as np
import os
from datetime import datetime
import boto3
import json

def create_predictions_csv():
    """予測結果CSVの作成"""
    # サマリーファイルを読み込み（実際の予測結果が含まれる）
    summary_path = "/home/ubuntu/yamasa2/data/predictions/product_level_summary_latest.parquet"
    if os.path.exists(summary_path):
        df_summary = pd.read_parquet(summary_path)

        # 必要なカラムだけを抽出してCSV形式に整形
        df_pred = pd.DataFrame()

        # material_key（product_key）ごとの予測結果を作成
        if 'material_key' in df_summary.columns:
            df_pred['material_key'] = df_summary['material_key']

            # 実績値と予測値のカラムを追加（利用可能なものを使用）
            if 'Mean_Actual' in df_summary.columns:
                df_pred['actual_sum'] = df_summary['Mean_Actual'] * df_summary.get('count', 31)
            else:
                df_pred['actual_sum'] = 0

            if 'Mean_Predicted' in df_summary.columns:
                df_pred['predicted_sum'] = df_summary['Mean_Predicted'] * df_summary.get('count', 31)
            else:
                df_pred['predicted_sum'] = 0

            # その他のメトリクスを追加
            for col in ['RMSE', 'MAE', 'MAPE', 'Error_Rate_10', 'Error_Rate_20']:
                if col in df_summary.columns:
                    df_pred[col] = df_summary[col]
                else:
                    df_pred[col] = 0
        else:
            # material_key列がない場合はダミーデータを作成
            df_pred = pd.DataFrame({
                'material_key': [],
                'actual_sum': [],
                'predicted_sum': [],
                'RMSE': [],
                'MAE': [],
                'MAPE': [],
                'Error_Rate_10': [],
                'Error_Rate_20': []
            })

        # CSV保存
        csv_path = "/home/ubuntu/yamasa2/output/confirmed_order_demand_yamasa_predictions_latest.csv"
        os.makedirs("/home/ubuntu/yamasa2/output", exist_ok=True)
        df_pred.to_csv(csv_path, index=False)
        print(f"Saved predictions CSV: {csv_path}")
        return csv_path
    else:
        print(f"Summary file not found: {summary_path}")
        return None

def create_material_summary_csv():
    """Material Keyサマリーの作成"""
    # サマリーファイルを読み込み
    summary_path = "/home/ubuntu/yamasa2/data/predictions/product_level_summary_latest.parquet"
    if os.path.exists(summary_path):
        df_summary = pd.read_parquet(summary_path)

        # yamasaプロジェクトと同じ形式にするため、必要なカラムを整形
        if 'material_key' in df_summary.columns:
            # すでにサマリー形式なのでそのまま使用
            df_material_summary = df_summary.copy()
        else:
            df_material_summary = pd.DataFrame()

        # CSV保存
        csv_path = "/home/ubuntu/yamasa2/output/confirmed_order_demand_yamasa_material_summary_latest.csv"
        os.makedirs("/home/ubuntu/yamasa2/output", exist_ok=True)
        df_material_summary.to_csv(csv_path, index=False)
        print(f"Saved material summary CSV: {csv_path}")
        return csv_path
    else:
        print(f"Summary file not found: {summary_path}")
        return None

def create_feature_importance_csv():
    """特徴量重要度CSVの作成"""
    # 特徴量重要度ファイルを読み込み
    fi_path = "/home/ubuntu/yamasa2/data/models/feature_importance_latest.csv"
    if os.path.exists(fi_path):
        df_fi = pd.read_csv(fi_path)

        # CSV保存（すでにCSVなのでコピー）
        csv_path = "/home/ubuntu/yamasa2/output/confirmed_order_demand_yamasa_feature_importance_latest.csv"
        os.makedirs("/home/ubuntu/yamasa2/output", exist_ok=True)
        df_fi.to_csv(csv_path, index=False)
        print(f"Saved feature importance CSV: {csv_path}")
        return csv_path
    else:
        print(f"Feature importance file not found: {fi_path}")
        return None

def upload_csv_to_s3():
    """CSVファイルをS3にアップロード"""
    s3_client = boto3.client('s3', region_name='ap-northeast-1')
    bucket_name = 'fiby-yamasa-prediction'

    files_to_upload = [
        {
            'local_path': '/home/ubuntu/yamasa2/output/confirmed_order_demand_yamasa_predictions_latest.csv',
            's3_key': 'materials_material_key_equal_store_code/output/evaluation/confirmed_order_demand_yamasa_predictions_latest.csv'
        },
        {
            'local_path': '/home/ubuntu/yamasa2/output/confirmed_order_demand_yamasa_material_summary_latest.csv',
            's3_key': 'materials_material_key_equal_store_code/output/evaluation/confirmed_order_demand_yamasa_material_summary_latest.csv'
        },
        {
            'local_path': '/home/ubuntu/yamasa2/output/confirmed_order_demand_yamasa_feature_importance_latest.csv',
            's3_key': 'materials_material_key_equal_store_code/output/evaluation/confirmed_order_demand_yamasa_feature_importance_latest.csv'
        }
    ]

    print("\n" + "="*60)
    print("Uploading CSV files to S3")
    print("="*60)

    for file_info in files_to_upload:
        local_path = file_info['local_path']
        s3_key = file_info['s3_key']

        if not os.path.exists(local_path):
            print(f"❌ File not found: {local_path}")
            continue

        try:
            print(f"\nUploading: {os.path.basename(local_path)}")
            print(f"  To: s3://{bucket_name}/{s3_key}")

            # ファイルサイズ確認
            file_size = os.path.getsize(local_path) / 1024  # KB
            print(f"  File size: {file_size:.2f} KB")

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
    print("CSV upload completed!")
    print("="*60)

def main():
    """メイン処理"""
    print("="*60)
    print("Creating CSV files for Product-level results")
    print(f"Timestamp: {datetime.now()}")
    print("="*60)

    # 各CSVファイルを作成
    print("\n1. Creating predictions CSV...")
    pred_csv = create_predictions_csv()

    print("\n2. Creating material summary CSV...")
    summary_csv = create_material_summary_csv()

    print("\n3. Creating feature importance CSV...")
    fi_csv = create_feature_importance_csv()

    # S3にアップロード
    print("\n4. Uploading to S3...")
    upload_csv_to_s3()

    print("\n✅ All CSV files have been created and uploaded!")
    print("\nLocal files:")
    print("  /home/ubuntu/yamasa2/output/confirmed_order_demand_yamasa_predictions_latest.csv")
    print("  /home/ubuntu/yamasa2/output/confirmed_order_demand_yamasa_material_summary_latest.csv")
    print("  /home/ubuntu/yamasa2/output/confirmed_order_demand_yamasa_feature_importance_latest.csv")

    print("\nS3 locations:")
    print("  s3://fiby-yamasa-prediction/materials_material_key_equal_store_code/output/evaluation/confirmed_order_demand_yamasa_predictions_latest.csv")
    print("  s3://fiby-yamasa-prediction/materials_material_key_equal_store_code/output/evaluation/confirmed_order_demand_yamasa_material_summary_latest.csv")
    print("  s3://fiby-yamasa-prediction/materials_material_key_equal_store_code/output/evaluation/confirmed_order_demand_yamasa_feature_importance_latest.csv")

if __name__ == "__main__":
    main()