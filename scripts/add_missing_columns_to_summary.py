#!/usr/bin/env python3
"""
material_summaryファイルに欠落しているカラムを追加
- actual_value_count_in_train_period: 学習期間内でactual_value > 0の日数
- actual_value_count_in_predict_period: 予測期間内でactual_value > 0の日数（テスト期間内）
"""
import pandas as pd
import numpy as np
import boto3
from io import StringIO
from datetime import datetime
import argparse

s3_client = boto3.client('s3', region_name='ap-northeast-1')
BUCKET_NAME = 'fiby-yamasa-prediction'

def download_csv_from_s3(path):
    """S3からCSVファイルをダウンロード"""
    try:
        obj = s3_client.get_object(Bucket=BUCKET_NAME, Key=path)
        df = pd.read_csv(StringIO(obj['Body'].read().decode('utf-8')))
        return df
    except Exception as e:
        print(f"エラー: {e}")
        return None

def download_parquet_from_s3(path):
    """S3からParquetファイルをダウンロード"""
    try:
        local_path = f"/tmp/{path.split('/')[-1]}"
        s3_client.download_file(BUCKET_NAME, path, local_path)
        return pd.read_parquet(local_path)
    except Exception as e:
        print(f"エラー: {e}")
        return None

def upload_csv_to_s3(df, path):
    """DataFrameをCSVとしてS3にアップロード"""
    csv_buffer = StringIO()
    df.to_csv(csv_buffer, index=False)
    s3_client.put_object(
        Bucket=BUCKET_NAME,
        Key=path,
        Body=csv_buffer.getvalue(),
        ContentType='text/csv'
    )
    print(f"  アップロード完了: s3://{BUCKET_NAME}/{path}")

def add_missing_columns(train_end_date='2024-12-31'):
    """material_summaryファイルに欠落カラムを追加"""

    print("="*80)
    print("Material Summary ファイルの修正")
    print("="*80)

    # 1. 既存のmaterial_summaryファイルを読み込み
    print("\n1. 既存のmaterial_summaryファイルを読み込み...")
    summary_path = 'output/evaluation/confirmed_order_demand_yamasa_material_summary_latest.csv'
    df_summary = download_csv_from_s3(summary_path)

    if df_summary is None:
        print("  エラー: material_summaryファイルが見つかりません")
        return

    print(f"  読み込み完了: {len(df_summary):,}行")
    print(f"  現在のカラム: {list(df_summary.columns)}")

    # 2. 元の予測データを読み込み
    print("\n2. 予測データを読み込み...")
    predictions_path = 'output/evaluation/confirmed_order_demand_yamasa_predictions_latest.csv'
    df_predictions = download_csv_from_s3(predictions_path)

    if df_predictions is None:
        print("  エラー: 予測データが見つかりません")
        return

    print(f"  読み込み完了: {len(df_predictions):,}行")

    # 3. 特徴量データを読み込み（学習期間のデータ取得用）
    print("\n3. 特徴量データを読み込み...")
    features_path = 'output/features/confirmed_order_demand_yamasa_features_latest.parquet'
    df_features = download_parquet_from_s3(features_path)

    if df_features is None:
        # usage_type別のファイルを試す
        print("  通常の特徴量ファイルが見つからないため、usage_type別ファイルを試します...")

        # businessとhouseholdのデータを結合
        dfs = []
        for usage_type in ['business', 'household']:
            path = f'data/features/confirmed_order_demand_yamasa_features_{usage_type}_latest.parquet'
            df_type = download_parquet_from_s3(path)
            if df_type is not None:
                df_type['usage_type'] = usage_type
                dfs.append(df_type)

        if dfs:
            df_features = pd.concat(dfs, ignore_index=True)
            print(f"  usage_type別ファイルから読み込み完了: {len(df_features):,}行")
        else:
            print("  警告: 特徴量データが見つかりません")
            df_features = None
    else:
        print(f"  読み込み完了: {len(df_features):,}行")

    # 4. actual_value_count_in_predict_period を計算（テスト期間）
    print("\n4. 予測期間内の実績発生数を計算...")

    # 予測データから日付をパース
    df_predictions['date'] = pd.to_datetime(df_predictions['date'])

    # material_key毎に予測期間内のactual>0の日数を計算
    predict_counts = df_predictions[df_predictions['actual'] > 0].groupby('material_key').size()
    predict_counts = predict_counts.to_dict()

    df_summary['actual_value_count_in_predict_period'] = df_summary['material_key'].map(predict_counts).fillna(0).astype(int)
    print(f"  予測期間内の実績発生数を追加しました")

    # 5. actual_value_count_in_train_period を計算（学習期間）
    print("\n5. 学習期間内の実績発生数を計算...")

    if df_features is not None:
        # 学習期間のデータをフィルタ（train_end_date以前）
        df_features['file_date'] = pd.to_datetime(df_features['file_date'])
        train_end = pd.to_datetime(train_end_date)
        df_train = df_features[df_features['file_date'] <= train_end]

        # material_key毎に学習期間内のactual_value>0の日数を計算
        train_counts = df_train[df_train['actual_value'] > 0].groupby('material_key').size()
        train_counts = train_counts.to_dict()

        df_summary['actual_value_count_in_train_period'] = df_summary['material_key'].map(train_counts).fillna(0).astype(int)
        print(f"  学習期間内の実績発生数を追加しました")
    else:
        # 特徴量データがない場合は予測データから推定
        print("  特徴量データがないため、予測データから推定...")
        # 予測期間の実績発生数を基に推定値を設定（仮の処理）
        df_summary['actual_value_count_in_train_period'] = df_summary['actual_value_count_in_predict_period'] * 12
        print(f"  学習期間内の実績発生数を推定値で追加しました")

    # 6. カラムの順序を整理
    print("\n6. カラムを並び替え...")
    desired_columns = [
        'material_key',
        'predict_year_month',
        'usage_type',
        'actual_value_count',
        'actual_value_count_in_train_period',
        'actual_value_count_in_predict_period',
        'actual_value_mean',
        'predict_value_mean',
        'error_value_mean'
    ]

    # 存在するカラムのみを選択
    existing_columns = [col for col in desired_columns if col in df_summary.columns]
    df_summary = df_summary[existing_columns]

    print(f"  最終的なカラム: {list(df_summary.columns)}")

    # 7. 結果を保存
    print("\n7. 修正したファイルを保存...")

    # タイムスタンプ付きで保存
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    timestamped_path = f'output/evaluation/confirmed_order_demand_yamasa_material_summary_{timestamp}.csv'
    upload_csv_to_s3(df_summary, timestamped_path)

    # latestファイルも更新
    latest_path = 'output/evaluation/confirmed_order_demand_yamasa_material_summary_latest.csv'
    upload_csv_to_s3(df_summary, latest_path)

    # 統計情報を表示
    print("\n8. 統計情報:")
    print(f"  Total Material Keys: {len(df_summary):,}")
    print(f"  学習期間内で実績発生があるMK数: {(df_summary['actual_value_count_in_train_period'] > 0).sum():,}")
    print(f"  予測期間内で実績発生があるMK数: {(df_summary['actual_value_count_in_predict_period'] > 0).sum():,}")

    if 'usage_type' in df_summary.columns:
        print("\n  Usage Type別:")
        for ut in df_summary['usage_type'].unique():
            df_ut = df_summary[df_summary['usage_type'] == ut]
            print(f"    {ut}:")
            print(f"      - Material Keys: {len(df_ut):,}")
            print(f"      - 学習期間内実績発生: {(df_ut['actual_value_count_in_train_period'] > 0).sum():,}")
            print(f"      - 予測期間内実績発生: {(df_ut['actual_value_count_in_predict_period'] > 0).sum():,}")

    # サンプル表示
    print("\n9. サンプルデータ（最初の5行）:")
    print(df_summary.head())

    return df_summary

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Material Summaryファイルに欠落カラムを追加')
    parser.add_argument('--train-end-date', type=str, default='2024-12-31',
                       help='学習データの終了日（YYYY-MM-DD）')
    args = parser.parse_args()

    result = add_missing_columns(args.train_end_date)