#!/usr/bin/env python3
"""
モデル評価用の公式スクリプト
誤差率平均の正しい計算方法で予測精度を評価
"""
import boto3
import pandas as pd
import numpy as np
from pathlib import Path
import json
from datetime import datetime
from io import StringIO
import argparse

s3_client = boto3.client('s3', region_name='ap-northeast-1')
BUCKET_NAME = 'fiby-yamasa-prediction'

def download_from_s3(s3_path, local_dir="/tmp"):
    """S3からファイルをダウンロード"""
    local_path = Path(local_dir) / Path(s3_path).name
    s3_client.download_file(BUCKET_NAME, s3_path, str(local_path))
    return str(local_path)

def read_csv_from_s3(path):
    """S3からCSVファイルを直接読み込み"""
    try:
        obj = s3_client.get_object(Bucket=BUCKET_NAME, Key=path)
        df = pd.read_csv(StringIO(obj['Body'].read().decode('utf-8')))
        return df
    except Exception as e:
        print(f"エラー: {path} の読み込みに失敗 - {e}")
        return None

def read_parquet_from_s3(path):
    """S3からParquetファイルを読み込み"""
    try:
        local_path = download_from_s3(path)
        df = pd.read_parquet(local_path)
        return df
    except Exception as e:
        print(f"エラー: {path} の読み込みに失敗 - {e}")
        return None

def calculate_error_metrics(df, model_name="", verbose=True):
    """
    誤差率平均を正しく計算

    Parameters:
    -----------
    df : DataFrame
        予測結果のDataFrame（actual, predicted, material_key列が必要）
    model_name : str
        モデル名（表示用）
    verbose : bool
        詳細出力を行うかどうか

    Returns:
    --------
    dict : 評価指標の辞書
    """
    if verbose:
        print(f"\n{'='*60}")
        print(f"{model_name}の評価結果")
        print(f"{'='*60}")
        print(f"  総レコード数: {len(df):,}")

    # カラム名の正規化
    if 'actual_value' in df.columns:
        df['actual'] = df['actual_value']
    if 'predicted_value' in df.columns:
        df['predicted'] = df['predicted_value']

    # material_key毎にグループ化
    grouped = df.groupby('material_key')

    error_rates = []
    mk_stats = []

    for mk, group in grouped:
        # 実績値>0のレコードのみ
        positive_actual = group[group['actual'] > 0]
        if len(positive_actual) > 0:
            # 各レコードの誤差率を計算
            mk_error_rates = np.abs(positive_actual['predicted'] - positive_actual['actual']) / positive_actual['actual']
            # material_keyの誤差率平均
            error_rate_mean = mk_error_rates.mean()
            error_rates.append(error_rate_mean)

            # Material Key毎の統計情報
            mk_stats.append({
                'material_key': mk,
                'error_rate_mean': error_rate_mean,
                'n_records': len(positive_actual),
                'total_actual': positive_actual['actual'].sum(),
                'total_predicted': positive_actual['predicted'].sum()
            })

    error_rates = np.array(error_rates)

    if verbose:
        print(f"  評価対象Material Key数: {len(error_rates):,}")

    # 統計値の計算
    if len(error_rates) > 0:
        error_rate_mean = error_rates.mean()
        error_rate_median = np.median(error_rates)
        error_rate_std = error_rates.std()

        # 閾値別カウント
        within_20 = (error_rates <= 0.2).sum()
        within_30 = (error_rates <= 0.3).sum()
        within_50 = (error_rates <= 0.5).sum()
        within_100 = (error_rates <= 1.0).sum()
        total_mks = len(error_rates)

        if verbose:
            print(f"\n【誤差率平均の統計】")
            print(f"  平均値: {error_rate_mean:.2%}")
            print(f"  中央値: {error_rate_median:.2%}")
            print(f"  標準偏差: {error_rate_std:.2%}")

            print(f"\n【誤差率平均の閾値別割合】")
            print(f"  ≤20%: {within_20:,}/{total_mks:,} ({within_20/total_mks*100:.1f}%)")
            print(f"  ≤30%: {within_30:,}/{total_mks:,} ({within_30/total_mks*100:.1f}%)")
            print(f"  ≤50%: {within_50:,}/{total_mks:,} ({within_50/total_mks*100:.1f}%)")
            print(f"  ≤100%: {within_100:,}/{total_mks:,} ({within_100/total_mks*100:.1f}%)")

        # 分位点
        percentiles = [10, 25, 50, 75, 90]
        pcts = np.percentile(error_rates, percentiles)

        if verbose:
            print(f"\n【誤差率平均の分位点】")
            for p, v in zip(percentiles, pcts):
                print(f"  {p}パーセンタイル: {v:.2%}")
    else:
        error_rate_mean = error_rate_median = error_rate_std = 0
        within_20 = within_30 = within_50 = within_100 = 0
        total_mks = 0
        pcts = [0] * 5

    # 基本的な誤差指標（全レコード）
    mae = np.abs(df['predicted'] - df['actual']).mean()
    rmse = np.sqrt(((df['predicted'] - df['actual']) ** 2).mean())

    # 実績値>0のレコードでの誤差指標
    positive_df = df[df['actual'] > 0]
    if len(positive_df) > 0:
        mae_positive = np.abs(positive_df['predicted'] - positive_df['actual']).mean()
        rmse_positive = np.sqrt(((positive_df['predicted'] - positive_df['actual']) ** 2).mean())
        mape = (np.abs(positive_df['predicted'] - positive_df['actual']) / positive_df['actual']).mean()
    else:
        mae_positive = rmse_positive = mape = 0

    if verbose:
        print(f"\n【全体的な誤差指標】")
        print(f"  MAE（全データ）: {mae:.2f}")
        print(f"  RMSE（全データ）: {rmse:.2f}")
        print(f"  MAE（実績値>0）: {mae_positive:.2f}")
        print(f"  RMSE（実績値>0）: {rmse_positive:.2f}")
        print(f"  MAPE（実績値>0）: {mape:.2%}")

    return {
        'model_name': model_name,
        'total_records': int(len(df)),
        'total_material_keys': int(total_mks),
        'mae': float(mae),
        'rmse': float(rmse),
        'mae_positive': float(mae_positive),
        'rmse_positive': float(rmse_positive),
        'mape': float(mape),
        'error_rate_mean': float(error_rate_mean),
        'error_rate_median': float(error_rate_median),
        'error_rate_std': float(error_rate_std),
        'within_20_percent': int(within_20),
        'within_20_percent_ratio': float(within_20 / total_mks) if total_mks > 0 else 0,
        'within_30_percent': int(within_30),
        'within_30_percent_ratio': float(within_30 / total_mks) if total_mks > 0 else 0,
        'within_50_percent': int(within_50),
        'within_50_percent_ratio': float(within_50 / total_mks) if total_mks > 0 else 0,
        'within_100_percent': int(within_100),
        'within_100_percent_ratio': float(within_100 / total_mks) if total_mks > 0 else 0,
        'percentiles': {
            '10': float(pcts[0]) if len(pcts) > 0 else 0,
            '25': float(pcts[1]) if len(pcts) > 1 else 0,
            '50': float(pcts[2]) if len(pcts) > 2 else 0,
            '75': float(pcts[3]) if len(pcts) > 3 else 0,
            '90': float(pcts[4]) if len(pcts) > 4 else 0,
        },
        'mk_stats': mk_stats if verbose else []
    }

def compare_models(results, output_file=None):
    """
    複数のモデルの結果を比較

    Parameters:
    -----------
    results : list of dict
        各モデルの評価結果のリスト
    output_file : str
        結果を保存するファイルパス（オプション）
    """
    print("\n" + "="*80)
    print("モデル比較サマリー")
    print("="*80)

    # 比較表の作成
    comparison_df = pd.DataFrame(results)

    # 主要指標の表示
    print("\n【主要指標の比較】")
    metrics_to_show = [
        ('error_rate_mean', '誤差率平均'),
        ('error_rate_median', '誤差率中央値'),
        ('within_30_percent_ratio', '誤差率≤30%の割合'),
        ('mae_positive', 'MAE（実績値>0）'),
        ('mape', 'MAPE')
    ]

    for metric, label in metrics_to_show:
        if metric in comparison_df.columns:
            print(f"\n{label}:")
            for _, row in comparison_df.iterrows():
                if metric.endswith('_ratio') or metric == 'mape':
                    print(f"  {row['model_name']}: {row[metric]:.1%}")
                elif metric.startswith('error_rate'):
                    print(f"  {row['model_name']}: {row[metric]:.2%}")
                else:
                    print(f"  {row['model_name']}: {row[metric]:.2f}")

    # 最良モデルの特定
    best_model = comparison_df.loc[comparison_df['error_rate_mean'].idxmin()]
    print(f"\n【最良モデル】")
    print(f"  {best_model['model_name']} (誤差率平均: {best_model['error_rate_mean']:.2%})")

    # 結果の保存
    if output_file:
        results_with_timestamp = {
            'evaluation_time': datetime.now().isoformat(),
            'results': results,
            'best_model': best_model['model_name']
        }

        with open(output_file, 'w') as f:
            json.dump(results_with_timestamp, f, indent=2, ensure_ascii=False)
        print(f"\n評価結果を {output_file} に保存しました。")

    return comparison_df

def evaluate_latest_predictions():
    """最新の予測結果を評価"""
    # 最新のCSVファイルを評価
    csv_path = "output/confirmed_order_demand_yamasa_predictions_latest.csv"

    print("最新の予測結果を評価中...")
    df = read_csv_from_s3(csv_path)

    if df is not None:
        metrics = calculate_error_metrics(df, "最新モデル", verbose=True)

        # 結果をJSONファイルに保存
        output_file = "evaluation_results_latest.json"
        with open(output_file, 'w') as f:
            json.dump(metrics, f, indent=2, ensure_ascii=False)

        print(f"\n評価結果を {output_file} に保存しました。")
        return metrics
    else:
        print("予測結果ファイルが見つかりません。")
        return None

def main():
    """メイン処理"""
    parser = argparse.ArgumentParser(description='モデル評価スクリプト')
    parser.add_argument('--compare', action='store_true',
                      help='複数のモデルを比較')
    parser.add_argument('--model-paths', nargs='+',
                      help='評価するモデルのパス（CSVまたはParquet）')
    parser.add_argument('--model-names', nargs='+',
                      help='モデル名のリスト')
    parser.add_argument('--output', type=str, default='evaluation_results.json',
                      help='結果出力ファイル')
    parser.add_argument('--latest', action='store_true',
                      help='最新の予測結果を評価')

    args = parser.parse_args()

    if args.latest:
        # 最新の予測結果を評価
        evaluate_latest_predictions()

    elif args.compare and args.model_paths:
        # 複数のモデルを比較
        results = []

        for i, path in enumerate(args.model_paths):
            model_name = args.model_names[i] if args.model_names and i < len(args.model_names) else f"Model_{i+1}"

            print(f"\n{model_name} を評価中...")

            # ファイル形式に応じて読み込み
            if path.endswith('.csv'):
                df = read_csv_from_s3(path) if path.startswith('output/') else pd.read_csv(path)
            else:
                df = read_parquet_from_s3(path) if path.startswith('output/') else pd.read_parquet(path)

            if df is not None:
                metrics = calculate_error_metrics(df, model_name, verbose=True)
                results.append(metrics)

        # 結果を比較
        if results:
            compare_models(results, args.output)

    else:
        # デフォルト: 最新の予測結果を評価
        evaluate_latest_predictions()

if __name__ == '__main__':
    main()