#!/usr/bin/env python3
"""
閾値48版と96版の精度比較スクリプト
"""
import json
import pandas as pd
import boto3
from io import BytesIO
import sys
from pathlib import Path

def load_metrics_from_s3(bucket_name: str, key: str) -> dict:
    """S3からメトリクスを読み込む"""
    try:
        s3_client = boto3.client('s3')
        response = s3_client.get_object(Bucket=bucket_name, Key=key)
        return json.loads(response['Body'].read().decode('utf-8'))
    except Exception as e:
        print(f"Error loading {key}: {e}")
        return None

def compare_metrics():
    """2つのバージョンの精度を比較"""
    bucket_name = "fiby-yamasa-prediction"

    # メトリクスの読み込み
    metrics_48 = load_metrics_from_s3(bucket_name, "output/evaluation/threshold48_metrics.json")
    metrics_96 = load_metrics_from_s3(bucket_name, "output/evaluation/threshold96_metrics.json")

    if not metrics_48:
        print("閾値48版のメトリクスがまだ生成されていません")
        return

    if not metrics_96:
        print("閾値96版のメトリクスがまだ生成されていません")
        return

    print("=" * 80)
    print(" 閾値による精度比較結果")
    print("=" * 80)
    print()

    # 比較表を作成
    comparison_data = []

    # 主要メトリクスの比較
    metrics_to_compare = [
        ('RMSE', 'RMSE'),
        ('MAE', 'MAE'),
        ('MAPE (%)', 'MAPE'),
        ('R²スコア', 'R2'),
        ('Material Key数', 'n_material_keys'),
        ('予測件数', 'n_predictions')
    ]

    print("【主要メトリクスの比較】")
    print("-" * 60)
    print(f"{'指標':<20} {'閾値48':<15} {'閾値96':<15} {'差分':<15}")
    print("-" * 60)

    for display_name, metric_key in metrics_to_compare:
        val_48 = metrics_48.get(metric_key, 0)
        val_96 = metrics_96.get(metric_key, 0)

        if metric_key in ['RMSE', 'MAE', 'MAPE']:
            # エラー系の指標（小さいほど良い）
            diff = val_96 - val_48
            if val_48 > 0:
                diff_pct = (diff / val_48) * 100
                diff_str = f"{diff:+.2f} ({diff_pct:+.1f}%)"
            else:
                diff_str = f"{diff:+.2f}"

            if diff < 0:
                diff_str += " ✓"  # 改善
        elif metric_key == 'R2':
            # R²スコア（大きいほど良い）
            diff = val_96 - val_48
            if abs(val_48) > 0:
                diff_pct = (diff / abs(val_48)) * 100
                diff_str = f"{diff:+.4f} ({diff_pct:+.1f}%)"
            else:
                diff_str = f"{diff:+.4f}"

            if diff > 0:
                diff_str += " ✓"  # 改善
        else:
            # その他の情報
            diff = val_96 - val_48
            diff_str = f"{diff:+,d}"

        if isinstance(val_48, (int, float)):
            if metric_key in ['n_material_keys', 'n_predictions']:
                print(f"{display_name:<20} {val_48:>14,d} {val_96:>14,d} {diff_str:<15}")
            elif metric_key == 'R2':
                print(f"{display_name:<20} {val_48:>14.4f} {val_96:>14.4f} {diff_str:<15}")
            else:
                print(f"{display_name:<20} {val_48:>14.2f} {val_96:>14.2f} {diff_str:<15}")

    print()
    print("【分析結果】")
    print("-" * 60)

    # 精度改善の分析
    rmse_diff = metrics_96.get('RMSE', 0) - metrics_48.get('RMSE', 0)
    mae_diff = metrics_96.get('MAE', 0) - metrics_48.get('MAE', 0)
    r2_diff = metrics_96.get('R2', 0) - metrics_48.get('R2', 0)

    if rmse_diff < 0 and mae_diff < 0:
        print("✓ 閾値96版の方が全体的に精度が向上しています")
        print(f"  - RMSE: {abs(rmse_diff):.2f} 改善")
        print(f"  - MAE: {abs(mae_diff):.2f} 改善")
        if r2_diff > 0:
            print(f"  - R²: {r2_diff:.4f} 向上")
    elif rmse_diff > 0 and mae_diff > 0:
        print("✓ 閾値48版の方が全体的に精度が高いです")
        print(f"  - RMSE: {rmse_diff:.2f} 悪化")
        print(f"  - MAE: {mae_diff:.2f} 悪化")
        if r2_diff < 0:
            print(f"  - R²: {abs(r2_diff):.4f} 低下")
    else:
        print("△ 指標によって結果が異なります")
        if rmse_diff < 0:
            print(f"  - RMSE: {abs(rmse_diff):.2f} 改善")
        else:
            print(f"  - RMSE: {rmse_diff:.2f} 悪化")

        if mae_diff < 0:
            print(f"  - MAE: {abs(mae_diff):.2f} 改善")
        else:
            print(f"  - MAE: {mae_diff:.2f} 悪化")

        if r2_diff > 0:
            print(f"  - R²: {r2_diff:.4f} 向上")
        else:
            print(f"  - R²: {abs(r2_diff):.4f} 低下")

    # データ量の影響
    print()
    n_mk_48 = metrics_48.get('n_material_keys', 0)
    n_mk_96 = metrics_96.get('n_material_keys', 0)
    mk_reduction = ((n_mk_48 - n_mk_96) / n_mk_48 * 100) if n_mk_48 > 0 else 0

    print(f"• Material Key数の削減: {n_mk_48:,d} → {n_mk_96:,d} ({mk_reduction:.1f}%削減)")

    n_pred_48 = metrics_48.get('n_predictions', 0)
    n_pred_96 = metrics_96.get('n_predictions', 0)
    pred_reduction = ((n_pred_48 - n_pred_96) / n_pred_48 * 100) if n_pred_48 > 0 else 0

    print(f"• 予測データ数の削減: {n_pred_48:,d} → {n_pred_96:,d} ({pred_reduction:.1f}%削減)")

    print()
    print("【推奨事項】")
    print("-" * 60)

    # 推奨の判定
    if rmse_diff < -5 and mae_diff < -3 and mk_reduction > 20:
        print("◎ 閾値96を推奨します")
        print("  理由: 精度が向上し、かつデータ量も大幅に削減できるため")
    elif rmse_diff > 5 and mae_diff > 3:
        print("◎ 閾値48を推奨します")
        print("  理由: より多くのMaterial Keyをカバーでき、精度も高いため")
    elif mk_reduction > 50 and abs(rmse_diff) < 2:
        print("○ 閾値96を推奨します")
        print("  理由: 精度の低下が小さく、データ量を大幅に削減できるため")
    else:
        print("△ 用途に応じて選択してください")
        print("  - 精度重視の場合: 閾値48")
        print("  - 処理速度重視の場合: 閾値96")

    print()
    print("=" * 80)

if __name__ == "__main__":
    compare_metrics()