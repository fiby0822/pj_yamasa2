#!/usr/bin/env python3
"""
以前の特徴量と現在の特徴量での精度比較・検証・考察
"""
import json
import pandas as pd
import boto3
from datetime import datetime
from io import BytesIO

def load_metrics_from_s3(bucket_name: str, key: str) -> dict:
    """S3からメトリクスを読み込む"""
    try:
        s3_client = boto3.client('s3')
        response = s3_client.get_object(Bucket=bucket_name, Key=key)
        return json.loads(response['Body'].read().decode('utf-8'))
    except s3_client.exceptions.NoSuchKey:
        print(f"ファイルが見つかりません: {key}")
        return None
    except Exception as e:
        print(f"エラー: {e}")
        return None

def save_comparison_to_s3(comparison_result: dict, bucket_name: str, key: str):
    """比較結果をS3に保存"""
    s3_client = boto3.client('s3')
    s3_client.put_object(
        Bucket=bucket_name,
        Key=key,
        Body=json.dumps(comparison_result, indent=2, ensure_ascii=False)
    )

def compare_accuracy():
    """精度比較と考察"""
    bucket_name = "fiby-yamasa-prediction-2"

    # 以前の精度データを探す
    s3_client = boto3.client('s3')

    # baseline_metrics.txtまたは以前の評価結果を確認
    previous_metrics = None
    current_metrics = None

    # まず、ローカルのbaseline_metrics.txtを確認
    try:
        with open('baseline_metrics.txt', 'r') as f:
            lines = f.readlines()
            previous_metrics = {}
            for line in lines:
                if 'RMSE:' in line:
                    previous_metrics['RMSE'] = float(line.split(':')[1].strip())
                elif 'MAE:' in line:
                    previous_metrics['MAE'] = float(line.split(':')[1].strip())
                elif 'MAPE:' in line:
                    previous_metrics['MAPE'] = float(line.split(':')[1].replace('%', '').strip())
                elif 'R²:' in line:
                    previous_metrics['R2'] = float(line.split(':')[1].strip())
            previous_metrics['source'] = 'baseline_metrics.txt (以前の特徴量)'
    except FileNotFoundError:
        pass

    # 最新の評価結果を取得
    current_metrics = load_metrics_from_s3(
        bucket_name,
        "output/evaluation/confirmed_order_demand_yamasa_metrics_latest.json"
    )

    if current_metrics:
        current_metrics['source'] = 'latest (dow_mean特徴量を含む)'

    # 比較結果の生成
    print("=" * 80)
    print(" 特徴量による精度比較検証レポート")
    print("=" * 80)
    print(f"\n生成日時: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()

    comparison_result = {
        "generated_at": datetime.now().isoformat(),
        "previous_metrics": previous_metrics,
        "current_metrics": current_metrics,
        "improvements": {},
        "analysis": {}
    }

    if previous_metrics and current_metrics:
        print("【精度指標の比較】")
        print("-" * 60)
        print(f"{'指標':<15} {'以前':<15} {'現在':<15} {'改善度':<20}")
        print("-" * 60)

        metrics_to_compare = [
            ('RMSE', False),  # 小さいほど良い
            ('MAE', False),
            ('MAPE', False),
            ('R2', True),    # 大きいほど良い
        ]

        for metric, higher_is_better in metrics_to_compare:
            prev_val = previous_metrics.get(metric, 0)
            curr_val = current_metrics.get(metric, 0)

            if prev_val != 0:
                if higher_is_better:
                    improvement = ((curr_val - prev_val) / abs(prev_val)) * 100
                else:
                    improvement = ((prev_val - curr_val) / prev_val) * 100

                comparison_result['improvements'][metric] = improvement

                # 表示
                if metric == 'MAPE':
                    print(f"{metric:<15} {prev_val:<14.2f}% {curr_val:<14.2f}% ", end="")
                else:
                    print(f"{metric:<15} {prev_val:<14.4f} {curr_val:<14.4f} ", end="")

                if improvement > 0:
                    print(f"{improvement:+.2f}% ✓ 改善")
                elif improvement < 0:
                    print(f"{improvement:+.2f}% ✗ 悪化")
                else:
                    print("変化なし")

        # 分析と考察
        print("\n【分析結果】")
        print("-" * 60)

        # 全体的な傾向を分析
        improvements = comparison_result['improvements']
        improved_count = sum(1 for v in improvements.values() if v > 0)
        worsened_count = sum(1 for v in improvements.values() if v < 0)

        if improved_count > worsened_count:
            print("✓ 全体的に精度が向上しています")
            comparison_result['analysis']['overall'] = "改善"
        elif improved_count < worsened_count:
            print("△ 全体的に精度が低下しています")
            comparison_result['analysis']['overall'] = "悪化"
        else:
            print("△ 精度の変化は混在しています")
            comparison_result['analysis']['overall'] = "混在"

        # 個別の改善点を分析
        print("\n【詳細分析】")
        for metric, improvement in improvements.items():
            if abs(improvement) > 5:  # 5%以上の変化
                if improvement > 0:
                    print(f"• {metric}: {improvement:+.2f}% の大幅改善")
                else:
                    print(f"• {metric}: {improvement:+.2f}% の大幅悪化")
            elif abs(improvement) > 1:  # 1-5%の変化
                if improvement > 0:
                    print(f"• {metric}: {improvement:+.2f}% の改善")
                else:
                    print(f"• {metric}: {improvement:+.2f}% の悪化")

        # 考察
        print("\n【考察】")
        print("-" * 60)

        # dow_mean特徴量の効果について
        if 'RMSE' in improvements and 'MAE' in improvements:
            avg_improvement = (improvements['RMSE'] + improvements['MAE']) / 2

            if avg_improvement > 5:
                print("1. dow_mean特徴量の追加による効果:")
                print("   - Material Key×曜日の過去平均特徴量が大きく精度向上に寄与")
                print("   - 曜日パターンの学習が予測精度を改善")
                comparison_result['analysis']['dow_mean_effect'] = "highly_effective"

            elif avg_improvement > 0:
                print("1. dow_mean特徴量の追加による効果:")
                print("   - 曜日別の平均特徴量が一定の効果を示している")
                print("   - 週次パターンの考慮が予測に貢献")
                comparison_result['analysis']['dow_mean_effect'] = "moderately_effective"

            else:
                print("1. dow_mean特徴量の追加による効果:")
                print("   - 期待したほどの効果が見られない")
                print("   - 他の要因が精度に影響している可能性")
                comparison_result['analysis']['dow_mean_effect'] = "limited_effect"

        # R²スコアについて
        if 'R2' in improvements:
            if improvements['R2'] > 0:
                print("\n2. モデルの説明力:")
                print("   - R²スコアが向上し、データの変動をより良く説明")
            else:
                print("\n2. モデルの説明力:")
                print("   - R²スコアが低下、過学習の可能性を検討すべき")

        # 推奨事項
        print("\n【推奨事項】")
        print("-" * 60)

        if avg_improvement > 5:
            print("◎ 現在の特徴量設定を採用することを強く推奨")
            print("  - dow_mean特徴量が効果的に機能している")
            comparison_result['recommendation'] = "strongly_adopt_current"

        elif avg_improvement > 0:
            print("○ 現在の特徴量設定の採用を推奨")
            print("  - 小幅ながら改善が見られる")
            comparison_result['recommendation'] = "adopt_current"

        else:
            print("△ 追加の検証が必要")
            print("  - さらなる特徴量エンジニアリングを検討")
            print("  - ハイパーパラメータの最適化を試す")
            comparison_result['recommendation'] = "needs_further_validation"

        # 今後の改善案
        print("\n【今後の改善案】")
        print("-" * 60)
        print("1. 月次・季節性の特徴量追加")
        print("2. 外れ値処理の最適化")
        print("3. モデルのアンサンブル化")
        print("4. より長期の学習データの活用")

    else:
        if not previous_metrics:
            print("⚠️ 以前の精度データが見つかりません")
        if not current_metrics:
            print("⚠️ 現在の精度データがまだ生成されていません")
            print("   学習が完了するまでお待ちください...")

    # 結果をS3に保存
    if comparison_result['previous_metrics'] and comparison_result['current_metrics']:
        save_key = f"output/analysis/feature_comparison_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        save_comparison_to_s3(comparison_result, bucket_name, save_key)
        print(f"\n比較結果を保存しました: s3://{bucket_name}/{save_key}")

    print("\n" + "=" * 80)

    return comparison_result

if __name__ == "__main__":
    result = compare_accuracy()