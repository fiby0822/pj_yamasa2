#!/usr/bin/env python3
"""
誤差率計算方法のシミュレーション
Material Key単位での Error_Rate 計算を実演
"""

import pandas as pd
import numpy as np

def simulate_material_key_predictions():
    """
    あるMaterial Keyの予測結果をシミュレーション
    実績発生数4件（actual_value > 0 が4日）のケース
    """
    print("="*70)
    print("誤差率計算シミュレーション")
    print("="*70)
    print("\n【設定】")
    print("- Material Key: A01101A_T111443")
    print("- 期間: 2025年1月（31日間）")
    print("- 実績発生数: 4件（actual_value > 0 の日数）")
    print()

    # 31日分のデータを生成
    dates = pd.date_range('2025-01-01', '2025-01-31', freq='D')

    # シミュレーションデータ
    # 実績発生は4日のみ（5日、12日、19日、26日）
    data = []
    for i, date in enumerate(dates):
        if date.day in [5, 12, 19, 26]:
            # 実績発生日
            if date.day == 5:
                actual, predicted = 15, 18
            elif date.day == 12:
                actual, predicted = 8, 25
            elif date.day == 19:
                actual, predicted = 42, 38
            else:  # 26日
                actual, predicted = 10, 22
        else:
            # 実績なしの日（actual = 0）
            actual = 0
            # 予測値は0〜5のランダム値
            np.random.seed(i)
            predicted = np.random.randint(0, 6)

        data.append({
            'date': date,
            'material_key': 'A01101A_T111443',
            'actual': actual,
            'predicted': predicted,
            'abs_error': abs(predicted - actual)
        })

    df = pd.DataFrame(data)

    # 実績発生数の確認
    active_count = sum(df['actual'] > 0)
    print(f"■ 実績発生数の確認")
    print(f"  actual_value > 0 の日数: {active_count}日")
    print(f"  実績発生日: {', '.join([str(d.day) + '日' for d in df[df['actual'] > 0]['date']])}")
    print()

    # 主要な日のデータを表示
    print("■ 主要な日のデータ")
    print("-" * 60)
    print(f"{'日付':^12} {'実績値':>8} {'予測値':>8} {'絶対誤差':>10} {'実績発生':>10}")
    print("-" * 60)

    for _, row in df.head(31).iterrows():
        if row['actual'] > 0 or row['date'].day in [8, 15, 22]:  # 実績発生日と一部の実績なし日を表示
            is_active = '✓' if row['actual'] > 0 else '✗'
            print(f"{row['date'].strftime('%Y-%m-%d'):^12} {row['actual']:8.0f} {row['predicted']:8.0f} "
                  f"{row['abs_error']:10.0f} {is_active:>10}")

    print(f"{'（他の日は省略）':^12}")
    print()

    # Error_Rate の計算
    print("■ Error_Rate の計算（全31日分で計算）")
    print("-" * 60)

    thresholds = [5, 10, 20, 30, 50]
    error_rates = {}

    for threshold in thresholds:
        # 絶対誤差が閾値を超える日数をカウント
        over_threshold = sum(df['abs_error'] > threshold)
        error_rate = over_threshold / len(df) * 100
        error_rates[f'Error_Rate_{threshold}'] = error_rate

        print(f"Error_Rate_{threshold:2d}: 絶対誤差>{threshold:2d}の日数 = {over_threshold:2d}日 / {len(df)}日 = {error_rate:5.1f}%")

    print()

    # Material Keyの評価判定
    print("■ このMaterial Keyの評価")
    print("-" * 60)
    print(f"実績発生数: {active_count}件（step_count×4の条件を満たす）")
    print()

    for threshold in [20, 30, 50]:
        error_rate = error_rates[f'Error_Rate_{threshold}']
        if error_rate == 0:
            result = f"✓ 誤差{threshold}以内"
        else:
            result = f"✗ 誤差{threshold}超が{error_rate:.1f}%"
        print(f"Error_Rate_{threshold}: {error_rate:5.1f}% → {result}")

    print()

    # 全体集計のシミュレーション
    print("■ 全体集計でのカウント方法")
    print("-" * 60)
    print("仮に1000個のMaterial Keyがあった場合：")
    print()

    # ランダムに1000個のMaterial Keyの結果を生成
    np.random.seed(42)
    material_keys = []
    for i in range(1000):
        # 各Material KeyのError_Rate_20をランダムに生成
        # 多くは0%（誤差20以内）になるように設定
        if np.random.random() < 0.888:  # 88.8%は誤差20以内
            error_20 = 0.0
        else:
            error_20 = np.random.uniform(1, 50)

        if np.random.random() < 0.952:  # 95.2%は誤差30以内
            error_30 = 0.0
        else:
            error_30 = np.random.uniform(1, 30)

        if np.random.random() < 0.981:  # 98.1%は誤差50以内
            error_50 = 0.0
        else:
            error_50 = np.random.uniform(1, 20)

        material_keys.append({
            'material_key': f'MK_{i:04d}',
            'Error_Rate_20': error_20,
            'Error_Rate_30': error_30,
            'Error_Rate_50': error_50
        })

    mk_df = pd.DataFrame(material_keys)

    # 集計
    within_20 = sum(mk_df['Error_Rate_20'] == 0)
    within_30 = sum(mk_df['Error_Rate_30'] == 0)
    within_50 = sum(mk_df['Error_Rate_50'] == 0)

    print(f"Error_Rate_20 = 0% のMaterial Key: {within_20:4d}個")
    print(f"Error_Rate_30 = 0% のMaterial Key: {within_30:4d}個")
    print(f"Error_Rate_50 = 0% のMaterial Key: {within_50:4d}個")
    print()
    print("【集計結果】")
    print(f"誤差率20%以内: {within_20:4d}/1000 = {within_20/10:5.1f}%")
    print(f"誤差率30%以内: {within_30:4d}/1000 = {within_30/10:5.1f}%")
    print(f"誤差率50%以内: {within_50:4d}/1000 = {within_50/10:5.1f}%")
    print()

    print("="*70)
    print("【まとめ】")
    print("="*70)
    print("1. 実績発生数 = actual_value > 0 のレコード数")
    print("2. Error_Rate_XX = 絶対誤差がXXを超える日の割合(%)")
    print("3. 誤差率XX%以内 = Error_Rate_XX が 0% のMaterial Key数 ÷ 全Material Key数")
    print("4. Material Key単位で集計してから、全体の割合を計算")


if __name__ == '__main__':
    simulate_material_key_predictions()