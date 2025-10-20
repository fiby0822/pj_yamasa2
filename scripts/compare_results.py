#!/usr/bin/env python3
"""
20251009データと20251016データの精度比較スクリプト
"""
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import warnings
warnings.filterwarnings('ignore')

def calculate_metrics(y_true, y_pred, label=""):
    """各種評価指標を計算"""
    # ゼロ除算を避けるため、実績値が0の場合は除外
    mask = y_true > 0
    y_true_filtered = y_true[mask]
    y_pred_filtered = y_pred[mask]

    if len(y_true_filtered) == 0:
        return None

    # 基本指標
    rmse = np.sqrt(mean_squared_error(y_true_filtered, y_pred_filtered))
    mae = mean_absolute_error(y_true_filtered, y_pred_filtered)

    # 誤差率
    error_rate = np.abs(y_true_filtered - y_pred_filtered) / y_true_filtered * 100
    mean_error_rate = np.mean(error_rate)
    median_error_rate = np.median(error_rate)

    # 相関係数とR²
    correlation = np.corrcoef(y_true_filtered, y_pred_filtered)[0, 1]
    r2 = r2_score(y_true_filtered, y_pred_filtered)

    # 誤差率分析
    within_20 = np.sum(error_rate <= 20)
    within_30 = np.sum(error_rate <= 30)
    within_50 = np.sum(error_rate <= 50)
    total_count = len(error_rate)

    return {
        "label": label,
        "RMSE": round(rmse, 2),
        "MAE": round(mae, 2),
        "誤差率平均値(%)": round(mean_error_rate, 2),
        "誤差率中央値(%)": round(median_error_rate, 2),
        "20%以内の割合(%)": round(within_20 / total_count * 100, 2),
        "30%以内の割合(%)": round(within_30 / total_count * 100, 2),
        "50%以内の割合(%)": round(within_50 / total_count * 100, 2),
        "相関係数": round(correlation, 4),
        "R²スコア": round(r2, 4),
        "評価対象数": total_count
    }

def main():
    print("=" * 80)
    print("精度比較: 20251009データ vs 20251016データ")
    print("=" * 80)

    # 20251009データの結果を読み込み
    try:
        df_2009 = pd.read_parquet("data/output/predictions_20251009.parquet")
        if 'predicted' in df_2009.columns:
            df_2009['prediction'] = df_2009['predicted']
        print(f"\n20251009データ読み込み完了: {len(df_2009):,}レコード")
        metrics_2009 = calculate_metrics(
            df_2009['actual'].values,
            df_2009['prediction'].values,
            "20251009データ"
        )
    except Exception as e:
        print(f"20251009データ読み込みエラー: {e}")
        metrics_2009 = None

    # 20251016データ（最新）の結果を読み込み
    try:
        df_2016 = pd.read_parquet("data/output/confirmed_order_demand_yamasa_predictions_latest.parquet")
        if 'predicted' in df_2016.columns:
            df_2016['prediction'] = df_2016['predicted']
        print(f"20251016データ読み込み完了: {len(df_2016):,}レコード")
        metrics_2016 = calculate_metrics(
            df_2016['actual'].values,
            df_2016['prediction'].values,
            "20251016データ"
        )
    except Exception as e:
        print(f"20251016データ読み込みエラー: {e}")
        metrics_2016 = None

    # 結果の表示
    print("\n" + "=" * 80)
    print("【基本評価指標の比較】")
    print("=" * 80)

    if metrics_2009 and metrics_2016:
        # 表形式で表示
        print(f"\n{'指標':<25} {'20251009データ':>20} {'20251016データ':>20} {'改善率':>15}")
        print("-" * 80)

        # RMSE
        rmse_improve = ((metrics_2009['RMSE'] - metrics_2016['RMSE']) / metrics_2009['RMSE'] * 100)
        print(f"{'RMSE':<25} {metrics_2009['RMSE']:>20.2f} {metrics_2016['RMSE']:>20.2f} {rmse_improve:>14.1f}%")

        # MAE
        mae_improve = ((metrics_2009['MAE'] - metrics_2016['MAE']) / metrics_2009['MAE'] * 100)
        print(f"{'MAE':<25} {metrics_2009['MAE']:>20.2f} {metrics_2016['MAE']:>20.2f} {mae_improve:>14.1f}%")

        # 誤差率平均値
        mean_err_improve = ((metrics_2009['誤差率平均値(%)'] - metrics_2016['誤差率平均値(%)']) / metrics_2009['誤差率平均値(%)'] * 100)
        print(f"{'誤差率平均値(%)':<25} {metrics_2009['誤差率平均値(%)']:>20.2f} {metrics_2016['誤差率平均値(%)']:>20.2f} {mean_err_improve:>14.1f}%")

        # 誤差率中央値
        median_err_improve = ((metrics_2009['誤差率中央値(%)'] - metrics_2016['誤差率中央値(%)']) / metrics_2009['誤差率中央値(%)'] * 100)
        print(f"{'誤差率中央値(%)':<25} {metrics_2009['誤差率中央値(%)']:>20.2f} {metrics_2016['誤差率中央値(%)']:>20.2f} {median_err_improve:>14.1f}%")

        print("\n" + "=" * 80)
        print("【誤差率分布の比較】")
        print("=" * 80)
        print(f"\n{'誤差範囲':<25} {'20251009データ':>20} {'20251016データ':>20} {'改善':>15}")
        print("-" * 80)

        # 20%以内
        within20_diff = metrics_2016['20%以内の割合(%)'] - metrics_2009['20%以内の割合(%)']
        print(f"{'20%以内の割合(%)':<25} {metrics_2009['20%以内の割合(%)']:>19.2f}% {metrics_2016['20%以内の割合(%)']:>19.2f}% {within20_diff:>13.1f}pt")

        # 30%以内
        within30_diff = metrics_2016['30%以内の割合(%)'] - metrics_2009['30%以内の割合(%)']
        print(f"{'30%以内の割合(%)':<25} {metrics_2009['30%以内の割合(%)']:>19.2f}% {metrics_2016['30%以内の割合(%)']:>19.2f}% {within30_diff:>13.1f}pt")

        # 50%以内
        within50_diff = metrics_2016['50%以内の割合(%)'] - metrics_2009['50%以内の割合(%)']
        print(f"{'50%以内の割合(%)':<25} {metrics_2009['50%以内の割合(%)']:>19.2f}% {metrics_2016['50%以内の割合(%)']:>19.2f}% {within50_diff:>13.1f}pt")

        print("\n" + "=" * 80)
        print("【その他の指標】")
        print("=" * 80)
        print(f"\n{'指標':<25} {'20251009データ':>20} {'20251016データ':>20}")
        print("-" * 80)
        print(f"{'相関係数':<25} {metrics_2009['相関係数']:>20.4f} {metrics_2016['相関係数']:>20.4f}")
        print(f"{'R²スコア':<25} {metrics_2009['R²スコア']:>20.4f} {metrics_2016['R²スコア']:>20.4f}")
        print(f"{'評価対象数':<25} {metrics_2009['評価対象数']:>20,} {metrics_2016['評価対象数']:>20,}")

        # 総合評価
        print("\n" + "=" * 80)
        print("【総合評価】")
        print("=" * 80)

        improvements = []
        if rmse_improve > 0:
            improvements.append(f"RMSE: {rmse_improve:.1f}%改善")
        if mae_improve > 0:
            improvements.append(f"MAE: {mae_improve:.1f}%改善")
        if within20_diff > 0:
            improvements.append(f"高精度予測: {within20_diff:.1f}pt増加")

        if improvements:
            print("\n✅ 改善点:")
            for imp in improvements:
                print(f"  - {imp}")

        deteriorations = []
        if rmse_improve < 0:
            deteriorations.append(f"RMSE: {abs(rmse_improve):.1f}%悪化")
        if mae_improve < 0:
            deteriorations.append(f"MAE: {abs(mae_improve):.1f}%悪化")
        if within20_diff < 0:
            deteriorations.append(f"高精度予測: {abs(within20_diff):.1f}pt減少")

        if deteriorations:
            print("\n⚠️ 悪化点:")
            for det in deteriorations:
                print(f"  - {det}")

        # 最終判定
        overall_score = (rmse_improve + mae_improve + within20_diff) / 3
        print(f"\n総合スコア: {overall_score:.1f}")
        if overall_score > 5:
            print("判定: 🎉 大幅に改善されました")
        elif overall_score > 0:
            print("判定: ✅ 改善されました")
        elif overall_score > -5:
            print("判定: 📊 ほぼ同等の性能です")
        else:
            print("判定: ⚠️ 性能が低下しました")

    elif metrics_2009:
        print("\n20251016データの結果がまだ生成されていません。")
    elif metrics_2016:
        print("\n20251009データの結果が見つかりません。")
    else:
        print("\n両方のデータが見つかりません。")

    print("\n" + "=" * 80)

if __name__ == "__main__":
    main()