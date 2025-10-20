"""
精度検証スクリプト
予測結果と実績値を比較して、各種精度指標を計算する
"""

import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import json
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

def calculate_metrics(y_true, y_pred):
    """各種評価指標を計算"""
    # ゼロ除算を避けるため、実績値が0の場合は除外
    mask = y_true > 0
    y_true_filtered = y_true[mask]
    y_pred_filtered = y_pred[mask]

    if len(y_true_filtered) == 0:
        return {}

    # 基本指標
    rmse = np.sqrt(mean_squared_error(y_true_filtered, y_pred_filtered))
    mae = mean_absolute_error(y_true_filtered, y_pred_filtered)

    # 誤差率
    error_rate = np.abs(y_true_filtered - y_pred_filtered) / y_true_filtered * 100
    mean_error_rate = np.mean(error_rate)
    median_error_rate = np.median(error_rate)

    # MAPE
    mape = np.mean(error_rate)

    # 相関係数
    correlation = np.corrcoef(y_true_filtered, y_pred_filtered)[0, 1]

    # R²スコア
    r2 = r2_score(y_true_filtered, y_pred_filtered)

    # 誤差率分析
    within_20 = np.sum(error_rate <= 20)
    within_30 = np.sum(error_rate <= 30)
    within_50 = np.sum(error_rate <= 50)
    total_count = len(error_rate)

    return {
        "基本評価指標": {
            "RMSE": round(rmse, 2),
            "MAE": round(mae, 2),
            "誤差率平均値(%)": round(mean_error_rate, 2),
            "誤差率中央値(%)": round(median_error_rate, 2)
        },
        "誤差率分析": {
            "20%以内": {
                "件数": within_20,
                "割合(%)": round(within_20 / total_count * 100, 2)
            },
            "30%以内": {
                "件数": within_30,
                "割合(%)": round(within_30 / total_count * 100, 2)
            },
            "50%以内": {
                "件数": within_50,
                "割合(%)": round(within_50 / total_count * 100, 2)
            },
            "評価対象数": total_count
        },
        "追加指標": {
            "MAPE(%)": round(mape, 2),
            "相関係数": round(correlation, 4),
            "R²スコア": round(r2, 4)
        }
    }

def analyze_by_material_key(df):
    """Material Key別の精度分析"""
    results = []

    for mk in df['material_key'].unique():
        mk_data = df[df['material_key'] == mk]

        if len(mk_data) == 0 or mk_data['actual'].sum() == 0:
            continue

        y_true = mk_data['actual'].values
        y_pred = mk_data['prediction'].values

        # 誤差率計算（実績値が0でない場合のみ）
        mask = y_true > 0
        if mask.sum() == 0:
            continue

        error_rate = np.abs(y_true[mask] - y_pred[mask]) / y_true[mask] * 100

        results.append({
            'material_key': mk,
            'actual_sum': y_true.sum(),
            'predicted_sum': y_pred.sum(),
            'mean_error_rate': np.mean(error_rate),
            'median_error_rate': np.median(error_rate),
            'max_error_rate': np.max(error_rate),
            'min_error_rate': np.min(error_rate)
        })

    return pd.DataFrame(results)

def main():
    print("=" * 60)
    print("精度検証開始")
    print("=" * 60)

    # 予測結果ファイルを読み込み
    predictions_path = Path("data/output/confirmed_order_demand_yamasa_predictions_latest.parquet")

    if not predictions_path.exists():
        print(f"予測結果ファイルが見つかりません: {predictions_path}")
        return

    print(f"\n予測結果ファイルを読み込み中: {predictions_path}")
    df = pd.read_parquet(predictions_path)

    # 'prediction'カラムを'predicted'カラムに合わせる
    if 'predicted' in df.columns and 'prediction' not in df.columns:
        df['prediction'] = df['predicted']

    print(f"データ件数: {len(df):,}")
    print(f"Material Key数: {df['material_key'].nunique():,}")
    if 'date' in df.columns:
        print(f"予測期間: {df['date'].min()} - {df['date'].max()}")
    elif 'year_month' in df.columns:
        print(f"予測期間: {df['year_month'].min()} - {df['year_month'].max()}")

    # 全体の精度評価
    print("\n" + "=" * 60)
    print("全体の精度評価")
    print("=" * 60)

    overall_metrics = calculate_metrics(df['actual'].values, df['prediction'].values)

    for category, metrics in overall_metrics.items():
        print(f"\n【{category}】")
        if isinstance(metrics, dict):
            for key, value in metrics.items():
                if isinstance(value, dict):
                    print(f"  {key}:")
                    for k, v in value.items():
                        print(f"    {k}: {v}")
                else:
                    print(f"  {key}: {value}")
        else:
            print(f"  {metrics}")

    # Material Key別の分析
    print("\n" + "=" * 60)
    print("Material Key別分析")
    print("=" * 60)

    mk_analysis = analyze_by_material_key(df)

    # 誤差率の低い順にTop10を表示
    print("\n【精度の良いMaterial Key Top10】")
    top10 = mk_analysis.nsmallest(10, 'mean_error_rate')
    for idx, row in top10.iterrows():
        print(f"  {row['material_key']}: 誤差率 {row['mean_error_rate']:.2f}% (実績合計: {row['actual_sum']:.0f})")

    # 誤差率の高い順にTop10を表示
    print("\n【精度の悪いMaterial Key Top10】")
    bottom10 = mk_analysis.nlargest(10, 'mean_error_rate')
    for idx, row in bottom10.iterrows():
        print(f"  {row['material_key']}: 誤差率 {row['mean_error_rate']:.2f}% (実績合計: {row['actual_sum']:.0f})")

    # 実績値の規模別分析
    print("\n" + "=" * 60)
    print("実績値規模別分析")
    print("=" * 60)

    # 実績値の四分位数で分類
    q1 = mk_analysis['actual_sum'].quantile(0.25)
    q2 = mk_analysis['actual_sum'].quantile(0.50)
    q3 = mk_analysis['actual_sum'].quantile(0.75)

    print(f"\n実績値の分布:")
    print(f"  第1四分位: {q1:.0f}")
    print(f"  中央値: {q2:.0f}")
    print(f"  第3四分位: {q3:.0f}")

    # 各四分位での平均誤差率
    small = mk_analysis[mk_analysis['actual_sum'] <= q1]['mean_error_rate'].mean()
    medium_low = mk_analysis[(mk_analysis['actual_sum'] > q1) & (mk_analysis['actual_sum'] <= q2)]['mean_error_rate'].mean()
    medium_high = mk_analysis[(mk_analysis['actual_sum'] > q2) & (mk_analysis['actual_sum'] <= q3)]['mean_error_rate'].mean()
    large = mk_analysis[mk_analysis['actual_sum'] > q3]['mean_error_rate'].mean()

    print(f"\n規模別平均誤差率:")
    print(f"  小規模 (≤{q1:.0f}): {small:.2f}%")
    print(f"  中規模-低 ({q1:.0f}-{q2:.0f}): {medium_low:.2f}%")
    print(f"  中規模-高 ({q2:.0f}-{q3:.0f}): {medium_high:.2f}%")
    print(f"  大規模 (>{q3:.0f}): {large:.2f}%")

    # 結果をファイルに保存
    results = {
        "overall_metrics": overall_metrics,
        "summary": {
            "total_records": len(df),
            "material_keys": df['material_key'].nunique(),
            "prediction_period": f"{df['year_month'].min()} - {df['year_month'].max()}",
            "scale_analysis": {
                "small": {"threshold": f"≤{q1:.0f}", "mean_error_rate": round(small, 2)},
                "medium_low": {"threshold": f"{q1:.0f}-{q2:.0f}", "mean_error_rate": round(medium_low, 2)},
                "medium_high": {"threshold": f"{q2:.0f}-{q3:.0f}", "mean_error_rate": round(medium_high, 2)},
                "large": {"threshold": f">{q3:.0f}", "mean_error_rate": round(large, 2)}
            }
        }
    }

    output_path = Path("data/output/accuracy_evaluation_results.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"\n評価結果を保存しました: {output_path}")

    # Material Key別詳細をCSVで保存
    csv_path = Path("data/output/material_key_accuracy_analysis.csv")
    mk_analysis.to_csv(csv_path, index=False)
    print(f"Material Key別詳細を保存しました: {csv_path}")

    print("\n" + "=" * 60)
    print("精度検証完了")
    print("=" * 60)

if __name__ == "__main__":
    main()