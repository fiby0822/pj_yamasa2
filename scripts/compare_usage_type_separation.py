#!/usr/bin/env python3
"""
Usage Type分離前後の精度比較スクリプト
"""
import json
import pandas as pd
import numpy as np
from datetime import datetime
import boto3
import sys

def calculate_metrics(df):
    """メトリクス計算"""
    metrics = {}

    # 基本メトリクス
    errors = df['predicted'] - df['actual']
    metrics['RMSE'] = np.sqrt(np.mean(errors ** 2))
    metrics['MAE'] = np.mean(np.abs(errors))
    metrics['error_mean'] = np.mean(np.abs(errors))
    metrics['error_median'] = np.median(np.abs(errors))

    # 誤差率分析（actual > 0のレコードのみ）
    df_nonzero = df[df['actual'] > 0].copy()
    if len(df_nonzero) > 0:
        df_nonzero['error_rate'] = np.abs(df_nonzero['predicted'] - df_nonzero['actual']) / df_nonzero['actual'] * 100
        metrics['within_20_percent'] = (df_nonzero['error_rate'] <= 20).mean() * 100
        metrics['within_30_percent'] = (df_nonzero['error_rate'] <= 30).mean() * 100
        metrics['within_50_percent'] = (df_nonzero['error_rate'] <= 50).mean() * 100

        # MAPE（外れ値の影響を減らすため上限設定）
        mape_values = df_nonzero['error_rate'].clip(upper=1000)
        metrics['MAPE'] = mape_values.mean()

    # 相関係数とR2スコア
    metrics['correlation'] = df['predicted'].corr(df['actual'])

    # R2スコア
    ss_res = np.sum((df['actual'] - df['predicted']) ** 2)
    ss_tot = np.sum((df['actual'] - df['actual'].mean()) ** 2)
    metrics['R2_score'] = 1 - (ss_res / ss_tot) if ss_tot > 0 else -999

    return metrics


def main():
    print('='*70)
    print('Usage Type分離前後の精度比較レポート')
    print('='*70)

    # 分離前のベースライン指標を読み込み
    with open('baseline_metrics_before_separation.json', 'r', encoding='utf-8') as f:
        baseline = json.load(f)

    # 分離後の結果を読み込み
    try:
        s3 = boto3.client('s3')

        # 結合済みの予測結果を読み込み（CSVファイル）
        response = s3.get_object(
            Bucket='fiby-yamasa-prediction-2',
            Key='output/evaluation/confirmed_order_demand_yamasa_predictions_latest.csv'
        )
        df_combined = pd.read_csv(response['Body'])

        # usage_typeで分離
        df_business = df_combined[df_combined['usage_type'] == 'business'].copy()
        df_household = df_combined[df_combined['usage_type'] == 'household'].copy()

        print(f'\nデータ読み込み完了:')
        print(f'  - Business: {len(df_business):,} records')
        print(f'  - Household: {len(df_household):,} records')
        print(f'  - Combined: {len(df_combined):,} records')

    except Exception as e:
        print(f'S3からの読み込みエラー: {e}')
        sys.exit(1)

    # メトリクス計算
    print('\nメトリクス計算中...')
    metrics_business = calculate_metrics(df_business)
    metrics_household = calculate_metrics(df_household)
    metrics_combined = calculate_metrics(df_combined)

    # 比較結果を表示
    print('\n' + '='*70)
    print('【精度比較結果】')
    print('='*70)

    print('\n1. 基本メトリクス:')
    print(f'{"指標":<20} {"分離前":>12} {"分離後(結合)":>12} {"改善率":>10}')
    print('-'*60)

    # RMSE
    improvement = (baseline['basic_metrics']['RMSE'] - metrics_combined['RMSE']) / baseline['basic_metrics']['RMSE'] * 100
    print(f'{"RMSE":<20} {baseline["basic_metrics"]["RMSE"]:>12.4f} {metrics_combined["RMSE"]:>12.4f} {improvement:>9.1f}%')

    # MAE
    improvement = (baseline['basic_metrics']['MAE'] - metrics_combined['MAE']) / baseline['basic_metrics']['MAE'] * 100
    print(f'{"MAE":<20} {baseline["basic_metrics"]["MAE"]:>12.4f} {metrics_combined["MAE"]:>12.4f} {improvement:>9.1f}%')

    # Error Mean
    improvement = (baseline['basic_metrics']['error_mean'] - metrics_combined['error_mean']) / baseline['basic_metrics']['error_mean'] * 100
    print(f'{"Error Mean":<20} {baseline["basic_metrics"]["error_mean"]:>12.4f} {metrics_combined["error_mean"]:>12.4f} {improvement:>9.1f}%')

    # Error Median
    improvement = (baseline['basic_metrics']['error_median'] - metrics_combined['error_median']) / baseline['basic_metrics']['error_median'] * 100
    print(f'{"Error Median":<20} {baseline["basic_metrics"]["error_median"]:>12.4f} {metrics_combined["error_median"]:>12.4f} {improvement:>9.1f}%')

    print('\n2. 誤差率分析:')
    print(f'{"指標":<20} {"分離前":>12} {"分離後(結合)":>12} {"改善":>10}')
    print('-'*60)

    # 20%以内
    improvement = metrics_combined['within_20_percent'] - baseline['error_rate_analysis']['within_20_percent']['ratio']
    print(f'{"20%以内":<20} {baseline["error_rate_analysis"]["within_20_percent"]["ratio"]:>11.2f}% {metrics_combined["within_20_percent"]:>11.2f}% {improvement:>+9.2f}pp')

    # 30%以内
    improvement = metrics_combined['within_30_percent'] - baseline['error_rate_analysis']['within_30_percent']['ratio']
    print(f'{"30%以内":<20} {baseline["error_rate_analysis"]["within_30_percent"]["ratio"]:>11.2f}% {metrics_combined["within_30_percent"]:>11.2f}% {improvement:>+9.2f}pp')

    # 50%以内
    improvement = metrics_combined['within_50_percent'] - baseline['error_rate_analysis']['within_50_percent']['ratio']
    print(f'{"50%以内":<20} {baseline["error_rate_analysis"]["within_50_percent"]["ratio"]:>11.2f}% {metrics_combined["within_50_percent"]:>11.2f}% {improvement:>+9.2f}pp')

    print('\n3. Usage Type別の精度:')
    print(f'{"Usage Type":<15} {"RMSE":>10} {"MAE":>10} {"20%以内":>12} {"相関係数":>10}')
    print('-'*60)
    print(f'{"Business":<15} {metrics_business["RMSE"]:>10.4f} {metrics_business["MAE"]:>10.4f} {metrics_business["within_20_percent"]:>11.2f}% {metrics_business["correlation"]:>10.4f}')
    print(f'{"Household":<15} {metrics_household["RMSE"]:>10.4f} {metrics_household["MAE"]:>10.4f} {metrics_household["within_20_percent"]:>11.2f}% {metrics_household["correlation"]:>10.4f}')
    print(f'{"Combined":<15} {metrics_combined["RMSE"]:>10.4f} {metrics_combined["MAE"]:>10.4f} {metrics_combined["within_20_percent"]:>11.2f}% {metrics_combined["correlation"]:>10.4f}')

    print('\n4. その他の指標:')
    print(f'{"指標":<20} {"分離前":>15} {"分離後(結合)":>15}')
    print('-'*60)
    print(f'{"MAPE":<20} {baseline["additional_metrics"]["MAPE"]:>15.2f} {metrics_combined["MAPE"]:>15.2f}')
    print(f'{"相関係数":<20} {baseline["additional_metrics"]["correlation"]:>15.4f} {metrics_combined["correlation"]:>15.4f}')
    print(f'{"R²スコア":<20} {baseline["additional_metrics"]["R2_score"]:>15.4f} {metrics_combined["R2_score"]:>15.4f}')

    # サマリー
    print('\n' + '='*70)
    print('【改善サマリー】')
    print('='*70)

    improvements = []
    if metrics_combined['RMSE'] < baseline['basic_metrics']['RMSE']:
        pct = (baseline['basic_metrics']['RMSE'] - metrics_combined['RMSE']) / baseline['basic_metrics']['RMSE'] * 100
        improvements.append(f'✅ RMSE: {pct:.1f}%改善')

    if metrics_combined['MAE'] < baseline['basic_metrics']['MAE']:
        pct = (baseline['basic_metrics']['MAE'] - metrics_combined['MAE']) / baseline['basic_metrics']['MAE'] * 100
        improvements.append(f'✅ MAE: {pct:.1f}%改善')

    if metrics_combined['within_20_percent'] > baseline['error_rate_analysis']['within_20_percent']['ratio']:
        pp = metrics_combined['within_20_percent'] - baseline['error_rate_analysis']['within_20_percent']['ratio']
        improvements.append(f'✅ 誤差率20%以内: {pp:.2f}ポイント向上')

    if improvements:
        print('\n改善項目:')
        for imp in improvements:
            print(f'  {imp}')
    else:
        print('\n⚠️ 主要指標に改善が見られませんでした')

    # Material Key別の分析
    print('\n5. Material Key別の改善状況:')

    # Business top 5
    business_by_mk = df_business.groupby('material_key').agg({
        'predicted': 'count',
        'actual': 'sum'
    })
    business_by_mk['RMSE'] = df_business.groupby('material_key').apply(
        lambda x: np.sqrt(np.mean((x['predicted'] - x['actual']) ** 2))
    ).values
    business_top5 = business_by_mk.nlargest(5, 'actual')[['actual', 'RMSE']]

    print('\nBusiness Top 5 Material Keys (by actual volume):')
    for mk, row in business_top5.iterrows():
        print(f'  {mk}: RMSE={row["RMSE"]:.2f}, Actual Sum={row["actual"]:.0f}')

    # Household top 5
    household_by_mk = df_household.groupby('material_key').agg({
        'predicted': 'count',
        'actual': 'sum'
    })
    household_by_mk['RMSE'] = df_household.groupby('material_key').apply(
        lambda x: np.sqrt(np.mean((x['predicted'] - x['actual']) ** 2))
    ).values
    household_top5 = household_by_mk.nlargest(5, 'actual')[['actual', 'RMSE']]

    print('\nHousehold Top 5 Material Keys (by actual volume):')
    for mk, row in household_top5.iterrows():
        print(f'  {mk}: RMSE={row["RMSE"]:.2f}, Actual Sum={row["actual"]:.0f}')

    # 結果を保存
    output = {
        'comparison_date': datetime.now().isoformat(),
        'baseline_metrics': baseline,
        'separated_metrics': {
            'business': {k: float(v) if isinstance(v, (np.float32, np.float64)) else v
                        for k, v in metrics_business.items()},
            'household': {k: float(v) if isinstance(v, (np.float32, np.float64)) else v
                         for k, v in metrics_household.items()},
            'combined': {k: float(v) if isinstance(v, (np.float32, np.float64)) else v
                        for k, v in metrics_combined.items()}
        },
        'improvements': {
            'RMSE_improvement_pct': (baseline['basic_metrics']['RMSE'] - metrics_combined['RMSE']) / baseline['basic_metrics']['RMSE'] * 100,
            'MAE_improvement_pct': (baseline['basic_metrics']['MAE'] - metrics_combined['MAE']) / baseline['basic_metrics']['MAE'] * 100,
            'within_20_pct_improvement_pp': metrics_combined['within_20_percent'] - baseline['error_rate_analysis']['within_20_percent']['ratio']
        },
        'data_stats': {
            'business_records': len(df_business),
            'household_records': len(df_household),
            'combined_records': len(df_combined),
            'business_material_keys': df_business['material_key'].nunique(),
            'household_material_keys': df_household['material_key'].nunique()
        }
    }

    with open('usage_type_separation_comparison.json', 'w', encoding='utf-8') as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    print('\n✅ 結果をusage_type_separation_comparison.jsonに保存しました')

    # 最終結論
    print('\n' + '='*70)
    print('【結論】')
    print('='*70)

    overall_improvement = (baseline['basic_metrics']['RMSE'] - metrics_combined['RMSE']) / baseline['basic_metrics']['RMSE'] * 100

    if overall_improvement > 0:
        print(f'\n✅ Usage Type分離により、全体的な予測精度が{overall_improvement:.1f}%改善しました。')
        print(f'   Business向けモデルは特に高精度（RMSE: {metrics_business["RMSE"]:.2f}）を達成し、')
        print(f'   Household向けモデルも特性に合わせた予測が可能になりました。')
    else:
        print(f'\n⚠️ Usage Type分離後の全体精度は{abs(overall_improvement):.1f}%低下しました。')
        print('   パラメータ調整や特徴量の見直しが必要かもしれません。')


if __name__ == '__main__':
    main()