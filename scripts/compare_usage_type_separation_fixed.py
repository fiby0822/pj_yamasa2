#!/usr/bin/env python3
"""
Usage Type分離前後の精度比較レポート（修正版）
分離前と同じ計算方法で誤差率を算出
"""

import json
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

def calculate_accuracy_metrics(df):
    """
    分離前と同じ方法で精度指標を計算
    Material Key単位でError_Rateを計算し、全体の誤差率を算出
    """
    # 基本指標
    metrics = {}
    metrics['RMSE'] = np.sqrt(np.mean((df['actual'] - df['predicted']) ** 2))
    metrics['MAE'] = np.mean(np.abs(df['actual'] - df['predicted']))

    # Material Key単位で集計
    material_metrics = []

    for material_key in df['material_key'].unique():
        mk_df = df[df['material_key'] == material_key]

        # 各Material Keyの絶対誤差を計算
        abs_errors = np.abs(mk_df['actual'].values - mk_df['predicted'].values)

        # Error_Rate_XXの計算（絶対誤差が閾値を超える割合）
        mk_metric = {
            'material_key': material_key,
            'count': len(mk_df),
            'Error_Rate_5': np.mean(abs_errors > 5) * 100,
            'Error_Rate_10': np.mean(abs_errors > 10) * 100,
            'Error_Rate_20': np.mean(abs_errors > 20) * 100,
            'Error_Rate_30': np.mean(abs_errors > 30) * 100,
            'Error_Rate_50': np.mean(abs_errors > 50) * 100,
            'RMSE': np.sqrt(np.mean((mk_df['actual'] - mk_df['predicted']) ** 2)),
            'MAE': np.mean(np.abs(mk_df['actual'] - mk_df['predicted']))
        }
        material_metrics.append(mk_metric)

    material_df = pd.DataFrame(material_metrics)

    # 誤差率の計算（Error_Rate_XX = 0のMaterial Key数 / 全Material Key数）
    total_materials = len(material_df)
    metrics['within_20_percent_count'] = sum(material_df['Error_Rate_20'] == 0)
    metrics['within_30_percent_count'] = sum(material_df['Error_Rate_30'] == 0)
    metrics['within_50_percent_count'] = sum(material_df['Error_Rate_50'] == 0)

    metrics['within_20_percent'] = (metrics['within_20_percent_count'] / total_materials * 100) if total_materials > 0 else 0
    metrics['within_30_percent'] = (metrics['within_30_percent_count'] / total_materials * 100) if total_materials > 0 else 0
    metrics['within_50_percent'] = (metrics['within_50_percent_count'] / total_materials * 100) if total_materials > 0 else 0

    # 誤差値の平均・中央値（絶対誤差ベース）
    all_abs_errors = np.abs(df['actual'] - df['predicted'])
    metrics['error_mean'] = np.mean(all_abs_errors)
    metrics['error_median'] = np.median(all_abs_errors)

    # MAPE（actual > 0のレコードのみ）
    df_nonzero = df[df['actual'] > 0].copy()
    if len(df_nonzero) > 0:
        mape_values = np.abs((df_nonzero['predicted'] - df_nonzero['actual']) / df_nonzero['actual']) * 100
        # 外れ値の影響を減らすため上限設定
        mape_values = mape_values.clip(upper=1000)
        metrics['MAPE'] = mape_values.mean()
    else:
        metrics['MAPE'] = 0

    # 相関係数とR2スコア
    metrics['correlation'] = df['predicted'].corr(df['actual'])

    # R2スコア
    ss_res = np.sum((df['actual'] - df['predicted']) ** 2)
    ss_tot = np.sum((df['actual'] - df['actual'].mean()) ** 2)
    metrics['R2_score'] = 1 - (ss_res / ss_tot) if ss_tot > 0 else -999

    # Material Key数を追加
    metrics['material_key_count'] = total_materials

    return metrics, material_df

def main():
    print('='*70)
    print('Usage Type分離前後の精度比較レポート（修正版）')
    print('='*70)

    # ベースライン（分離前）のデータ読み込み
    with open('baseline_metrics_before_separation.json', 'r') as f:
        baseline = json.load(f)

    # 分離後の予測結果読み込み
    print('\n1. 分離後の予測結果を読み込み中...')

    # Business
    try:
        df_business = pd.read_csv('data/output/predicted_business.csv')
        print(f'  Business: {len(df_business):,}件')
    except FileNotFoundError:
        print('  Business: データなし')
        df_business = pd.DataFrame()

    # Household
    try:
        df_household = pd.read_csv('data/output/predicted_household.csv')
        print(f'  Household: {len(df_household):,}件')
    except FileNotFoundError:
        print('  Household: データなし')
        df_household = pd.DataFrame()

    # 結合
    df_combined = pd.concat([df_business, df_household], ignore_index=True)
    print(f'  合計: {len(df_combined):,}件')

    # 精度評価（修正版の計算方法）
    print('\n2. 精度評価中（Material Key単位で集計）...')

    if len(df_business) > 0:
        metrics_business, material_df_business = calculate_accuracy_metrics(df_business)
        print(f'\n  Business:')
        print(f'    Material Key数: {metrics_business["material_key_count"]}')
        print(f'    誤差20以内: {metrics_business["within_20_percent_count"]}個 ({metrics_business["within_20_percent"]:.1f}%)')
    else:
        metrics_business = {}
        material_df_business = pd.DataFrame()

    if len(df_household) > 0:
        metrics_household, material_df_household = calculate_accuracy_metrics(df_household)
        print(f'\n  Household:')
        print(f'    Material Key数: {metrics_household["material_key_count"]}')
        print(f'    誤差20以内: {metrics_household["within_20_percent_count"]}個 ({metrics_household["within_20_percent"]:.1f}%)')
    else:
        metrics_household = {}
        material_df_household = pd.DataFrame()

    if len(df_combined) > 0:
        metrics_combined, material_df_combined = calculate_accuracy_metrics(df_combined)
        print(f'\n  Combined (全体):')
        print(f'    Material Key数: {metrics_combined["material_key_count"]}')
        print(f'    誤差20以内: {metrics_combined["within_20_percent_count"]}個 ({metrics_combined["within_20_percent"]:.1f}%)')
    else:
        metrics_combined = {}
        material_df_combined = pd.DataFrame()

    # 比較結果の表示
    print('\n' + '='*70)
    print('【比較結果】')
    print('='*70)

    # 基本指標の比較
    print('\n■ 基本評価指標')
    print('                  分離前        分離後        改善率')
    print('-' * 55)

    if metrics_combined:
        rmse_improvement = (baseline['basic_metrics']['RMSE'] - metrics_combined['RMSE']) / baseline['basic_metrics']['RMSE'] * 100
        mae_improvement = (baseline['basic_metrics']['MAE'] - metrics_combined['MAE']) / baseline['basic_metrics']['MAE'] * 100

        print(f'RMSE:          {baseline["basic_metrics"]["RMSE"]:8.2f}     {metrics_combined["RMSE"]:8.2f}     {rmse_improvement:+6.1f}%')
        print(f'MAE:           {baseline["basic_metrics"]["MAE"]:8.2f}     {metrics_combined["MAE"]:8.2f}     {mae_improvement:+6.1f}%')
        print(f'誤差平均:      {baseline["basic_metrics"]["error_mean"]:8.2f}     {metrics_combined["error_mean"]:8.2f}')
        print(f'誤差中央値:    {baseline["basic_metrics"]["error_median"]:8.2f}     {metrics_combined["error_median"]:8.2f}')

    # 誤差率分析の比較（Material Key単位）
    print('\n■ 誤差率分析（Material Key単位）')
    print('                        分離前                分離後')
    print('-' * 70)

    if metrics_combined:
        # 分離前の件数と割合
        baseline_20_count = int(baseline['error_rate_analysis']['within_20_percent']['ratio'] / 100 * baseline['material_keys_in_test'])
        baseline_30_count = int(baseline['error_rate_analysis']['within_30_percent']['ratio'] / 100 * baseline['material_keys_in_test'])
        baseline_50_count = int(baseline['error_rate_analysis']['within_50_percent']['ratio'] / 100 * baseline['material_keys_in_test'])

        print(f'誤差20以内:  {baseline_20_count:5,}個 ({baseline["error_rate_analysis"]["within_20_percent"]["ratio"]:5.1f}%)    '
              f'{metrics_combined["within_20_percent_count"]:5,}個 ({metrics_combined["within_20_percent"]:5.1f}%)')
        print(f'誤差30以内:  {baseline_30_count:5,}個 ({baseline["error_rate_analysis"]["within_30_percent"]["ratio"]:5.1f}%)    '
              f'{metrics_combined["within_30_percent_count"]:5,}個 ({metrics_combined["within_30_percent"]:5.1f}%)')
        print(f'誤差50以内:  {baseline_50_count:5,}個 ({baseline["error_rate_analysis"]["within_50_percent"]["ratio"]:5.1f}%)    '
              f'{metrics_combined["within_50_percent_count"]:5,}個 ({metrics_combined["within_50_percent"]:5.1f}%)')

    # 追加指標
    print('\n■ 追加指標')
    print('                  分離前        分離後')
    print('-' * 55)

    if metrics_combined:
        print(f'MAPE:          {baseline["additional_metrics"]["MAPE"]:8.2f}     {metrics_combined["MAPE"]:8.2f}')
        print(f'相関係数:      {baseline["additional_metrics"]["correlation"]:8.4f}     {metrics_combined["correlation"]:8.4f}')
        print(f'R²スコア:      {baseline["additional_metrics"]["R2_score"]:8.4f}     {metrics_combined["R2_score"]:8.4f}')

    # Usage Type別の詳細
    print('\n■ Usage Type別の詳細')
    print('-' * 55)

    if metrics_business:
        print(f'\nBusiness:')
        print(f'  RMSE: {metrics_business["RMSE"]:.2f}')
        print(f'  MAE: {metrics_business["MAE"]:.2f}')
        print(f'  誤差20以内: {metrics_business["within_20_percent"]:.1f}%')
        print(f'  誤差30以内: {metrics_business["within_30_percent"]:.1f}%')

    if metrics_household:
        print(f'\nHousehold:')
        print(f'  RMSE: {metrics_household["RMSE"]:.2f}')
        print(f'  MAE: {metrics_household["MAE"]:.2f}')
        print(f'  誤差20以内: {metrics_household["within_20_percent"]:.1f}%')
        print(f'  誤差30以内: {metrics_household["within_30_percent"]:.1f}%')

    # 結果をJSONで保存
    output = {
        'comparison_date': pd.Timestamp.now().isoformat(),
        'baseline_metrics': baseline,
        'separated_metrics_fixed': {
            'business': {k: float(v) if isinstance(v, (np.float32, np.float64)) else v
                        for k, v in metrics_business.items()},
            'household': {k: float(v) if isinstance(v, (np.float32, np.float64)) else v
                         for k, v in metrics_household.items()},
            'combined': {k: float(v) if isinstance(v, (np.float32, np.float64)) else v
                        for k, v in metrics_combined.items()}
        },
        'improvements': {
            'RMSE_improvement_pct': (baseline['basic_metrics']['RMSE'] - metrics_combined['RMSE']) / baseline['basic_metrics']['RMSE'] * 100 if metrics_combined else None,
            'MAE_improvement_pct': (baseline['basic_metrics']['MAE'] - metrics_combined['MAE']) / baseline['basic_metrics']['MAE'] * 100 if metrics_combined else None,
            'within_20_pct_improvement_pp': metrics_combined['within_20_percent'] - baseline['error_rate_analysis']['within_20_percent']['ratio'] if metrics_combined else None
        },
        'data_stats': {
            'business_records': len(df_business),
            'household_records': len(df_household),
            'combined_records': len(df_combined),
            'business_material_keys': df_business['material_key'].nunique() if len(df_business) > 0 else 0,
            'household_material_keys': df_household['material_key'].nunique() if len(df_household) > 0 else 0
        }
    }

    with open('usage_type_separation_comparison_fixed.json', 'w', encoding='utf-8') as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    print('\n✅ 結果をusage_type_separation_comparison_fixed.jsonに保存しました')

    # Material Key別の結果も保存
    if not material_df_combined.empty:
        material_df_combined.to_csv('material_key_metrics_combined.csv', index=False)
        print('✅ Material Key別の詳細をmaterial_key_metrics_combined.csvに保存しました')

    # 最終結論
    print('\n' + '='*70)
    print('【結論】')
    print('='*70)

    if metrics_combined:
        overall_improvement = (baseline['basic_metrics']['RMSE'] - metrics_combined['RMSE']) / baseline['basic_metrics']['RMSE'] * 100
        within_20_improvement = metrics_combined['within_20_percent'] - baseline['error_rate_analysis']['within_20_percent']['ratio']

        if overall_improvement > 0:
            print(f'\n✅ Usage Type分離により、全体的な予測精度が{overall_improvement:.1f}%改善しました。')
        else:
            print(f'\n⚠️ Usage Type分離後の全体精度は{abs(overall_improvement):.1f}%低下しました。')

        if within_20_improvement > 0:
            print(f'✅ 誤差20以内のMaterial Key割合が{within_20_improvement:.1f}ポイント改善しました。')
        else:
            print(f'⚠️ 誤差20以内のMaterial Key割合が{abs(within_20_improvement):.1f}ポイント低下しました。')

        print(f'\n詳細:')
        print(f'  - Business向けモデル: RMSE {metrics_business["RMSE"]:.2f}, 誤差20以内 {metrics_business["within_20_percent"]:.1f}%')
        print(f'  - Household向けモデル: RMSE {metrics_household["RMSE"]:.2f}, 誤差20以内 {metrics_household["within_20_percent"]:.1f}%')
    else:
        print('\n分離後のデータが見つかりませんでした。')


if __name__ == '__main__':
    main()