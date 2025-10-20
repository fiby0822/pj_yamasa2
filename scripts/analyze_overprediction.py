#!/usr/bin/env python3
"""
予測値が実績値より大きくなる傾向の原因分析
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# 日本語フォント設定
plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

def load_test_results():
    """テスト結果データの読み込み"""
    print("=== テスト結果データの読み込み ===")

    # 予測結果ファイルの読み込み
    predictions_path = Path("data/output/confirmed_order_demand_yamasa_predictions_latest.parquet")
    summary_path = Path("data/output/confirmed_order_demand_yamasa_material_summary_latest.parquet")

    if predictions_path.exists():
        predictions = pd.read_parquet(predictions_path)
        print(f"Predictions loaded: {len(predictions)} records")
        print(f"Columns: {predictions.columns.tolist()}")

        # dateカラムをfile_dateに変換し、2025年1月のデータのみフィルタリング
        if 'date' in predictions.columns:
            predictions['file_date'] = pd.to_datetime(predictions['date'])
            test_data = predictions[predictions['file_date'].dt.year == 2025]
            print(f"2025年1月のテストデータ: {len(test_data)} records")
            return test_data
        elif 'file_date' in predictions.columns:
            predictions['file_date'] = pd.to_datetime(predictions['file_date'])
            test_data = predictions[predictions['file_date'].dt.year == 2025]
            print(f"2025年1月のテストデータ: {len(test_data)} records")
            return test_data
        else:
            print("date/file_date column not found in predictions")
            # stepカラムでフィルタリング（step=1が2025年1月を意味する場合）
            if 'step' in predictions.columns:
                test_data = predictions[predictions['step'] == 1]
                print(f"step=1のテストデータ: {len(test_data)} records")
                return test_data
            return predictions
    else:
        print(f"File not found: {predictions_path}")
        return None

def calculate_overprediction_metrics(df):
    """過大予測の指標を計算"""
    print("\n=== 過大予測の分析 ===")

    if df is None or df.empty:
        print("No data available")
        return None

    # カラム名を修正
    if 'actual' in df.columns and 'predicted' in df.columns:
        df = df.rename(columns={
            'actual': 'actual_value',
            'predicted': 'predicted_value'
        })

    # 必要なカラムの確認
    required_cols = ['actual_value', 'predicted_value', 'material_key']
    for col in required_cols:
        if col not in df.columns:
            print(f"Column {col} not found. Available columns: {df.columns.tolist()}")
            return None

    # 基本統計
    df['prediction_ratio'] = df['predicted_value'] / df['actual_value'].replace(0, np.nan)
    df['absolute_error'] = df['predicted_value'] - df['actual_value']
    df['relative_error'] = df['absolute_error'] / df['actual_value'].replace(0, np.nan)

    # 全体的な傾向
    print("\n【全体統計】")
    print(f"平均実績値: {df['actual_value'].mean():.2f}")
    print(f"平均予測値: {df['predicted_value'].mean():.2f}")
    print(f"予測/実績比率の中央値: {df['prediction_ratio'].median():.2f}")
    print(f"予測/実績比率の平均値: {df['prediction_ratio'].mean():.2f}")

    # 過大予測の割合
    overpredicted = (df['predicted_value'] > df['actual_value']).sum()
    underpredicted = (df['predicted_value'] < df['actual_value']).sum()
    exact = (df['predicted_value'] == df['actual_value']).sum()

    print(f"\n【予測の傾向】")
    print(f"過大予測: {overpredicted} ({overpredicted/len(df)*100:.1f}%)")
    print(f"過小予測: {underpredicted} ({underpredicted/len(df)*100:.1f}%)")
    print(f"完全一致: {exact} ({exact/len(df)*100:.1f}%)")

    # 実績値の規模別分析
    print("\n【実績値の規模別分析】")
    df['actual_scale'] = pd.cut(df['actual_value'],
                                bins=[0, 10, 50, 100, 500, np.inf],
                                labels=['0-10', '10-50', '50-100', '100-500', '500+'])

    scale_analysis = df.groupby('actual_scale').agg({
        'prediction_ratio': ['mean', 'median', 'count'],
        'absolute_error': 'mean',
        'relative_error': 'mean'
    }).round(2)
    print(scale_analysis)

    return df

def analyze_by_material_key(df):
    """Material Key別の分析"""
    print("\n=== Material Key別分析 ===")

    if df is None or df.empty:
        return None

    # Material Key別の集計
    mk_stats = df.groupby('material_key').agg({
        'actual_value': ['sum', 'mean', 'std'],
        'predicted_value': ['sum', 'mean', 'std'],
        'prediction_ratio': ['mean', 'median'],
        'absolute_error': 'mean'
    }).round(2)

    # カラム名をフラット化
    mk_stats.columns = ['_'.join(col).strip() for col in mk_stats.columns]
    mk_stats = mk_stats.reset_index()

    # 過大予測が顕著なMaterial Key
    mk_stats['overprediction_score'] = mk_stats['predicted_value_sum'] / mk_stats['actual_value_sum']

    print("\n【過大予測Top10 Material Keys】")
    top_overpredicted = mk_stats.nlargest(10, 'overprediction_score')[
        ['material_key', 'actual_value_sum', 'predicted_value_sum', 'overprediction_score']
    ]
    print(top_overpredicted)

    print("\n【予測精度が良いTop10 Material Keys】")
    mk_stats['accuracy_score'] = 1 / (1 + abs(mk_stats['overprediction_score'] - 1))
    top_accurate = mk_stats.nlargest(10, 'accuracy_score')[
        ['material_key', 'actual_value_sum', 'predicted_value_sum', 'overprediction_score', 'accuracy_score']
    ]
    print(top_accurate)

    return mk_stats

def analyze_temporal_patterns(df):
    """時系列パターンの分析"""
    print("\n=== 時系列パターン分析 ===")

    if df is None or df.empty:
        return None

    if 'file_date' in df.columns:
        # 日付別の集計
        daily_stats = df.groupby('file_date').agg({
            'actual_value': 'sum',
            'predicted_value': 'sum',
            'prediction_ratio': 'mean'
        }).reset_index()

        daily_stats['overprediction_rate'] = daily_stats['predicted_value'] / daily_stats['actual_value']

        print("\n【日別の過大予測率】")
        print(daily_stats[['file_date', 'actual_value', 'predicted_value', 'overprediction_rate']].round(2))

        # 曜日効果の分析
        df['weekday'] = pd.to_datetime(df['file_date']).dt.dayofweek
        weekday_stats = df.groupby('weekday').agg({
            'prediction_ratio': 'mean',
            'absolute_error': 'mean'
        }).round(2)
        weekday_stats.index = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
        print("\n【曜日別の予測傾向】")
        print(weekday_stats)

        return daily_stats
    else:
        print("file_date column not found")
        return None

def create_diagnostic_plots(df, mk_stats):
    """診断用の可視化"""
    print("\n=== 診断用可視化の作成 ===")

    if df is None or df.empty:
        print("No data for plotting")
        return

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    # 1. 実績値 vs 予測値の散布図
    ax = axes[0, 0]
    ax.scatter(df['actual_value'], df['predicted_value'], alpha=0.5, s=10)
    max_val = max(df['actual_value'].max(), df['predicted_value'].max())
    ax.plot([0, max_val], [0, max_val], 'r--', label='Perfect prediction')
    ax.set_xlabel('Actual Value')
    ax.set_ylabel('Predicted Value')
    ax.set_title('Actual vs Predicted')
    ax.legend()

    # 2. 予測比率のヒストグラム
    ax = axes[0, 1]
    ratio_data = df['prediction_ratio'].dropna()
    ratio_data = ratio_data[ratio_data < 10]  # 外れ値除外
    ax.hist(ratio_data, bins=50, edgecolor='black')
    ax.axvline(x=1, color='r', linestyle='--', label='Perfect ratio')
    ax.set_xlabel('Prediction Ratio (Pred/Actual)')
    ax.set_ylabel('Frequency')
    ax.set_title('Distribution of Prediction Ratio')
    ax.legend()

    # 3. 実績値規模別の予測比率
    ax = axes[0, 2]
    scale_data = df.groupby('actual_scale')['prediction_ratio'].mean().sort_index()
    scale_data.plot(kind='bar', ax=ax)
    ax.axhline(y=1, color='r', linestyle='--')
    ax.set_xlabel('Actual Value Scale')
    ax.set_ylabel('Average Prediction Ratio')
    ax.set_title('Prediction Ratio by Scale')
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45)

    # 4. 絶対誤差の分布
    ax = axes[1, 0]
    abs_error = df['absolute_error'].abs()
    abs_error = abs_error[abs_error < abs_error.quantile(0.95)]  # 外れ値除外
    ax.hist(abs_error, bins=50, edgecolor='black')
    ax.set_xlabel('Absolute Error')
    ax.set_ylabel('Frequency')
    ax.set_title('Distribution of Absolute Error')

    # 5. Material Key別の過大予測スコア
    ax = axes[1, 1]
    if mk_stats is not None and not mk_stats.empty:
        top_mk = mk_stats.nlargest(20, 'overprediction_score')
        ax.barh(range(len(top_mk)), top_mk['overprediction_score'])
        ax.set_yticks(range(len(top_mk)))
        ax.set_yticklabels([mk[:10] for mk in top_mk['material_key']], fontsize=8)
        ax.axvline(x=1, color='r', linestyle='--')
        ax.set_xlabel('Overprediction Score')
        ax.set_title('Top 20 Overpredicted Material Keys')

    # 6. 残差プロット
    ax = axes[1, 2]
    residuals = df['predicted_value'] - df['actual_value']
    ax.scatter(df['predicted_value'], residuals, alpha=0.5, s=10)
    ax.axhline(y=0, color='r', linestyle='--')
    ax.set_xlabel('Predicted Value')
    ax.set_ylabel('Residual (Pred - Actual)')
    ax.set_title('Residual Plot')

    plt.tight_layout()
    plt.savefig('overprediction_diagnosis.png', dpi=150, bbox_inches='tight')
    print("診断プロットを overprediction_diagnosis.png に保存しました")
    plt.close()

    # 追加の時系列プロット
    if 'file_date' in df.columns:
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        # 日別の予測vs実績
        daily_data = df.groupby('file_date')[['actual_value', 'predicted_value']].sum().reset_index()
        ax = axes[0]
        ax.plot(daily_data['file_date'], daily_data['actual_value'], label='Actual', marker='o')
        ax.plot(daily_data['file_date'], daily_data['predicted_value'], label='Predicted', marker='s')
        ax.set_xlabel('Date')
        ax.set_ylabel('Total Value')
        ax.set_title('Daily Actual vs Predicted')
        ax.legend()
        ax.tick_params(axis='x', rotation=45)

        # 日別の過大予測率
        daily_data['overpred_rate'] = daily_data['predicted_value'] / daily_data['actual_value']
        ax = axes[1]
        ax.plot(daily_data['file_date'], daily_data['overpred_rate'], marker='o')
        ax.axhline(y=1, color='r', linestyle='--', label='Perfect prediction')
        ax.set_xlabel('Date')
        ax.set_ylabel('Overprediction Rate')
        ax.set_title('Daily Overprediction Rate')
        ax.legend()
        ax.tick_params(axis='x', rotation=45)

        plt.tight_layout()
        plt.savefig('temporal_analysis.png', dpi=150, bbox_inches='tight')
        print("時系列分析プロットを temporal_analysis.png に保存しました")
        plt.close()

def analyze_features():
    """特徴量レベルの分析"""
    print("\n=== 特徴量の分析 ===")

    # 特徴量ファイルの確認
    feature_path = Path("data/features/confirmed_order_demand_yamasa_features.parquet")

    if feature_path.exists():
        features = pd.read_parquet(feature_path)
        print(f"Features loaded: {len(features)} records, {len(features.columns)} columns")

        # 特徴量の統計情報
        numeric_cols = features.select_dtypes(include=[np.number]).columns

        # 欠損値の確認
        missing_ratio = features[numeric_cols].isnull().mean()
        high_missing = missing_ratio[missing_ratio > 0.1]
        if len(high_missing) > 0:
            print("\n【欠損値が多い特徴量（>10%）】")
            print(high_missing.sort_values(ascending=False))

        # 分散が極端に小さい特徴量
        low_variance = features[numeric_cols].var()
        near_zero_var = low_variance[low_variance < 0.01]
        if len(near_zero_var) > 0:
            print("\n【分散が極端に小さい特徴量】")
            print(near_zero_var.sort_values())

        # 相関の高い特徴量ペア
        corr_matrix = features[numeric_cols].corr()
        high_corr_pairs = []
        for i in range(len(corr_matrix.columns)):
            for j in range(i+1, len(corr_matrix.columns)):
                if abs(corr_matrix.iloc[i, j]) > 0.95:
                    high_corr_pairs.append((
                        corr_matrix.columns[i],
                        corr_matrix.columns[j],
                        corr_matrix.iloc[i, j]
                    ))

        if high_corr_pairs:
            print("\n【高相関の特徴量ペア（|r| > 0.95）】")
            for feat1, feat2, corr in high_corr_pairs[:10]:
                print(f"{feat1} <-> {feat2}: {corr:.3f}")

        return features
    else:
        print(f"Feature file not found: {feature_path}")
        return None

def generate_recommendations(df, mk_stats):
    """改善提案の生成"""
    print("\n" + "="*60)
    print("=== 改善提案 ===")
    print("="*60)

    recommendations = []

    # 1. 全体的な過大予測傾向への対処
    if df is not None and not df.empty:
        avg_ratio = df['prediction_ratio'].mean()
        if avg_ratio > 1.2:
            recommendations.append({
                'priority': 'HIGH',
                'category': 'モデル調整',
                'recommendation': f'予測値が平均で実績値の{avg_ratio:.1f}倍になっています。以下の対策を検討してください：\n'
                                '  - 予測値に0.8～0.9程度の補正係数を適用\n'
                                '  - モデルの正則化パラメータ（alpha, lambda等）を強化\n'
                                '  - 訓練データの外れ値処理を強化'
            })

    # 2. 実績値規模別の対処
    scale_bias = df.groupby('actual_scale')['prediction_ratio'].mean()
    if (scale_bias.max() - scale_bias.min()) > 0.5:
        recommendations.append({
            'priority': 'HIGH',
            'category': '規模別モデル',
            'recommendation': '実績値の規模によって予測精度が大きく異なります：\n'
                            f'  - 小規模（0-10）: 予測比率 {scale_bias.iloc[0]:.2f}\n'
                            f'  - 大規模（500+）: 予測比率 {scale_bias.iloc[-1]:.2f}\n'
                            '  → 実績値の規模別にモデルを分割することを推奨'
        })

    # 3. 特徴量エンジニアリング
    recommendations.append({
        'priority': 'MEDIUM',
        'category': '特徴量改善',
        'recommendation': '過大予測を改善する特徴量エンジニアリング：\n'
                        '  - 移動平均期間の調整（現在の期間が長すぎる可能性）\n'
                        '  - 季節性・トレンドの除去または正規化\n'
                        '  - 実績値の対数変換やBox-Cox変換の適用\n'
                        '  - 外れ値をクリップまたは除外する閾値の調整'
    })

    # 4. Material Key別の処理
    if mk_stats is not None and not mk_stats.empty:
        high_error_mk = mk_stats[mk_stats['overprediction_score'] > 2]
        if len(high_error_mk) > 0:
            recommendations.append({
                'priority': 'MEDIUM',
                'category': 'Material Key別処理',
                'recommendation': f'{len(high_error_mk)}個のMaterial Keyで2倍以上の過大予測が発生：\n'
                                '  - これらのMaterial Key専用のモデルを構築\n'
                                '  - 特殊な季節性パターンの個別学習\n'
                                '  - 異常値として別途処理'
            })

    # 5. モデルアンサンブル
    recommendations.append({
        'priority': 'LOW',
        'category': 'アンサンブル学習',
        'recommendation': '複数モデルのアンサンブルによる改善：\n'
                        '  - LightGBMに加えてXGBoost、CatBoostも学習\n'
                        '  - 予測値の中央値や加重平均を採用\n'
                        '  - スタッキングによるメタ学習器の導入'
    })

    # 6. 後処理の導入
    recommendations.append({
        'priority': 'HIGH',
        'category': '後処理',
        'recommendation': '予測値の後処理による補正：\n'
                        '  - 過去の予測誤差パターンを学習して補正\n'
                        '  - ベイズ的な事後補正の適用\n'
                        '  - 信頼区間を考慮した保守的な予測値の採用'
    })

    print("\n【優先度別改善提案】\n")
    for priority in ['HIGH', 'MEDIUM', 'LOW']:
        recs = [r for r in recommendations if r['priority'] == priority]
        if recs:
            print(f"\n■ 優先度: {priority}")
            print("-" * 50)
            for rec in recs:
                print(f"\n◆ {rec['category']}")
                print(rec['recommendation'])

    return recommendations

def main():
    """メイン実行関数"""
    print("="*60)
    print("予測値過大傾向の原因分析")
    print("="*60)

    # データ読み込み
    df = load_test_results()

    if df is not None and not df.empty:
        # 過大予測メトリクスの計算
        df = calculate_overprediction_metrics(df)

        # Material Key別分析
        mk_stats = analyze_by_material_key(df)

        # 時系列パターン分析
        temporal_stats = analyze_temporal_patterns(df)

        # 診断プロットの作成
        create_diagnostic_plots(df, mk_stats)

        # 特徴量分析
        features = analyze_features()

        # 改善提案の生成
        recommendations = generate_recommendations(df, mk_stats)

        # 分析結果の保存
        analysis_results = {
            'overall_stats': {
                'mean_actual': float(df['actual_value'].mean()),
                'mean_predicted': float(df['predicted_value'].mean()),
                'mean_prediction_ratio': float(df['prediction_ratio'].mean()),
                'overprediction_percentage': float((df['predicted_value'] > df['actual_value']).mean() * 100)
            },
            'scale_analysis': df.groupby('actual_scale')['prediction_ratio'].mean().to_dict() if 'actual_scale' in df.columns else {},
            'top_overpredicted_materials': mk_stats.nlargest(10, 'overprediction_score')[['material_key', 'overprediction_score']].to_dict('records') if mk_stats is not None else [],
            'recommendations': [{'priority': r['priority'], 'category': r['category']} for r in recommendations]
        }

        import json
        with open('overprediction_analysis.json', 'w') as f:
            json.dump(analysis_results, f, indent=2, ensure_ascii=False)

        print("\n" + "="*60)
        print("分析完了")
        print("="*60)
        print("\n生成されたファイル:")
        print("  - overprediction_diagnosis.png: 診断用プロット")
        print("  - temporal_analysis.png: 時系列分析プロット")
        print("  - overprediction_analysis.json: 分析結果のサマリー")

    else:
        print("データが見つかりませんでした。")

if __name__ == "__main__":
    main()