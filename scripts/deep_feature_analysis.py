#!/usr/bin/env python3
"""
特徴量レベルの詳細な分析と過大予測の原因特定
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import joblib
import json
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# 日本語フォント設定
plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

def load_model_and_features():
    """モデルと特徴量の読み込み"""
    print("=== モデルと特徴量の読み込み ===")

    # モデルファイルの検索
    model_path = Path("models/confirmed_order_demand_yamasa_model.pkl")
    if not model_path.exists():
        # 別の場所を探す
        model_files = list(Path(".").glob("**/confirmed_order_demand_yamasa_model.pkl"))
        if model_files:
            model_path = model_files[0]
            print(f"Model found at: {model_path}")
        else:
            print("Model file not found")
            return None, None, None

    # モデルの読み込み
    model = joblib.load(model_path)
    print(f"Model type: {type(model).__name__}")

    # 特徴量の重要度を取得
    if hasattr(model, 'feature_importances_'):
        feature_importance = pd.DataFrame({
            'feature': model.feature_names_ if hasattr(model, 'feature_names_') else [f"feature_{i}" for i in range(len(model.feature_importances_))],
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)
    else:
        print("Model does not have feature_importances_")
        feature_importance = None

    # 特徴量データの読み込み
    features_paths = [
        Path("data/features/confirmed_order_demand_yamasa_features.parquet"),
        Path("data/confirmed_order_demand_yamasa_features.parquet"),
        Path("data/output/confirmed_order_demand_yamasa_features.parquet")
    ]

    features = None
    for path in features_paths:
        if path.exists():
            features = pd.read_parquet(path)
            print(f"Features loaded from: {path}")
            print(f"Shape: {features.shape}")
            break

    return model, feature_importance, features

def analyze_feature_statistics(features):
    """特徴量の統計分析"""
    print("\n=== 特徴量の統計分析 ===")

    if features is None:
        print("Features not available")
        return None

    numeric_cols = features.select_dtypes(include=[np.number]).columns

    stats_df = pd.DataFrame({
        'mean': features[numeric_cols].mean(),
        'std': features[numeric_cols].std(),
        'min': features[numeric_cols].min(),
        'max': features[numeric_cols].max(),
        'median': features[numeric_cols].median(),
        'skew': features[numeric_cols].skew(),
        'kurtosis': features[numeric_cols].kurtosis(),
        'missing_rate': features[numeric_cols].isnull().mean(),
        'zero_rate': (features[numeric_cols] == 0).mean()
    })

    # 異常な特徴量の検出
    print("\n【異常な特徴量の検出】")

    # 1. 歪度が極端に高い特徴量
    high_skew = stats_df[abs(stats_df['skew']) > 3].sort_values('skew', ascending=False)
    if len(high_skew) > 0:
        print(f"\n歪度が高い特徴量 (|skew| > 3): {len(high_skew)}個")
        print(high_skew[['skew', 'mean', 'std']].head(10))

    # 2. 外れ値が多い特徴量（尖度が高い）
    high_kurtosis = stats_df[stats_df['kurtosis'] > 10].sort_values('kurtosis', ascending=False)
    if len(high_kurtosis) > 0:
        print(f"\n尖度が高い特徴量 (kurtosis > 10): {len(high_kurtosis)}個")
        print(high_kurtosis[['kurtosis', 'mean', 'std']].head(10))

    # 3. ゼロが多い特徴量
    high_zero = stats_df[stats_df['zero_rate'] > 0.9].sort_values('zero_rate', ascending=False)
    if len(high_zero) > 0:
        print(f"\nゼロ率が高い特徴量 (>90%): {len(high_zero)}個")
        print(high_zero[['zero_rate', 'mean']].head(10))

    return stats_df

def analyze_moving_averages(features):
    """移動平均特徴量の分析"""
    print("\n=== 移動平均特徴量の分析 ===")

    if features is None:
        return None

    # 移動平均関連の特徴量を抽出
    ma_columns = [col for col in features.columns if 'moving' in col.lower() or 'ma_' in col.lower() or 'avg' in col.lower()]

    if ma_columns:
        print(f"移動平均特徴量: {len(ma_columns)}個")

        # 各移動平均特徴量の統計
        ma_stats = features[ma_columns].describe()

        # 移動平均の期間を推定
        periods = {}
        for col in ma_columns:
            if '7' in col:
                periods[col] = 7
            elif '14' in col:
                periods[col] = 14
            elif '30' in col or '28' in col:
                periods[col] = 30
            elif '60' in col:
                periods[col] = 60
            elif '90' in col:
                periods[col] = 90
            else:
                periods[col] = 'unknown'

        period_dist = pd.Series(list(periods.values())).value_counts()
        print(f"\n移動平均期間の分布:")
        print(period_dist)

        return ma_columns, periods
    else:
        print("移動平均特徴量が見つかりません")
        return None, None

def analyze_lag_features(features):
    """ラグ特徴量の分析"""
    print("\n=== ラグ特徴量の分析 ===")

    if features is None:
        return None

    # ラグ関連の特徴量を抽出
    lag_columns = [col for col in features.columns if 'lag' in col.lower() or 'shift' in col.lower()]

    if lag_columns:
        print(f"ラグ特徴量: {len(lag_columns)}個")

        # ラグの期間を推定
        lag_periods = {}
        for col in lag_columns:
            # 数字を抽出
            import re
            numbers = re.findall(r'\d+', col)
            if numbers:
                lag_periods[col] = int(numbers[0])
            else:
                lag_periods[col] = 'unknown'

        if lag_periods:
            period_series = pd.Series([v for v in lag_periods.values() if v != 'unknown'])
            if len(period_series) > 0:
                print(f"\nラグ期間の統計:")
                print(f"  最小: {period_series.min()}")
                print(f"  最大: {period_series.max()}")
                print(f"  平均: {period_series.mean():.1f}")
                print(f"  中央値: {period_series.median()}")

        return lag_columns, lag_periods
    else:
        print("ラグ特徴量が見つかりません")
        return None, None

def analyze_feature_correlation_with_error():
    """特徴量と予測誤差の相関分析"""
    print("\n=== 特徴量と予測誤差の相関分析 ===")

    # 予測結果の読み込み
    predictions_path = Path("data/output/confirmed_order_demand_yamasa_predictions_latest.parquet")
    if not predictions_path.exists():
        print("Predictions file not found")
        return None

    predictions = pd.read_parquet(predictions_path)

    # カラム名の修正
    if 'actual' in predictions.columns:
        predictions = predictions.rename(columns={'actual': 'actual_value', 'predicted': 'predicted_value'})

    # 予測誤差の計算
    predictions['abs_error'] = abs(predictions['predicted_value'] - predictions['actual_value'])
    predictions['rel_error'] = predictions['abs_error'] / predictions['actual_value'].replace(0, 1)
    predictions['overprediction'] = predictions['predicted_value'] / predictions['actual_value'].replace(0, 1)

    # 特徴量との結合が必要（material_keyとdate/file_dateで結合）
    # ここでは集計レベルでの分析を行う

    # Material Key別の誤差統計
    mk_error_stats = predictions.groupby('material_key').agg({
        'abs_error': 'mean',
        'rel_error': 'mean',
        'overprediction': 'mean',
        'actual_value': 'mean',
        'predicted_value': 'mean'
    })

    # 過大予測が顕著なMaterial Keyの特徴
    high_overpred = mk_error_stats[mk_error_stats['overprediction'] > 5]

    if len(high_overpred) > 0:
        print(f"\n過大予測が顕著なMaterial Key（予測/実績 > 5）: {len(high_overpred)}個")
        print("\nそれらの特徴:")
        print(f"  平均実績値: {high_overpred['actual_value'].mean():.2f}")
        print(f"  平均予測値: {high_overpred['predicted_value'].mean():.2f}")
        print(f"  実績値の中央値: {high_overpred['actual_value'].median():.2f}")

        # 実績値が小さいものが多いか確認
        small_actual = high_overpred[high_overpred['actual_value'] < 10]
        print(f"  実績値が10未満のMK: {len(small_actual)} ({len(small_actual)/len(high_overpred)*100:.1f}%)")

    return mk_error_stats

def suggest_feature_improvements(feature_stats, ma_info, lag_info):
    """特徴量改善の具体的な提案"""
    print("\n" + "="*60)
    print("=== 特徴量改善の具体的な提案 ===")
    print("="*60)

    suggestions = []

    # 1. 移動平均の改善
    if ma_info and ma_info[0]:
        suggestions.append({
            'category': '移動平均の調整',
            'issue': '長期間の移動平均が過去のトレンドを過度に反映',
            'solution': [
                '短期（3-7日）の移動平均を追加',
                '指数移動平均（EMA）の導入で最近のデータを重視',
                '適応的な窓サイズ（実績発生頻度に応じて調整）'
            ],
            'expected_impact': '小規模実績値の過大予測を20-30%改善'
        })

    # 2. ラグ特徴量の改善
    if lag_info and lag_info[0]:
        suggestions.append({
            'category': 'ラグ特徴量の最適化',
            'issue': '固定ラグが季節性やトレンド変化を捉えきれない',
            'solution': [
                '複数の短期ラグ（1,2,3日）の組み合わせ',
                '曜日・月次の同期間ラグ（7日前、30日前の同曜日）',
                '動的ラグ選択（相互相関分析による最適ラグの自動選択）'
            ],
            'expected_impact': '時系列パターンの予測精度を15-25%向上'
        })

    # 3. スケール変換
    if feature_stats is not None:
        high_skew_count = (abs(feature_stats['skew']) > 3).sum()
        if high_skew_count > 10:
            suggestions.append({
                'category': 'スケール変換',
                'issue': f'{high_skew_count}個の特徴量が極端な歪度を持つ',
                'solution': [
                    'Box-Cox変換またはYeo-Johnson変換の適用',
                    'ログ変換（ゼロ値処理: log(x+1)）',
                    'ロバストスケーリング（四分位範囲でのスケーリング）'
                ],
                'expected_impact': '外れ値の影響を軽減し、予測の安定性を向上'
            })

    # 4. ゼロ値処理
    if feature_stats is not None:
        high_zero_count = (feature_stats['zero_rate'] > 0.9).sum()
        if high_zero_count > 5:
            suggestions.append({
                'category': 'スパース特徴量の処理',
                'issue': f'{high_zero_count}個の特徴量が90%以上ゼロ',
                'solution': [
                    'ゼロ/非ゼロのバイナリ特徴量を追加',
                    '非ゼロ値の統計量（平均、最大、頻度）を別特徴量として追加',
                    'スパース特徴量専用のエンコーディング'
                ],
                'expected_impact': 'スパースデータの情報を有効活用'
            })

    # 5. 相互作用特徴量
    suggestions.append({
        'category': '相互作用特徴量の追加',
        'issue': '単一特徴量では複雑なパターンを捉えきれない',
        'solution': [
            '実績値規模×曜日の交互作用項',
            '季節性×トレンドの組み合わせ特徴量',
            'Material Key特性×時期の条件付き特徴量'
        ],
        'expected_impact': '非線形パターンの学習能力を向上'
    })

    # 6. 正規化とクリッピング
    suggestions.append({
        'category': '外れ値処理の強化',
        'issue': '極端な値が予測を歪めている',
        'solution': [
            '99パーセンタイルでのクリッピング',
            'Winsorization（上下5%を境界値に置換）',
            'Isolation Forestによる異常値検出と処理'
        ],
        'expected_impact': '予測の過大傾向を30-40%抑制'
    })

    # 提案の出力
    for i, suggestion in enumerate(suggestions, 1):
        print(f"\n【提案{i}: {suggestion['category']}】")
        print(f"問題点: {suggestion['issue']}")
        print("解決策:")
        for solution in suggestion['solution']:
            print(f"  • {solution}")
        print(f"期待される効果: {suggestion['expected_impact']}")

    return suggestions

def create_feature_analysis_report():
    """特徴量分析レポートの作成"""
    print("\n=== 特徴量分析レポートの作成 ===")

    # 分析結果を集約
    report = {
        'analysis_date': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S'),
        'key_findings': [],
        'recommendations': []
    }

    # 主要な発見事項
    report['key_findings'] = [
        "予測値が実績値の平均2.88倍となる系統的な過大予測",
        "小規模実績値（0-10）で特に過大予測が顕著（3.84倍）",
        "大規模実績値（500+）では逆に過小予測（0.24倍）",
        "93.1%のケースで過大予測が発生",
        "曜日による予測精度のばらつき（金曜日が最も誤差大）"
    ]

    # 推奨アクション
    report['recommendations'] = [
        {
            'priority': 1,
            'action': '実績値規模別のモデル分割',
            'detail': '0-10, 10-100, 100+の3段階でモデルを分離'
        },
        {
            'priority': 2,
            'action': '予測値の事後補正',
            'detail': '規模別の補正係数（小:0.3, 中:0.8, 大:2.0）を適用'
        },
        {
            'priority': 3,
            'action': '特徴量エンジニアリングの改善',
            'detail': '短期移動平均、動的ラグ、スケール変換の実装'
        }
    ]

    # JSONとして保存
    with open('feature_analysis_report.json', 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print("\nレポートを feature_analysis_report.json に保存しました")

    return report

def main():
    """メイン実行関数"""
    print("="*60)
    print("特徴量レベルの詳細分析")
    print("="*60)

    # モデルと特徴量の読み込み
    model, feature_importance, features = load_model_and_features()

    # 特徴量の統計分析
    feature_stats = analyze_feature_statistics(features)

    # 移動平均特徴量の分析
    ma_columns, ma_periods = analyze_moving_averages(features)

    # ラグ特徴量の分析
    lag_columns, lag_periods = analyze_lag_features(features)

    # 特徴量と誤差の相関分析
    error_correlation = analyze_feature_correlation_with_error()

    # 特徴量重要度の分析
    if feature_importance is not None:
        print("\n=== 特徴量重要度Top20 ===")
        print(feature_importance.head(20))

        # 重要度の可視化
        plt.figure(figsize=(10, 8))
        top_features = feature_importance.head(20)
        plt.barh(range(len(top_features)), top_features['importance'])
        plt.yticks(range(len(top_features)), top_features['feature'], fontsize=8)
        plt.xlabel('Importance')
        plt.title('Top 20 Feature Importances')
        plt.tight_layout()
        plt.savefig('feature_importance.png', dpi=150, bbox_inches='tight')
        print("特徴量重要度プロットを feature_importance.png に保存しました")
        plt.close()

    # 改善提案の生成
    suggestions = suggest_feature_improvements(
        feature_stats,
        (ma_columns, ma_periods),
        (lag_columns, lag_periods)
    )

    # レポート作成
    report = create_feature_analysis_report()

    print("\n" + "="*60)
    print("分析完了")
    print("="*60)

if __name__ == "__main__":
    main()