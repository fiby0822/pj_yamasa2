import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import json
import warnings
warnings.filterwarnings('ignore')

def load_data(file_path):
    """予測結果CSVファイルを読み込む"""
    df = pd.read_csv(file_path)
    df['date'] = pd.to_datetime(df['date'])
    return df

def calculate_accuracy_metrics(df):
    """精度指標を計算"""
    accuracy_by_product = {}

    for product in df['material_key'].unique():
        product_df = df[df['material_key'] == product].copy()

        # MAPE (Mean Absolute Percentage Error) の計算
        # 実績値が0の場合を除外
        valid_rows = product_df[product_df['actual'] != 0]
        if len(valid_rows) > 0:
            mape = np.mean(np.abs((valid_rows['actual'] - valid_rows['predicted']) / valid_rows['actual']) * 100)
        else:
            mape = np.nan

        # MAE (Mean Absolute Error) の計算
        mae = np.mean(product_df['abs_error'])

        # RMSE (Root Mean Square Error) の計算
        rmse = np.sqrt(np.mean(product_df['error'] ** 2))

        # 精度 (100 - MAPE) として計算
        accuracy = 100 - mape if not np.isnan(mape) else np.nan

        accuracy_by_product[product] = {
            'mape': mape,
            'mae': mae,
            'rmse': rmse,
            'accuracy': accuracy,
            'n_samples': len(product_df)
        }

    return accuracy_by_product

def create_product_graph(df, product_code, output_dir):
    """商品コードごとの実績と予測の折れ線グラフを作成"""
    product_df = df[df['material_key'] == product_code].copy()
    product_df = product_df.sort_values('date')

    plt.figure(figsize=(12, 6))

    # 実績値（灰色）
    plt.plot(product_df['date'], product_df['actual'],
             color='gray', linewidth=2, marker='o', markersize=6,
             label='Actual', alpha=0.7)

    # 予測値（赤色）
    plt.plot(product_df['date'], product_df['predicted'],
             color='red', linewidth=2, marker='s', markersize=6,
             label='Predicted', alpha=0.7)

    plt.title(f'Product: {product_code} - Actual vs Predicted', fontsize=14)
    plt.xlabel('Date', fontsize=12)
    plt.ylabel('Demand', fontsize=12)
    plt.legend(loc='best')
    plt.grid(True, alpha=0.3)
    plt.xticks(rotation=45)
    plt.tight_layout()

    # グラフ保存
    output_path = output_dir / f'{product_code}.png'
    plt.savefig(output_path, dpi=100, bbox_inches='tight')
    plt.close()

    return output_path

def save_accuracy_summary(accuracy_metrics, output_dir):
    """精度サマリーを保存"""

    # DataFrameに変換
    summary_df = pd.DataFrame.from_dict(accuracy_metrics, orient='index')
    summary_df.index.name = 'product_code'
    summary_df = summary_df.reset_index()

    # 全体平均を計算（NaNを除外）
    overall_metrics = {
        'product_code': 'OVERALL_MEAN',
        'mape': summary_df['mape'].mean(skipna=True),
        'mae': summary_df['mae'].mean(),
        'rmse': summary_df['rmse'].mean(),
        'accuracy': summary_df['accuracy'].mean(skipna=True),
        'n_samples': summary_df['n_samples'].sum()
    }

    # 全体平均を追加
    summary_with_overall = pd.concat([
        summary_df,
        pd.DataFrame([overall_metrics])
    ], ignore_index=True)

    # CSV形式で保存
    csv_path = output_dir / 'accuracy_summary.csv'
    summary_with_overall.to_csv(csv_path, index=False)

    # JSON形式でも保存
    json_path = output_dir / 'accuracy_summary.json'
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump({
            'by_product': accuracy_metrics,
            'overall': overall_metrics
        }, f, ensure_ascii=False, indent=2, default=float)

    # テキスト形式のサマリーを作成
    txt_path = output_dir / 'accuracy_summary.txt'
    with open(txt_path, 'w', encoding='utf-8') as f:
        f.write("="*70 + "\n")
        f.write("精度サマリーレポート\n")
        f.write("="*70 + "\n\n")

        f.write("【全体サマリー】\n")
        f.write("-"*40 + "\n")
        f.write(f"商品コード数: {len(accuracy_metrics)}\n")
        f.write(f"総サンプル数: {overall_metrics['n_samples']}\n")
        f.write(f"平均MAPE: {overall_metrics['mape']:.2f}%\n")
        f.write(f"平均MAE: {overall_metrics['mae']:.2f}\n")
        f.write(f"平均RMSE: {overall_metrics['rmse']:.2f}\n")
        f.write(f"平均精度: {overall_metrics['accuracy']:.2f}%\n")
        f.write("\n")

        f.write("【商品コード別精度 (精度順)】\n")
        f.write("-"*40 + "\n")

        # 精度でソート
        sorted_products = sorted(
            [(k, v) for k, v in accuracy_metrics.items() if not np.isnan(v['accuracy'])],
            key=lambda x: x[1]['accuracy'],
            reverse=True
        )

        for product, metrics in sorted_products:
            f.write(f"\n商品コード: {product}\n")
            f.write(f"  精度: {metrics['accuracy']:.2f}%\n")
            f.write(f"  MAPE: {metrics['mape']:.2f}%\n")
            f.write(f"  MAE: {metrics['mae']:.2f}\n")
            f.write(f"  RMSE: {metrics['rmse']:.2f}\n")
            f.write(f"  サンプル数: {metrics['n_samples']}\n")

    return csv_path, json_path, txt_path

def main():
    # パス設定
    input_file = Path('/home/ubuntu/yamasa2/work/data/output/confirmed_order_demand_yamasa_predictions_latest_product_key.csv')
    graph_dir = Path('/home/ubuntu/yamasa2/work/visualization/graph')
    summary_dir = Path('/home/ubuntu/yamasa2/work/visualization/summary')

    # ディレクトリ作成
    graph_dir.mkdir(parents=True, exist_ok=True)
    summary_dir.mkdir(parents=True, exist_ok=True)

    # データ読み込み
    print("データを読み込んでいます...")
    df = load_data(input_file)
    print(f"データ読み込み完了: {len(df)}行")

    # 商品コードリスト取得
    product_codes = df['material_key'].unique()
    print(f"商品コード数: {len(product_codes)}")

    # 精度指標を計算
    print("\n精度指標を計算中...")
    accuracy_metrics = calculate_accuracy_metrics(df)

    # 精度サマリーを保存
    print("\n精度サマリーを保存中...")
    csv_path, json_path, txt_path = save_accuracy_summary(accuracy_metrics, summary_dir)
    print(f"精度サマリー保存完了:")
    print(f"  - CSV: {csv_path}")
    print(f"  - JSON: {json_path}")
    print(f"  - TXT: {txt_path}")

    # 各商品コードのグラフを作成
    print(f"\n{len(product_codes)}個の商品コードのグラフを作成中...")
    for i, product_code in enumerate(product_codes, 1):
        output_path = create_product_graph(df, product_code, graph_dir)
        if i % 10 == 0:
            print(f"  進捗: {i}/{len(product_codes)} 完了")

    print(f"\nすべてのグラフ作成完了: {graph_dir}")

    # 全体サマリーを表示
    overall = accuracy_metrics[list(accuracy_metrics.keys())[0]]
    overall_accuracy = np.mean([m['accuracy'] for m in accuracy_metrics.values() if not np.isnan(m['accuracy'])])
    overall_mape = np.mean([m['mape'] for m in accuracy_metrics.values() if not np.isnan(m['mape'])])

    print("\n" + "="*50)
    print("処理完了サマリー")
    print("="*50)
    print(f"処理商品数: {len(product_codes)}")
    print(f"全体平均精度: {overall_accuracy:.2f}%")
    print(f"全体平均MAPE: {overall_mape:.2f}%")
    print(f"グラフ出力先: {graph_dir}")
    print(f"精度サマリー出力先: {summary_dir}")

if __name__ == "__main__":
    main()