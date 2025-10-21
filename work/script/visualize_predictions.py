#!/usr/bin/env python3
"""
商品レベル予測結果の可視化スクリプト（パラメータ対応版）
- inputファイルパスをパラメータで指定
- outputパスをパラメータで指定
- 1~12ヶ月のデータに対応
- 商品コードごとに最初にデータがあったタイミングから最後まで表示
"""
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import json
import warnings
import argparse
from datetime import datetime
warnings.filterwarnings('ignore')

# 日本語フォント設定 - 利用可能なフォントのみを設定
import matplotlib
matplotlib.use('Agg')  # バックエンドを設定
# 警告を出さないようにするため、存在するフォントのみを使用
plt.rcParams['font.family'] = 'DejaVu Sans'
# 日本語が必要な場合は以下を有効化（警告を承知の上で）
# plt.rcParams['font.sans-serif'] = ['DejaVu Sans']

class PredictionVisualizer:
    """予測結果の可視化クラス"""

    def __init__(self, input_file, output_dir, historical_data_file=None):
        """
        初期化

        Args:
            input_file: 予測結果CSVファイルパス
            output_dir: 出力ディレクトリパス
            historical_data_file: 過去データのparquetファイルパス（商品ごとのデータ期間確認用）
        """
        self.input_file = Path(input_file)
        self.output_dir = Path(output_dir)
        self.historical_data_file = Path(historical_data_file) if historical_data_file else None

        # 出力用サブディレクトリ
        self.graph_dir = self.output_dir / 'graph'
        self.summary_dir = self.output_dir / 'summary'

        # ディレクトリ作成
        self.graph_dir.mkdir(parents=True, exist_ok=True)
        self.summary_dir.mkdir(parents=True, exist_ok=True)

    def load_prediction_data(self):
        """予測結果CSVファイルを読み込む"""
        print(f"予測結果を読み込んでいます: {self.input_file}")
        df = pd.read_csv(self.input_file)
        df['date'] = pd.to_datetime(df['date'])
        print(f"  読み込み完了: {len(df)}行, 期間: {df['date'].min()} ~ {df['date'].max()}")
        return df

    def load_historical_data(self):
        """過去データを読み込み、商品ごとのデータ開始日を取得"""
        if self.historical_data_file and self.historical_data_file.exists():
            print(f"\n過去データを読み込んでいます: {self.historical_data_file}")
            hist_df = pd.read_parquet(self.historical_data_file)

            # file_dateカラムを日付型に変換
            hist_df['file_date'] = pd.to_datetime(hist_df['file_date'])

            # 商品コード（material_key）ごとの最初のデータ日付を取得
            product_start_dates = hist_df.groupby('material_key')['file_date'].agg(['min', 'max']).to_dict()

            print(f"  過去データ期間: {hist_df['file_date'].min()} ~ {hist_df['file_date'].max()}")
            print(f"  商品コード数: {hist_df['material_key'].nunique()}")

            return hist_df, product_start_dates
        else:
            print("過去データファイルが指定されていないか、存在しません")
            return None, {}

    def calculate_accuracy_metrics(self, df):
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
                'n_samples': len(product_df),
                'actual_sum': product_df['actual'].sum(),
                'predicted_sum': product_df['predicted'].sum()
            }

        return accuracy_by_product

    def create_product_graph(self, pred_df, product_code, hist_df=None):
        """
        商品コードごとの実績と予測の折れ線グラフを作成
        過去データがある場合は、予測対象月の半年前から表示
        """
        # 予測期間のデータ
        product_pred_df = pred_df[pred_df['material_key'] == product_code].copy()
        product_pred_df = product_pred_df.sort_values('date')

        plt.figure(figsize=(14, 7))

        # 過去データがある場合は追加
        if hist_df is not None and product_code in hist_df['material_key'].unique():
            product_hist_df = hist_df[hist_df['material_key'] == product_code].copy()
            product_hist_df = product_hist_df.sort_values('file_date')

            # 予測対象月の最初の月から半年前の日付を計算
            pred_start_date = product_pred_df['date'].min()
            six_months_before = pred_start_date - pd.DateOffset(months=6)

            # 予測期間の最終日を取得（実際の予測がある最終日）
            pred_end_date = product_pred_df['date'].max()

            # 半年前から予測開始前日までのデータのみをフィルタリング（予測期間は含まない）
            product_hist_df = product_hist_df[
                (product_hist_df['file_date'] >= six_months_before) &
                (product_hist_df['file_date'] < pred_start_date)
            ]

            # 過去データの実績値（薄い灰色）
            if len(product_hist_df) > 0:
                plt.plot(product_hist_df['file_date'], product_hist_df['actual_value'],
                        color='lightgray', linewidth=1.5, marker='',
                        label='Historical Actual (6 months)', alpha=0.5)

        # 予測期間の実績値（濃い灰色）- 予測が存在する日付のみ表示
        # 予測値がNaNでない行のみをフィルタリング
        valid_pred_df = product_pred_df[product_pred_df['predicted'].notna()].copy()

        if len(valid_pred_df) > 0:
            # 実績値（濃い灰色）- 予測が存在する日付のみ
            plt.plot(valid_pred_df['date'], valid_pred_df['actual'],
                    color='gray', linewidth=2, marker='o', markersize=6,
                    label='Test Period Actual', alpha=0.7)

            # 予測値（赤色）- 予測が存在する日付のみ
            plt.plot(valid_pred_df['date'], valid_pred_df['predicted'],
                    color='red', linewidth=2, marker='s', markersize=6,
                    label='Predicted', alpha=0.7)

        # タイトルと軸ラベル
        title = f'Product: {product_code} - Actual vs Predicted'
        if len(valid_pred_df) > 0:
            accuracy = 100 - np.mean(valid_pred_df['percentage_error'])
            title += f' (Accuracy: {accuracy:.1f}%)'

        plt.title(title, fontsize=14)
        plt.xlabel('Date', fontsize=12)
        plt.ylabel('Demand', fontsize=12)
        plt.legend(loc='best')
        plt.grid(True, alpha=0.3)
        plt.xticks(rotation=45)

        # y軸の範囲を調整（負の値を避ける）
        y_min = 0
        y_max_values = []
        if hist_df is not None and product_code in hist_df['material_key'].unique():
            filtered_hist = hist_df[hist_df['material_key'] == product_code]
            if len(filtered_hist) > 0:
                y_max_values.append(filtered_hist['actual_value'].max())
        if len(valid_pred_df) > 0:
            y_max_values.extend([valid_pred_df['actual'].max(), valid_pred_df['predicted'].max()])
        if y_max_values:
            y_max = max(y_max_values) * 1.1
            plt.ylim(y_min, y_max)

        plt.tight_layout()

        # グラフ保存
        output_path = self.graph_dir / f'{product_code}.png'
        plt.savefig(output_path, dpi=100, bbox_inches='tight')
        plt.close()

        return output_path

    def save_accuracy_summary(self, accuracy_metrics):
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
            'n_samples': summary_df['n_samples'].sum(),
            'actual_sum': summary_df['actual_sum'].sum(),
            'predicted_sum': summary_df['predicted_sum'].sum()
        }

        # 全体平均を追加
        summary_with_overall = pd.concat([
            summary_df,
            pd.DataFrame([overall_metrics])
        ], ignore_index=True)

        # CSV形式で保存
        csv_path = self.summary_dir / 'accuracy_summary.csv'
        summary_with_overall.to_csv(csv_path, index=False)

        # JSON形式でも保存
        json_path = self.summary_dir / 'accuracy_summary.json'
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump({
                'by_product': accuracy_metrics,
                'overall': overall_metrics,
                'timestamp': datetime.now().isoformat()
            }, f, ensure_ascii=False, indent=2, default=float)

        # テキスト形式のサマリーを作成
        txt_path = self.summary_dir / 'accuracy_summary.txt'
        with open(txt_path, 'w', encoding='utf-8') as f:
            f.write("="*70 + "\n")
            f.write("予測精度サマリーレポート\n")
            f.write(f"作成日時: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("="*70 + "\n\n")

            f.write("【全体サマリー】\n")
            f.write("-"*40 + "\n")
            f.write(f"商品コード数: {len(accuracy_metrics)}\n")
            f.write(f"総サンプル数: {overall_metrics['n_samples']}\n")
            f.write(f"実績値合計: {overall_metrics['actual_sum']:,.0f}\n")
            f.write(f"予測値合計: {overall_metrics['predicted_sum']:,.0f}\n")
            f.write(f"平均MAPE: {overall_metrics['mape']:.2f}%\n")
            f.write(f"平均MAE: {overall_metrics['mae']:.2f}\n")
            f.write(f"平均RMSE: {overall_metrics['rmse']:.2f}\n")
            f.write(f"平均精度: {overall_metrics['accuracy']:.2f}%\n")
            f.write("\n")

            # 精度別の分布
            f.write("【精度分布】\n")
            f.write("-"*40 + "\n")
            valid_accuracies = [v['accuracy'] for v in accuracy_metrics.values() if not np.isnan(v['accuracy'])]
            if valid_accuracies:
                f.write(f"95%以上: {sum(1 for a in valid_accuracies if a >= 95)}/{len(valid_accuracies)} 商品\n")
                f.write(f"90-95%: {sum(1 for a in valid_accuracies if 90 <= a < 95)}/{len(valid_accuracies)} 商品\n")
                f.write(f"80-90%: {sum(1 for a in valid_accuracies if 80 <= a < 90)}/{len(valid_accuracies)} 商品\n")
                f.write(f"70-80%: {sum(1 for a in valid_accuracies if 70 <= a < 80)}/{len(valid_accuracies)} 商品\n")
                f.write(f"70%未満: {sum(1 for a in valid_accuracies if a < 70)}/{len(valid_accuracies)} 商品\n")
            f.write("\n")

            f.write("【商品コード別精度 TOP 10】\n")
            f.write("-"*40 + "\n")

            # 精度でソート（上位10件）
            sorted_products = sorted(
                [(k, v) for k, v in accuracy_metrics.items() if not np.isnan(v['accuracy'])],
                key=lambda x: x[1]['accuracy'],
                reverse=True
            )[:10]

            for product, metrics in sorted_products:
                f.write(f"\n商品コード: {product}\n")
                f.write(f"  精度: {metrics['accuracy']:.2f}%\n")
                f.write(f"  MAPE: {metrics['mape']:.2f}%\n")
                f.write(f"  MAE: {metrics['mae']:.2f}\n")
                f.write(f"  RMSE: {metrics['rmse']:.2f}\n")
                f.write(f"  サンプル数: {metrics['n_samples']}\n")

            f.write("\n")
            f.write("【商品コード別精度 BOTTOM 10】\n")
            f.write("-"*40 + "\n")

            # 精度でソート（下位10件）
            bottom_products = sorted(
                [(k, v) for k, v in accuracy_metrics.items() if not np.isnan(v['accuracy'])],
                key=lambda x: x[1]['accuracy']
            )[:10]

            for product, metrics in bottom_products:
                f.write(f"\n商品コード: {product}\n")
                f.write(f"  精度: {metrics['accuracy']:.2f}%\n")
                f.write(f"  MAPE: {metrics['mape']:.2f}%\n")
                f.write(f"  MAE: {metrics['mae']:.2f}\n")
                f.write(f"  RMSE: {metrics['rmse']:.2f}\n")
                f.write(f"  サンプル数: {metrics['n_samples']}\n")

        return csv_path, json_path, txt_path

    def run(self):
        """メイン処理を実行"""
        print("="*60)
        print("商品レベル予測結果の可視化")
        print(f"実行日時: {datetime.now()}")
        print("="*60)

        # 予測データ読み込み
        pred_df = self.load_prediction_data()

        # 過去データ読み込み（指定されている場合）
        hist_df = None
        if self.historical_data_file:
            hist_df, product_dates = self.load_historical_data()

        # 商品コードリスト取得
        product_codes = pred_df['material_key'].unique()
        print(f"\n商品コード数: {len(product_codes)}")

        # 精度指標を計算
        print("\n精度指標を計算中...")
        accuracy_metrics = self.calculate_accuracy_metrics(pred_df)

        # 精度サマリーを保存
        print("\n精度サマリーを保存中...")
        csv_path, json_path, txt_path = self.save_accuracy_summary(accuracy_metrics)
        print(f"精度サマリー保存完了:")
        print(f"  - CSV: {csv_path}")
        print(f"  - JSON: {json_path}")
        print(f"  - TXT: {txt_path}")

        # 各商品コードのグラフを作成
        print(f"\n{len(product_codes)}個の商品コードのグラフを作成中...")
        for i, product_code in enumerate(product_codes, 1):
            output_path = self.create_product_graph(pred_df, product_code, hist_df)
            if i % 10 == 0:
                print(f"  進捗: {i}/{len(product_codes)} 完了")

        print(f"\nすべてのグラフ作成完了: {self.graph_dir}")

        # 全体サマリーを表示
        overall_accuracy = np.mean([m['accuracy'] for m in accuracy_metrics.values() if not np.isnan(m['accuracy'])])
        overall_mape = np.mean([m['mape'] for m in accuracy_metrics.values() if not np.isnan(m['mape'])])

        print("\n" + "="*50)
        print("処理完了サマリー")
        print("="*50)
        print(f"処理商品数: {len(product_codes)}")
        print(f"全体平均精度: {overall_accuracy:.2f}%")
        print(f"全体平均MAPE: {overall_mape:.2f}%")
        print(f"グラフ出力先: {self.graph_dir}")
        print(f"精度サマリー出力先: {self.summary_dir}")

        return accuracy_metrics


def main():
    """メイン処理"""
    parser = argparse.ArgumentParser(
        description='商品レベル予測結果の可視化',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # デフォルト設定で実行
  python visualize_predictions.py

  # 入力ファイルと出力ディレクトリを指定
  python visualize_predictions.py \\
    --input /path/to/predictions.csv \\
    --output /path/to/output

  # 過去データも含めてグラフを作成
  python visualize_predictions.py \\
    --input /path/to/predictions.csv \\
    --output /path/to/output \\
    --historical /home/ubuntu/yamasa2/work/data/input/df_confirmed_order_input_yamasa_fill_zero.parquet
        """
    )

    parser.add_argument(
        '--input',
        type=str,
        default='/home/ubuntu/yamasa2/output/confirmed_order_demand_yamasa_predictions_latest_product_key.csv',
        help='予測結果CSVファイルのパス'
    )

    parser.add_argument(
        '--output',
        type=str,
        default='/home/ubuntu/yamasa2/work/visualization',
        help='出力ディレクトリのパス'
    )

    parser.add_argument(
        '--historical',
        type=str,
        default='/home/ubuntu/yamasa2/work/data/input/df_confirmed_order_input_yamasa_fill_zero.parquet',
        help='過去データのparquetファイルパス（商品ごとの全期間表示用）'
    )

    args = parser.parse_args()

    # ファイルの存在確認
    input_path = Path(args.input)
    if not input_path.exists():
        print(f"Error: 入力ファイルが見つかりません: {input_path}")
        return 1

    # Visualizerインスタンス作成
    visualizer = PredictionVisualizer(
        input_file=args.input,
        output_dir=args.output,
        historical_data_file=args.historical
    )

    # 実行
    try:
        accuracy_metrics = visualizer.run()
        print("\n✅ 可視化処理が正常に完了しました")
        return 0
    except Exception as e:
        print(f"\n❌ エラーが発生しました: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())