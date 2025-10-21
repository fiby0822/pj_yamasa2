#!/usr/bin/env python3
"""
商品レベル予測パイプラインの実行スクリプト（パラメータ指定可能版）
学習期間と検証期間を柔軟に設定可能
"""
import argparse
import subprocess
import sys
import os
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta

def validate_date(date_string):
    """日付文字列の検証"""
    try:
        return datetime.strptime(date_string, "%Y-%m-%d")
    except ValueError:
        raise argparse.ArgumentTypeError(f"Invalid date format: {date_string}. Use YYYY-MM-DD")

def run_pipeline(train_end_date, step_count):
    """
    パイプラインを実行

    Args:
        train_end_date: 学習データの終了日（YYYY-MM-DD形式）
        step_count: 予測月数（1-12）
    """
    print("="*60)
    print("Product-level Prediction Pipeline")
    print(f"Timestamp: {datetime.now()}")
    print("="*60)
    print(f"\n Configuration:")
    print(f"  - Train end date: {train_end_date}")
    print(f"  - Step count (months): {step_count}")

    # 検証期間の計算と表示
    train_end_dt = datetime.strptime(train_end_date, "%Y-%m-%d")
    test_start = train_end_dt + timedelta(days=1)
    test_end = test_start + relativedelta(months=step_count) - timedelta(days=1)
    print(f"  - Test period: {test_start.strftime('%Y-%m-%d')} to {test_end.strftime('%Y-%m-%d')}")
    print("="*60)

    scripts = [
        {
            'name': 'Data Preparation',
            'script': 'scripts/prepare_product_level_data_local.py',
            'args': []  # データ準備には期間パラメータ不要
        },
        {
            'name': 'Feature Generation',
            'script': 'work/script/generate_features_product_level.py',
            'args': ['--train_end_date', train_end_date]
        },
        {
            'name': 'Model Training & Prediction',
            'script': 'scripts/train_and_save_product_level.py',
            'args': ['--train_end_date', train_end_date, '--step_count', str(step_count)]
        },
        {
            'name': 'Save Results to CSV',
            'script': 'scripts/save_results_csv.py',
            'args': []
        }
    ]

    # 各スクリプトを順次実行
    for i, script_info in enumerate(scripts, 1):
        print(f"\n[{i}/{len(scripts)}] Running: {script_info['name']}")
        print("-" * 40)

        script_path = os.path.join('/home/ubuntu/yamasa2', script_info['script'])

        # コマンドを構築
        cmd = ['python', script_path] + script_info['args']

        try:
            # スクリプトを実行
            result = subprocess.run(
                cmd,
                check=True,
                capture_output=True,
                text=True,
                cwd='/home/ubuntu/yamasa2'
            )

            # 出力を表示
            if result.stdout:
                print(result.stdout)

            print(f"✅ {script_info['name']} completed successfully")

        except subprocess.CalledProcessError as e:
            print(f"❌ Error running {script_info['name']}")
            print(f"Error message: {e.stderr}")
            sys.exit(1)

    print("\n" + "="*60)
    print("✅ Pipeline completed successfully!")
    print("="*60)
    print(f"\nResults saved in /home/ubuntu/yamasa2/output/")
    print(f"  - Training period: up to {train_end_date}")
    print(f"  - Test period: {test_start.strftime('%Y-%m')} to {test_end.strftime('%Y-%m')}")

def main():
    parser = argparse.ArgumentParser(
        description='商品レベル予測パイプライン実行スクリプト',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # デフォルト設定（2024-12-31まで学習、2025年1月を予測）
  python scripts/run_product_level_pipeline.py

  # 2024-11-30まで学習、2024年12月を予測
  python scripts/run_product_level_pipeline.py --train_end_date 2024-11-30

  # 2024-10-31まで学習、2024年11月～2025年1月（3ヶ月）を予測
  python scripts/run_product_level_pipeline.py --train_end_date 2024-10-31 --step_count 3
        """
    )

    parser.add_argument(
        '--train_end_date',
        type=str,
        default='2024-12-31',
        help='学習データの終了日 (YYYY-MM-DD形式、デフォルト: 2024-12-31)'
    )

    parser.add_argument(
        '--step_count',
        type=int,
        default=1,
        choices=range(1, 13),
        metavar='N',
        help='予測する月数 (1-12、デフォルト: 1)'
    )

    args = parser.parse_args()

    # 日付の妥当性チェック
    try:
        validate_date(args.train_end_date)
    except argparse.ArgumentTypeError as e:
        print(f"Error: {e}")
        sys.exit(1)

    # パイプライン実行
    run_pipeline(args.train_end_date, args.step_count)

if __name__ == "__main__":
    main()