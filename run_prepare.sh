#!/bin/bash

# ヤマサ確定注文需要予測システム - データ準備スクリプト
# S3からExcelファイルを読み込み、加工・ゼロ値補完を実行

set -e  # エラーが発生したら停止

echo "=================================================="
echo " ヤマサ確定注文需要予測 - データ準備処理"
echo "=================================================="
echo ""

# 仮想環境の有効化
if [ -f "venv/bin/activate" ]; then
    echo "仮想環境を有効化しています..."
    source venv/bin/activate
else
    echo "警告: 仮想環境が見つかりません。継続しますが、依存関係のエラーが発生する可能性があります。"
fi

# 1. S3からExcelファイルを読み込み、加工・結合
echo ""
echo "STEP 1/2: Excelファイルの読み込みと加工"
echo "--------------------------------------------------"
python3 scripts/prepare/run_prepare_yamasa.py
if [ $? -ne 0 ]; then
    echo "エラー: prepare_data_yamasa.py の実行に失敗しました"
    exit 1
fi

# 2. ゼロ値レコードの補完
echo ""
echo "STEP 2/2: ゼロ値レコードの補完処理"
echo "--------------------------------------------------"
python3 scripts/prepare/run_prepare_fill_zero.py
if [ $? -ne 0 ]; then
    echo "エラー: prepare_data_fill_zero.py の実行に失敗しました"
    exit 1
fi

echo ""
echo "=================================================="
echo " データ準備処理が完了しました"
echo "=================================================="
echo ""
echo "出力ファイル:"
echo "  - s3://fiby-yamasa-prediction/output/df_confirmed_order_input_yamasa.parquet"
echo "  - s3://fiby-yamasa-prediction/output/df_confirmed_order_input_yamasa_fill_zero.parquet"
echo ""