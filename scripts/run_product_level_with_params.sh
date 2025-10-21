#!/bin/bash
# 商品レベル予測パイプライン実行スクリプト（パラメータ指定版）

# デフォルト値の設定
TRAIN_END_DATE="2024-12-31"
STEP_COUNT=1

# 使用方法の表示
usage() {
    echo "Usage: $0 [-t TRAIN_END_DATE] [-s STEP_COUNT]"
    echo "  -t TRAIN_END_DATE : 学習データの終了日 (YYYY-MM-DD形式、デフォルト: 2024-12-31)"
    echo "  -s STEP_COUNT     : 予測する月数 (1-12、デフォルト: 1)"
    echo ""
    echo "Examples:"
    echo "  $0                                    # デフォルト設定で実行"
    echo "  $0 -t 2024-11-30                     # 2024-11-30まで学習"
    echo "  $0 -t 2024-10-31 -s 3                # 2024-10-31まで学習、3ヶ月予測"
    exit 1
}

# コマンドライン引数の解析
while getopts "t:s:h" opt; do
    case $opt in
        t)
            TRAIN_END_DATE=$OPTARG
            ;;
        s)
            STEP_COUNT=$OPTARG
            ;;
        h)
            usage
            ;;
        \?)
            echo "Invalid option: -$OPTARG" >&2
            usage
            ;;
    esac
done

# パラメータの表示
echo "======================================================================"
echo "Product-level Prediction Pipeline (with parameters)"
echo "======================================================================"
echo "Train end date: $TRAIN_END_DATE"
echo "Step count (months): $STEP_COUNT"
echo "======================================================================"

# 作業ディレクトリに移動
cd /home/ubuntu/yamasa2

# Python仮想環境をアクティベート
source venv/bin/activate

# 各スクリプトを実行
echo ""
echo "[1/4] Data Preparation..."
echo "----------------------------------------------------------------------"
python scripts/prepare_product_level_data_local.py
if [ $? -ne 0 ]; then
    echo "Error: Data preparation failed"
    exit 1
fi

echo ""
echo "[2/4] Feature Generation..."
echo "----------------------------------------------------------------------"
python work/script/generate_features_product_level.py --train_end_date "$TRAIN_END_DATE"
if [ $? -ne 0 ]; then
    echo "Error: Feature generation failed"
    exit 1
fi

echo ""
echo "[3/4] Model Training & Prediction..."
echo "----------------------------------------------------------------------"
python scripts/train_and_save_product_level.py --train_end_date "$TRAIN_END_DATE" --step_count "$STEP_COUNT"
if [ $? -ne 0 ]; then
    echo "Error: Model training failed"
    exit 1
fi

echo ""
echo "[4/4] Save Results to CSV..."
echo "----------------------------------------------------------------------"
python scripts/save_results_csv.py
if [ $? -ne 0 ]; then
    echo "Error: Save results failed"
    exit 1
fi

echo ""
echo "======================================================================"
echo "✅ Pipeline completed successfully!"
echo "======================================================================"
echo "Results saved in /home/ubuntu/yamasa2/output/"
echo "  Training period: up to $TRAIN_END_DATE"
echo "  Test period: $STEP_COUNT month(s) from $(date -d "$TRAIN_END_DATE + 1 day" +%Y-%m-%d)"