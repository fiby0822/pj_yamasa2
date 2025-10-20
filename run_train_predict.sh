#!/bin/bash

# ヤマサ確定注文需要予測システム - 学習・予測実行スクリプト

set -e  # エラーが発生したら停止

echo "=================================================="
echo " ヤマサ確定注文需要予測 - モデル学習・予測"
echo "=================================================="
echo ""

# 引数の解析
MODE=${1:-"train"}  # デフォルトは学習モード
OPTIMIZE=${2:-"false"}  # デフォルトはハイパーパラメータ最適化なし

# 仮想環境の有効化
if [ -f "venv/bin/activate" ]; then
    echo "仮想環境を有効化しています..."
    source venv/bin/activate
else
    echo "警告: 仮想環境が見つかりません。継続しますが、依存関係のエラーが発生する可能性があります。"
fi

# モードに応じた処理
case $MODE in
    "train")
        echo ""
        echo "モデル学習を実行します"
        echo "--------------------------------------------------"

        if [ "$OPTIMIZE" = "true" ]; then
            echo "ハイパーパラメータ最適化: 有効"
            python3 scripts/train/train_model.py \
                --train-end-date "2024-12-31" \
                --step-count 1 \
                --use-optuna \
                --n-trials 50
        else
            echo "ハイパーパラメータ最適化: 無効"
            python3 scripts/train/train_model.py \
                --train-end-date "2024-12-31" \
                --step-count 1
        fi
        ;;

    "predict")
        echo ""
        echo "予測を実行します"
        echo "--------------------------------------------------"

        # 将来2週間の予測
        python3 scripts/predict/run_prediction.py \
            --mode future \
            --start-date "2025-01-01" \
            --end-date "2025-01-14" \
            --save-results
        ;;

    "walk-forward")
        echo ""
        echo "Walk-forward検証を実行します"
        echo "--------------------------------------------------"

        # Walk-forward予測の実行
        python3 scripts/predict/run_prediction.py \
            --mode walk-forward \
            --save-results
        ;;

    "full")
        echo ""
        echo "フルパイプライン（学習＋予測）を実行します"
        echo "--------------------------------------------------"

        # 1. モデル学習
        echo ""
        echo "STEP 1/2: モデル学習"
        echo "--------------------------------------------------"
        if [ "$OPTIMIZE" = "true" ]; then
            python3 scripts/train/train_model.py \
                --train-end-date "2024-12-31" \
                --step-count 1 \
                --use-optuna \
                --n-trials 50
        else
            python3 scripts/train/train_model.py \
                --train-end-date "2024-12-31" \
                --step-count 1
        fi

        # 2. 予測実行
        echo ""
        echo "STEP 2/2: 予測実行"
        echo "--------------------------------------------------"
        python3 scripts/predict/run_prediction.py \
            --mode future \
            --start-date "2025-01-01" \
            --end-date "2025-01-14" \
            --save-results
        ;;

    *)
        echo "使用方法: $0 [train|predict|walk-forward|full] [true|false]"
        echo ""
        echo "モード:"
        echo "  train         - モデルの学習のみ"
        echo "  predict       - 学習済みモデルで予測のみ"
        echo "  walk-forward  - Walk-forward検証"
        echo "  full          - 学習＋予測の両方を実行"
        echo ""
        echo "第2引数:"
        echo "  true          - ハイパーパラメータ最適化を有効化"
        echo "  false         - ハイパーパラメータ最適化を無効化（デフォルト）"
        exit 1
        ;;
esac

echo ""
echo "=================================================="
echo " 処理が完了しました"
echo "=================================================="
echo ""

# 出力ファイルの場所を表示
echo "出力ファイル:"
echo "  - モデル: s3://fiby-yamasa-prediction-2/output/models/"
echo "  - 予測結果: s3://fiby-yamasa-prediction-2/output/predictions/"
echo "  - 評価結果: s3://fiby-yamasa-prediction-2/output/evaluation/"
echo ""