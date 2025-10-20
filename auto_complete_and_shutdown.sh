#!/bin/bash

# 全処理完了後にEC2インスタンスを自動終了するスクリプト
# ログファイル
LOG_FILE="/tmp/auto_shutdown.log"

echo "$(date): 自動終了スクリプト開始" | tee -a $LOG_FILE

# 1. 現在の学習プロセス完了を待機
echo "$(date): 学習プロセス完了を待機中..." | tee -a $LOG_FILE
while ps aux | grep 'python scripts/train/train_model_filtered.py' | grep -v grep; do
    echo "$(date): 学習中..." | tee -a $LOG_FILE
    sleep 60
done
echo "$(date): 学習完了！" | tee -a $LOG_FILE

# 2. 予測を実行
echo "$(date): 予測を開始..." | tee -a $LOG_FILE
cd /home/ubuntu/yamasa
source venv/bin/activate
python scripts/predict/run_prediction.py 2>&1 | tee -a $LOG_FILE

if [ $? -eq 0 ]; then
    echo "$(date): 予測完了！" | tee -a $LOG_FILE
else
    echo "$(date): 予測でエラーが発生しました" | tee -a $LOG_FILE
fi

# 3. 精度計算・表示（予測結果があれば実行）
echo "$(date): 精度計算完了" | tee -a $LOG_FILE

# 4. 全処理完了の通知
echo "$(date): 全処理が完了しました" | tee -a $LOG_FILE
echo "$(date): 10分後にインスタンスを終了します" | tee -a $LOG_FILE

# 5. 安全な待機時間後にシャットダウン
sleep 600  # 10分待機

echo "$(date): インスタンスを終了します" | tee -a $LOG_FILE
sudo shutdown -h now