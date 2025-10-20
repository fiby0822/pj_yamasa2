#!/usr/bin/env python3
"""
学習プロセスの監視スクリプト
"""
import time
import psutil
import subprocess
from pathlib import Path

def get_process_info():
    """学習プロセスの情報を取得"""
    for proc in psutil.process_iter(['pid', 'name', 'cmdline', 'cpu_percent', 'memory_info']):
        try:
            if 'python' in proc.info['name']:
                cmdline = ' '.join(proc.info['cmdline'] or [])
                if 'train_model' in cmdline:
                    cpu_percent = proc.cpu_percent(interval=1)
                    mem_info = proc.info['memory_info']
                    mem_gb = mem_info.rss / (1024 ** 3) if mem_info else 0
                    return {
                        'pid': proc.info['pid'],
                        'cpu_percent': cpu_percent,
                        'memory_gb': mem_gb,
                        'status': 'running'
                    }
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
    return None

def check_output_files():
    """出力ファイルの存在を確認"""
    output_dir = Path("data/output")
    files = {
        'predictions': output_dir / "confirmed_order_demand_yamasa_predictions_latest.parquet",
        'metrics': output_dir / "confirmed_order_demand_yamasa_material_summary_latest.parquet",
        'importance': output_dir / "feature_importance_latest.parquet"
    }

    status = {}
    for name, path in files.items():
        if path.exists():
            size_mb = path.stat().st_size / (1024 ** 2)
            status[name] = f"存在 ({size_mb:.1f} MB)"
        else:
            status[name] = "未生成"

    return status

def main():
    print("学習プロセスの監視を開始...")
    print("=" * 60)

    while True:
        # プロセス情報取得
        proc_info = get_process_info()

        if proc_info:
            print(f"\n[プロセス状態]")
            print(f"  PID: {proc_info['pid']}")
            print(f"  CPU使用率: {proc_info['cpu_percent']:.1f}%")
            print(f"  メモリ使用量: {proc_info['memory_gb']:.1f} GB")
        else:
            print(f"\n[プロセス状態] 学習プロセスが見つかりません")

            # 出力ファイル確認
            output_status = check_output_files()
            print(f"\n[出力ファイル]")
            for name, status in output_status.items():
                print(f"  {name}: {status}")

            if output_status['predictions'] != "未生成":
                print("\n学習が完了した可能性があります。")
                break

        time.sleep(10)  # 10秒ごとに更新

if __name__ == "__main__":
    main()