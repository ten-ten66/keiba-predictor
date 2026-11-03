============================================
  競馬予想AI - ローカル学習パッケージ
============================================

■ 動作要件
  - Windows 10/11
  - Python 3.9〜3.13 (https://www.python.org/downloads/)
  - メモリ: 16GB以上 (48GB推奨)
  - GPU: NVIDIA GPU (オプション、あれば高速化)

■ セットアップ手順

  1. Python をインストール (3.9〜3.13 どれでもOK、最新版推奨)
     https://www.python.org/downloads/
     ★「Add Python to PATH」に必ずチェック！

  2. このフォルダをデスクトップに展開

  3. setup.bat をダブルクリック
     → 仮想環境作成 + パッケージインストール

  4. 学習開始:
     コマンドプロンプトを開いて:
       cd Desktop\keiba-training
       venv\Scripts\activate.bat
       python train_advanced.py

■ ファイル構成
  setup.bat            - セットアップスクリプト
  requirements.txt     - 必要パッケージ一覧
  train_advanced.py    - メイン学習スクリプト
  data/
    _all_races.csv     - 全レースデータ (130万件)
    bloodline.csv      - 血統データ
  models/              - 学習済みモデル出力先
  output/              - レポート・グラフ出力先

■ 学習後のサーバー反映
  学習完了後、models/keiba_model.pkl をサーバーにアップロード:

  方法1: SCP (WSLまたはGit Bash)
    scp models/keiba_model.pkl <user>@<your-vps-host>:/opt/keiba-predictor/models/

  方法2: WinSCPなどのSFTPクライアント
    接続先: <your-vps-host>
    パス: /opt/keiba-predictor/models/keiba_model.pkl

  アップロード後、APIサーバーを再起動:
    ssh <user>@<your-vps-host> "systemctl restart keiba-api"
    もしくは: kill + uvicorn再起動

■ 設定変更 (train_advanced.py 冒頭のCONFIG)
  use_all_data: True      全データ使用 / False=直近10年
  use_advanced_features: True   拡張特徴量
  use_gpu: True           GPU使用
  optuna_trials: 100      パラメータ探索回数 (多い=精度UP/時間UP)
  use_xgboost: True       XGBoostも使用
  use_stacking: True      スタッキング

■ 所要時間目安 (48GB RAM, NVIDIA GPU)
  全データ + Optuna 100試行: 約1-2時間
  全データ + Optuna  50試行: 約30-60分
  10年データ + Optuna 50試行: 約15-30分
