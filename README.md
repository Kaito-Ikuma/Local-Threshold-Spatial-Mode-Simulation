# Local Threshold Spatial-Mode Simulation

一次元局所二値しきい値モデルの空間モードを、理論・Gaussian map・microscopic dynamics の比較によって検証する研究用コードです。固有値スペクトル、モード減衰、パラメータスイープの CSV・図・アニメーションを生成します。

## ディレクトリ構成

```text
local_threshold/
├── src/          # シミュレーション本体と解析コード
├── scripts/      # 一括実行・再描画用スクリプト
├── results/      # 既存の計算結果、図、動画
├── tests/        # 標準ライブラリ unittest によるテスト
├── .vscode/      # 共有する VS Code 設定
├── requirements.txt
└── requirements-mpi.txt
```

`results/` の既存成果物は再現結果を確認できるよう Git 管理対象にしています。新しい試行結果を一時保存する場合は、Git 管理対象外の `results/runs/` を利用できます。

## セットアップ

Python 3.10 以上を推奨します。

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -r requirements.txt
```

MPI 版を使う場合は、Open MPI などの MPI 実装を用意したうえで追加依存をインストールします。

```bash
python3 -m pip install -r requirements-mpi.txt
```

MP4 を生成する場合は `ffmpeg` も必要です。GIF の生成には Pillow を使います。

## Spinodal Phase0

`src/spinodal_phase0.py` はシミュレーションを実行せず、Gaussian mean-field map のスピノーダルと、その準安定側にある固定点を理論的・数値的に確定します。スピノーダル条件

```text
F(m_sp) = m_sp,    F'(m_sp) = 1
```

から `z_spinodal = ±sqrt(2 ln B)` を用いるため、スピノーダルが存在するのは `B > 1` の場合だけです。既存の空間モード解析の既定値 `B=0.72` は変更していません。

```bash
python3 src/spinodal_phase0.py \
  --B 2.0 \
  --R 12 \
  --sigma-J 1.0 \
  --sigma-phi 0.06 \
  --phi-bar 0.0 \
  --a 1.0 \
  --branch stay_to_evacuate \
  --delta-list 1e-2,3e-3,1e-3,3e-4,1e-4,3e-5,1e-5 \
  --output-dir results/runs/phase0_B2_R12
```

主な出力は次のとおりです。

- `phase0_summary.json`: 入力値、解析的スピノーダル、条件式の残差、再現用メタデータ
- `phase0_delta_table.csv`: 各 `delta` の準安定固定点と `Gamma0_theory`、`tau0_theory`、`kappa_R_theory`、`xi_theory`
- `phase0_fixed_point_branch.png`: 安定・不安定固定点枝、対象枝、スピノーダル
- `phase0_theory_scales.png`: 後続Phaseの条件決定に使う理論スケール

これらのスケールは Gaussian map による **理論予測** であり、シミュレーションの測定値ではありません。Phase0は計算量が小さいためMPIを使いませんが、数値計算・描画・ファイル出力を分離し、`Phase0Task` ごとに `run_phase0_case()` を独立実行できる設計です。将来は `(B, R, delta_list, branch)` のtask列をMPI rankへ巡回分配できます。

Phase0のテスト:

```bash
python3 -m unittest discover -s tests -v
```

## Spinodal Phase1-2

Phase1は `q=0` の一様モード、Phase2は `q!=0` の有限波数モードについて、Phase0で得た準安定固定点と理論値を入力として緩和率 `Gamma(q)=-ln|lambda(q)|` を測定します。Phase2では厳密関係

```text
Gamma(q) - Gamma(0) = -ln|K_hat_R(q)|
```

を確認した後、低波数点だけで `Gamma(q)=Gamma0+Dq^2` をfitします。

ここで使う `deterministic_closure` は連続値 `u_i` を平均場写像で同期更新する決定論的計算です。二値状態を確率的に再サンプリングする既存の `dynamics="gaussian_map"` とは別実装であり、既存モードの名前・意味・出力は変更していません。

serial実行:

```bash
python3 src/spinodal_phase12_mpi.py \
  --phase0-dir results/runs/phase0_B2_R12 \
  --N 1024 \
  --modes 0,1,2,3,4,5,6 \
  --output-dir results/runs/phase12_B2_R12_serial
```

4-core MPI実行:

```bash
./scripts/run_spinodal_phase12_mpi_4cores.sh
```

MPIは空間を分割せず、独立な `(delta, mode, epsilon_fraction)` taskをrank間でround-robin分配します。`mpi4py` がない環境では同じdriverがserial fallbackで動作します。`--epsilon-fraction-scan 0.10,0.05,0.025` を指定すると、代表delta・modeに対するepsilon収束確認も実行します。

主な出力:

- `phase12_mode_results.csv`: 各 `(delta, mode)` の3種類のfit、理論値、信頼性診断
- `phase12_dispersion_fits.csv`: `Gamma0` 切片、`D_fit`、標準誤差、`kappa_R`との差
- `phase12_kernel_relation.csv`: 厳密kernel関係の誤差
- `phase12_validation_summary.json`: Phase1・2の自動sanity checkと実行情報
- `timeseries/`: 全site profileを含まないcompactな `A_q(t)` 時系列
- `phase1_*.png`, `phase2_*.png`: 一様緩和、分散、kernel関係、`D_fit` の確認図

Phase1・2は後続解析用データの確立までを担当し、Phase3・4の臨界指数fitやdata collapseは含みません。

## 実行例

軽量な動作確認:

```bash
python3 src/spatial_mode_ensemble_validation.py \
  --N 48 --R 3 --T 30 --ensemble 120 \
  --modes 0,1,2,4 --animate-mode 2 --format gif \
  --output-dir results/runs/example_gaussian
```

発表資料用のパラメータスイープ:

```bash
./scripts/run_spatial_mode_sweeps.sh
```

4プロセスの MPI 版:

```bash
./scripts/run_spatial_mode_sweeps_mpi_4cores.sh
```

既存 CSV から図を再生成:

```bash
python3 scripts/replot_R_sweep_normalized_overlay.py
python3 scripts/replot_R_sweep_selected_mode_decay_overlay.py
python3 scripts/replot_lambda_summary_vertical.py \
  --input results/presentation_materials/delta_h_minus_phi_sweep_lambda_aggregate.csv
```

各コマンドの引数は `--help` で確認できます。

## 主なコード

- `src/spatial_mode_ensemble_validation.py`: 単一条件でのアンサンブル検証
- `src/spinodal_phase0.py`: スピノーダルと後続Phase用理論スケールの計算
- `src/spinodal_phase12.py`: 決定論的closureの数値コアとPhase1・2解析
- `src/spinodal_phase12_mpi.py`: serial/MPI task sweep driver
- `src/spatial_mode_presentation_materials_sweeps_v2.py`: 複数 seed・パラメータスイープ
- `src/spatial_mode_presentation_materials_sweeps_mpi.py`: MPI 並列版
- `scripts/replot_*.py`: 保存済み CSV から図を再作成

## GitHub への初回公開

このフォルダはローカル Git リポジトリとして初期化済みです。GitHub 側で空のリポジトリを作成した後、次のように接続できます。

```bash
git add .
git commit -m "Organize simulation project"
git branch -M main
git remote add origin https://github.com/USER/REPOSITORY.git
git push -u origin main
```

`USER/REPOSITORY` は作成したリポジトリ名に置き換えてください。
