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

MPI 版を使う場合は、Intel MPI などの MPI 実装を用意したうえで追加依存をインストールします。SQUIDのPhase5は `BaseCPU` / Intel MPIを使用します。

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

## Spinodal Phase3-4

Phase3・4は新しいsimulationではなく、既存の決定論的Gaussian closureのPhase1・2 CSVを読むserial post-processingです。数十行の表に対する回帰なのでMPIは使用しません。入力がなければPhase1・2を暗黙に再実行せず、明示的なエラーで停止します。

Phase3ではprimary observableを mode_index=0, task_group=main, reliable=True の Gamma_from_lambda に固定し、Gamma0 は delta の1/2乗、tau0=1/Gamma0 は delta の-1/2乗というscalingを調べます。primary asymptotic windowは事前指定した delta <= 3e-4 であり、指数が理論値に近くなるよう自動選択しません。nearest 3〜7点のnested windowもすべて保存し、回帰standard errorはpower lawからの残差診断として扱います。tau0はGamma0から作るderived quantityであり、独立な指数測定ではありません。

mean-field spinodal近傍の緩和時間に関する背景は Mori, Miyashita, and Rikvold, *Phys. Rev. E* **81**, 011135 (2010), [DOI:10.1103/PhysRevE.81.011135](https://doi.org/10.1103/PhysRevE.81.011135) を参照してください。係数 C_Gamma=sqrt(2|z_spinodal|/sigma_eff) はこのGaussian map固有です。

Phase4ではPhase2の分散から xi_dyn=sqrt(D/Gamma0) を定義し、xi_dyn が delta の-1/4乗、tau0がxi_dynの2乗となること、および tau(q)/tau0=1/[1+(q xi_dyn)^2] のfinite-q collapseを検証します。これはPhase1・2の同じdynamic dataから導く内部整合性確認であり、xiやz=2の独立測定ではありません。構造はModel-A-likeな長波長relaxationですが、離散時間・非平衡mapを厳密なHohenberg–Halperin Model Aとは同一視しません。

一般的背景は Hohenberg and Halperin, *Rev. Mod. Phys.* **49**, 435 (1977), [DOI:10.1103/RevModPhys.49.435](https://doi.org/10.1103/RevModPhys.49.435) を参照してください。実空間での独立検証は将来のPhase6 fixed-boundary responseによる xi_boundary との比較です。

有限波数窓による約0.4%の D_fit-kappa_R 差については、同じmodeでのexact-kernel slopeと、-ln Khat_R(q)=kappa_R q^2+c4_R q^4+O(q^6) を使う q^2+q^4 fitを併記し、finite-q systematicとして評価します。

実行:

    python3 src/spinodal_phase34.py \
      --phase0-dir results/runs/phase0_B2_R12 \
      --phase12-dir results/runs/phase12_B2_R12 \
      --primary-delta-max 3e-4 \
      --qR-max-collapse 0.35 \
      --output-dir results/runs/phase34_B2_R12

または ./scripts/run_spinodal_phase34.sh を使用できます。主な出力は phase3_scaling_table.csv, phase3_powerlaw_fits.csv, phase3_window_stability.csv, phase3_effective_exponents.csv, phase4_length_table.csv, phase4_scaling_fits.csv, phase4_dispersion_systematics.csv, phase4_collapse.csv, phase34_validation_summary.json と10枚の診断図です。

## Spinodal Phase5

Phase5は元の二値microscopic dynamicsによるPhase0〜4 Gaussian closureのrobustness / breakdown testです。同じ指数を強制せず、Gaussian-centered deltaを固定したまま、spinodal位置のshift、finite-R rounding、metastable escape、Gamma(q)とDのずれを測ります。有限range系にはtrue spinodalという表現を自動的に使わず、必要ならeffective microscopic transitionまたはpseudospinodal-like crossoverと記述します。

reference kernel direct_J は既存実装と同じく、2R本のannealed Gaussian J場をstepごとに生成し、plus/minusへ同じJ、初期一様乱数、quenched thresholdを共有します。削除せずcorrectness referenceとして保持しています。

production既定の aggregated_exact は、現在のspin pairを固定した条件付き分布を厳密に積分します。c=2R、局所平均をm_plus, m_minus、局所overlapをrhoとすると、noise varianceは sigma_J^2/c、plus/minus covarianceは sigma_J^2 rho/cです。二つの標準Gaussian Z1,Z2から、plus noiseを sigma_J Z1/sqrt(c)、minus noiseを sigma_J[rho Z1+sqrt(1-rho^2)Z2]/sqrt(c) と生成します。これはcentral-limit approximationではありません。周期近傍和はcumulative sumによりblock×Nに対してO(block N)で計算します。

乱数はNumPy Philoxを用い、base seed、delta index、mode index、epsilon index、block IDだけからstreamを決めます。MPI rank数はseedへ入りません。この設計の背景は Salmon, Moraes, Dror, and Shaw, “Parallel Random Numbers: As Easy as 1, 2, 3,” SC11 (2011), [Random123](https://random123.com/) を参照してください。

MPIはsite空間を分割しません。work unitは (delta, mode, epsilon, block ID) で、各rankはblockを最後まで独立実行します。rank0はestimated cost降順のLPT greedy assignmentを作り、time step中のallreduce、gather、halo exchangeは行いません。1 block完了ごとにblocksディレクトリへatomic checkpointを書き、同じcommandの --resume で有効なblockをskipします。15時間jobでは --max-runtime-seconds 53400 により新規block開始を早めに止め、再投入できます。

### SQUID実行前の初回セットアップ

Phase5 SQUID runs do not depend on the old `evac_sim` Conda environment. 過去の実地形simulationと分離し、次の専用venvのPythonをactivateせず絶対パスで実行します。

```bash
module purge
module load BasePy/2026
module load BaseCPU/2026

export PHASE5_VENV=/sqfs/work/cm9029/$USER/phase5_venv
export PHASE5_PY="$PHASE5_VENV/bin/python"
python3 -m venv "$PHASE5_VENV"
"$PHASE5_PY" --version
```

`BasePy/2026` はvenv Pythonが必要とするPython 3.13 runtime、`BaseCPU/2026` はIntel compiler / Intel MPIを供給します。必要なpackageは計算job内ではなく、外部networkに接続できるSQUIDフロントエンドで事前にinstallします。

```bash
cd /path/to/Local-Threshold-Spatial-Mode-Simulation
"$PHASE5_PY" -m pip install --upgrade pip setuptools wheel
"$PHASE5_PY" -m pip install -r requirements.txt
```

Intel MPIに対して `mpi4py` をsource buildする場合は次を実行します。すでに後述のserial / 2-rank checkが通る場合は再install不要です。

```bash
which mpicc
which mpirun
mpirun --version

MPI4PY_BUILD_MPICC="$(which mpicc)" \
"$PHASE5_PY" -m pip install \
  --no-cache-dir \
  --no-binary=mpi4py \
  mpi4py
```

### 各ログイン後の環境確認

以下のコマンドはリポジトリのrootで実行します。共通helperがmoduleとvenvの既定値を設定し、Python共有library、package、Intel MPI、Python絶対パスを検査します。

```bash
cd /path/to/Local-Threshold-Spatial-Mode-Simulation
source scripts/phase5_squid_env.sh
source scripts/phase5_squid_preflight.sh
```

手動のserial import checkは次のとおりです。`Python:` は `/sqfs/work/cm9029/<user>/phase5_venv/bin/python` を指し、MPI libraryはIntel MPIでなければなりません。

```bash
"$PHASE5_PY" -c '
import sys
import numpy
import scipy
import pandas
import mpi4py
from mpi4py import MPI

print("Python:", sys.executable)
print("NumPy:", numpy.__version__)
print("SciPy:", scipy.__version__)
print("pandas:", pandas.__version__)
print("mpi4py:", mpi4py.__version__)
print("MPI:")
print(MPI.Get_library_version())
'

"$PHASE5_PY" scripts/check_squid_mpi_env.py \
  --expected-flavor intelmpi \
  --expected-python "$PHASE5_PY"
```

本計算前の2-rank Python-path確認は必須です。次のhelperをフロントエンドで実行するか、2 coreのPBS smoke jobを投入します。

```bash
MPI_TEST_RANKS=2 scripts/check_phase5_mpi_python.sh

# または計算ノードで確認
qsub scripts/run_phase5_squid_env_smoke.sh
qstat
```

内部で実行するコマンドは次と同義です。

```bash
mpirun ${NQSV_MPIOPTS:-} -np 2 "$PHASE5_PY" -c '
from mpi4py import MPI
import socket
import sys

print(
    "rank=", MPI.COMM_WORLD.Get_rank(),
    "host=", socket.gethostname(),
    "python=", sys.executable,
)
'
```

両rankの `python=` がどちらも `/sqfs/work/cm9029/<user>/phase5_venv/bin/python` なら成功です。rankごとに異なるPython、system Python、Conda Pythonが表示された場合はbenchmarkへ進んではいけません。

SQUID側に `results/runs/phase0_B2_R12` と `results/runs/phase12_B2_R12` も必要です。`results/runs/` はGit管理対象外なので、ローカルで作成した場合はSQUIDへ別途転送してください。

```bash
test -f results/runs/phase0_B2_R12/phase0_summary.json
test -f results/runs/phase0_B2_R12/phase0_delta_table.csv
test -f results/runs/phase12_B2_R12/phase12_mode_results.csv
test -f results/runs/phase12_B2_R12/phase12_dispersion_fits.csv
```

### benchmark → pilot → production

#### Step 1: local unit tests

```bash
python3 -m unittest discover -s tests -v
```

#### Step 2: kernel・block size・MPI scaling benchmark

最初に1 coreで `direct_J` と `aggregated_exact`、block size 16/32/64/128を比較します。

```bash
qsub scripts/run_phase5_squid_benchmark.sh
qstat
```

job終了後に結果を確認します。

```bash
"$PHASE5_PY" -m json.tool \
  results/runs/phase5_B2_R12/benchmarks/phase5_kernel_benchmark.json
cat results/runs/phase5_B2_R12/benchmarks/phase5_block_size_benchmark.csv
```

次に同一条件を19/38/57/76 ranksで実行します。kernel benchmarkの完了後に投入してください。

```bash
qsub scripts/run_phase5_squid_scaling_benchmark.sh
qstat
```

```bash
cat results/runs/phase5_scaling_benchmark/phase5_mpi_scaling_benchmark.csv
```

`aggregated_exact` のthroughputが最も高く、メモリ使用量に余裕があるblock sizeを選びます。production rank数は、単に76を固定せず `trial_site_steps_per_sec` と `parallel_efficiency` の実測値から選んでください。

#### Step 3: epsilon・M convergence pilot

Step 2で選んだblock sizeとrank数を [scripts/run_phase5_squid_pilot.sh](scripts/run_phase5_squid_pilot.sh) の `--block-size`と `-np` へ反映したうえでpilotを投入します。

```bash
qsub scripts/run_phase5_squid_pilot.sh
qstat
```

15時間内に全blockが完了しなかった場合もcheckpointは残ります。同じscriptを再度 `qsub` すると `--resume` により未完了blockだけを続行します。完了後は次を確認します。

```bash
"$PHASE5_PY" -m json.tool \
  results/runs/phase5_B2_R12_pilot/phase5_run_state.json
"$PHASE5_PY" -m json.tool \
  results/runs/phase5_B2_R12_pilot/phase5_validation_summary.json
cat results/runs/phase5_B2_R12_pilot/phase5_epsilon_convergence.csv
cat results/runs/phase5_B2_R12_pilot/phase5_M_convergence.csv
```

`all_complete=true`であること、epsilonを0.025/0.05/0.10と変えてもGammaが統計誤差内で安定すること、Mを増やしたときにGammaと誤差が収束することを確認します。加えて `escape_fraction`、`baseline_drift`、`preparation_drift`、fit-window dependenceを確認します。substantial escapeや `reliable=false` が残る場合はproductionへ進まず、preparation protocol、Delta範囲、M、Tを再検討してください。

##### preparation survival / microscopic pseudospinodal診断

pilotで全条件が `escape_t0=1` になった場合は、Gammaやepsilon/M convergenceを再計算する前に、次の1-rank `direct_J` controlでspinodalから離れた領域を調べます。標準Phase0/12出力は上書きせず、同じGaussian-map式から診断用の `Delta` と `m_star` をMatplotlib非依存で計算します。

```bash
qsub scripts/run_phase5_squid_pseudospinodal_direct_j.sh
qstat
```

既定条件は `delta=0.01,0.02,0.03,0.05,0.10`、`N=1024`、`M=64`、`block-size=32`、`mode=0`、`epsilon-fraction=0.05`、`T=2`、1 MPI rankです。計算後は次を確認します。

```bash
cat results/runs/phase5_pseudospinodal_direct_J/phase5_pseudospinodal_scan.csv

"$PHASE5_PY" -m json.tool \
  results/runs/phase5_pseudospinodal_direct_J/phase5_pseudospinodal_detail.json
```

`preparation_survives=true` は準備直後の `escape_t0<=0.1` を意味します。初めてfalseからtrueへ変わるdelta区間をmicroscopic pseudospinodalの追加診断範囲とし、この結果だけからtrue spinodalとは呼びません。

##### shifted-delta epsilon/M convergence pilot

50-step survival診断で `delta=0.060` と `0.065` の間に10% escape境界が見つかった後は、次のscriptをそのまま投入します。このscriptは標準Phase0/12 CSVにないdeltaのGaussian referenceをmemory上で生成するため、追加の前処理やMatplotlibは必要ありません。

```bash
cd /path/to/Local-Threshold-Spatial-Mode-Simulation
source scripts/phase5_squid_env.sh
qsub scripts/run_phase5_squid_shifted_pilot.sh
qstat
```

現在の固定条件は生存側候補の `delta=0.065,0.070,0.080`、mode `0,1,2,3,4,5,6`、epsilon fraction `0.025,0.05,0.10`、`M=8192`、`block-size=64`、`T=50`、fit window `0:3`、`qR<=0.45`、`aggregated_exact`、57 MPI ranksです。`T=50` の時系列はescape監視に残し、Gammaの初期緩和fitにはmethod B/Cが一致する短い窓を使います。

```bash
"$PHASE5_PY" -m json.tool \
  results/runs/phase5_B2_R12_shifted_dispersion_M8192/phase5_run_state.json
"$PHASE5_PY" -m json.tool \
  results/runs/phase5_B2_R12_shifted_dispersion_M8192/phase5_validation_summary.json
cat results/runs/phase5_B2_R12_shifted_dispersion_M8192/phase5_epsilon_convergence.csv
cat results/runs/phase5_B2_R12_shifted_dispersion_M8192/phase5_M_convergence.csv
cat results/runs/phase5_B2_R12_shifted_dispersion_M8192/phase5_mode_results.csv
cat results/runs/phase5_B2_R12_shifted_dispersion_M8192/phase5_dispersion_fits.csv
```

`delta>=0.065` のprimary epsilonで `reliable=true`、全時刻の `escape_fraction<=0.1`、epsilon/Mに対するGammaの安定を確認してから次へ進みます。`--analytic-references` は同じPhase0式とtop-hat kernelからreferenceを作り、標準Phase0/12出力は変更しません。

#### Phase5 follow-up diagnostics

M=8192 shifted runの次は、Gaussian closureへ結果を合わせるのではなく、microscopic relaxationの時間依存、準備手順依存、escape混入を順に分解します。一般的な時間依存緩和の背景として Glauber, “Time-Dependent Statistics of the Ising Model,” *J. Math. Phys.* **4**, 294 (1963), [DOI:10.1063/1.1703954](https://doi.org/10.1063/1.1703954)、および Hohenberg and Halperin, “Theory of Dynamic Critical Phenomena,” *Rev. Mod. Phys.* **49**, 435 (1977), [DOI:10.1103/RevModPhys.49.435](https://doi.org/10.1103/RevModPhys.49.435) を参照してください。現在の離散時間・非平衡・annealed-Gaussian-J modelを厳密なGlauber modelやModel Aと同一視しません。

`Gamma_eff(t)=-ln|A(t+1)/A(t)|` はsingle-exponential近似が成立する時間領域の診断です。CSVには符号付きの `lambda_eff=A(t+1)/A(t)` と `sign_flip` も残し、絶対値によってzero crossingやoscillationを隠しません。block間SEから `A_q_snr=|A_q|/SE` を作り、既定の `SNR>=5` を数値的な推奨領域とします。この閾値は物理法則ではなく `--gamma-eff-min-snr` で変更できます。Gamma_effの不確かさはblock bootstrapで評価します。

実行順は必ずA→B→C→Dです。

##### Step A: 既存checkpointだけによるtime diagnostics

新規simulationは行いません。schema-v1の8064 checkpointを直接読みます。

```bash
"$PHASE5_PY" scripts/analyze_phase5_time_diagnostics.py time \
  --input-dir results/runs/phase5_B2_R12_shifted_dispersion_M8192 \
  --output-dir results/runs/phase5_B2_R12_followup \
  --gamma-eff-min-snr 5 \
  --bootstrap-replicates 1000
```

ローカル転送時に `results/runs/runs/` となっている場合は `--input-dir` だけをその実在パスへ変更します。出力は `phase5_gamma_eff.csv` と `phase5_fit_window_extended.csv` です。fixed windows `0:3,0:5,1:3,1:5,2:5,2:7,3:7` をすべて保存し、closureに近いwindowを自動選択しません。plateau判定もprimary `0:3` を置き換えません。

##### Step B: preparation dependence

```bash
qsub scripts/run_phase5_squid_preparation_scan.sh
qstat
```

`burn_steps_per_stage=8,16,32` を別々のoutput/checkpointへ保存し、`delta=0.065,0.070,0.080`、mode `0,4`、epsilon `0.05`、M=8192で比較します。比較表は `results/runs/phase5_B2_R12_followup/phase5_preparation_scan.csv` です。既存のsingle Philox streamを維持するためthreshold realizationはstable IDで対応しますが、burn長が変わるとmeasurement開始時のRNG位置は変わります。

##### Step C: survival-conditioned response

Step Bの結果を確認し、必要ならscriptの `--burn-steps-per-stage` とoutput directoryを変更してから投入します。defaultはbaselineと同じ8であり、Bの結果を見ずに16/32を自動採用しません。

```bash
qsub scripts/run_phase5_squid_survival_conditioned.sh
qstat
```

既存の `escape_fraction` は各時刻にthreshold外にいる割合であり、意味を変更しません。新しいcumulative escapeは「一度でもescapeしたtrialは戻さない」first-passage量です。`A_surviving_current` は時刻tまで生存した集合、`A_survive_to_T` は最終時刻まで生存する固定cohortです。後者はfuture informationを使うbasin-internal trajectory診断で、unconditional responseではありません。block集約はblock平均ではなく、保存したamplitude numeratorの総和をsurvivor総数で割ります。出力は `phase5_survival_conditioned.csv` です。

checkpoint schemaは、従来ファイルをv1、survival sufficient statistics付き新規ファイルをv2として扱います。通常解析とStep Aはv1を読み込めます。v1へsurvival解析を要求すると、再実行が必要であることを明示して停止します。

##### Step D: unperturbed fine pseudospinodal scan

Step Bで準備手順を確認後に投入します。primary scanはepsilon=0であり、paired plus/minusは同一初期状態とcommon noiseにより同一trajectoryを保ちます。

```bash
qsub scripts/run_phase5_squid_pseudospinodal_fine.sh
qstat
```

固定gridは `delta=0.058,0.060,0.062,0.064,0.066,0.068,0.070`、M=8192、T=50です。operational criterionは事前固定した

```text
P_esc^cum(T_obs=50) = 0.10
```

であり、これはtrue spinodalではなく「operational microscopic pseudospinodal-like 10%-escape crossover」です。局所的にmonotonicな隣接2点が10%を挟む場合だけ線形補間し、bracketがなければnullを返します。block bootstrapでescape probabilityと補間位置のSE・95%区間を計算し、同じrunから `T_obs=10,20,30,40,50` の時間依存も出します。

有限range系のspinodal-like dynamicsの背景は Loscar et al., “Nonequilibrium characterization of spinodal points using short time dynamics,” *J. Chem. Phys.* **131**, 024120 (2009), [DOI:10.1063/1.3168404](https://doi.org/10.1063/1.3168404)、および Mori, Miyashita, and Rikvold, *Phys. Rev. E* **81**, 011135 (2010), [DOI:10.1103/PhysRevE.81.011135](https://doi.org/10.1103/PhysRevE.81.011135) を参照してください。

Step A–Dの共通summaryは `phase5_followup_validation_summary.json`、fine scanは `phase5_pseudospinodal_fine_scan.csv`、観測時間依存は `phase5_pseudospinodal_time_dependence.csv` です。SQUIDでは全scriptが `--no-figures` を維持します。CSVをローカルへ転送後、8種類の図を生成します。

```bash
python3 scripts/replot_phase5_followup.py \
  --input-dir results/runs/phase5_B2_R12_followup
```

#### Step 4: production

pilot結果から決めた `--block-size`、`--M-total`、`--epsilon-fraction`、`-np`、`--T-fixed` または `--tau-multiplier` を [scripts/run_phase5_squid_intelmpi.sh](scripts/run_phase5_squid_intelmpi.sh) へ反映し、Intel MPI版productionを投入します。

```bash
qsub scripts/run_phase5_squid_intelmpi.sh
```

途中終了した場合は同じscriptを再投入します。全block完了後に以下を確認します。

```bash
"$PHASE5_PY" -m json.tool results/runs/phase5_B2_R12/phase5_run_state.json
"$PHASE5_PY" -m json.tool results/runs/phase5_B2_R12/phase5_validation_summary.json
cat results/runs/phase5_B2_R12/phase5_mode_results.csv
cat results/runs/phase5_B2_R12/phase5_dispersion_fits.csv
```

2 nodeまたは4 nodeへ増やす場合は、1 nodeベンチマークで不足が確認できた場合に限ります。2 nodeなら `#PBS -b 2`、`#PBS -l cpunum_job=76`、`mpirun -np 152`、4 nodeなら `#PBS -b 4`、`#PBS -l cpunum_job=76`、`mpirun -np 304` とします。どのrank数でもlauncherに同じ `"$PHASE5_PY"` を渡すため、Python pathはrank番号やnode数に依存しません。

通常のSQUID Phase5 PBS scriptは `--no-figures` を指定し、`matplotlib` をimportせずcheckpoint、CSV、JSONのみを生成します。pseudospinodal診断driverは図生成機能自体を持ちません。PNG図はSQUID出力をローカルへ転送し、同じ物理オプションと `--resume --figures --max-runtime-seconds 0` で後から生成できます。

### Troubleshooting

`phase5_venv/bin/python: error while loading shared libraries: libpython3.13.so.1.0: cannot open shared object file` は、`mpi4py` より前にPython interpreter自身が起動できていない状態です。`mpi4py` を再installする問題ではありません。

```bash
module purge
module load BasePy/2026
module load BaseCPU/2026
ldd "$PHASE5_VENV/bin/python" | grep -E "python|not found"
"$PHASE5_PY" --version
```

`libpython3.13.so.1.0 => not found` と表示される場合は、`BasePy/2026` がloadされているか、venv作成時と実行時のPython moduleが同じかを確認してください。

現行の2026年度SQUID環境ではPython 3.13.5に `BasePy/2026`、Intel oneAPI 2023.2 / Intel MPI 2021.11に `BaseCPU/2026` を使用します。詳細は大阪大学D3センターの[2026年度software update](https://www.hpc.cmc.osaka-u.ac.jp/maintenance/20260420/)、[Python手順](https://www.hpc.cmc.osaka-u.ac.jp/system/manual/squid-use/python/)、[Intel MPI手順](https://www.hpc.cmc.osaka-u.ac.jp/en/system/manual/squid-use/intelmpi-3/)を参照してください。

主要script:

- scripts/run_phase5_squid_benchmark.sh: single-core kernel/block-size benchmark
- scripts/run_phase5_squid_scaling_benchmark.sh: 19/38/57/76 rank benchmark
- scripts/run_phase5_squid_pilot.sh: 3 delta × 3 mode × 3 epsilon pilot
- scripts/run_phase5_squid_intelmpi.sh: BasePy / BaseCPU / Intel MPI production template
- scripts/run_phase5_squid_production.sh: production preflight / qsub案内wrapper
- scripts/phase5_squid_env.sh: BasePy / BaseCPU / PHASE5_PY共通設定
- scripts/phase5_squid_preflight.sh: Python package、MPI、PBS context検査
- scripts/check_phase5_mpi_python.sh: rankごとのPython絶対パス検査
- scripts/run_phase5_squid_env_smoke.sh: 2-rank preflight PBS job
- scripts/run_phase5_squid_pseudospinodal_direct_j.sh: 1-rank direct_J preparation-survival scan
- scripts/run_phase5_squid_shifted_pilot.sh: shifted-delta epsilon/M convergence pilot
- scripts/run_phase5_squid_preparation_scan.sh: burn=8/16/32 preparation comparison
- scripts/run_phase5_squid_survival_conditioned.sh: schema-v2 first-passage/survivor run
- scripts/run_phase5_squid_pseudospinodal_fine.sh: epsilon=0 fine crossover scan
- scripts/analyze_phase5_time_diagnostics.py: schema-v1/v2 checkpoint-only follow-up analysis
- scripts/replot_phase5_followup.py: follow-up CSVからローカル図を再生成

Mac上の参考benchmark（SQUID性能ではありません、N=1024, R=12, block=32, 50 steps, float64）では、direct_Jが6.47e6、aggregated_exactが3.51e7 trial-site-steps/sで、kernel部分のspeedupは5.43倍でした。block-size結果は16,32,64,128をCSVへ保存しましたが、production値はSQUID上の再測定後に決めてください。

Phase5のprimary出力はphase5_mode_results.csv、phase5_dispersion_fits.csv、phase5_scaling_summary.csv、phase5_validation_summary.jsonです。pilotではphase5_epsilon_convergence.csvとphase5_M_convergence.csvも生成します。full structure factorは既定OFFで、代表debug条件に限り --save-structure-factor で保存できます。

### SQUIDで毎回実行する最小手順

```bash
cd /path/to/Local-Threshold-Spatial-Mode-Simulation
source scripts/phase5_squid_env.sh

"$PHASE5_PY" scripts/check_squid_mpi_env.py \
  --expected-flavor intelmpi \
  --expected-python "$PHASE5_PY"

MPI_TEST_RANKS=2 scripts/check_phase5_mpi_python.sh
qsub scripts/run_phase5_squid_benchmark.sh
```

推奨順は、(1) 専用venv確認、(2) package import、(3) Intel MPI / `mpi4py`、(4) 2-rank Python path、(5) benchmark、(6) block size / scaling確認、(7) pilot、(8) epsilon / M convergence確認、(9) productionです。

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
- `src/spinodal_phase34.py`: Phase1・2 CSVからPhase3・4 scalingを解析するserial post-processing
- `src/spinodal_phase5_core.py`: MPI非依存microscopic kernels、block simulation、checkpoint
- `src/spinodal_phase5_mpi.py`: weighted ensemble-block MPI driverとserial fallback
- `src/spinodal_phase5_analysis.py`: block bootstrap、closure比較、分散・図のrank0解析
- `src/spinodal_phase5_followup_analysis.py`: Gamma_eff、preparation、survival、operational crossover解析
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
