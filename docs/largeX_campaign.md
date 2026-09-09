# Phase 5 large-X microscopic campaign

## 1. 目的と解釈上の制限

この campaign は finite-R microscopic model について、長波長式

\[
\Gamma(q)=\Gamma_0+Dq^2+O(q^4),\quad
\xi_{\rm cal}=\sqrt{D_{\rm cal}/\Gamma_0},\quad
\frac{\tau(q)}{\tau(0)}=\frac{1}{1+(q\xi_{\rm cal})^2}
\]

を held-out mode で検証する。最初の対象は R=24, N=4096, qR<=0.35 であり、段階目標は reliable な X_max>=0.3、次に X_max>=0.5 である。q を長波長域の外へ広げるのではなく、measured Gamma0 が小さく、survival quality を保つ operational pseudospinodal 近傍を探して xi を大きくする。

finite-R microscopic model に真の spinodal があるとは仮定しない。座標は

- `delta_G = abs(Delta - Delta_sp_Gaussian)`
- `delta_ps(R)` は `P_esc_cum(T_obs=50)=0.10` の operational crossover
- `s = delta_G-delta_ps(R)` は matched/operational coordinate のみ

である。最終選別には s ではなく measured Gamma0、survival、escape を使う。

背景は Mori, Miyashita, Rikvold, *Phys. Rev. E* **81**, 011135 (2010), DOI `10.1103/PhysRevE.81.011135`、および Hohenberg, Halperin, *Rev. Mod. Phys.* **49**, 435 (1977), DOI `10.1103/RevModPhys.49.435` を参照する。Philox の根拠は既存コードと同じ Salmon et al., SC11 (2011) である。

## 2. 既存 Phase 5 から再利用するもの

`spinodal_phase5_core.py` の paired +/-epsilon、`aggregated_exact`、`direct_J`、Philox、stable block checkpoint、corrupt/config mismatch 検査、rank-independent assignment、append-only M extension、escape/survival sufficient statistics をそのまま使う。本番 kernel は Gaussian-J model に対して conditional exact な `aggregated_exact` である。小条件の比較は同じ CLI の `--kernel direct_J` で実行できる。

既存結果の基準値は次で再生成できる。

```bash
.venv/bin/python src/spinodal_phase5_largex_analysis.py baseline \
  --poster-dir results/runs/poster_ABCD \
  --output-dir results/runs/phase5_largeX
```

R=24 の最も近い reliable 条件は N=2048, M=32768, epsilon_fraction=0.05, fit=[0,3], delta_G=0.0323932441046, s≈0.001, survival=0.9230957, Gamma0=0.7421541, D=128.8539, xi=13.17655, X_max=0.1617006 である。資料中の X≈0.165 は R=48 の global maximum 0.1647247 であり、R=24 の値ではない。詳細な Gamma(q) は `baseline_largeX_summary.json` に保存される。

## 3. q-grid と独立 validation

grid は hard-code せず `qR=2*pi*n*R/N` から作る。R=24,N=4096 では mode 0..9 となる。

| set | mode | qR |
|---|---:|---:|
| q0 | 0 | 0 |
| calibration | 1,2,3,4 | 0.036816, 0.073631, 0.110447, 0.147262 |
| validation only | 5,6,7,8,9 | 0.184078, 0.220893, 0.257709, 0.294524, 0.331340 |

calibration は qR<=0.15 のみを使い、primary fit は `Gamma(q)-Gamma0=D_cal*q^2` の原点通過 fit とする。free intercept は diagnostic のみ。validation set は D または xi の fit に使わず、固定した xi_cal から X と `Y=Gamma0/Gamma(q)` を作る。D の 95% paired-bootstrap CI が 0 を跨ぐ場合は `xi_cal unresolved` として validation success から除外する。

## 4. survival と fit protocol

pre-specified quality criterion は fit end における全 mode の最小 survival >=0.90、sensitivity は >=0.80 である。0.90 は普遍的な物理定数ではない。unconditional、時刻ごとの current-survivor、fit end で母集団を固定する fixed-survivor cohort の Gamma をすべて出す。unconditional と fixed-survivor の相対差が 10% を超える条件は primary success にしない。conditional 結果だけでは成功にならない。

scout の初期 fit end=5 は既存 R=24 baseline Gamma0 から `ceil(3/Gamma0)` として事前に置く。scout 後は q0 の preliminary [0,3] Gamma0 に基づく `ceil(3/Gamma0)` と survival>=0.90 の最終時刻の小さい方を候補とし、production 投入前に `LARGEX_FIT_END` を固定する。解析コードは goodness-of-fit を最大化して窓を選ばない。production checkpoint では `survivor_cohort_end=LARGEX_FIT_END` を fingerprint に含める。

## 5. common random numbers と bootstrap

default の `independent_modes` は従来の entropy

```text
[base_seed, delta_index, mode_index, epsilon_index, block_id]
```

を一字一句変えない。large-X の opt-in `common_modes` は

```text
[base_seed, delta_index, epsilon_index, block_id]
```

とし、同じ condition/epsilon/block の mode 間で threshold、preparation draw、annealed draw sequence を共有する。task/checkpoint fingerprint には mode index、RNG mode、survival criteria、q set、q-grid signature、fit protocol、campaign version、harmonic/cohort settings、epsilon scout 検証済みフラグを残す。RNG identity と checkpoint identity は別である。

解析は mode ごとに独立 resample せず、同じ block ID の mode 0..9 を一組として resample する。各 replicate で Gamma0、Gamma(q)、D_cal、xi_cal、X、Y、residual を再計算する。出力には SE、percentile 95% CI、weighted RMSE、および bootstrap covariance が安定な場合だけ chi-square 相当量を含む。

## 6. epsilon、higher harmonics、kernel

epsilon scout は 0.025, 0.05, 0.10 を比較し、最小 epsilon の Gamma と combined 95% bootstrap uncertainty 内で整合する最大 epsilon を候補にする。必要なら同じ script の `--epsilon-fractions` に 0.15 を追加する。production は選定値を明示して投入する。3q,5q は離散 Nyquist folding 後の mode amplitude を checkpoint に保存する。

全 grid で exact `-ln|K_R(q)|`, `K_R=(1/R)sum cos(qra)` と `kappa_R q^2` の相対差を保存する。q^4 correction を microscopic scaling failure と混同しない。

## 7. SQUID 環境と benchmark

repo で稼働実績のある group `cm9029`、`#PBS -T intmpi`、`scripts/phase5_squid_env.sh` の `BasePy/2026`, `BaseCPU/2026` と venv を再利用する。OpenMPI と決め打ちしない。各 script は OMP/OpenBLAS/MKL/NUMEXPR thread を 1 に固定する。大規模 simulation を frontend で直接実行しない。

まず DBG で 4/19/38/57/76 ranks を測る。既存の N=256 benchmark では 57 ranks が約 `2.466e8 trial-site-steps/s` で、76 ranks の約 `1.943e8` より速かった。ただし N=4096,block=32 の large-X workload へその順位を外挿せず、新しい DBG 結果で決める。

```bash
DBG_ID=$(qsub jobs/largeX/00_dbg_environment.sh)
qstat "$DBG_ID"
```

結果は `results/runs/phase5_largeX/benchmarks/largeX_mpi_benchmark.csv`。wall time、throughput、per-rank peak RSS、LPT load imbalance を比較し、最速かつ安全な rank 数を選ぶ。large-X 実測前に 57 や 76 を最適とは主張しないため、後続 script は `LARGEX_RANKS` 未指定なら停止する。

R=24,N=4096,block=32 の保守的 live-array 見積りは約 26.0 MiB/rank、76 ranks 合計約 1.93 GiB（Python/allocator/MPI 固定 overhead を除く）。実際の安全性は DBG の `peak_rss_mb_per_rank` で判定する。R=48,N=8192 は block=16 として site 数を同程度に保つ。

## 8. 実行順序

以下で `38` は例ではなく、必ず benchmark CSV から選んだ値へ置換する。

1. q0 scout:

```bash
qsub -v LARGEX_RANKS=<BENCHMARKで選んだ値> jobs/largeX/01_q0_scout_R24.sh
```

2. q0 出力の survival/Gamma と提案 fit end を確認後、epsilon scout:

```bash
qsub -v LARGEX_RANKS=<値>,LARGEX_FIT_END=<事前固定値> \
  jobs/largeX/02_epsilon_scout_R24.sh
```

3. scout をローカルでも図示する場合:

```bash
LARGEX_INPUT=results/runs/phase5_largeX/R024/q0_scout \
LARGEX_DELTA_PS=0.032274036590158366 \
LARGEX_BOOTSTRAP_REPLICATES=5000 \
  jobs/largeX/90_analysis.sh
```

epsilon scout についても `LARGEX_INPUT` を `epsilon_scout` に替える。

4. survival primary を通り measured Gamma0 が小さい condition、linearity を通る最大 epsilon、fit end を凍結して production:

```bash
qsub -v LARGEX_RANKS=<値>,LARGEX_DELTA_G=<選択した1条件>,\
LARGEX_EPSILON=<0.025|0.05|0.10>,LARGEX_FIT_END=<整数> \
  jobs/largeX/03_largeX_production_R24.sh
```

5. production 解析:

```bash
LARGEX_INPUT=results/runs/phase5_largeX/R024/production/primary \
LARGEX_DELTA_PS=0.032274036590158366 \
  jobs/largeX/90_analysis.sh
```

6. `precision_goal_met=false` の condition だけを対象に、同一の delta_G/epsilon/fit/ranks/output tag を保って append-only extension を行う。M 以外を変えると mismatch で停止する。複数 condition を production する場合は `LARGEX_OUTPUT_TAG` を変えて一条件ずつ投入するため、十分精度が出た condition を増やさずに済む。

```bash
qsub -v LARGEX_RANKS=<値>,LARGEX_DELTA_G=<同じ値>,LARGEX_EPSILON=<同じ値>,\
LARGEX_FIT_END=<同じ値>,LARGEX_M=16384 jobs/largeX/04_extend_M_R24.sh
# 解析後、必要な場合だけ 32768、次に 65536
```

7. R=48 は R=24 summary が quality criteria 付き X>=0.5 を通った場合だけ script gate を通る。N=8192 の operational delta_ps(T=50) を別途測定し、R48 benchmark も確認してから実行する。

```bash
qsub -v LARGEX_RANKS=<R48 benchmark値>,LARGEX_R48_DELTA_PS=<実測値>,\
LARGEX_R48_OFFSET=<R24後に固定した1 offset>,LARGEX_EPSILON=<検証値>,LARGEX_FIT_END=<固定値> \
  jobs/largeX/05_replication_R48.sh
```

R=6 は最初の production 対象にしない。

## 9. checkpoint、計算量、dry-run

同一 command の再投入では valid completed block を skip し、corrupt/incomplete block は未完了として再計算し、物理・解析 protocol mismatch は reject する。M のみ 8192→16384→32768→65536 と追加できる。

R=24 の 1 condition、10 modes、M=8192、N=4096、48 preparation steps + 50 observation steps の重み付き規模は約 `3.29e10 trial-site-steps`。condition 数に比例する。実 wall time は DBG/short benchmark からのみ見積もる。

simulation を行わない task 解決確認:

```bash
.venv/bin/python src/spinodal_phase5_mpi.py \
  --analytic-references --B 2 --R 24 --N 4096 --deltas 0.033 \
  --auto-q-grid --epsilon-fraction 0.05 --M-total 8192 --block-size 32 \
  --T-fixed 50 --fit-end 5 --survivor-cohort-end 5 --track-survival \
  --rng-coupling-mode common_modes --harmonic-orders 3,5 \
  --fit-protocol q0_tau_survival_frozen_v1 \
  --campaign-version 2026.09.09-phase5-largeX-v1 \
  --dry-run --no-figures --output-dir /tmp/phase5_largeX_dryrun
```

## 10. outputs と成功・失敗判定

解析は `largeX_mode_results.csv`, `largeX_scaling_results.csv`, `largeX_survival_cohort_timeseries.csv`, `largeX_epsilon_linearity.csv`, `largeX_M_convergence.csv`, `largeX_kernel_diagnostic.csv`, `largeX_common_random_covariance.csv`, `largeX_validation_summary.json` と 01–09 の PNG を生成する。primary precision goal は各 validation mode で `SE[Y]<=0.02` かつ `relative SE<=5%`。これは二条件の厳しい方を満たす実装である。

success は survival>=0.90、unconditional/fixed agreement、common RNG checksum/entropy 一致、D_cal の 95% CI>0、held-out validation の存在をすべて要求する。その上で X>=0.3/0.5 を判定する。D unresolved、X 未到達、survival failure、epsilon dependence、fit instability、common-RNG mismatch、conditional bias、M 非収束、大きな q^4 correction は summary に残し、criterion を後から緩めて成功扱いにしない。

将来の stronger test は fixed-boundary microscopic `xi_micro,bnd` を独立に測り、periodic finite-q relaxation を `X=q xi_micro,bnd` で検証することである。本 campaign の xi_cal は calibration dispersion 自身から得る内部的 length であり、これだけで universality class や真の spinodal を証明しない。
