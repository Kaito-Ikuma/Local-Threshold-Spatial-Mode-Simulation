# Presentation figures

このディレクトリは、既存の最終解析 PNG を発表用に集約したものです。画像は元ファイルから `cp -p` で複製しており、シミュレーションの再実行・再描画・画像加工は行っていません。

## Main figures

| destination | original source | class | slide purpose |
|---|---|---|---|
| `01_MAIN_phase3_gamma0_scaling.png` | `results/runs/phase34_B2_R12/phase3_gamma0_scaling.png` | MAIN | スピノーダルへの接近に伴う臨界減速を示し、緩和率が `Gamma0 ~ delta^(1/2)` に従うことを提示する。数値フィットは理論指数 `1/2` と整合する。 |
| `02_MAIN_phase4_xi_scaling.png` | `results/runs/phase34_B2_R12/phase4_xi_scaling.png` | MAIN | 有限波数分散から導いた動的長さが `xi ~ delta^(-1/4)` に従うことを示す。なお `xi` と動的指数 `z` は同じ `Gamma` と `D` から導く内部整合性の検証であり、独立測定ではない。 |
| `03_MAIN_phase4_data_collapse.png` | `results/runs/phase34_B2_R12/phase4_data_collapse.png` | MAIN | 有限波数の緩和時間を `q xi` で整理するとデータが共通曲線へ collapse し、有限 `q` の動的スケーリングを支持することを示す。 |
| `04_MAIN_phase5_R96_extension.png` | `results/runs/phase5_final_validation/final_R96_extension.png` | MAIN | 相互作用距離 `R` を大きくすると有限範囲の microscopic dynamics が Gaussian closure に近づくことを、`R=96` までの拡張で示す。ここで位置づけるのは operational pseudospinodal であり、真の microscopic spinodal の証明ではない。 |
| `05_MAIN_phase5_D_over_kappa.png` | `results/runs/phase5_final_validation/final_D_over_kappa_high_precision.png` | MAIN | `R=12,24,48` の `D_micro/kappa_R` は誤差範囲で `1` と整合し、有意な系統偏差は確認されない。ただし、これは `D=kappa_R` の厳密な証明ではない。 |

## Backup figures

| destination | original source | class | slide purpose |
|---|---|---|---|
| `B01_final_finite_size_gamma.png` | `results/runs/phase5_final_validation/final_finite_size_gamma.png` | BACKUP | 有限サイズ変更に対する緩和率の頑健性を確認する。 |
| `B02_final_finite_size_ps.png` | `results/runs/phase5_final_validation/final_finite_size_ps.png` | BACKUP | operational pseudospinodal の有限サイズ依存性を確認する。 |
| `B03_final_observation_time_all_R.png` | `results/runs/phase5_final_validation/final_observation_time_all_R.png` | BACKUP | 全 `R` に対する観測時間依存性を比較する。 |
| `B04_final_relative_rounding_vs_R.png` | `results/runs/phase5_final_validation/final_relative_rounding_vs_R.png` | BACKUP | `R` の増加に伴う相対的な rounding の低下を示す。 |
| `B05_final_seed_reproducibility.png` | `results/runs/phase5_final_validation/final_seed_reproducibility.png` | BACKUP | 独立 seed に対する最終結果の再現性を確認する。 |
| `B06_high_precision_dispersion_panels.png` | `results/runs/phase5_final_validation/high_precision_dispersion_panels.png` | BACKUP | 高精度 microscopic dispersion fit の各 `R` の詳細を提示する。 |
| `B07_kernel_relation_high_precision.png` | `results/runs/phase5_final_validation/kernel_relation_high_precision.png` | BACKUP | 高精度データによる kernel relation の比較を提示する。 |
| `B08_phase3_exponent_window_stability.png` | `results/runs/phase34_B2_R12/phase3_exponent_window_stability.png` | BACKUP | `Gamma0` の指数がフィット窓に対して安定かを確認する。 |
| `B09_phase4_D_systematics.png` | `results/runs/phase34_B2_R12/phase4_D_systematics.png` | BACKUP | 分散係数 `D` の系統依存性を確認する。 |
| `B10_phase4_tau_vs_xi.png` | `results/runs/phase34_B2_R12/phase4_tau_vs_xi.png` | BACKUP | `tau` と `xi` の関係から動的スケーリングの内部整合性を確認する。`xi` と `z` は同じ `Gamma` と `D` に由来し、独立な観測量ではない。 |
| `B11_phase1_gamma0_theory_vs_numeric.png` | `results/runs/phase12_B2_R12/phase1_gamma0_theory_vs_numeric.png` | BACKUP | `q=0` 緩和率の理論値と数値値を比較する。 |
| `B12_phase2_exact_kernel_relation.png` | `results/runs/phase12_B2_R12/phase2_exact_kernel_relation.png` | BACKUP | deterministic closure における exact kernel relation を確認する。 |
| `B13_phase0_fixed_point_branch.png` | `results/runs/phase0_B2_R12/phase0_fixed_point_branch.png` | BACKUP | Phase0 の固定点分枝と解析対象の準安定枝を示す。 |

## Strict high-precision D validation

`05_MAIN_phase5_D_over_kappa.png` は `high_precision_D_validation_summary.json` の strict validation が完了していることを確認したうえで採用しています。必須の `R=12,24,48` はすべて `M_total=65536`、`production_run_finalized=true`、`production_precision_status=precision_target_met` です。

| R | M_total | D_micro/kappa_R | SE | production precision |
|---:|---:|---:|---:|---|
| 12 | 65536 | 0.717462 | 0.238736 | `precision_target_met` |
| 24 | 65536 | 1.070947 | 0.191801 | `precision_target_met` |
| 48 | 65536 | 0.850197 | 0.154983 | `precision_target_met` |

3 点はいずれも uncertainty interval が `1` を含みます。この結果は統計誤差内での整合性を意味し、有限精度のデータから `D=kappa_R` を恒等式として証明するものではありません。
