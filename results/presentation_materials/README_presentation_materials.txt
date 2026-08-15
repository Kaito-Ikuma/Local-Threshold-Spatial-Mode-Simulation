作成された主資料
----------------
1. integrated_lambda_spectrum.png
   理論・gaussian_map・microscopic を同一図上で比較した固有値スペクトル。
   中抜きマーカーは low-SNR / unreliable 判定。

2. integrated_normalized_spectrum.png
   lambda(q)/lambda(0) と Khat_R(q) の比較。
   空間カーネル形状の頑健性を示す主図。

3. selected_mode_n*_decay_comparison.png
   選択モードの A_q(t)/A_q(0) の時間発展。
   理論線 lambda(q)^t と、gaussian_map / microscopic の平均±SE を比較。

4. negative_mode_n*_signflip.png
   負の固有値モードの符号反転しながらの減衰。

5. linearity_epsilon_scan.png（指定時のみ）
   epsilon 依存性。lambda_fit が epsilon に依らないことを確認する資料。

6. R_scan_normalized_spectrum.png（指定時のみ）
   R 依存性。lambda(q)/lambda(0) が Khat_R(q) に従うことを確認する資料。

主要CSV
-------
- gaussian_lambda_aggregate.csv
- microscopic_lambda_aggregate.csv
- gaussian_ratio_aggregate.csv
- microscopic_ratio_aggregate.csv
- gaussian_selected_mode_n*_decay.csv
- microscopic_selected_mode_n*_decay.csv

発表での推奨使用法
------------------
- 口頭発表本編：1,2,3
- 補足またはバックアップ：4,5,6
- ポスター：1 を主図、2 と 3 を右側に添える
