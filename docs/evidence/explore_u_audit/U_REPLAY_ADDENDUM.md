# Raw Hessian replay addendum

The initial uniform 1e-10 absolute-error diagnostic check failed only when replaying AIMNet-environment frequency diagonalization under the different xtbflow NumPy/BLAS environment. This was an additional investigator check, not a handoff prespecified gate. It was not hidden, and its log is retained as u-npz-replay.log.

Follow-up diagnostic recomputed all 1,115 successful NPZ records in BOTH approved environments without any new force/energy calculation. In the original acquisition environment, GFN2 metrics replay exactly (all errors zero under xtbflow); AIMNet2 metrics replay exactly (all errors zero under aimnet). Across environments, the largest AIMNet2 frequency discrepancy under xtbflow is 1.0084910684327042e-10 cm^-1, overlap discrepancy1.9872992140790302e-14, c_u discrepancy9.094947017729282e-13 kcal/mol/Angstrom^2. Neither imaginary-frequency counts nor curvature signs change. Both full diagnostic JSONs are retained; no acceptance threshold, scientific result or raw output was changed.

The earlier U_NUMERICAL_AUDIT pending NPZ diagonalization item is now resolved for same-environment replay. Independent force/Hessian acquisition is still a different validation surface and was not claimed. Raw NPZ hashes cover557 GFN2 and558 AIMNet2 matrices.

## 中文说明
附加的数值复算最初把两个环境混用，导致一个接近机器精度的频率差略高于临时检查值，这个失败日志已经保留，没有删掉。随后分别用原来的计算环境读取全部矩阵，GFN2 和 AIMNet2 的数值都能精确重现；跨环境的微小差异没有改变虚频数量或曲率正负。这里只验证已保存矩阵的分析，不把它说成重新计算了量子化学力，也没有修改任何科学门槛或原结果。
