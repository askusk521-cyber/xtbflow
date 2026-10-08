# 生成中物理筛选：冻结设计 smc-1

本设计全部实验为探索性，正式结果尚未产生。机器可读的完整定义在 `configs/explore/smc_selection.json`；不得根据结果调整定义，变更需要 smc-2 并同时报告两版。

## 问题与范围
比较生成中剪枝—克隆与生成后的同一 relaxed 筛选，区分物理选择和随机重启的作用。只使用 V1a 开发集 62 母体、21 分子式组、训练 seed 0/1/2；不使用 screen、V1b、DFT，不合并 PR。基于 explore/physics-guidance 的 ffca40f。

## 种群、打分和重启
每个母体每个 seed 32 条，保留 16 条。geometry_lead3，50 步，在 s=0.6/0.7/0.8/0.9（k=30/35/40/45）选择。256 批大小，复用 pg-1 查询顺序与初始噪声，不跨种群切批。保存检查点和终点的实际状态与连续预测终点。

raw 为 xTB 能量减母体锚定反应物能量。relaxed 在预测终点进行 20 步反射力鞍点弛豫：力为负梯度，Hartree/bohr 转换为 kcal/mol/Å；每步重算 event_direction，以 saddle_reflect 得到方向，步系数 0.001 Å²/(kcal/mol)，全步缩放保证每原子不超过 0.05 Å，去质心。最终能量减锚定能量；h=0.005 Å，曲率为负力差沿方向的中心差分。任一阶段间距小于 0.5 Å、SCF 失败或无效方向记失败。成功且负曲率优于非负曲率，失败最后；层内按势垒，平局按 sha256(parent_id|seed|proposal)。S0 全部 558 条真实 TS 中成功负曲率比例小于 50% 时，统一删除曲率分层。失败包含在该比例分母，不能通过剔除失败提高比例。

每个幸存者恰好一个克隆，重启公式为 `(1-t_b)*(b_r+sigma_b*epsilon_b)+t_b*b_hat` 和 `centered((1-t_x)*(x_r+sigma_x*epsilon_x)+t_x*x_hat)`。sigma_b=1、sigma_x=0.5。复用 symmetric_noise/geometry_noise，噪声地址为 explore_smc_clone、joint_b/joint_x、query、训练 seed、幸存者 proposal、checkpoint_step，不含臂名。幸存者占原行，按排名排列的克隆填入按排名排列的被剪行。随机臂按 sha256(parent_id|seed|proposal|k_c) 排名，其余相同。从原检查点实际状态或克隆状态调用 sampler.rollout(start=k_c)。

13 臂为 none 和四检查点各 relaxed/raw/random。主臂 relaxed_0.7；0.7 与 lead3 都使用过开发集信息。逻辑网络步数各臂相同，实际复用 none 前缀和幸存者重算的成本另行报告。每个成功 relaxed 评分通常 24 次调用：20 梯度、1 最终能量、2 曲率梯度，另加 1 初始 raw 能量诊断；原文“约23次”不作为精确计费值。

## 终点评估和推断
所有终点用同一解码/匹配流程及 relaxed 评分。只对合法事件按最好候选的层/势垒排序，不同事件平局按 sha256(channel_id)。hit@K 使用 K=1/2/4，H 为三者均值。次指标：hit@∞、种群平均事件效用、不同合法事件数、合法率、best-reference 命中率。先每个母体/seed，再平均三个 seed，配对母体差以 split_group 为分组调用 cluster_summary 给双侧95%区间。

比较每个 relaxed/raw 对 none 和同期 random，relaxed 对 raw，random 对 none。对 none 下界>0 且 random 下界>0 为 IN_GENERATION_SELECTION_HELPS；对 none 下界>0 且 random 区间含0为 NOT_PHYSICS_SPECIFIC；none 区间含0为 NO_GAIN_OVER_POSTHOC；none 上界<0为 SELECTION_HURTS。原表没有定义“优于 none 但显著差于 random”，若发生明确记为未分类，不擅自改写表。

诊断包含检查点评分与 none 终点已知目录势垒的同母体相关（至少4条、至少2目标值），按 seed 算后对完整三个 seed 的母体平均；none 有最佳通道时幸存者丢掉最佳通道的误剪率（每母体先平均合格 seed）；克隆与祖先事件不同率；仅克隆有最佳通道的种群比例；评分分层、失败、降能分位数、调用数和耗时。

## 门禁、证据及资源
S0 使用参考目录锚定顺序的 x_ts/b_p；逐条 raw 与检查A误差<=1e-6 kcal/mol，事件 min 汇总 rho 与 0.7491015704725383 差<1e-9。C1 none 事件和 proxy_status 必须与 pg-1 100%一致；四检查点 raw 误差<=1e-6。C2 各臂幸存者事件复现>=99.9%，同时报告最大坐标偏差。任一不通过写 STATUS 并停止正式下游阶段。输入哈希不符同样停止。

开发仅在独立 dev-clone，正式计算只用已推送提交的干净 source-SHA 克隆。manifest 保存源码、配置与输入哈希、dirty=false、Slurm号。analyze 必须原始输出重跑逐字节一致。CPU 使用 nice 10、最多48 worker、一次一个重任务，启动前负载不超过120。GPU只通过 main 分区单 pro6000，smc- 名称、每作业<=60分钟，有已有作业则依赖排队。估计总 GPU 超1小时或 CPU 超40核时停止。

冻结提交推送后，先 CPU 两母体/32 proposals/seed0 冒烟，不输出或查看科学指标；再正式 S0/G1/C1/G2/C2/analyze。交付 REPORT_ZH.md、results.json、各阶段 manifest、SHA256SUMS 和未合并 PR（base physics-guidance）。全量验证：validate_bootstrap、tests/reusable unittest、排除 reusable 的完整 pytest。

## 局限
开发集多次重复使用；检查点与时钟已经受到开发集结果影响；闭世界目录外通道计未命中；xTB 与 ωB97X 的势垒排序不一致。代理命中不是经过验证的过渡态，不作化学认证结论。
