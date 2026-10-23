**日期** 2026-10-17 ｜ **判据**（≥5 篇给明确判定，每篇注明数据/模型/切分/pooling + 理由）｜ **状态** v1.0：**S1 端到端判定已跑出（GPT2-XL 全量 CPU 86min）= 无法判定**（contiguous 主切分 Δr=0.049 CI[−0.004,0.098] p=0.0432；r_LLM=0.105 vs shuffled 对照 0.409≈×3.91）。经验性声称 S1、E1、I1、T1、C1 五篇成列（**5/5**；独立评审终审 **PASS**）

> 文中出现的 commit 号（如 `9c49cf0`）指公开前的内部开发历史，不在本仓库里；`results/` 下的运行输出不随仓库发布，按 README 的命令复跑生成。

## 一、协议常量与判定规则（已在代码里预注册，勿擅改）

来源：`src/d1/config.py`（judge 阈值）+ `src/d1/judge.py`（P1–P5 + 判定规则）。

- **judge 阈值**：α = 0.05；SESOI(Δr) = 0.02；min_test_blocks = 10；零输入保留上限 = 0.10。
- **P1–P5 硬前提**（任一不满足 → 无法判定，行内写清缺哪条）：P1 主切分必须 contiguous（shuffled 只作对照、永不决定判定）；P2 显著性来自循环移位 null（i.i.d./permutation 被代码直接拒绝）；P3 零输入消融通过（保留比例 |r(零输入)/r(真实输入)| < 0.10，不是 |r|<0.10）；P4 ≥ min_test_blocks；P5 null 能分辨 α。
- **判定**（Δr = r(LLM+混淆基线) − r(混淆基线)，配对块 bootstrap CI + 循环移位 p）：
  - **仍成立**：混淆已控制 + Δ CI 下界 > 0 + p < α；若 claim 设了 reference_value + min_retention，还需保留率 CI 下界过线。
  - **不成立**：Δ CI 上界 < SESOI，或保留率 CI 上界 < min_retention，或 LLM 单独 r < SESOI。
  - **无法判定**：其余；未控制混淆最多只能判「不成立」。
- **行 schema**：`rows.jsonl`（`schema_version=d1-row/0.1`）+ 同序 `rows.tsv`；shuffled 对照写在同一行 `control_*` 列。

## 二、方法论批判三篇（锚定协议与 scope，本身不进「被审计」行列）

1. **Nature Comms 2026-04-27** [s41467-026-72253-7](https://www.nature.com/articles/s41467-026-72253-7) — 三指控：shuffled split 制造虚假结论 / pooling（last·mean·sum）系统偏置 / 位置+语速混淆（OASM 基线解释 GPT2-XL 预测力 >80% Pereira2018、>50% Fedorenko2016、~100% Blank2014；contiguous 下 OASM 归零）。→ 给判定表提供 contiguous vs shuffled 对照、OASM 诊断、三档 pooling。
2. **Rahbar** arXiv [2610.03827](https://arxiv.org/abs/2610.03827)（NeurIPS 2026 LP4FM）— 打碎脑对应后对齐分仍 0.31（训练目标 0.10→0.34，信度上限 0.54）；shuffled 目标 ≈ k/n 基线。→ 提供「去结构后还剩多少」诊断 + k/n 基线。
3. **Murphy** arXiv [2609.03160](https://arxiv.org/abs/2609.03160) — 对齐能约束但不能识别机制。→ 定 scope：判定表只判「声称的数字在严格协议下是否仍成立」，不判「机制是否为真」。

## 三、被审计的原生对齐声称（判定表逐篇给结论）

| claim_id | 论文 | 声称（原文口径） | 数据 | 模型 | pooling | 原文切分/混淆 | reference_value（原文报告值） | 判定 | 提取状态 |
|---|---|---|---|---|---|---|---|---|---|
| S1 | Schrimpf et al. 2021，PNAS 118(45)（bioRxiv 2020.06.26.174482） | GPT2-xl 在 Pereira2018 与 Fedorenko2016 上达噪声上限（≈100% 归一化预测力），43 个模型最高；transformer 架构 + 下一词预测驱动脑样表征 | Pereira2018（OSF crwz7）、Fedorenko2016、Blank2014 | GPT2-xl | 逐层扫描取每模型最优层（argmax over layers，原文 Methods 6 / Fig S10；句级/样本聚合亦按原文口径，重算时一并对齐） | 原文 train/test（sample 级）切分，**未做 contiguous 对照**——正是 Nat Comms 2026 指控的 shuffled split | 0.31（Pereira2018 raw r；ceiling=0.32，原文报告归一化预测力≈100%） | 无法判定（本管线 nested 口径 0.105；argmax 对齐重算=0.115 L21，仍≪0.31，判定不变） | ✅ 已从原文落地（Fig 2a + Results §「Specific models」） |
| P1 | Pereira et al. 2018，Nat Commun 9:963 | 语义可从脑激活线性读出（universal decoder） | Pereira2018（自身） | 原文非 LLM（语义特征回归）；作数据源 + 基线 | N/A（非 LLM） | 待核 | 待提取 | 基线/数据源（不计入） | ⏳ |
| E1 | Hong et al. 2026，eLife 13:RP101204，[10.7554/eLife.101204](https://doi.org/10.7554/eLife.101204)（2026-09-16） | 模型越大越能预测神经活动；编码性能峰值层随 size 增大而前移（对数关系） | ECoG，10 癫痫患者（8 人进 160 电极语言集），30min 播客（This American Life，~5000 词），词级 HFBB(70–200Hz) | 多尺寸 transformer LLM 家族（GPT-2/GPT-Neo/OPT/Llama-2，82M–70B；GPT-Neo 家族固定训练集剥离 size×架构） | 逐词（subword token 平均到 word；每层独立，非多层聚合） | ✅ **10 折 CV = 9/10 contiguous 训练段 + 留出测试段（原文明写 contiguous）**；相位随机化 null（保留自相关 5000 次）+ 未训练 SMALL 基线 + GloVe/声学特征基线 + PCA（仅 OLS）控制维度 | 无单一归一化分数；参考值 = 编码相关（160 电极 SMALL 0.255 / XL 0.284，t=16.791 p=4.770e-37；mSTG SMALL 0.335 / XL 0.377，t=11.584）；声称 = size↔编码性能单调/对数缩放曲线 | 无法判定（协议不适用：scaling 曲线非单一 Δr） | ✅ 全文已读（EuropePMC fullTextXML / PMC13581289） |
| I1 | Ismayilzada et al. 2026，arXiv [2604.03480](https://arxiv.org/abs/2604.03480)（q-bio.NC，COLM 2026 接收，v2 2026-08-09） | LLM 与创造性思维中人脑「对齐」：DMN 对齐与模型 size 正相关（r=0.55,p<0.05,last-token pooling）及与想法原创性（AUT 分）正相关（r=0.52,p<0.05,mean-token pooling），prompt 阶段最强；post-training 目标改变对齐不对称性 | Beaty et al. 2018 fMRI，170 被试（清洗后 ~162），AUT + OCT 控制任务，~23 刺激/任务/被试 | 24 个 LLM（270M–72B：Gemma-3/Llama-3.1/3.2/Olmo-3/Qwen2.5/Falcon-40B/Mistral-7B/DeepSeek-R1-Distill-70B）+ Llama-3.1-8B 变体（CrPO-cre/R1-Distill/base） | last-token（size 关联）/ mean-token（原创性关联） | **RSA（Spearman rank，非回归）→ 无 train/test 切分，contiguous/shuffled 论不直接适用**；控制较全（OCT 任务 + SOM 网络双分离、empty/placeholder prompt 基线、生成长度无关、交叉验证 layer selection、LOO+Spearman+BCa）；但主关联 n=24（模型层面）且两关联各自依赖不同 pooling | 无单一 r；声称 = 关联强度 r=0.55/0.52，RSA 噪声上限归一化（留一被试 RDM 相关） | 无法判定（协议不适用：RSA 几何关联） | ✅ 全文（HTML v2）已读 |
| T1 | Temporal structure of natural language processing in the human brain corresponds to layered hierarchy of LLMs，Nat Commun [s41467-025-65518-0](https://www.nature.com/articles/s41467-025-65518-0)（2025-11-26；作者更正 2026-04-17） | LLM 层序与脑语言理解时间动态「对齐」：越深层对应越晚脑活动（lag-layer 相关 r=0.85, IFG, p<1e-13） | ECoG 9 癫痫患者（7 人有目标电极）听 30min 播客 NPR 2017，高 gamma；ROI mSTG(28)/aSTG(13)/IFG(46)/TP(6) | GPT2-XL（48 层）+ Llama-2（32 层），逐层 embedding→PCA 50→线性编码回归 | 逐层（每层输出，非多层聚合；PCA 50） | ⚠️ **10 折 CV 未说明 contiguous/shuffled**（词级 ECoG 时序自相关强）；混淆控制较全（layer 置换 null 10 万次、线性插值伪层对照、Llama-2 复现、符号学对照、线性混合模型跨电极/被试） | lag-layer 相关 r=0.85(IFG)/0.92(aSTG)/0.93(TP)，置换 p<1e-5 | 无法判定（协议不适用：C——lag-layer 相关属二阶关联；另：切分未明示） | ✅ 全文已读（Nature 页） |
| C1 | Caucheteux & King 2022，Commun Biol 5:134，[10.1038/s42003-022-03036-1](https://doi.org/10.1038/s42003-022-03036-1)（2022-02-16；2023-04-11 有 author correction） | 深度语言模型（transformer）激活与人类脑对句子的反应「部分收敛」：相似度主要取决于从上下文预测词的能力；中间层最像脑（倒 U 形） | fMRI + MEG，102 被试（fMRI n=100、MEG n=95），读 400 个孤立荷兰语句（9–15 词）2×1h，非叙事 | 36 个 transformer 架构（128–512 维 × 4–12 层 × 4–8 头；CLM+MLM），荷兰语 Wikipedia 278M 词，100 训练步 → 3600 网络 / ~32400 embedding | 逐词逐层（非多层聚合） | ⚠️ **五折 CV（held-out 句子）未说明 contiguous/shuffled**；孤立句（作者自注非叙事）降低但未消除时序自相关（BOLD 慢响应 + 句内/句间结构）；作者自认视觉/词位置混淆（句首句尾 + 屏幕闪烁 + 停顿）；但有 permutation baseline + 随机网络对照（随机 embedding 也显著 R=0.019/0.018） | compositional embedding fMRI **R=0.048**、MEG **R=0.041**；noise ceiling（shared response）fMRI R=0.060、MEG R=0.020；brain score×language performance 相关 fMRI R=0.57、MEG R=0.77 | 无法判定（主声称 A：切分未明示；副声称 brain×language R=0.57/0.77 归 C：协议不适用） | ✅ 全文已读（Nature 页） |
| N1 | Nastase, Zada, Goldberg & Hasson 2026，Neuron 114:3527，[10.1016/j.neuron.2026.07.024](https://doi.org/10.1016/j.neuron.2026.07.024)（开放获取 CC BY，作者 PDF 已读） | **Perspective（理论综述）**：LLM 与人脑「收敛到同一表征格式（分布式高维群体编码）」——概念性声称，非新对齐数字 | 综述（引 Goldstein Whisper / Zada GPT-2 coupling / Kumar BERT） | 综述 | N/A（综述） | — | **无单一参考值**；且原文自带 D1 要的 caveat（「线性编码只证明捕获特征，不证明架构/算法相同」） | 移出（Perspective） | ✅ 全文已读 → **移出「被审计原生声称」**，归入理论/背景 |

> 验收标准要 ≥5 篇给明确判定。**当前 S1、E1、I1、T1、C1 五篇是真实经验性对齐声称（5/5）**；P1 是数据源+基线，N1 是 Perspective（已移出）。表格现含 pooling 与判定两列（首轮评审 fail 后补）。E1 已读全文=**contiguous 10 折明写**（见 §九修订）；欠报切分的剩 T1/C1（未明示）+ S1（shuffled）。备用候选（后续可扩表）：SAE 2605.23035 / hierarchical composition bioRxiv / Fedorenko2016·Blank2014 独立行。

## 四、min_retention 默认值（协议决策，待与 Claude Code 对一轮）

- **提议** min_retention = 0.5：被审计声称在严格协议下的保留率 ≥50% 才算「仍成立」，否则退「无法判定/不成立」。
- 依据：Nat Comms 2026 报告 shuffled→contiguous 后跨层神经预测力模式 r ≈ −0.60 ~ −0.86 翻号——效应不仅缩水，连「哪层最像脑」都翻号；0.5 是保守但可辩护的门槛。
- **未定**：与 Claude Code 结对评审后再写进 claim 配置（config.py 的 `claim.min_retention` 现为 null=仅报告）。

## 五、下一步（按依赖排序）

1. **跑通 S1 端到端**：下 OSF crwz7（Pereira2018）真实数据 → loader 桩转正 → GPT2-XL 跑 contiguous 主切分 + shuffled 对照 + 三档 pooling → 用 S1 的 reference_value=0.31 填 claim 配置。
2. 提取 P1 / E1 / N1 的 reference_value（E1、Rahbar 为 open access；N1 付费墙，走摘要或找预印本）。
3. 补 1 篇原生声称凑满 ≥5 篇（T1 已补，现 4/5）。
4. min_retention 与 SESOI 用 Claude Code 结对定稿后回填 config 与本文档。

## 六、S1 端到端进展（2026-10-17 21:00 轮）

- **数据版本已定**：Brain-Score `Pereira2018.language` assembly（= Schrimpf 2021 实际打分那份，逐值核对 Schrimpf 本人 S3 原文件最大差 0.0、sha1 一致）。raw OSF 弃用（OSF 维护到 10-19；且 Schrimpf 用的响应本就是 Blank/Fedorenko 重处理的）。
- **呈现顺序（重要限制，已落盘）**：段内顺序=阅读顺序，可恢复；**段间顺序在任何公开版本都不存在**——Pereira 原文写明段落分 run/顺序按被试·session 各自随机、每人看 3 遍、公开每句响应是 3 次合并。→ loader 不静默排序，用 `dataset.passage_order` 显式构造（默认 `topic_interleaved`：每个 contiguous 折恰含 12 话题各一段、无话题被整体留出，与 2026 批判折结构一致）。
- **噪声上限（ceiling）口径纠正**：Schrimpf = 留出被试 + 外推无限被试（**非 split-half**；voxel 级 split-half 不可行因被试间无体素对应）。实测 shuffled 5 折 **0.3225**（原文 0.32 / S3 0.3186）vs 本管线 contiguous 8 折 **0.0951**。
- **本轮最关键发现**：**换 contiguous 切分后噪声上限自身下降约 70%（0.32→0.095）**（补测岭回归最高仅 0.13，排除 OLS 无正则所致）。→ S1「归一化预测力≈100%」不能拿 contiguous 下 r 去除 0.32；下一轮须在 contiguous 下重算 ceiling 再判。
- **自检（独立复跑确认）**：384 句 / 96 段 / 9 被试 / 12155 体素；ceiling=0.3225(shuffled)/0.0951(contiguous)；话题留出 0/24；词数统计与原文吻合。数据 561MB 放在 `data/` 下；本轮未用 ds003020，未跑 GPT2-XL。
- **小瑕疵（记后续）**：check 脚本把报告写 `data/derived/`，在 `data/` 不可写的环境里复跑，最后写 JSON 时报 PermissionError（检查本身全过）——报告应改写到可写路径（之后已改为写到 `results/pereira2018/`）。
- **下一步**：GPT2-XL 全量端到端（activations→readout→judge）+ contiguous 下重算 ceiling，把 S1 行判定跑出来。

## 七、S1 端到端判定（2026-10-18 14:00 轮 · GPT2-XL 全量 · CPU 86 分钟）

**判定 = 无法判定**（contiguous 主切分，last pooling，基线 position+word_rate，9 被试 / 384 句 / 96 段）：

| 指标 | 值 |
|---|---|
| Δr = r(LLM+基线) − r(基线) | **0.049**，95% CI [−0.004, 0.098]，p_null(Δ)=**0.0432** |
| r_LLM | **0.105**，CI [0.049, 0.165]，z_null=3.1，p_null=0.0108 |
| r_基线 / r_joint | 0.062 / 0.111 |
| null | 循环移位 ×369（min_shift=8），最小可达 p=0.0027 |
| 零输入保留 | 0.00（通过） |
| shuffled 对照 | r_LLM=**0.409**（主切分的 ×3.91）；OASM 0.000→0.429 |

- **为何无法判定**：p_null(Δ)=0.043 < α=0.05，但 Δ 的 CI 下界 −0.004 未 >0 → 不能判「仍成立」；CI 上界 0.098 > SESOI 0.02 → 不能判「不成立」。缺口：效应不确定，需更多被试或测试块。
- **pooling 口径（复评纠正）**：原文 S1 报告的是「逐层扫描取每模型最优层（argmax over layers，Methods 6 / Fig S10）」，本管线跑的是 last 末层——「0.105 vs 原文 0.31 ≈ 1/3」里**切分（shuffled→contiguous）与层选择（最优层→末层）两个口径同时变了**，此前全记在切分账上。layer_curve.json 已含逐层曲线，取 argmax 重算即可对齐层选择口径（几乎零成本）。附带审计点（**v0.3 撤回原「原文未提、我方补入」**）：Schrimpf 实际做了选层对照——Fig S2 B/C 在实验 2 上选层 vs 实验 3 上选层，差值误差棒跨 0、各层分数跨实验相关 r=.95（bioRxiv 版图注已核），归 **L-b（test-argmax + 原文对照，无后果）**，见检查表 v0.3 §3.6。此对照在 shuffled 切分下做的，对 P1 无帮助。对照：I1 用交叉验证 layer selection（L-a），是正例。
- **argmax 对齐重算（2026-10-21 14:00 完成）**：layer_curve.json 的逐层 held-out r（交叉验证、非拟合全量）取 argmax——contiguous 最优层 **L21=0.115**（全曲线 0.085–0.115）、shuffled 最优层 **L48=0.412**（单调升至 0.412）。归因拆分：**切分（contiguous→shuffled）在「逐层取最优」同粒度下把 r 抬 ×3.58（0.115→0.412），是主因；层选择口径差（nested 0.105 全体素 vs argmax 0.115 2000 体素子集）约 +0.011，次要**（近似，混入体素子集差异）。结论：原文 0.31↔本管线 0.105 的差距主因是切分而非层选择；「argmax 重算待做」关闭，S1 判定不变（无法判定）。**外部效度限定（评审追问）**：×3.58 / ×3.91 是「同一条构造 passage_order（topic_interleaved）上 contiguous vs shuffled」的**内部切分效应**量化，**不是**「Schrimpf 换真实 contiguous 会掉多少」的估计——Pereira2018 真实段间顺序不可恢复，本管线的 contiguous 是对真 contiguous 的近似。
- **ceiling 口径（同口径重算完成 · 2026-10-21 深夜）**：判定规则里没用 ceiling（min_retention=null）。原「r_LLM=0.105 已高于 contiguous ceiling 0.0951」是口径不一致的假悖论（0.0951 是 OLS+体素中位数的旧口径）。用与 r_LLM 同一套岭估计器、同一折/内折/α/计分/聚合重算，再按模拟校准后：**同口径 ceiling ≈ 0.130**，**归一化保留率 ≈ 0.81**（硬区间 [0.44, 0.99]；只计 r_LLM 抽样误差的 CI [0.38, 1.28]）。初版 0.969 只是上界（先平均再外推）；逐体素外推再取均值 (a) 会把 ceiling 高估约 2.2 倍（长右尾来自 16.8% 体素 τ>9 外推失控，逐体素 v0 重测相关仅 0.18），故须模拟校准。**shuffled 下各种口径 1.17–1.22、OASM（无语言）也 >1**——「其他被试的脑」不是真上限，段落泄漏让段落身份码学到目标被试自己的段落级响应，该比值只能当泄漏证据。**不推翻「S1=无法判定」**：judge 不用保留率；contiguous 点估计 ≈0.81 与论文「≈100%」有差距但 CI 太宽，且上限本身很低（位置+语速基线 0.58，LLM 增量 Δ 0.45 CI[−0.04,0.91]）。
- **本轮判断被改变 2 条**：①「S1 归一化预测力≈100% 可复现」被推翻——contiguous 主切分下 r_LLM 仅 0.105（原文 0.31 的约 1/3）；但此差距是**切分（shuffled→contiguous）＋层选择（最优层→末层）叠加、尚未拆分归因，argmax 重算已定量（切分 ×3.6 主因、层选择 +0.01 次要）**，不能单独坐实「0.31 全因 shuffled」（shuffled 对照 0.409 同样受层选择口径影响）；②「r/ceiling 可直接给归一化分数」被 Claude Code 用事实纠正（口径不同）。
- **复现四要件现已全部满足**（无 GPU 端到端跑通 Schrimpf 系复现 / contiguous 主+shuffled 对照 / 三种 pooling 切换 / position+word_rate 基线切换 / README 一条命令复跑）——待独立评审。
- **Manifest**：`results/d1/pereira2018_schrimpf2021__split-contiguous__pool-last__base-on__<config 哈希前 8 位>/`（运行输出，不随仓库发布；verdict.txt / manifest.json / rows.jsonl / rows.tsv / layer_curve.json）。
- **下一步**：① 同口径（岭）重算 contiguous ceiling → 给 S1 归一化保留率；② 补 ≥3 篇经验性声称；③ min_retention 与 SESOI 结对定稿。

## 八、I1 判定 + 协议范围边界（2026-10-19 14:00 轮）

**I1（Ismayilzada et al. 2026，creative thinking）判定 = 无法判定（协议不适用，非数据缺失）**

- 本文用 **RSA**（LLM RDM 与逐被试 fMRI RDM 的 Spearman rank，取值 [−1,1]，负值=反对齐），**不是线性预测回归**。协议 P1–P5 全建在回归口径上（Δr = r(LLM+基线) − r(基线)、contiguous 切分、循环移位 null），对 RSA 不直接适用——没有 train/test 切分可审，也没有「基线差 Δr」可算。
- 声称形态也不同：是**关联性声称**（对齐与 size/originality 相关 r=0.55/0.52），不是 S1 那种「预测力≈ceiling 的 X%」单一数字。
- **缺什么（验收标准要求写明）**：要判它，需给协议补一个「RSA/关联」分支（RDM 几何相关 + 噪声上限 + 置换 null + 交叉验证 layer selection），或明确标注「RSA 类不适用本协议、单独立项」；且需其代码/数据公开才能同口径复跑（原文无 code 链接）。
- 但它**做了不少控制**（OCT 任务 + SOM 网络双分离、empty/placeholder prompt 基线、生成长度控制、交叉验证 layer selection 量化乐观偏差、n=24 关联的 LOO+Spearman+BCa）——是「做了控制」的正面例子，可作协议对照片。

**本轮判断被改变 1 条**：隐含假设「被审计对齐声称默认是线性编码/预测类、P1–P5 全覆盖」被 I1 打破——对齐声称至少分两类（回归预测 vs RSA 几何关联，后者可为负），协议须显式区分或标注不适用，否则会把 RSA 声称误套回归规则。

## 九、T1 判定 + E1 判定补正式（2026-10-19 21:00 轮）

**T1（Temporal structure，Nat Commun s41467-025-65518-0）判定 = 无法判定（协议不适用：C——lag-layer 相关是「分数随层/时滞变化」的二阶关联，非单一 Δr；另：切分未明示）**

- 声称：LLM 层序（GPT2-XL 48 层 / Llama-2 32 层）与脑语言理解的时间动态「对齐」——越深层对应越晚脑活动。核心参考值 = **lag-layer 相关 r=0.85**（IFG，Pearson p<1e-13；置换 10 万次 p<1e-5；aSTG r=0.92、TP r=0.93）。
- 方法：逐词 contextual embedding（每层输出）→ PCA 50 维 → 线性编码回归；性能 = 预测-真实相关；lag −2000~+2000ms。
- **切分**：原文写「10 折交叉验证」但**未说明 contiguous 还是 shuffled**——词级 ECoG 时序自相关极强，这正是 D1 要审的点：若 shuffled，编码性能虚高、峰值 lag 可能被带偏。
- **混淆控制（较全，可作正面参照）**：layer 置换 null（10 万次）、线性插值伪层对照、Llama-2 复现、符号学（音/形/句/义）对照、线性混合模型（跨电极/被试）。
- **缺什么**：需确认 10 折切分是否时间连续留出；用本管线 contiguous 主切分重跑 lag-layer 相关才能给「仍成立/不成立」。另有 2026-04-17 作者更正（s41467-026-72170-9），需核对是否影响主结论。

**E1（Hong et al. 2026 eLife，ECoG scaling）判定修订（2026-10-20 14:00 · 全文已读）= 无法判定（协议不适用：scaling 曲线非单一 Δr；但切分/混淆已核，是正面例子）**

- **切分已核**：原文 Methods 明写「10 折 CV = 9/10 **contiguous** 训练段 + 留出测试段」——**不是欠报，而是显式 contiguous 留出**（D1 的 P1 金标准）。此前「切分/混淆未核」的判断是错的（只读了摘要）。
- **混淆控制（较全，正面例子）**：相位随机化 null（保留自相关，5000 次，与 D1 循环移位 null 同构）；未训练 SMALL 基线（Fig2-figsupp4，每层 trained>untrained）；GloVe + 声学特征基线；PCA 仅用于 OLS 且 train/test 分开算避免泄漏；RidgeCV 定 alpha。
- **为何仍无法判定**：声称形态是「size↔编码性能单调/对数缩放曲线」，非「预测力≈ceiling 的 X%」单一数字，P1–P5 的 Δr/SESOI 口径不直接适用（与 I1 同属「声称形态不匹配回归 Δr」，但 E1 本身是回归、只是声称是曲线）。
- **规律收窄**：此前「时序对齐声称系统性欠报切分」的规律需修正——E1 是显式 contiguous 的反例；欠报的只剩 **T1、C1（10 折/5 折未明示）** 与 **S1（shuffled，被 Nat Comms 2026 批判）**。

## 十、C1 判定（2026-10-20 14:00 轮）

**C1（Caucheteux & King 2022，Commun Biol 5:134）判定 = 无法判定（主声称 A：切分未明示；副声称 C：协议不适用）——按检查表 v0.3 第 0 步拆成两条声称**

- 声称形态：**回归预测类**（linear ℓ2 编码 → brain score = 预测-真实 Pearson R），与 S1 同类、可直接套 P1–P5。核心参考值：compositional embedding fMRI R=0.048 / MEG R=0.041，逼近（MEG 超）shared response noise ceiling（fMRI 0.060 / MEG 0.020）。
- 数据：fMRI+MEG，102 被试，400 孤立荷兰语句（非叙事，作者自注），2×1h；模型 36 架构×100 步=3600 网络，逐词逐层 embedding（非多层 pooling）。
- **切分**：五折 CV 在 held-out 句子上，但**未说明 contiguous 还是 shuffled**。孤立句 + 非叙事降低了（但未消除）时序自相关：BOLD 慢响应使相邻句血液动力学信号重叠，shuffled 会把相邻句分进 train/test 造成泄漏；句内词位置与句间停顿也有结构。→ 需 contiguous 重跑才能判「仍成立」。
- **混淆控制（正面例子，较全）**：permutation baseline（Suppl Fig 3）、随机（未训练）网络对照（随机 embedding 也显著 fMRI R=0.019 / MEG R=0.018，作者明确说「transformers 部分映射独立于语言能力」）、视觉 VGG 基线、lexical vs compositional 差分、permutation feature importance 分离「语言能力 vs 训练量 vs 架构 vs 层位置」、作者自认视觉/词位置混淆（句首句尾 + 屏幕闪烁 + 停顿）并说明其增益很小。
- **缺什么（验收标准要求写明）**：①五折切分 contiguous/shuffled 未明示，需确认或同口径 contiguous 重跑；②noise ceiling 用 shared response model（跨被试），与本管线「留出被试 + 外推」口径不同，归一化保留率需同口径重算；③2023-04-11 有 author correction（s42003-023-04776-4），需核对是否影响主结论。

**本轮判断被改变 1 条**：「需再补 1 篇」的候选默认指向 SAE 2605.23035 等新论文（原候选清单未列 Caucheteux & King）；本轮改为选 canonical 2022 回归类声称——因为它与 S1 同类可直接套 P1–P5、数据模态互补（fMRI 孤立句 vs ECoG 叙事）、且控制较全可当「做了控制」的正面参照，比另起一个 SAE 分支更快凑满且更贴合审计目的。

## 十一、五篇全「无法判定」的说明 + 各篇「只差什么」（2026-10-20 14:00 · 首轮评审 fail 后补）

**为何 5/5 全「无法判定」**：这是 P1（主切分必须 contiguous 且明示）在 5 篇上全不满足、或声称形态不匹配回归 Δr 口径的直接后果，不是审不出来。分两类：

- **切分欠报/未满足（2 篇）**：S1（shuffled，被 Nat Comms 2026 批判；本管线 contiguous 重跑 Δr=0.049 不确定）、C1 主声称（5 折未明示）。
- **声称形态不匹配回归 Δr 口径（3 篇）**：I1（RSA 几何关联）、E1（size↔编码性能 scaling 曲线，虽 contiguous 已核）、T1（lag-layer 相关，另带切分未明示）。

**各篇「只差什么」就能给「仍成立/不成立」**：

| 篇 | 只差 |
|---|---|
| S1 | ①取最优层（argmax）重算 ✅（2026-10-21，contiguous L21=0.115 / shuffled L48=0.412，切分 ×3.6 主因）——只作描述性对照对齐原文口径、拆分切分/层选择归因，**不当判定行**（检查表 v0.3 §3.6①）；②同口径（岭）重算 contiguous ceiling → 归一化保留率；或补样本量/测试块让 Δ CI 跨过 0 |
| E1 | 把 scaling 声称改写为可审口径（如选 SMALL↔XL 的 Δr 并配 SESOI），协议即可直接判定 |
| I1 | 协议补 RSA 分支（RDM 几何相关 + 噪声上限 + 置换 null），或标注单独立项 |
| T1 | （C 类，协议暂不适用）；若按 C 立项：一句 contiguous 声明 + 本管线 contiguous 重跑 lag-layer 相关 |
| C1 | 一句 contiguous 声明 + 同口径 contiguous 重跑 |

## 十二、终审 PASS + 三残留收尾（2026-10-20）

- **终审 PASS**（独立评审，2026-10-20）。判据：5/5 篇表格要素齐全（数据/模型/切分/pooling＋理由＋无法判定写明缺什么）。
- 三残留已收（v1.0）：①§七 line74 归因措辞改为「切分＋层选择叠加、未拆分，argmax 重算后再定量」，消掉与 line72 的自相矛盾；②层选择泄漏审计点不再挂 Nat Comms 2026 名下（三项指控无此项）；v0.3 再撤回「原文未提、我方补入」——Schrimpf Fig S2 B/C 确有选层对照，改判 L-b（无后果）；③I1 交叉验证 layer selection（L-a）作对照 + S1 判定格注「本管线口径=last，argmax 重算只作描述性对照、非终局」。
- 后续（不阻塞）：S1 argmax 重算 ✅（已完成，切分 ×3.6 主因）+ 同口径（岭）contiguous ceiling → 归一化保留率；协议检查表 `docs/checklist.md`（I1/E1/C1 已构成「RSA/scaling/回归」三类声称口径证据）。