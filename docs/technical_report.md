# LLM–脑「对齐」声称的统一稳健性审计：技术报告 v0.1

**作者** Nin ｜ **许可** CC-BY 4.0（代码 MIT）｜ **日期** 2026-10 ｜ **仓库** 见本文件所在目录的 README.md

> 本报告是「LLM × 认知科学 × 脑」项目 D1（对齐研究稳健性审计）+ D7（产学研接口地图）的可引用技术报告。每个数字都可追溯到三处来源：判定表（`docs/judgement_table.md`）、协议检查表（`docs/checklist.md`）、复现管线的 `manifest.json`。凡未在这三处落盘的，本报告不作声称。
>
> 文中出现的 commit 号（如 `9c49cf0`）指公开前的内部开发历史，不在本仓库里；`results/` 下的运行输出不随仓库发布，按 README 的命令复跑生成。

---

## 摘要

2026 年对齐研究经历了一次方法论清算（Nature Comms 2026 证明相当一部分「LLM 与脑对齐」结论来自 shuffled split、pooling 口径与未控制的混淆），而基础模型在真实脑任务上只边际胜出（Meta FAIR NeuralBench，arXiv 2605.08495：36 任务 / 94 数据集 / 9,478 被试）。本项目不预设「LLM 与人脑相似」，而是把「用脑数据检验对齐声称」本身做成一门可复现的审计：**先定协议（可执行代码），再逐篇判定已发表结论是否仍成立**。

本报告交付三样硬产出：

1. **统一审计协议**（`docs/checklist.md` v0.3，与 `src/d1/judge.py` 同源）——第三方评审者仅凭它就能对一篇对齐论文给出 仍成立 / 不成立 / 无法判定，全程不联系作者。
2. **五篇已发表对齐声称的判定表**（`docs/judgement_table.md` v1.0）——S1、E1、I1、T1、C1，**5/5 全部「无法判定」**，并逐篇写明「只差什么」就能定论。
3. **一条命令可复现的管线**（`src/d1/`）——无 GPU、全 CPU，端到端重跑 Schrimpf 系复现。

**最关键的方法学发现**：把 Schrimpf et al. 2021 的「归一化预测力 ≈ 100%」放到 contiguous（时间连续留出）切分下重跑，r_LLM 从原文 0.31 掉到 **0.105**（约 1/3）；进一步用同口径岭噪声上限重算，归一化保留率 ≈ **0.81**，而 shuffled 切分下保留率 > 1、OASM > 1，构成段落泄漏的直接证据——「其他被试的脑」在 shuffled 切分下不是真上限。

---

## ① 统一审计协议（摘要；全文见 `docs/checklist.md`）

协议从 `src/d1/judge.py` 的 `decide()` 与 `src/d1/config.py` 预注册转录而来，与代码同源；协议常量与 P1–P5 不得擅改。

### 声称分类（第 0 步决策树）

审计对象是**一条声称**（claim），不是一篇论文。按 Q0.0 → Q0.3 顺序问，答案落到某类即停：

- **Q0.0** 非经验性（综述 / Perspective / 只引用别人数字）→ **X**，移出。
- **Q0.1** 不是「拟合映射 + 留出评估」的几何比较（RSA / RDM 相关 / CKA）→ **B**。
- **Q0.2** 解码方向 / 分类精度 → **D**。
- **Q0.3** 分数随模型侧变量（size/层/训练）或脑侧变量（时滞/ROI）变化 → **C**；否则（单一模型相对基线的分数多高）→ **A**。

**A 是代码的原生路径**（回归 Δr 口径）；B/C/D 尚未进代码，判定 = 「无法判定（协议不适用）」，只在检查表给文字规则。

### 五个硬前提（P1–P5，任一不过 → 无法判定）

| 前提 | 内容 | 代码对应 |
|---|---|---|
| **P1** | 主切分必须 **contiguous**（时间连续留出）；shuffled 只作对照、**永不决定判定** | `splits.py` |
| **P2** | 显著性来自**循环移位 null**（i.i.d./permutation 被 `validate()` 直接拒绝——破坏 BOLD 自相关会虚报显著性） | `null_test.method` |
| **P3** | 零输入消融通过：保留比例 \|r(零输入)/r(真实输入)\| < 0.10（不是 \|r\| < 0.10） | `judge.max_zero_input_retained` |
| **P4** | 测试块数 ≥ 10；**block = 段落/故事**，数的是全数据集不同 block 数（S1=96 段） | `judge.min_test_blocks` |
| **P5** | null 能分辨 α | — |

### 判定规则（Δr = r(LLM+基线) − r(基线)）

- **仍成立**：混淆已控制 + Δ 的 CI 下界 > 0 + p < α。
- **不成立**：Δ 的 CI 上界 < SESOI，或保留率 CI 上界 < min_retention，或 LLM 单独 r < SESOI。
- **无法判定**：其余；未控制混淆最多只能判「不成立」。

### 协议常量（预注册，勿擅改）

α = 0.05 ｜ SESOI(Δr) = 0.02 ｜ min_test_blocks = 10 ｜ 零输入保留上限 = 0.10 ｜ **min_retention = null**（只报告、不进判定）。「混淆已控制」显式定义 = position + word_rate 基线进了联合模型；OASM、词频、未训练模型只作诊断项。reference_value 固定取原文的原始分数。

---

## ② 五篇判定（全文见 `docs/judgement_table.md`）

| claim | 论文 | 声称 | 数据 / 模型 | 切分 | 判定 |
|---|---|---|---|---|---|
| **S1** | Schrimpf et al. 2021，PNAS 118(45) | GPT2-xl 达噪声上限（≈100% 归一化预测力） | Pereira2018 / GPT2-xl | shuffled（原文），被 Nat Comms 2026 批判 | **无法判定**（切分欠报 + Δ CI 跨 0） |
| **E1** | Hong et al. 2026，eLife 13:RP101204 | 模型越大越能预测神经活动（scaling 曲线） | ECoG 10 患者 / GPT-2…Llama-2 82M–70B | ✅ 原文明写 9/10 contiguous + 留出段（正面例子） | **无法判定**（协议不适用：scaling 曲线非单一 Δr） |
| **I1** | Ismayilzada et al. 2026，arXiv 2604.03480 | LLM 与创造性思维「对齐」（r=0.55/0.52） | fMRI 170 被试 / 24 LLM 270M–72B | RSA 几何关联，无 train/test 切分 | **无法判定**（协议不适用：RSA） |
| **T1** | Nat Commun s41467-025-65518-0 | 层序与脑时间动态对齐（lag-layer r=0.85） | ECoG 9 患者 / GPT2-XL + Llama-2 | ⚠️ 10 折 CV 未说明 contiguous/shuffled | **无法判定**（C 类二阶关联 + 切分未明示） |
| **C1** | Caucheteux & King 2022，Commun Biol 5:134 | transformer 与脑「部分收敛」（R=0.048/0.041） | fMRI+MEG 102 被试 / 36 架构 | ⚠️ 五折 CV 未说明 contiguous/shuffled | **无法判定**（主声称 A 切分未明示；副声称 C 不适用） |

**为何 5/5 全「无法判定」**：这是 P1（主切分必须 contiguous 且明示）在五篇上全不满足、或声称形态不匹配回归 Δr 口径的直接后果，不是审不出来。分两类：

- **切分欠报 / 未满足（2 篇）**：S1（shuffled）、C1 主声称（5 折未明示）。
- **声称形态不匹配回归 Δr 口径（3 篇）**：I1（RSA）、E1（scaling 曲线，虽 contiguous 已核）、T1（lag-layer 相关，另带切分未明示）。

**各篇「只差什么」就能定论**：

| 篇 | 只差 |
|---|---|
| S1 | 同口径（岭）重算 contiguous ceiling → 归一化保留率（已完成，≈0.81）；或补样本量/测试块让 Δ CI 跨过 0 |
| E1 | 把 scaling 声称改写为可审口径（如选 SMALL↔XL 的 Δr 配 SESOI） |
| I1 | 协议补 RSA 分支（RDM 几何相关 + 噪声上限 + 置换 null），或标注单独立项 |
| T1 | 一句 contiguous 声明 + 本管线 contiguous 重跑 lag-layer 相关 |
| C1 | 一句 contiguous 声明 + 同口径 contiguous 重跑 |

---

## ③ 关键发现

### 3.1 S1 端到端重跑（GPT2-XL 全量，CPU 86 分钟）

contiguous 主切分、last pooling、基线 position+word_rate、9 被试 / 384 句 / 96 段：

| 指标 | 值 |
|---|---|
| Δr = r(LLM+基线) − r(基线) | **0.049**，95% CI [−0.004, 0.098]，p_null(Δ)=0.0432 |
| r_LLM | **0.105**，CI [0.049, 0.165]，p_null=0.0108 |
| r_基线 / r_joint | 0.062 / 0.111 |
| null | 循环移位 ×369（min_shift=8） |
| 零输入保留 | 0.00（通过） |
| shuffled 对照 | r_LLM=**0.409**（主切分的 ×3.91）；OASM 0.000→0.429 |

**0.31 → 0.105 的归因**（argmax 口径对齐重算）：原文报告的是「逐层扫描取最优层」，本管线跑 last 末层——两个口径同时变了。用 layer_curve 逐层曲线取 argmax 重算后：contiguous 最优层 **L21 = 0.115**、shuffled 最优层 **L48 = 0.412**。**切分（contiguous→shuffled）在同一粒度下把 r 抬 ×3.58，是主因；层选择口径差约 +0.011，次要**。注意（外部效度限定）：×3.58 / ×3.91 是「同一条构造 passage_order 上 contiguous vs shuffled」的**内部切分效应**量化，不是「Schrimpf 换真实 contiguous 会掉多少」的估计——Pereira2018 真实段间顺序不可恢复，本管线的 contiguous 是对真 contiguous 的近似。

### 3.2 同口径岭噪声上限与归一化保留率

判定规则本身不用 ceiling（min_retention=null），但归一化保留率是 S1 原始声称（「≈100% 预测力」）的直接对应物。旧 ceiling 用最小范数 OLS + 体素中位数，与 r_LLM 的岭估计器 / 均值聚合口径不一致，会出现「r_LLM 高于 ceiling」的假悖论。用与 r_LLM 同一套岭估计器、同一折 / 内折 / α / 计分 / 聚合重算，并按模拟校准后：

- **contiguous 同口径 ceiling ≈ 0.130**，**归一化保留率 ≈ 0.81**（硬区间 [0.44, 0.99]；只计 r_LLM 抽样误差的 CI [0.38, 1.28]）。
- **shuffled 下各口径保留率 1.17–1.22、OASM（无语言）也 > 1**——「其他被试的脑」不是真上限，段落泄漏让段落身份码学到目标被试自己的段落级响应，该比值只能当**泄漏证据**。

### 3.3 切分改变噪声上限本身

换 contiguous 切分后，噪声上限自身下降约 **70%**：shuffled 5 折 ceiling 0.3225 → contiguous 8 折 OLS 0.0951（同口径岭 0.1080、模拟校准后 ≈0.130，仍远低于 shuffled 的 0.32，排除 OLS 无正则所致）。这意味着「S1 归一化预测力 ≈ 100%」不能拿 contiguous 下的 r 去除 0.32 那个 shuffled ceiling。

### 3.4 三条不推翻判定、但必须写明的限定

1. S1 的「无法判定」不因保留率 ≈ 0.81 而改变：judge 不用保留率；且 Δ 的 CI 跨 0（效应不确定）。
2. 位置+语速基线在 contiguous 下的保留率为 0.58（r_conf 0.0624 ÷ ceiling 0.1080），LLM 增量 Δ 的保留率仅 0.45（CI [−0.04, 0.91]）——上限本身很低。
3. E1 是「显式 contiguous 留出」的反例（此前「时序对齐声称系统性欠报切分」的规律不成立）；欠报的只剩 T1、C1、S1。

---

## ④ D7 产学研接口地图摘要（全文未随本仓库公开）

D7 回答「D1 的判定表 / 检查表该先给谁看、最后谁付钱」。首批 **26 家**，按三类角色分：

- **角色① 有数据（8 家）**：OpenNeuro、DANDI Archive、Allen Institute、Brain-Score（S1 的直接裁判对象，接触价值最高）、Meta AI、HCP、UK Biobank、Fedorenko 实验室（MIT）。真实需求 = 数据复用与复现率，防止 shuffled split 类方法学错误污染下游结论。
- **角色② 有通道（10 家）**：Neuralink、Synchron、Precision Neuroscience、Paradromics、Blackrock、Science Corporation、INBRAIN、Axoft、Newronika、Neuracle（中国）。其「神经解码优越性 / 脑对齐」营销声称是 D1 检查表最直接的可商业化对象——监管申报与市场宣传都要求方法可复现、无混淆。
- **角色③ 会付钱（8 家）**：DARPA、NIH BRAIN、CZI、Templeton（直接命中「LLM 是否与人脑对齐」类断言）、Simons、AMED（日本）、战略/产业资本（Tether→Blackrock、Bezos→Synchron 等）、Flourish（Bezos + Google Ventures）。

**7 条具体到入口的接触路径**：Brain-Score→GitHub + Martin Schrimpf（EPFL）；Neuralink→patient-registry + clinicaltrials.gov「PRIME」；Synchron→「COMMAND」；OpenNeuro/DANDI 门户；UK Biobank 申请门户；AMED 公募入口；Templeton/CZI/Simons 资助 RFA。

**商业化起点**（两层）：①出资方需要独立方法学尽调（决定投不投）；②临床公司本身愿意为「证明自己的声称可复现」付费。

---

## ⑤ 可复现性声明

### 一条命令复跑（无 GPU，全 CPU）

```bash
bash scripts/00_setup_env.sh                                   # 建 venv（data/venv）并装依赖，结尾自检
data/venv/bin/python scripts/run_d1.py --smoke                 # 合成数据自检，~15s
data/venv/bin/python scripts/check_pereira2018.py --download   # 下载 Pereira2018（约 36 MB）并校验
data/venv/bin/python scripts/run_d1.py --config configs/pereira2018_schrimpf2021.yaml   # 真实数据 + GPT2-XL（CPU 约 86 分钟）
```

环境自证：`scripts/00_setup_env.sh`（幂等）＋ `--verify` 自检 torch/nibabel/numpy/scipy/sklearn 可 import，venv 落在 `data/venv`（`data/` 可以是指向大容量盘的软链）。单测 `python -m unittest discover -s tests`，28/28。

### 数据与溯源

- 数据 = Brain-Score `Pereira2018.language` assembly（即 Schrimpf 2021 实际打分那份；逐值核对 Schrimpf 本人 S3 原文件最大差 0.0）。来源 URL、sha1（`f8434b4022f5b2c862f0ff2854d5b3f5f2a7fb96`，已验证）记录在 manifest。原始出处 = Pereira et al. 2018，Nat Commun 9:963（OSF crwz7）。
- 规模：9 被试 / 384 句 / 96 段 / 24 话题 / 12155 体素。

### 判定可追溯

每次运行的 `manifest.json` 全量记录 run_id、config_sha256、git_commit、split_kind（primary/control）、pooling、baselines_enabled、dataset_meta、逐层曲线、判定 decision（verdict / reasons / missing / retention）。判定行另有 `rows.jsonl` / `rows.tsv`（schema `d1-row/0.1`）与 `verdict.txt`。

### 关键限制（如实声明）

1. **Pereira2018 的段间呈现顺序在任何公开版本都不存在**（原文写明段落分 run、顺序按被试×session 随机）。管线的「contiguous」用显式构造的 `passage_order`（默认 `topic_interleaved`），是对真 contiguous 的近似，不是原始实验时间线。
2. 用 9 个被试做外推，ceiling 的模型不确定性约 15–20%，比 r_LLM 自身约 ±50% 的抽样误差小。
3. 层选择的泄漏审计点：Schrimpf 实际做了选层对照（Fig S2 B/C），归 **L-b**（test-argmax + 原文对照，无后果）；I1 用交叉验证 layer selection（L-a）是正例。

---

## 附录：产出文件索引

- `docs/checklist.md` —— 《alignment 研究协议检查表》v0.3（与 `src/d1/judge.py` 同源）
- `docs/judgement_table.md` —— D1 判定表 · claim 注册表 v1.0（5/5 无法判定 + 各自缺什么）
- `src/d1/` —— 复现管线（10 模块 + 入口 `scripts/run_d1.py`）
- `results/d1/<run_id>/` —— S1 端到端运行（manifest / verdict / rows / layer_curve；不随仓库发布，复跑生成）
- `results/d1/ceiling_ridge__<run_id>/` —— 同口径岭噪声上限与保留率重算（summary.md / aggregation/summary.md；由 `scripts/ceiling_*_pereira2018.py` 生成）
