# llm-cog-brain：LLM–脑「对齐」声称的统一稳健性审计

把已发表的「LLM 表征与人脑响应对齐」结论放到同一套预注册协议下重跑：主切分必须是时间连续留出（contiguous），显著性来自保留时间自相关的循环移位 null，位置与语速混淆进联合模型。每条声称输出一行判定：**仍成立 / 不成立 / 无法判定**，并写明判定依据的数字和「还缺什么」。管线只做推理和岭回归，不训练任何东西，也不用 GPU。

## 内容

| 路径 | 内容 |
|---|---|
| `docs/technical_report.md` | 技术报告 v0.1：协议、五篇判定、关键发现、可复现性声明 |
| `docs/checklist.md` | 《alignment 研究协议检查表》v0.3：第三方评审者只凭这张表就能对一篇论文走完判定 |
| `docs/judgement_table.md` | 判定表 v1.0：S1、E1、I1、T1、C1 五篇，5/5「无法判定」，以及各篇还缺什么 |
| `src/d1/` | 审计管线，即协议的可执行版本；判定规则在 `judge.py` |
| `configs/` | `smoke.yaml`（合成数据）、`pereira2018_schrimpf2021.yaml`（S1） |
| `scripts/` | 环境搭建、入口、数据自检、噪声上限重算 |
| `tests/` | 单测 |

**主要结论**（详见技术报告）：Schrimpf et al. 2021 声称 GPT2-xl 在 Pereira2018 上达到噪声上限。换成 contiguous 切分重跑后，r_LLM 从原文的 0.31 降到 0.105。LLM 相对位置 + 语速基线的增量 Δr = 0.049，95% CI [−0.004, 0.098]，判定为无法判定（证据不足）。在同口径的岭噪声上限下，归一化保留率约 0.81。

## 复现

需要 Python 3.12。venv、pip 缓存和模型权重都放在 `data/` 下；仓库所在的盘不够大时，先把 `data/` 建成指向大容量盘的软链。

```bash
bash scripts/00_setup_env.sh                     # 建 data/venv，按钉死的版本装依赖（含 torch，体积大），结尾自检
data/venv/bin/python scripts/run_d1.py --smoke   # 一条命令自检：合成数据，CPU，约 15 秒
```

退出码 0 表示管线跑通、自检全部通过（判定结果本身是什么都算跑通）。最后一行是判定和数字，完整输出在 `tmp/smoke/out/<run_id>/`。smoke 不需要 torch：只装了 numpy 和 PyYAML 的 Python 3.12 也能直接 `python3 scripts/run_d1.py --smoke`。

```bash
data/venv/bin/python -m unittest discover -s tests          # 单测
data/venv/bin/python scripts/run_d1.py --smoke --matrix     # 三个开关的 2×3×2 共 12 种组合 → tmp/smoke/out/matrix.tsv
data/venv/bin/python scripts/check_hf_backend.py            # 用随机初始化的微型 GPT-2 检查 transformers 后端，不联网
```

**三个开关**（在 YAML 里改一行，或用命令行覆盖；命令行覆盖会记进 manifest，这样的运行不当判定行）：

| 开关 | 取值 | 备注 |
|---|---|---|
| `split.kind` | `contiguous` / `shuffled` | contiguous 是主切分；shuffled 只作对照，永远不据此判定 |
| `pooling` | `last` / `mean` / `sum` | 只对当前句的 token 做 pooling，前文只作为上下文输入 |
| `baselines.enabled` | `true` / `false` | 混淆基线：位置、语速；OASM 只作诊断 |

### S1：Pereira2018 + GPT2-XL

```bash
data/venv/bin/python scripts/check_pereira2018.py --download      # 下载约 36 MB 到 data/raw/pereira2018/ 并校验，再跑加载器与噪声上限自检（CPU，约 2 分钟）
data/venv/bin/python scripts/run_d1.py --config configs/pereira2018_schrimpf2021.yaml   # GPT2-XL 全量，CPU 约 86 分钟；首次运行从 Hugging Face 下载约 6 GB 权重
data/venv/bin/python scripts/ceiling_ridge_pereira2018.py --run results/d1/<run_id>    # 同口径岭噪声上限与保留率（CPU，约 20–25 分钟）
```

判定行在 `results/d1/<run_id>/rows.jsonl`（看 `verdict / verdict_reason / missing` 三个字段），一行人读版在 `verdict.txt`，完整溯源在 `manifest.json`。

注意：
- `run_id` 末尾的配置哈希把激活缓存目录的绝对路径也算了进去，所以换一台机器或换一个目录，哈希都会和文档里的不同。
- `manifest.json` 会记录本机的绝对路径。分享运行输出之前，请先自己检查一遍。
- `scripts/ceiling_aggregation_pereira2018.py` 目前写死了原始运行的 `run_id`（脚本里的 `RUN` 常量），复跑前需要改成自己的。

## 数据来源

- `--smoke`：`d1.data.synthetic` 生成的合成数据，形状模仿 Pereira2018（段落 × 句子），真值已知：语义 + 位置 + 语速 + 段落级漂移 + 噪声。同时会跑一遍去掉语义信号的负对照。
- Pereira2018（Pereira et al. 2018, *Nat Commun* 9:963；原始项目在 OSF [`crwz7`](https://osf.io/crwz7/)）：用的是 Brain-Score 的 `Pereira2018.language` assembly（[S3](https://brainscore-storage.s3.amazonaws.com/brainscore-language/assy_Pereira2018_language.nc)，sha1 `f8434b40…`，加载时校验）。它的数值与 Schrimpf 2021 原始 assembly 的语言网络部分完全一致。**段间呈现顺序在任何公开版本里都不存在**，所以段落顺序是显式构造的（`dataset.passage_order`），见 `src/d1/data.py`。它**不是** OpenNeuro ds003020（那是 LeBel et al. 2023 的听故事数据）。

## 模块

`config.py`（开关 + 校验；拒绝 i.i.d. null）· `data.py`（合成数据；Pereira2018 加载器）· `splits.py` · `activations.py`（按 (model, layer, pooling, split) 缓存）· `readout.py`（一次分解扫遍 alpha 的岭回归、嵌套 CV、块 bootstrap）· `baselines.py` · `null.py`（循环移位）· `ceiling.py`（被试池外推噪声上限，OLS 或岭回归；split-half 诊断）· `judge.py`（判定规则 + 行格式）· `pipeline.py`。

## 许可

代码：MIT（见 `LICENSE`）。文档（`docs/`）：CC-BY 4.0。作者：Nin。
