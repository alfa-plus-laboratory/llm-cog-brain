#!/usr/bin/env bash
# =============================================================================
# scripts/00_setup_env.sh — llm-cog-brain 的 Python 环境：搭建（幂等）+ 自检
#
# 复跑命令（在仓库根目录执行；三步都应退 0）：
#   bash scripts/00_setup_env.sh && bash scripts/00_setup_env.sh && bash scripts/00_setup_env.sh --verify
#
# 用法：
#   bash scripts/00_setup_env.sh            建/补 venv、装依赖；结尾自动跑一遍 --verify
#   bash scripts/00_setup_env.sh --verify   只检查、不安装、不写盘：torch/nibabel/numpy/scipy/sklearn
#                                           能否从 venv 里 import。全过退 0，否则退 1
#                                           并逐项打印 PASS/FAIL。未安装时它失败是预期的。
#   日常使用：source data/venv/bin/activate   或直接用 data/venv/bin/python
#
# 前提：
#   - Python 3.12（依赖按 cp312 钉版本）。默认用 PATH 里的 python3，可用 PYTHON=/path/to/python3.12 指定。
#   - venv、pip 缓存、模型权重全部放在 data/ 下，装好后约 6–10 GB（估计，未实测），GPT2-XL 权重另需约 6 GB。
#     仓库所在的盘不够大时，先把 data/ 建成指向大容量盘的软链，再跑本脚本。
#   - D1 管线只用 CPU，不需要 GPU。
# =============================================================================
set -euo pipefail

log() { printf '[setup_env] %s\n' "$*"; }
die() { printf '[setup_env] 错误：%s\n' "$*" >&2; exit 1; }
usage() { echo "用法：bash scripts/00_setup_env.sh [--verify]（详见脚本头部注释）"; }

# ---- 路径 ---------------------------------------------------------------------
SCRIPT_PATH="$(readlink -f -- "${BASH_SOURCE[0]}")"
WS="$(dirname -- "$(dirname -- "$SCRIPT_PATH")")"     # 工作区根 = scripts/ 的上一级
DATA_LINK="$WS/data"
# 用解析后的真实路径，不用 data/ 软链路径：venv 会把绝对路径写进 pyvenv.cfg、
# activate 和每个 console script 的 shebang；用真实路径就不依赖工作区那条软链。
DATA="$(readlink -f -- "$DATA_LINK")" || die "解析不了 $DATA_LINK"
SYS_PY="${PYTHON:-python3}"

# ---- venv 放哪：固定为 data/venv --------------------------------------------------
# torch 2.14.1 会连带拉 CUDA 13 一整套（cuda-toolkit、cudnn、nccl、nvshmem、cusparselt、triton），
# 再加 transformers/scipy/pandas 等，体积大；和数据、缓存放在同一个 data/ 下，换盘只需要换一条软链。
# 不做成可配置项：多一个开关只会多一种跑出不同结果的方式。
VENV="$DATA/venv"
PY="$VENV/bin/python"

# ---- 缓存 / 临时目录：全部指到 data/ 底下 ------------------------------------------------
# 安装期：PIP_CACHE_DIR（torch 那套 wheel 有几 GB）、TMPDIR（pip 先把 wheel 下到临时目录再解包）。
#   不重定向的话会落到 ~/.cache 和系统临时目录。
# 运行期：HF_HOME（transformers 模型权重，几 GB 到几十 GB）、XDG_CACHE_HOME（torch hub 等的默认缓存根）、
#   TRITON_CACHE_DIR（triton 默认用 ~/.triton，不跟 XDG）、NILEARN_DATA（默认 ~/nilearn_data）。
#   这些由 write_runtime_env 固化进 venv（见下），不靠使用者记得 export。
# 故意不碰任何 CUDA_* / LD_LIBRARY_PATH / CUDA_VISIBLE_DEVICES：这类设置留给具体实验自己决定。
CACHE="$DATA/cache"
export PIP_CACHE_DIR="$CACHE/pip"
export XDG_CACHE_HOME="$CACHE/xdg"
export HF_HOME="$CACHE/huggingface"
export TRITON_CACHE_DIR="$CACHE/triton"
export NILEARN_DATA="$DATA/nilearn_data"
export TMPDIR="$DATA/tmp"
RUNTIME_ENV_KEYS=(PIP_CACHE_DIR XDG_CACHE_HOME HF_HOME TRITON_CACHE_DIR NILEARN_DATA TMPDIR)

# pip 行为：不做自身版本检查（少一次联网，结果不随时间变）。
export PIP_DISABLE_PIP_VERSION_CHECK=1

# ---- 依赖清单 -----------------------------------------------------------------------
# 全部显式安装，不假设环境里已有任何包。
# 版本：精确钉为 2026-10-16 PyPI 上看到的最新版，这样两次跑、换人跑结果一致。
#   升级 = 改这里的版本号再跑一遍（stamp 会变，触发重装）。
# torch 用 PyPI 默认 wheel（依赖 CUDA 13.0 那套 nvidia 库，全部作为 pip 包装进 venv，不碰系统 CUDA 和驱动）。
#   不额外挂 download.pytorch.org 的 index，免得两个源混装。D1 管线只用 CPU，有没有 GPU 都能跑。
# 不装 pycortex / datalad：它们要用系统包管理器装依赖（git-annex 等），本项目用不到。
PIP_PIN="pip==26.2.1"
REQS=(
  torch==2.14.1
  numpy==2.5.4
  scipy==1.18.1
  scikit-learn==1.9.1
  pandas==3.0.6
  nibabel==5.4.2
  nilearn==0.14.1
  transformers==5.19.0
  h5py==3.16.0
  himalaya==0.4.11
  tqdm==4.70.1
  pyyaml==6.0.3
)
VERIFY_MODULES=(torch nibabel numpy scipy sklearn)

# --verify 里每个模块的检查：能 import，而且确实是从这个 venv 里 import 的（不是系统的 numpy 1.26.4 之类）。
IMPORT_CHECK_PY='
import importlib, os, sys
name = sys.argv[1]
if sys.prefix == sys.base_prefix:
    sys.exit("解释器不在 venv 里")
mod = importlib.import_module(name)
path = os.path.realpath(mod.__file__)
if not path.startswith(os.path.realpath(sys.prefix) + os.sep):
    sys.exit(f"{name} 不是从 venv 导入的：{path}")
print(getattr(mod, "__version__", "?"), path)
'

VERIFY_FAILS=0
vpass() { printf '[verify] PASS  %s\n' "$*"; }
vfail() { printf '[verify] FAIL  %s\n' "$*"; VERIFY_FAILS=$((VERIFY_FAILS + 1)); }

# 只读：不安装、不写盘（python 用 -B 不写 .pyc）。
verify() {
  local m out
  VERIFY_FAILS=0
  printf '[verify] venv = %s\n' "$VENV"

  # 判据 1：venv 存在
  if [ ! -d "$VENV" ]; then
    vfail "venv 位置：目录不存在（$VENV）——还没装？先跑：bash scripts/00_setup_env.sh"
  else
    vpass "venv 位置：$(readlink -f -- "$VENV")"
  fi

  # 判据 2：各模块能 import。-I：不让 cwd / PYTHONPATH 里的同名目录冒充；-B：不写 .pyc。
  if [ ! -x "$PY" ]; then
    for m in "${VERIFY_MODULES[@]}"; do vfail "import $m：venv 解释器不存在（$PY）"; done
  else
    for m in "${VERIFY_MODULES[@]}"; do
      if out="$("$PY" -I -B -c "$IMPORT_CHECK_PY" "$m" 2>&1)"; then
        vpass "import $m：$out"
      else
        vfail "import $m：$(printf '%s\n' "$out" | tail -n 1)"
      fi
    done
    # 只作参考、不计入判定（D1 管线不用 GPU）。
    # 用 NVML 判断 CUDA 是否可用：不在 GPU 上建 context，不占显存。
    out="$(PYTORCH_NVML_BASED_CUDA_CHECK=1 "$PY" -I -B -c 'import torch; print(f"torch {torch.__version__}, CUDA {torch.version.cuda}, cuda.is_available()={torch.cuda.is_available()}")' 2>&1 | tail -n 1)" || true
    printf '[verify] INFO  %s\n' "$out"
  fi

  if [ "$VERIFY_FAILS" -eq 0 ]; then
    printf '[verify] 结果：全部通过\n'
    return 0
  fi
  printf '[verify] 结果：%d 项失败\n' "$VERIFY_FAILS"
  return 1
}

# 把 RUNTIME_ENV_KEYS 固化进 venv，两条路都要覆盖到：
#   - bin/activate 末尾 source 一个 env 文件：给交互式 `source data/venv/bin/activate` 的人；
#   - site-packages 里放一个 .pth（以 import 开头的行会在解释器启动时执行）：给不 activate、直接调用
#     data/venv/bin/python 的进程（nohup、cron 等）。用 setdefault，调用方显式
#     设过的值优先。不用 sitecustomize.py，因为 Ubuntu 的 stdlib 里已经有一个，会把 venv 里那个遮住。
# 每次都整文件重写（内容只由本脚本决定 → 幂等）；activate 钩子按标记行只加一次。
# 注意：deactivate 不会撤销这些变量。
write_runtime_env() {
  local envf="$VENV/setup_env.runtime.sh"
  local mark="# >>> llm-cog-brain runtime env (scripts/00_setup_env.sh) >>>"
  local site pth k pairs=""
  site="$("$PY" -I -c 'import sysconfig; print(sysconfig.get_path("purelib"))')"
  pth="$site/00-llm-cog-brain-env.pth"
  {
    echo "# 由 scripts/00_setup_env.sh 生成，不要手改"
    for k in "${RUNTIME_ENV_KEYS[@]}"; do printf 'export %s=%q\n' "$k" "${!k}"; done
  } > "$envf"
  for k in "${RUNTIME_ENV_KEYS[@]}"; do pairs+="('$k', '${!k}'), "; done
  printf 'import os; [os.environ.setdefault(k, v) for k, v in (%s)]\n' "$pairs" > "$pth"
  if ! grep -qF -- "$mark" "$VENV/bin/activate"; then
    printf '\n%s\n. %q\n' "$mark" "$envf" >> "$VENV/bin/activate"
  fi
}

setup() {
  # -- 0. 前置检查 --
  mkdir -p -- "$DATA" || die "建不了 $DATA"
  [ -w "$DATA" ] || die "$DATA 不可写"
  command -v -- "$SYS_PY" >/dev/null || die "找不到 $SYS_PY（用 PYTHON=/path/to/python3.12 指定）"
  "$SYS_PY" -I -c 'import sys; sys.exit(sys.version_info[:2] != (3, 12))' \
    || die "系统 python 不是 3.12（依赖版本是按 cp312 钉的）：$("$SYS_PY" -V 2>&1)"

  # -- 互斥：两次 setup 并发时别同时写同一个 venv。阻塞等待而不是报错，这样并发跑也都能退 0 --
  # 没有 flock（util-linux）的系统上跳过互斥，只是不要并发跑。
  if command -v flock >/dev/null; then
    exec 9>"$DATA/.setup_env.lock"
    flock 9
  else
    log "没有 flock，跳过并发互斥"
  fi

  mkdir -p "$PIP_CACHE_DIR" "$XDG_CACHE_HOME" "$HF_HOME" "$TRITON_CACHE_DIR" "$NILEARN_DATA" "$TMPDIR"

  # -- 1. 建 venv --
  # 先用标准的 `python3 -m venv`。有的发行版没装 ensurepip（Debian/Ubuntu 缺 python3-venv），这一步会失败；
  # 这时改用 `venv --without-pip` 建一个不带 pip 的 venv（只用标准库），下一步再用系统 pip 的 `--python`
  # 选项（pip ≥ 22.3）把 pip 装进去。pip 会切到 venv 的解释器里运行，只往 venv 里写；目标是 venv，
  # 所以也不会触发系统的 EXTERNALLY-MANAGED 限制。不用 --system-site-packages：系统里的旧版 numpy/scipy
  # 会漏进 venv，版本来源混乱。
  # 只在解释器不存在时才创建：已存在就不动（幂等），也不会冲掉 activate 里加的钩子。
  if [ ! -x "$PY" ]; then
    log "创建 venv：$VENV"
    if ! "$SYS_PY" -I -m venv "$VENV"; then
      log "python -m venv 失败（多半缺 ensurepip），改用 --without-pip"
      "$SYS_PY" -I -m venv --without-pip "$VENV"
    fi
  else
    log "venv 已存在：$VENV"
  fi

  # -- 2. 给 venv 装 pip（仅当 venv 里还没有 pip）--
  # PYTHONDONTWRITEBYTECODE：这一步是 venv 解释器去 import 系统目录里的 pip 源码，不加的话可能往系统的
  # pip/__pycache__ 写 .pyc（工作区外，不允许）。外层用 -B，内层靠环境变量。
  # --only-binary=:all:：pip 本身只用 wheel 装。
  if ! "$PY" -I -m pip --version >/dev/null 2>&1; then
    log "用系统 pip 的 --python 选项把 $PIP_PIN 装进 venv"
    PYTHONDONTWRITEBYTECODE=1 "$SYS_PY" -I -B -m pip --python "$PY" install --only-binary=:all: "$PIP_PIN" \
      || die "venv 里没有 pip，系统 pip 也装不进去（需要系统 pip ≥ 22.3，或装好 ensurepip 后删掉 $VENV 重跑）"
  fi

  # -- 3. 装依赖（幂等）--
  # stamp = 解释器版本 + 依赖清单的 sha256。stamp 命中、并且 `pip check`（离线）通过 → 跳过 pip install：
  # 第二次跑不联网，也不受 PyPI 当时状态的影响，结果是确定的。清单一改，stamp 对不上 → 重装。
  # 上次装到一半中断：stamp 只在全部成功之后才写，所以下次会接着装完。
  # --prefer-binary 而不是 --only-binary=:all:：重的包（torch/numpy/scipy/h5py/pyyaml…）都有 cp312 wheel；
  # 但我没能逐个核实每个钉住的版本都有 wheel（比如纯 Python 的 himalaya），不希望因为缺一个 wheel 就整体失败。
  # 纯 Python 的 sdist 会在 TMPDIR（data/ 下）的隔离环境里构建。
  local stamp="$VENV/.setup_env.stamp" want avail_gb
  want="$( { "$PY" -I -V; printf '%s\n' "$PIP_PIN" "${REQS[@]}"; } | sha256sum | cut -d' ' -f1 )"
  if [ -f "$stamp" ] && [ "$(cat "$stamp")" = "$want" ] && "$PY" -I -m pip check >/dev/null 2>&1; then
    log "依赖已按当前清单装好（stamp 命中，pip check 通过），跳过安装"
  else
    avail_gb="$(df -P -BG -- "$DATA" | awk 'NR==2 { sub(/G$/, "", $4); print $4 }')"
    [ "$avail_gb" -ge 30 ] || die "data/ 所在的盘只剩 ${avail_gb}G，不到 30G（torch+CUDA 的 venv 加 pip 缓存要十几 GB）"
    log "安装依赖（torch 那套 CUDA 13 wheel 有好几 GB，要等一阵）"
    "$PY" -I -m pip install --prefer-binary --progress-bar off "$PIP_PIN" "${REQS[@]}"
    "$PY" -I -m pip check
    # 实际装上的完整版本（含传递依赖）留档：这就是事实上的 lock 文件
    "$PY" -I -m pip freeze --all > "$VENV/pip-freeze.txt"
    printf '%s\n' "$want" > "$stamp"
  fi

  # -- 4. 运行期的缓存位置固化进 venv --
  write_runtime_env

  # -- 5. 自检：setup 退 0 就意味着 --verify 也会通过 --
  log "自检："
  verify || die "安装流程走完了，但自检没通过（见上面的 FAIL 行）"
  log "完成。使用：source $VENV/bin/activate   （或直接用 $PY）"
}

# ---- 入口 ----------------------------------------------------------------------------
[ $# -le 1 ] || { usage; die "参数太多"; }
case "${1:-}" in
  "")
    setup
    ;;
  --verify)
    if verify; then exit 0; else exit 1; fi
    ;;
  -h|--help)
    usage
    ;;
  *)
    usage
    die "未知参数：$1"
    ;;
esac
