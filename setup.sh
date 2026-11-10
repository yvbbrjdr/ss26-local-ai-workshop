#!/bin/bash
# One-time setup (needs the internet): the SGLang image and Qwen3.8 weights for Step 2, the sandbox image, Qwen3-0.6B
# for the engines, four sabotaged copies of mini-sglang in env1-env4, and the untouched original that ./verify.sh
# compares against. Rerunning keeps existing env directories; delete one (rm -rf env1) to get a fresh copy.
# Output is copied to setup.log, and .setup-done appears once everything is in place.
set -euo pipefail
cd "$(dirname "$0")"

# One setup at a time: the launchable starts one at first boot, and a second run would race it.
exec 9> .setup.lock
flock -n 9 || { echo "setup is already running; follow it with: tail -f $PWD/setup.log" >&2; exit 1; }
rm -f .setup-done
exec > >(tee setup.log) 2>&1
trap '[ -e .setup-done ] || echo "setup FAILED: see the error above, fix it and rerun ./setup.sh"' EXIT

docker pull lmsysorg/sglang:v0.5.22-runtime
docker build -t workshop-env --build-arg UID=$(id -u) --build-arg GID=$(id -g) image
docker network inspect workshop-net > /dev/null 2>&1 || docker network create --internal workshop-net

mkdir -p ~/.cache/huggingface models
docker run --rm -e HF_HOME=/hf -e HF_HUB_OFFLINE=0 -v ~/.cache/huggingface:/hf -v "$PWD/models":/models \
  workshop-env python -c "
from huggingface_hub import snapshot_download
snapshot_download('nvidia/Qwen3.8-27B-NVFP4')
snapshot_download('RadixArk/Qwen3.8-27B-DSpark')
snapshot_download('Qwen/Qwen3-0.6B', local_dir='/models/Qwen3-0.6B')"

SRC=$(mktemp -d)
git clone -q https://github.com/sgl-project/mini-sglang.git $SRC
copy() {  # <dir> [patch]: mini-sglang at commit 9a91cfa with the patch applied, without git history
  [ -d $1 ] && { echo "$1 exists, keeping it"; return; }
  git clone -q $SRC $1
  git -C $1 checkout -q 9a91cfa
  [ -z "${2:-}" ] || git -C $1 apply "$PWD/patches/$2.patch"
  rm -rf $1/.git
}
copy .original
copy env1 chunked-prefill
copy env2 cuda-graph
copy env3 radix
copy env4 overlap
rm -rf $SRC
touch .setup-done
echo "setup done: env1 env2 env3 env4"
