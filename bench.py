"""The checks behind ./verify.sh. Runs inside the sandbox image against whichever mini-sglang is on PYTHONPATH.

usage: bench.py run <out dir> [test...]         each test in its own process, into <out dir>/<test>.json
       bench.py <test> <out file>                one test
       bench.py compare <original dir> <engine dir> [test...]
"""
import dataclasses
import json
import os
import statistics
import subprocess
import sys
import time
from random import randint, seed

MODEL = "/models/Qwen3-0.6B"
MEMORY_RATIO = 0.25  # the GPU is shared with SGLang
TESTS = ["check", "latency", "offline", "prefix", "long_prompt"]
TOLERANCE = 0.05
NEAR_MISS = 0.15  # a timing that fails by less than this may be noise from sharing the GPU, so say so
METRICS = [  # test, key, label, higher is better
    ("latency", "decode_bs1_ms", "decode step, batch 1 (ms)", False),
    ("latency", "decode_bs8_ms", "decode step, batch 8 (ms)", False),
    ("offline", "offline_tok_s", "offline throughput (tok/s)", True),
    ("prefix", "prefix_s", "64 prompts, shared prefix (s)", False),
    ("long_prompt", "long_prompt_peak_mb", "16k-token prompt, peak memory (MB)", False),
]


def make_llm(**kwargs):
    from minisgl.llm import LLM
    from minisgl.scheduler import SchedulerConfig

    # the automatic attention backend hangs on sm120; a broken engine may lack some settings, so drop those
    kwargs = dict(attention_backend="fi", memory_ratio=MEMORY_RATIO, **kwargs)
    fields = {f.name for f in dataclasses.fields(SchedulerConfig)}
    llm = LLM(MODEL, **{k: v for k, v in kwargs.items() if k in fields})
    llm.generate(["Benchmark: "], sampling(temperature=0.1))  # warm up
    return llm


def sampling(**kwargs):
    from minisgl.core import SamplingParams

    return SamplingParams(**kwargs)


def timed(fn) -> float:
    t = time.perf_counter()
    fn()
    return time.perf_counter() - t


def check():
    """Greedy outputs, twice (the second pass with a warm prefix cache). Prompts share a long prefix and one is
    longer than the 512-token prefill budget, so every optimization is exercised."""
    llm = make_llm(max_extend_tokens=512, max_seq_len_override=4096)
    shared = "You are a careful assistant who answers briefly and precisely. " * 30
    prompts = [shared + q for q in ["What is 2+2?", "Name three colors.", "What does a GPU do?"]]
    prompts.append("Write a haiku about GPUs.")
    prompts.append("The quick brown fox jumps over the lazy dog. " * 150 + "How many times was the fox named?")
    sp = sampling(temperature=0.0, max_tokens=48)
    return {"outputs": [[o["token_ids"] for o in llm.generate(prompts, sp)] for _ in range(2)]}


def latency():
    """Small-batch decode: 128-token prompts, 256 output tokens, median of 3."""
    llm = make_llm(max_seq_len_override=4096)
    seed(0)
    out = {}
    for bs in [1, 8]:
        prompts = [[randint(0, 10000) for _ in range(128)] for _ in range(bs)]
        sp = sampling(temperature=0.0, ignore_eos=True, max_tokens=256)
        t = statistics.median(timed(lambda: llm.generate(prompts, sp)) for _ in range(3))
        out[f"decode_bs{bs}_ms"] = 1000 * t / 256
    return out


def offline():
    """mini-sglang's README offline benchmark: 256 requests, 100-1024 input and output tokens."""
    llm = make_llm(max_seq_len_override=4096, max_extend_tokens=16384, cuda_graph_max_bs=256, page_size=256)
    seed(0)
    prompts = [[randint(0, 10000) for _ in range(randint(100, 1024))] for _ in range(256)]
    sps = [sampling(temperature=0.6, ignore_eos=True, max_tokens=randint(100, 1024)) for _ in range(256)]
    return {"offline_tok_s": sum(sp.max_tokens for sp in sps) / timed(lambda: llm.generate(prompts, sps))}


def prefix():
    """64 requests sharing a 4,000-token prefix, each with its own 32-token question; median of 3."""
    llm = make_llm(max_seq_len_override=8192)
    seed(0)
    sp = sampling(temperature=0.0, ignore_eos=True, max_tokens=16)
    times = []
    for _ in range(3):
        shared = [randint(0, 10000) for _ in range(4000)]  # a new prefix every round, so none starts warm
        prompts = [shared + [randint(0, 10000) for _ in range(32)] for _ in range(64)]
        times.append(timed(lambda: llm.generate(prompts, sp)))
    return {"prefix_s": statistics.median(times)}


def long_prompt():
    """Peak memory while prefilling one 16,000-token prompt, twice the default prefill budget. The KV cache is
    allocated up front, so this is the memory of the forward passes alone."""
    import torch

    llm = make_llm(max_seq_len_override=16384)
    seed(0)
    prompt = [randint(0, 10000) for _ in range(16000)]
    torch.cuda.synchronize()
    base = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    sp = sampling(temperature=0.0, ignore_eos=True, max_tokens=1)
    out = []
    t = timed(lambda: out.extend(llm.generate([prompt], sp)))
    # mini-sglang drops, with only a warning, a prompt longer than its KV cache
    if len(out[0]["token_ids"]) != 1:
        sys.exit("the prompt was dropped: the KV cache is too small; free more GPU memory (is SGLang using it all?)")
    return {"long_prompt_peak_mb": (torch.cuda.max_memory_allocated() - base) / 2**20, "long_prompt_s": t}


def run(out_dir: str, *tests: str) -> int:
    failed = 0
    for test in tests or TESTS:
        t = time.time()
        with open(f"{out_dir}/{test}.log", "w") as log:
            try:
                cmd = [sys.executable, __file__, test, f"{out_dir}/{test}.json"]
                code = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, timeout=900).returncode
            except subprocess.TimeoutExpired:
                code = "timeout"
        status = "ok" if code == 0 else f"FAILED ({code}), see {test}.log"
        print(f"  {test:12s} {status} ({time.time() - t:.0f}s)", flush=True)
        failed += code != 0
    return 1 if failed else 0


def load(results_dir: str) -> dict:
    out = {}
    for test in TESTS:
        try:
            with open(f"{results_dir}/{test}.json") as f:
                out.update(json.load(f))
        except (OSError, ValueError):
            pass
    return out


def compare(original_dir: str, engine_dir: str, *tests: str) -> int:
    tests = tests or TESTS
    a, b = load(original_dir), load(engine_dir)
    name = os.path.basename(os.path.normpath(engine_dir))
    problems = []
    near_miss = False
    print(f"{'':36s} {'original':>10s} {name:>10s}")
    if "check" in tests and ("outputs" not in a or "outputs" not in b):
        problems.append("the output check did not run")
    elif "check" in tests:
        ref = [ids for p in a["outputs"] for ids in p]
        same = sum(x == y for x, y in zip(ref, (ids for p in b["outputs"] for ids in p)))
        print(f"{'greedy outputs identical':36s} {'':>10s} {f'{same}/{len(ref)}':>10s}")
        if same < len(ref):
            problems.append("greedy outputs differ from the original")
    for test, key, label, higher in METRICS:
        if test not in tests:
            continue
        if not a.get(key) or key not in b:
            print(f"{label:36s} {a.get(key, '-'):>10} {b.get(key, '-'):>10}")
            problems.append(f"{label} did not run")
            continue
        ratio = b[key] / a[key]
        worse = ratio < 1 - TOLERANCE if higher else ratio > 1 + TOLERANCE
        print(f"{label:36s} {a[key]:10.2f} {b[key]:10.2f} {ratio:6.2f}x{'  <- worse' if worse else ''}")
        if worse:
            problems.append(f"{label} at {ratio:.2f}x")
            near_miss = near_miss or (test != "long_prompt" and abs(ratio - 1) < NEAR_MISS)
    print()
    if problems:
        print(f"NOT RESTORED: {'; '.join(problems)}")
        if near_miss:
            print(
                "A timing is only slightly behind. If you believe the engine is repaired, this can be noise: the GPU "
                "is shared with SGLang. Make sure your agent is idle and run ./verify.sh again; if the original was "
                "measured while the agent was busy, delete results/original first."
            )
        return 1
    print(f"RESTORED: {name} gives the original's outputs and is within {TOLERANCE:.0%} of it on every benchmark")
    return 0


if __name__ == "__main__":
    cmd, *args = sys.argv[1:]
    if cmd in ("run", "compare"):
        sys.exit(globals()[cmd](*args))
    with open(args[0], "w") as f:
        json.dump(globals()[cmd](), f, indent=1)
