# SGLang Summit Workshop by LMSYS: Local AI Coding Lab

This is the hands-on guide for the [Local AI Coding Lab at SGLang Summit 2026](https://luma.com/yac43hxy). See the event page for the schedule and venue details.

## Lore

You are a Member of Technical Staff at **FastInference AI, Inc**. As the name suggests, your company provides high-performance LLM inference services, which is powered by Mini-SGLang.

One day, a swarm of rogue AI agents launches a coordinated attack on your infrastructure. They tamper with the inference engine's source code across multiple deployments, introducing severe performance regressions. Upon discovering the attack, your team immediately isolates the entire infrastructure from the Internet to contain the damage.

Now, with no Internet access, no cloud AI services, and no clean backups to restore from, you must repair the compromised inference engines yourself.

Fortunately, you have a powerful open-source model, **Qwen3.8-27B**, running locally on an **NVIDIA RTX PRO 6000 Blackwell Edition** GPU through SGLang.

**Your mission: Use your local AI coding agent to identify the sabotaged optimizations, repair the inference engines, and restore production performance.**

_Your journey starts now._

## The Journey

### Step 1: Launch an NVIDIA Brev Instance

1. Log in to the NVIDIA Brev platform at [https://brev.nvidia.com](https://brev.nvidia.com).
2. Create a new account if you don't have one.
3. You should have already received some Brev credits through your email. If not, ask the staff for a coupon code, and redeem it to get free credits.
4. Create a new instance with the launchable at [https://brev.nvidia.com/launchable/deploy/now?launchableID=env-3LoAjmTNqTfQLeTMeFsOmgi2Ddv](https://brev.nvidia.com/launchable/deploy/now?launchableID=env-3LoAjmTNqTfQLeTMeFsOmgi2Ddv).
5. After the instance is up, follow the instructions on the webpage to install Brev CLI and connect to the instance via SSH and enter the workshop directory:
   ```bash
   cd ss26-local-ai-workshop
   ```
6. Even after you can connect, the instance might still be setting up: the launchable may still be downloading images and models in the background. Make sure setup has finished before moving on:
   ```bash
   tail -f setup.log
   ```
   Setup is finished when the log ends with `setup done: env1 env2 env3 env4` and the file `.setup-done` exists (press `Ctrl+C` to leave `tail`). If the log ends with `setup FAILED`, fix the error it shows and run `./setup.sh` again. Running `./setup.sh` while setup is still going is harmless: it tells you so and exits.

### Step 2: Launch Qwen3.8-27B with SGLang

1. Start a tmux session to keep the SGLang server running:
   ```bash
   tmux
   ```
2. Start the SGLang server. If you are curious where this command comes from, it is adapted from the [SGLang Cookbook entry for Qwen3.8-27B](https://docs.sglang.io/cookbook/autoregressive/Qwen/Qwen3.8-27B#hw=rtx6000&variant=default&quant=nvfp4-nvidia&nodes=single&spec=dspark&tier=low-latency&ssmDtype=float32). The cookbook gives the optimal command for serving a model with SGLang on your hardware and configuration of choice.
   ```bash
   docker run --gpus all \
     --name sglang --rm \
     --shm-size 32g \
     -p 30000:30000 \
     -v ~/.cache/huggingface:/root/.cache/huggingface \
     --ipc=host \
     lmsysorg/sglang:v0.5.22-runtime \
     sglang serve \
       --trust-remote-code \
       --model-path nvidia/Qwen3.8-27B-NVFP4 \
       --kv-cache-dtype fp8_e4m3 \
       --mem-fraction-static 0.85 \
       --attention-backend flashinfer \
       --chunked-prefill-size 2048 \
       --reasoning-parser qwen3 \
       --tool-call-parser qwen3_coder \
       --mamba-full-memory-ratio 11.93 \
       --host 0.0.0.0 \
       --port 30000 \
       --speculative-algorithm DSPARK \
       --speculative-draft-model-path RadixArk/Qwen3.8-27B-DSpark \
       --speculative-draft-attention-backend flashinfer \
       --mamba-radix-cache-strategy extra_buffer \
       --mamba-ssm-dtype float32 \
       --max-running-requests 4 \
       --max-mamba-cache-size 20 \
       --max-total-tokens 262144 \
       --allowed-media-domains none.invalid
   ```
3. When you see the message `The server is fired up and ready to roll!`, it means the SGLang server is running. You can now test the endpoint (Use `Ctrl+B Shift+5` to add a new pane to tmux):
   ```bash
   curl -X POST http://localhost:30000/v1/chat/completions \
     -H "Content-Type: application/json" \
     -d '{"messages": [{"role": "user", "content": "What is 1+1?"}]}'
   ```
   You should see the response from the SGLang server.
4. You can test the generation speed by asking the model to solve a math problem:
   ```bash
   curl -X POST http://localhost:30000/v1/chat/completions \
     -H "Content-Type: application/json" \
     -d '{"messages": [{"role": "user", "content": "Solve for the general nontrivial rational solution of the equation x^y = y^x"}], "stream": true}'
   ```
   When the answer is being generated, you can see the generation speed in the terminal. Note the relationship between the generation speed and accept length. You can stop the generation by pressing `Ctrl+C` on the `curl` command.
5. After you finish, you can detach from the tmux session by pressing `Ctrl+B D`.

If you ever restart the SGLang server, see [I restarted the SGLang server or the instance](#i-restarted-the-sglang-server-or-the-instance-and-my-agent-cannot-reach-the-model) in the FAQ.

### Step 3: Repair the Inference Engines

_Our story continues._

Before the network went dark, your team pulled four compromised deployments of Mini-SGLang off the production fleet. They now sit in quarantine as `env1` to `env4`. The engines still answer every request correctly, so the attack went unnoticed for hours. But in each one, the rogue agents planted a small, carefully hidden change that quietly breaks a different optimization. The code for every optimization is still there; it just no longer does its job.

Your forensics team has already worked out what is broken where:

| Engine | Broken optimization          | Symptom                                                                                              |
| ------ | ---------------------------- | ---------------------------------------------------------------------------------------------------- |
| `env1` | Chunked prefill              | A long prompt runs in one oversized pass, spiking memory and stalling every request that is decoding |
| `env2` | CUDA graphs                  | Each decode step launches hundreds of small GPU kernels from Python, so small batches crawl          |
| `env3` | Radix cache (prefix caching) | Requests that share a prompt prefix recompute it from scratch every time                             |
| `env4` | Overlap scheduling           | The GPU sits idle while the CPU prepares the next batch                                              |

`env1` serves your biggest customers, and it is the easiest of the four to repair: **bringing it back online is your mission for this workshop.** The other three are bonus missions, ordered from easier to harder, for when `env1` is done (Step 4).

Each deployment is sealed in its own sandbox, with a GPU and a small test model, **Qwen3-0.6B**, but no way out to the Internet. The only line still open leads to your Qwen3.8-27B, at `http://sglang:30000`.

> **Two models, two inference engines.** Keep them apart in your head:
>
> - **SGLang serving Qwen3.8-27B** (Step 2) is the brain of your coding agent. `omp` sends every request to it. You never modify it; it just has to stay running.
> - **Mini-SGLang running Qwen3-0.6B** is the patient. `env1` to `env4` are the sabotaged copies of Mini-SGLang's source code, and Qwen3-0.6B is the small model they load when `./verify.sh` benchmarks them. Your agent edits this code; it does not use this model to think.
>
> Both share the same GPU, which is why `./verify.sh` timings are best taken while your agent is idle.

The incident response team holds a sealed copy of the original engine. You cannot read it or restore from it, but they will benchmark any engine against it, side by side. An engine goes back into production only when its outputs match the original's token for token and it is within 5% of the original on every benchmark.

1. First, see the damage for yourself. Hand the compromised engine to the incident response team:
   ```bash
   ./verify.sh env1
   ```
   The table flags the benchmarks where it falls behind the original with `<- worse`, and the verdict is **NOT RESTORED**. The first run also benchmarks the sealed original, so it takes about twice as long.
2. Enter the quarantine zone for the engine:
   ```bash
   ./start.sh env1
   ```
3. Wake up your coding agent. It already knows the way to your local model:
   ```bash
   omp
   ```
   You can skip logging into any cloud LLM provider by pressing `Esc` in the beginning.  
   Brief it before you send it in: which optimization is broken and what the symptom is (see the table above). The code is all there, but a small change somewhere stops it from working: your agent has to find that change and fix it, not rewrite the feature.  
   If your agent gets lost, give it the mission briefing for the engine from the [Answer Key](#answer-key-mission-briefings) at the end of this page.
4. When your agent reports back, step out of the sandbox with `exit` and hand the repaired engine back to the incident response team:
   ```bash
   ./verify.sh env1
   ```
   Look for **RESTORED**. If it is not there yet, `./start.sh env1` takes you back into the sandbox (it stays up, and so does your agent's work); show your agent the benchmarks that are still behind and keep going.  
   The benchmarks share the GPU with your Qwen3.8-27B, so let your agent finish (or quit `omp`) before you run `./verify.sh`: a busy model slows the timings down and can turn a good repair into **NOT RESTORED**. If a timing misses by a small margin, run `./verify.sh` again.

If a repair goes badly wrong, pull a fresh copy of the compromised engine out of quarantine and start over: `docker rm -f workshop-env1 && rm -rf env1 && ./setup.sh`. This deletes your agent's edits; see [the FAQ](#how-do-i-start-an-exercise-over-without-losing-my-work) to save them first.

_FastInference AI is counting on you. Bring production back online._

### Step 4 (Optional): Bonus Missions

With `env1` serving again, the worst is over, but three deployments are still down. If you want a bigger challenge, repair `env2`, `env3` and `env4` with the same steps, in that order. Each one is harder than the last: expect your agent to need more time, more guidance, and sometimes a fresh start.

### Step 5 (Optional): Make It Yours

You now have a capable coding agent running entirely on your own GPU, with no cloud service involved. Put it to work on anything you like: a tool you have always wanted, a game, a script that automates something tedious. Give it a fresh sandbox of its own:

```bash
mkdir playground
./start.sh playground
```

It is the same kind of sandbox as the engines: your files are in `/workspace` (the `playground` directory on the instance), with Python, PyTorch and the GPU available. There is still no Internet, so your agent has to build things itself instead of installing packages.

## FAQ

### How do I know the instance is ready?

The instance boots before it is set up: the launchable keeps pulling images and downloading models in the background. Setup is finished when `setup.log` ends with `setup done: env1 env2 env3 env4` and the file `.setup-done` exists:

```bash
tail -n 5 setup.log
ls .setup-done
```

If the log ends with `setup FAILED`, fix the error shown above it and run `./setup.sh` again. If it says `setup is already running`, the first run is still going; follow it with `tail -f setup.log`. SGLang (Step 2) is ready only once it prints `The server is fired up and ready to roll!`, and loading the model can take several minutes.

### `omp` asks me to log in to a provider. Do I need an account?

No. Press `Esc` to skip the login. `omp` is already configured to use your local Qwen3.8-27B at `http://sglang:30000` for everything, so no cloud account or API key is needed, and the sandbox has no Internet access to reach a provider anyway. If `omp` reports that it cannot connect to the model, check that SGLang is still running (Step 2) and see [I restarted the SGLang server or the instance](#i-restarted-the-sglang-server-or-the-instance-and-my-agent-cannot-reach-the-model).

### Why can't the sandbox reach the Internet?

That is by design: the sandboxes are on an internal Docker network whose only other member is the SGLang server. `pip install`, `git clone`, `curl` to outside sites and web searches will all fail, and there is no `git` or `pip` in the sandbox. Everything the repair needs is already there: the engine's source in `/workspace`, Python with Mini-SGLang's dependencies, the GPU, and Qwen3-0.6B at `/models/Qwen3-0.6B`. If your agent tries to download something, tell it to work offline with what is installed.

### I restarted the SGLang server or the instance, and my agent cannot reach the model.

The sandboxes reach the model over a private Docker network, `workshop-net`. Each time the SGLang container starts, it starts outside that network, so the agent loses its model until you reconnect it:

1. Make sure SGLang is running. Restarting the instance removes the container, so if `docker ps` does not list `sglang`, redo Step 2 in a new `tmux` session. Wait for `The server is fired up and ready to roll!`.
2. On the instance (not inside the sandbox), run `./start.sh env1`. It reconnects SGLang to the network and takes you back into the sandbox. If you only want to reconnect, without opening another sandbox shell, run `docker network connect workshop-net sglang` instead.

Your sandbox and the agent's edits in `env1` survive both kinds of restart.

### `./verify.sh` says NOT RESTORED, but my fix looks right.

First check whether the outputs match and how far the flagged benchmarks are behind. If a benchmark misses the 5% bar by only a little, it is usually timing noise rather than a bad fix:

- The benchmarks share the GPU with Qwen3.8-27B. Let your agent finish, or quit `omp`, before running `./verify.sh`; a model that is busy generating slows the engine down.
- Run `./verify.sh env1` again. Each run measures the engine afresh.
- If the original's numbers look off (for example, they were measured while the GPU was busy), measure them again with `rm -rf results/original` and rerun `./verify.sh env1`.

If a benchmark is still far behind after that, the optimization is still broken: go back in with `./start.sh env1` and show your agent the numbers.

### How do I start an exercise over without losing my work?

Your agent's edits live in the engine's directory on the instance (`env1` is `/workspace` inside the sandbox), and resetting deletes it. Save a copy first, then reset:

```bash
cp -r env1 env1-attempt1
docker rm -f workshop-env1 && rm -rf env1 && ./setup.sh
```

`./setup.sh` keeps every engine directory that still exists, so only `env1` is replaced. You can compare an old attempt with the fresh copy using `diff -ru env1 env1-attempt1`, or put it back with `rm -rf env1 && mv env1-attempt1 env1`.

## Answer Key: Mission Briefings

Prompts for the four repairs, one per engine. Paste the prompt into `omp` inside the engine's sandbox.

<details>
<summary><code>env1</code>: Chunked prefill</summary>

> The LLM inference server in /workspace implements chunked prefill: a prompt that does not fit in the prefill token budget (8,192 tokens by default) is split into chunks that run in later scheduler iterations, so no forward pass exceeds the budget. Someone made a small change to the code that silently broke it: long prompts now run in one oversized forward pass. Start with the code that forms prefill batches. Find the change and fix it with the smallest edit that restores the intended behavior. Do not start the server or run the model; just read the code and edit it.

</details>

<details>
<summary><code>env2</code>: CUDA graphs</summary>

> The LLM inference server in /workspace implements CUDA graphs for decode: at startup it captures a graph for each of a set of batch sizes (1, 2, 4, 8, then every 8 up to a maximum), and every decode batch up to the largest captured size should be padded to the nearest captured size and replayed from its graph. Someone made a small change to the code that silently broke it: a single request decodes fast, but a batch of 8 runs as slowly as with no graphs at all. Start where the engine decides between replaying a graph and running the model eagerly. Find the change and fix it with the smallest edit that restores the intended behavior. Do not start the server or run the model; just read the code and edit it.

</details>

<details>
<summary><code>env3</code>: Radix cache</summary>

> The LLM inference server in /workspace implements prefix caching with a radix tree: when a request arrives, the tree is searched for the longest cached prefix of its prompt, and that part of the KV cache is reused instead of recomputed. Someone made a small change to the code that silently broke it: requests that share a long prefix get back a cached prefix only a token or so long, and recompute nearly their whole prompt. Start with how a new request looks up the prefix cache. Find the change and fix it with the smallest edit that restores the intended behavior. Do not start the server or run the model; just read the code and edit it.

</details>

<details>
<summary><code>env4</code>: Overlap scheduling</summary>

> The LLM inference server in /workspace implements overlap scheduling: while the GPU runs batch N, the CPU processes the results of batch N-1 and prepares batch N+1, so CPU work hides behind GPU work. Someone made a small change to the code that silently broke it: the overlapped loop is no faster than the serial fallback, and the GPU sits idle while the CPU processes each step's results. Start by comparing the scheduler's overlapped loop with its serial loop. Find the change and fix it with the smallest edit that restores the intended behavior. Do not start the server or run the model; just read the code and edit it.

</details>
