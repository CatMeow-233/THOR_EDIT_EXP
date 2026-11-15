# ResearchRL — Modular Research Framework for LLM Training and Reinforcement Learning

ResearchRL is a small research layer on top of Hugging Face **TRL / Transformers / Accelerate**.
TRL does the generic training (SFT, DPO, GRPO, RLOO, on-policy distillation); ResearchRL adds the
parameter-level operations, update rules, experiment management and extension contracts that many
independent research projects need, without forking or copying TRL trainers.

It lives in `research_rl/` next to the existing THOR / verl code in this repository, which it does
not import or modify.

> **Status of results.** Everything in this repository was verified on **CPU with tiny, randomly
> initialised models** (a few thousand parameters, character tokenizer, built in-process). These
> tests verify *algorithmic and engineering correctness* against independent reference
> implementations. They say nothing about real LLM quality. All GPU performance numbers are
> `NOT_TESTED` until measured with the scripts in `scripts/gpu/` on a CUDA machine (see
> [DEPLOY_GPU.md](DEPLOY_GPU.md)).

## Install

```bash
# CPU (CI / laptop)
pip install "torch==2.15.0" --index-url https://download.pytorch.org/whl/cpu
pip install -e ".[dev]"
```

```bash
# CUDA box (RTX 5090 needs a CUDA 12.8+ build; the default PyPI torch 2.15 wheel is CUDA 13)
pip install "torch==2.15.0"
pip install -e ".[dev]"
```

Pinned versions (the ones the test-suite ran against): torch 2.15.0, transformers 5.23.0,
trl 1.18.0, accelerate 1.15.0, peft 0.21.3, datasets 5.2.0, safetensors 0.8.0.

## Layout

| Package | What it contains | V1.0 status |
|---|---|---|
| `core/` | `ExperimentConfig` (versioned, JSON/YAML), plugin `Registry`s, `ExperimentRecorder`, research checkpoint state | implemented |
| `trainers/` | `TrainerAdapter` + `build_run`; adapters `sft`, `dpo`, `grpo`, `rloo`, `distillation`; `ResearchTrainerMixin` (installs `HookedOptimizer` into any TRL trainer); `multi_teacher` utilities | implemented |
| `optim/` | `HookedOptimizer`, `UpdateRule` contract, `MaskRule`, backward-time gradient masks, task-vector priors, `P2PriorRule` | implemented |
| `models/` | SwiGLU introspection, `ParameterMask`, FFN channel operations, `ExpertManager`, tiny offline models | implemented |
| `rollout/` | trajectories, policy versions, agent-environment contract (TRL `environment_factory`) | implemented / contract |
| `rewards/` | deterministic `MathVerifier`, TRL reward adapter, **disabled-by-default** code sandbox interface | implemented / interface |
| `state/` | recurrent-state contract (`FastSlowMemory`, `StateStore`), compute controllers and ledger | contract + reference impls |
| `evaluation/` | metrics (pass@k, bootstrap CI, NLL), capability-retention suite and report | implemented |
| `utils/` | logging, profiling (GPU numbers only when measured on CUDA) | implemented |

## Quick start

```python
from datasets import Dataset
from research_rl.core.config import ExperimentConfig
from research_rl.trainers import build_run
from research_rl.rewards import MathVerifier, as_trl_reward_func

exp = ExperimentConfig.load("examples/configs/grpo_math_tiny.yaml")
exp.model.name_or_path = "Qwen/Qwen2.5-0.5B-Instruct"        # any HF causal LM
data = Dataset.from_list([{"prompt": "1+1=?", "answer": "2"}])  # your RLVR data
run = build_run(exp, train_dataset=data,
                reward_funcs=[as_trl_reward_func(MathVerifier(), answer_key="answer")])
record = run.train()          # -> runs/<study_id>/<run_name>/record.json
```

Every example in `examples/` runs offline on CPU by default (tiny model) and on a real model with
`--model <hf id or path> --bf16`:

```bash
python examples/sft_tiny.py --max-steps 2
```

```bash
python examples/grpo_math_tiny.py --max-steps 2
```

The other examples are `dpo_tiny.py`, `rloo_math_tiny.py` and `distillation_tiny.py`.

## Algorithms and what they are

| Adapter | TRL trainer | Is it RL? | Verified against an independent reference |
|---|---|---|---|
| `sft` | `SFTTrainer` | no (supervised) | completion-only token CE, gradient-accumulation token weighting, BF16 path, bit-exact resume |
| `dpo` | `DPOTrainer` | no (offline preference optimisation) | `-logsigmoid(β·Δ)` from summed completion log-probs (default `loss_type=sigmoid`) |
| `distillation` | `DistillationTrainer` | no (on-policy distillation) | generalized JSD for several β / temperatures, student gradients |
| `grpo` | `GRPOTrainer` | **yes** | completion masks, group advantages (`scale_rewards` group/batch/none, unbiased std + 1e-4), policy-gradient gradients for `dapo` and `grpo` loss types |
| `rloo` | `RLOOTrainer` | **yes** | leave-one-out advantages (+ normalisation), KL-shaped reward, per-sequence loss gradients |

RL-specific bookkeeping (GRPO/RLOO):

* **Policy version** = number of parameter updates actually applied; stamped on every rollout batch
  (`rollout/policy_version`, `rollout/policy_lag` in the logs).
* **Zero-advantage windows**: with `optim.skip_zero_advantage_steps: true`, an optimizer step whose
  whole gradient-accumulation window had only zero advantages is skipped exactly (no weight decay,
  no Adam state change). `record.json` keeps `nominal_optimizer_steps`, `applied_updates` and
  `skipped_steps` separately. A skip would also drop gradient that does not come from the
  advantage, so the flag is refused (`ValueError`) with GRPO `beta != 0`, the GRPO entropy bonus,
  the MoE router aux loss, or an `UpdateRule` with `needs_every_step = True` (checked at train
  begin). RLOO `beta != 0` is allowed: TRL folds its KL penalty into the reward, i.e. into the advantage.
* **Advantage shaping**: pass `advantage_shapers=[...]` to `build_run` (objects implementing
  `research_rl.trainers.rl_common.AdvantageShaper`, or names registered in
  `ADVANTAGE_SHAPERS`). They rewrite TRL's advantages (`(B,)`, or `(B, T)` for GRPO) before the
  signal flags and the trajectory log; the "no signal" decision then comes from the last shaper's
  `signal`, or from "advantage exactly non-zero" if it returns none. Advantages edited later (e.g.
  by a subclass) are detected per micro-batch and fall back to the exact non-zero test, with a warning.
  TRL's completions table (`log_completions`) still shows TRL's unshaped advantages; the shaped ones
  are in `trajectory_buffer` and the `rollout/*` metrics. Shapers with `state_dict`/`load_state_dict`
  are checkpointed; other shapers restart from their constructor state on resume.
* **TRL 1.18 facts found while testing**: a custom `rollout_func` is used verbatim (no cut at the
  first EOS) — wrap it with `research_rl.trainers.rl_common.eos_truncating_rollout`; GRPO's default
  `loss_type` is `dapo`; RLOO's default `beta=0.05` reloads the reference model from
  `config._name_or_path` (in-memory models need `beta=0`); TRL configs default to `bf16=True`, so the
  adapters set `bf16=False` unless you ask for it.

## Optimizer extension (`HookedOptimizer`)

A normal `TrainerCallback` cannot change what Adam writes into the weights. `HookedOptimizer` wraps
the optimizer TRL creates (via `ResearchTrainerMixin.create_optimizer`, so it works for every TRL
trainer) and runs `UpdateRule`s at four points of every optimizer step:

```
grads ready ─► post_grad ─► [clip inside wrapper] ─► inner step (Adam preconditioning, weight decay)
          ─► transform_update(candidate Δ, fp32) ─► pre_write(final value) ─► post_step(actual Δ)
```

Guarantees (each has a numerical test): bit-identical to the inner optimizer when no rule changes
anything; masks restore frozen elements with `torch.where` (exact under Adam momentum and AdamW
decay, also in BF16, also for `requires_grad=False` params with stale grads); hooks run once per
optimizer step under gradient accumulation; rule state and step counters live in the optimizer
state file (weights-only loadable) so HF checkpoint resume restores them bit-exactly.

```python
from research_rl.optim import UpdateRule

class TrustRegion(UpdateRule):            # DeltaHippo / STABLE-style constraint on the Adam update
    name = "trust_region"
    def __init__(self, tau): self.tau = tau
    def tracks(self, name, param): return True
    def transform_update(self, ctx, name, param, old, delta):
        return delta.clamp(-self.tau, self.tau)

run = build_run(exp, train_dataset=data, update_rules=[TrustRegion(1e-3)])
```

Gradient clipping: HF clips *before* `optimizer.step()`. Rules that must see unclipped gradients
(e.g. `P2PriorRule`) or masks that must not influence the clip norm use
`optim.clip_in_wrapper: <norm>` (the TRL `max_grad_norm` is then forced to 0 and the wrapper clips
after the `post_grad` hooks) or backward-time masks (`register_gradient_masks`).

Precision: a parameter *stored* in bf16 (model loaded with `dtype: bfloat16`, no fp32 master copy)
silently loses every update below ≈ `|w|·2⁻⁸`, because AdamW writes it back with round-to-nearest
(HF `bf16: true` autocast keeps fp32 parameters and is not affected). `HookedOptimizer` warns about
this at construction (`optim.precision_check: warn | error | off`). `optim.master_weights: true`
keeps an fp32 master of every bf16/fp16 parameter: the inner optimizer (fp32 state) and the
`transform_update`/`pre_write` rules run on the master, the model receives
`round_to(master, dtype, optim.write_rounding)` (`nearest`, or unbiased `stochastic`, seeded from the
experiment seed), `post_step` sees fp32 master displacements, and the masters are checkpointed in the
optimizer state (exact HF resume). `P2PriorRule` refuses bf16 prior modules without masters
(`allow_low_precision=True` to override). The record stores `precision` (trainable dtypes, options).
Writes to the weights from outside the optimizer (`restore_frozen`, `rollback_channels`,
`model.load_state_dict`, ...) reach the master for every element whose value changed; an element
rewritten with its current bits keeps its master, so for an exact rollback call
`optimizer.set_master(...)` or `resync_masters()`.
See `research_rl/optim/precision.py` and the `HookedOptimizer` docstring.

## Parameter-level research tools

* **FFN channels** (`models/ffn_channels.py`): a SwiGLU channel *c* = row *c* of `gate_proj`, row *c*
  of `up_proj`, column *c* of `down_proj`. `ChannelSelection`, `channel_mask` / `freeze_mask`,
  `setup_channel_training` (MaskRule + backward-time masks), `channel_diff` / `top_changed_channels`,
  `rollback_channels`, `ablate_channels`, `verify_modification_scope`.
* **Experts** (`models/expert_manager.py`, TLC): `ExpertManager.attach(layers)`, `create_expert`,
  `set_trainable`, `freeze_backbone` / `assert_backbone_unchanged` (bit-exact fingerprint),
  `activate` / `using` / per-sample `routed`, `save_expert` / `load_expert` (safetensors + compatibility
  metadata), `export_merged` (vanilla HF checkpoint). Response-level expert selection — **not**
  token-level MoE. `find_swiglu_ffns(model, expert="name")` lets the channel tools operate inside one
  expert.
* **Parameter priors** (`optim/parameter_prior.py`): `TaskVector`, `TaskVectorPrior`,
  `project_onto_span`, `PriorSubspaceRule`, and `P2PriorRule` — the two-channel update of P²-MOPD
  (arXiv 2611.03953): `W = B + Σ α_k τ_k` per module, coefficient gradient `⟨G, τ⟩` from the unclipped
  gradient, separate clipping, Adam on α, AdamW (decay 0) on W, `W ← W_AdamW + Σ Δα·τ`. This is the
  parameter-prior part only. The P²-MOPD distillation *loss* (per-prompt routed,
  support-restricted reverse KL inside a policy-gradient surrogate) is **not** implemented in V1.0;
  `trainers/multi_teacher.py` provides routed aggregation / masked-KL helpers (tested offline) and a
  static-weight `TeacherEnsemble` for TRL's full-vocabulary JSD `DistillationTrainer`.

## Experiment records

`runs/<study_id>/<run_name>/` contains `config.json`, `metrics.jsonl` and `record.json` with: study
id, config (+ sha256), model identity (class, HF commit hash, weights sha256 for small models),
dataset identity (row count, columns, content sha256), seed, trainable-parameter count, nominal /
applied / skipped optimizer steps, training tokens, final loss / reward / grad-norm, checkpoints,
evaluation outputs, environment (package versions, device). `hardware.gpu_metrics` stays
`"NOT_TESTED"` unless `record_gpu_metrics` is called during a CUDA run.

## Extension contracts (V1.0: contract + tested reference implementation, not full papers)

| Contract | Module | Reference implementation |
|---|---|---|
| Parameter prior (P²-MOPD) | `optim/parameter_prior.py` | `TaskVectorPrior`, `PriorSubspaceRule`, `P2PriorRule` |
| Recurrent state | `state/recurrent_interface.py` | `FastSlowMemory`, `StateStore` (no cross-sample sharing). `FastSlowMemory` computes its updates in at least fp32 and stores the slow state in at least fp32 (`slow_dtype=torch.float32`, so a bf16 state reads out as fp32), because a low-rate EMA increment is otherwise rounded away in bf16; `slow_dtype=None` restores same-dtype storage and warns when a rate is below the dtype's precision (larger rates can still stall near their target in half precision) |
| Compute controller | `state/compute_controller.py` | `FixedDepth`, `ThresholdHalting`, `run_looped`, `ComputeLedger`, `best_of_depth` |
| Agent environment | `rollout/agent_env.py` | TRL `environment_factory`-compatible `ResearchEnvironment`, `CalculatorEnv`, `EpisodeLog` credit math. OpenEnv (`trl.experimental.async_grpo.openenv_harness`) needs the separate `openenv` package — not installed, not tested |
| Multi-teacher distillation | `trainers/multi_teacher.py` | teacher aggregation (mixture / product, per-sample weights for routing), masked KLs, `TeacherEnsemble` usable as TRL `teacher_model` (global static weights only: TRL does not pass a per-prompt domain to the teacher). `AsyncDistillationTrainer` needs a vLLM server — not tested |
| Code-execution reward | `rewards/code_sandbox.py` | disabled by default; refuses to run without an isolated backend that declares no-network + resource limits; no local executor ships |

Plugins register themselves in `research_rl.core.registry` (`TRAINERS`, `UPDATE_RULES`, `VERIFIERS`,
`RECURRENT_STATES`, `COMPUTE_CONTROLLERS`, `PARAMETER_PRIORS`, `ENVIRONMENTS`), so a paper's code can
live in its own package and be selected by name from an experiment config.

## Tests and CI

```bash
pytest -m "not integration" -q          # unit tests
```

```bash
pytest -m integration -q                # tiny-model TRL training runs
```

```bash
ruff format --check && ruff check && mypy
```

GitHub Actions (`.github/workflows/research_rl_ci.yml`) runs format/lint/type checks, the unit
tests, the integration tests and the example smoke runs on CPU with `HF_HUB_OFFLINE=1`.

See [API.md](API.md) for the API reference (generated from the docstrings) and [DEPLOY_GPU.md](DEPLOY_GPU.md) for running on the two RTX 5090 hosts.
