# ResearchRL API reference

Generated from the module and object docstrings of `research_rl` (public names only).
See [README.md](README.md) for concepts and [DEPLOY_GPU.md](DEPLOY_GPU.md) for GPU runs.

## `research_rl.core.checkpoint`

Checkpoint contents and ResearchRL-specific resume state.

- **class `StatefulComponent`**
  - `load_state_dict(self, state: 'Mapping[str, Any]') -> 'None'`
  - `state_dict(self) -> 'dict[str, Any]'`
- `checkpoint_inventory(checkpoint_dir: 'str | Path') -> 'dict[str, bool]'` — Which resume-relevant artefacts exist in a checkpoint directory.
- `latest_checkpoint(output_dir: 'str | Path') -> 'str | None'`
- `load_components(checkpoint_dir: 'str | Path', components: 'Mapping[str, StatefulComponent]') -> 'list[str]'` — Restore registered components; returns the names that were restored.
- `load_research_state(checkpoint_dir: 'str | Path') -> 'dict[str, Any]'`
- `save_components(checkpoint_dir: 'str | Path', components: 'Mapping[str, StatefulComponent]') -> 'Path | None'`
- `save_research_state(checkpoint_dir: 'str | Path', state: 'Mapping[str, Any]') -> 'Path'`

## `research_rl.core.config`

Versioned experiment configuration.

- **class `ConfigError`**
- **class `DataSpec`** — Identity of the training data (the dataset object itself is passed at run time).
- **class `ExperimentConfig`** — ExperimentConfig(study_id: 'str', model: 'ModelSpec', data: 'DataSpec', trainer: 'TrainerSpec', seed: 'int' = 0, output_dir: 'str' = 'runs', run_name: 'str | None' = None, optim: 'OptimSpec' = <factory>, tags: 'list[str]' = <factory>, extra: 'dict[str, Any]' = <factory>, config_version: 'str' = '1.0')
  - `from_dict(cls, raw: 'dict[str, Any]') -> 'ExperimentConfig'`
  - `load(cls, path: 'str | Path') -> 'ExperimentConfig'`
  - `resolved_run_name`
  - `run_dir`
  - `save(self, path: 'str | Path') -> 'Path'`
  - `to_dict(self) -> 'dict[str, Any]'`
- **class `ModelSpec`** — Where the policy weights come from.
- **class `OptimSpec`** — ResearchRL optimizer-extension settings (see :mod:`research_rl.optim`).
- **class `TrainerSpec`** — Which registered trainer adapter to use and the raw TRL config arguments.

## `research_rl.core.experiment`

Unified experiment records.

- **class `ExperimentRecorder`** — Collects identity, metrics, checkpoints and evaluations for one run.
  - `add_checkpoint(self, path: 'str | Path', step: 'int') -> 'None'`
  - `add_evaluation(self, name: 'str', results: 'Mapping[str, Any]', step: 'int | None' = None) -> 'None'`
  - `finalize(self, status: 'str' = 'completed') -> 'dict[str, Any]'`
  - `history(self, key: 'str') -> 'list[float]'`
  - `load(run_dir: 'str | Path') -> 'dict[str, Any]'`
  - `log_metrics(self, step: 'int', logs: 'Mapping[str, Any]') -> 'None'`
  - `record_gpu_metrics(self, metrics: 'Mapping[str, Any]') -> 'None'` — Store measured GPU metrics. Refuses unless running on CUDA right now.
  - `resume_from(self, global_step: 'int') -> 'None'` — Continue the record of an interrupted run from checkpoint ``global_step``.
  - `set_dataset(self, dataset: 'Any', name: 'str | None' = None) -> 'None'`
  - `set_extra(self, key: 'str', value: 'Any') -> 'None'` — Store an algorithm-specific top-level field (e.g. reference/teacher model identity).
  - `set_model(self, model: 'Any', revision: 'str | None' = None, hash_weights: 'bool | None' = None) -> 'None'`
  - `set_steps(self, *, nominal: 'int', applied: 'int', skipped: 'int', global_step: 'int') -> 'None'`
- `dataset_identity(dataset: 'Any', name: 'str | None' = None, max_rows_for_hash: 'int' = 200000) -> 'dict[str, Any]'` — Stable identity of a dataset: row count, column names and a content hash.
- `environment_info() -> 'dict[str, Any]'`
- `model_identity(model: 'Any', revision: 'str | None' = None, hash_weights: 'bool | None' = None) -> 'dict[str, Any]'` — Model revision info: HF commit hash if known, and (for small models) a weights hash.

## `research_rl.core.registry`

Small, explicit plugin registries.

- **class `Registry`** — A name -> object mapping with decorator-based registration.
  - `add(self, name: 'str', obj: 'T', *, overwrite: 'bool' = False) -> 'None'`
  - `build(self, name: 'str', *args: 'Any', **kwargs: 'Any') -> 'Any'` — Look up ``name`` and call it (useful when entries are classes or factories).
  - `get(self, name: 'str') -> 'T'`
  - `names(self) -> 'list[str]'`
  - `register(self, name: 'str | None' = None, *, overwrite: 'bool' = False) -> 'Callable[[T], T]'` — Decorator registering ``obj`` under ``name`` (defaults to ``obj.__name__``).
- **class `RegistryError`** — Raised for unknown or duplicate registry entries.

## `research_rl.evaluation.capability_retention`

Capability retention: does RL(VR) training degrade other capabilities, and does recovery help?

- **class `CapabilityRetention`** — Retention numbers of one capability. ``None`` marks a quantity that is undefined.
  - `compute(cls, name: 'str', base: 'float', after_training: 'float', after_recovery: 'float | None' = None, *, drop_tolerance: 'float' = 0.0) -> 'CapabilityRetention'`
- **class `KLResult`** — Mean per-token ``KL(base || current)`` over probe prompt positions (natural log).
- **class `ProbeTask`** — One capability probe: ``(prompt, answer)`` examples scored by ``scorer``.
  - `prompts`
  - `scorer(prediction: 'str', answer: 'str') -> 'float'` — 1.0 if the whitespace-normalised prediction equals the answer, else 0.0.
- **class `RetentionReport`** — Comparison ``base -> after_training -> after_recovery`` over all capabilities.
  - `from_scores(cls, base: 'Mapping[str, float]', after_training: 'Mapping[str, float]', after_recovery: 'Mapping[str, float] | None' = None, *, drop_tolerance: 'float' = 0.0, kl_after_training: 'Mapping[str, float] | None' = None, kl_after_recovery: 'Mapping[str, float] | None' = None) -> 'RetentionReport'`
  - `log_to(self, recorder: 'Any', name: 'str' = 'capability_retention', step: 'int | None' = None) -> 'None'` — Append this report to an :class:`~research_rl.core.experiment.ExperimentRecorder`.
  - `summary(self) -> 'dict[str, Any]'` — Unweighted means over capabilities; ratio means use only capabilities where defined.
  - `to_dict(self) -> 'dict[str, Any]'`
- **class `RetentionSuite`** — A set of :class:`ProbeTask` evaluated with deterministic greedy decoding.
  - `evaluate(self, model: 'nn.Module', tokenizer: 'Any') -> 'SuiteResult'`
  - `logit_kl(self, base_model: 'nn.Module', model: 'nn.Module', tokenizer: 'Any') -> 'dict[str, float]'` — Per-task mean per-token KL plus ``"all"`` (token-weighted over every probe prompt).
  - `report(self, base_model: 'nn.Module', trained_model: 'nn.Module', tokenizer: 'Any', recovered_model: 'nn.Module | None' = None, *, with_kl: 'bool' = True, drop_tolerance: 'float' = 0.0) -> 'RetentionReport'` — Evaluate all models and build a :class:`RetentionReport`.
- **class `SuiteResult`** — Per-capability mean score plus the raw predictions they were computed from.
- `exact_match_scorer(prediction: 'str', answer: 'str') -> 'float'` — 1.0 if the whitespace-normalised prediction equals the answer, else 0.0.
- `greedy_generate(model: 'nn.Module', tokenizer: 'Any', prompts: 'Sequence[str]', *, max_new_tokens: 'int', batch_size: 'int' = 1) -> 'list[str]'` — Deterministic greedy continuations of ``prompts`` (decoded, EOS and everything after removed).
- `logit_kl(base_model: 'nn.Module', model: 'nn.Module', tokenizer: 'Any', prompts: 'Iterable[str]') -> 'KLResult'` — Token-weighted mean of ``KL(p_base || p_model)`` over every position of every prompt.
- `prefix_match_scorer(prediction: 'str', answer: 'str') -> 'float'` — 1.0 if the whitespace-normalised prediction *starts with* the answer.

## `research_rl.evaluation.metrics`

Evaluation metrics: exact match, unbiased pass@k, bootstrap confidence intervals, token-level NLL.

- **class `MeanCI`** — Sample mean with a percentile-bootstrap confidence interval.
  - `to_dict(self) -> 'dict[str, float | int]'`
- **class `NLLResult`** — Token-level negative log-likelihood (natural log) of a causal LM.
  - `to_dict(self) -> 'dict[str, Any]'`
- `exact_match(prediction: 'str', reference: 'str', *, normalize: 'bool' = True, case_sensitive: 'bool' = True) -> 'bool'` — String equality, optionally after :func:`normalize_answer`.
- `exact_match_accuracy(predictions: 'Sequence[str]', references: 'Sequence[str]', *, normalize: 'bool' = True, case_sensitive: 'bool' = True) -> 'float'` — Fraction of predictions that exactly match their reference.
- `mean_pass_at_k(num_samples: 'Sequence[int]', num_correct: 'Sequence[int]', k: 'int') -> 'float'` — Average of :func:`pass_at_k` over problems (``num_samples[i]``, ``num_correct[i]``).
- `mean_with_bootstrap_ci(values: 'Sequence[float] | np.ndarray | torch.Tensor', *, confidence: 'float' = 0.95, n_resamples: 'int' = 2000, seed: 'int' = 0) -> 'MeanCI'` — Mean and percentile-bootstrap CI, deterministic given ``seed``.
- `normalize_answer(text: 'str', *, case_sensitive: 'bool' = True) -> 'str'` — Strip surrounding whitespace and collapse internal whitespace runs to one space.
- `pass_at_k(n: 'int', c: 'int', k: 'int') -> 'float'` — Unbiased pass@k estimator (Chen et al., 2021): ``1 - C(n-c, k) / C(n, k)``.
- `sequence_nll(model: 'nn.Module', tokenizer: 'Any', pairs: 'Sequence[tuple[str, str]]', *, completion_only: 'bool' = True, batch_size: 'int' = 8, add_bos: 'bool' = False) -> 'NLLResult'` — Summed / mean NLL of ``(prompt, completion)`` pairs under a causal LM.

## `research_rl.models.expert_manager`

Expert Manager: a shared backbone with independently trained, response-level FFN experts.

- **class `BackboneChangedError`** — The shared backbone differs from its recorded fingerprint.
- **class `ExpertCompatibilityError`** — An expert file / checkpoint does not match this model.
- **class `ExpertError`** — Unknown expert, invalid name or invalid manager operation.
- **class `ExpertFFN`** — Drop-in replacement for ``layer.mlp`` holding several SwiGLU experts.
  - `extra_repr(self) -> 'str'`
  - `forward(self, x: 'torch.Tensor') -> 'torch.Tensor'`
- **class `ExpertManager`** — Manage response-level FFN experts on an HF decoder-only model with SwiGLU FFNs.
  - `activate(self, name: 'str') -> 'None'` — Use expert ``name`` for every row (clears per-sample routing).
  - `active`
  - `assert_backbone_unchanged(self, reference: 'str | None' = None) -> 'None'`
  - `attach(self, layers: 'Iterable[int] | None' = None) -> 'list[int]'` — Replace ``layer.mlp`` of the given decoder layers (default: all) by an :class:`ExpertFFN`.
  - `attached`
  - `backbone_fingerprint(self) -> 'str'`
  - `backbone_parameter_names(self) -> 'list[str]'` — Every parameter that does not belong to any expert (the shared backbone).
  - `clear_routing(self) -> 'None'`
  - `create_expert(self, name: 'str', source: 'str' = 'base') -> 'None'` — Add expert ``name`` as an exact copy of expert ``source`` in every attached layer.
  - `delete_expert(self, name: 'str') -> 'None'`
  - `detach(self, name: 'str' = 'base') -> 'None'` — Put expert ``name`` back as the vanilla ``layer.mlp`` (same parameter objects) and drop the other experts. Parameter ``requires_grad`` flags are left as they are.
  - `expert_fingerprint(self, name: 'str') -> 'str'`
  - `expert_names`
  - `expert_parameter_names(self, name: 'str') -> 'list[str]'`
  - `expert_parameters(self, name: 'str') -> 'list[nn.Parameter]'`
  - `export_merged(self, name: 'str', path: 'str | Path', **save_kwargs: 'Any') -> 'Path'` — Save a *vanilla* HF checkpoint (standard ``model.layers.i.mlp.*`` keys) with expert ``name``.
  - `ffn_modules(self) -> 'dict[int, ExpertFFN]'` — The attached :class:`ExpertFFN` modules by decoder-layer index (a new dict, same modules).
  - `freeze_backbone(self, *, rebaseline: 'bool' = False) -> 'str'` — Freeze everything except the trainable expert (if any) and return the backbone reference fingerprint used by :meth:`assert_backbone_unchanged`.
  - `layers`
  - `load_expert(self, path: 'str | Path', name: 'str | None' = None, *, overwrite: 'bool' = False, check_backbone: 'bool' = True) -> 'str'` — Load an expert file into this model; validates everything before changing the model.
  - `load_state_dict(self, state: 'Mapping[str, Any]') -> 'None'` — Recreate the expert structure (attach, create experts, trainability, selection).
  - `merged_model(self, name: 'str') -> 'nn.Module'` — A deep copy of the model with expert ``name`` as a plain ``layer.mlp`` (all params trainable).
  - `merged_state_dict(self, name: 'str') -> 'dict[str, torch.Tensor]'` — State dict with vanilla keys in which expert ``name`` is the FFN (no copies made).
  - `read_expert_file(path: 'str | Path') -> 'tuple[dict[str, torch.Tensor], dict[str, str]]'`
  - `routed(self, names: 'Sequence[str]') -> 'Iterator[None]'` — Temporarily route row ``b`` through expert ``names[b]``.
  - `routing`
  - `rows(self, names: 'Sequence[str]') -> 'Iterator[None]'` — Temporarily select the experts of a batch whose row ``b`` belongs to expert ``names[b]``.
  - `save_expert(self, name: 'str', path: 'str | Path') -> 'Path'` — Write one expert (all attached layers) as safetensors with compatibility metadata.
  - `save_pretrained(self, path: 'str | Path', **save_kwargs: 'Any') -> 'Path'` — Save the model *with experts attached* plus a manager sidecar (see :func:`load_model_with_experts`).
  - `set_routing(self, names: 'Sequence[str] | None') -> 'None'` — Per-sample (response-level) routing: ``names[b]`` is the expert of batch row ``b``.
  - `set_trainable(self, name: 'str | None', *, rebaseline: 'bool' = False) -> 'None'` — Make expert ``name`` the only trainable parameters and activate it (``None``: freeze all).
  - `state_dict(self) -> 'dict[str, Any]'` — Structure only (no tensors: expert weights live in the model state dict).
  - `trainable_expert`
  - `trainable_parameter_names(self) -> 'list[str]'`
  - `using(self, name: 'str') -> 'Iterator[None]'` — Temporarily use expert ``name`` for all rows.
- `config_hash(config: 'Any') -> 'str'` — SHA-256 of the architecture-relevant part of an HF config (stable across save/load).
- `load_model_with_experts(path: 'str | Path', model_factory: 'Callable[[Any, torch.dtype], nn.Module] | None' = None) -> 'tuple[nn.Module, ExpertManager]'` — Load a checkpoint saved with experts attached (``ExpertManager.save_pretrained`` or an HF Trainer checkpoint) and re-attach its experts. Fails if keys do not match exactly.
- `model_class_name(model: 'nn.Module') -> 'str'`

## `research_rl.models.ffn_channels`

Channel-level parameter operations on SwiGLU FFNs (selective fine-tuning, diffing, rollback).

- **class `ChannelDiff`** — Per-channel change statistics for one layer (float64).
  - `changed_channels(self) -> 'list[int]'`
  - `relative`
  - `score(self, metric: 'Metric') -> 'torch.Tensor'`
- **class `ChannelSelection`** — Immutable mapping ``layer_index -> sorted unique channel indices``.
  - `all_channels(cls, model: 'nn.Module', layers: 'Iterable[int] | None' = None, *, expert: 'str | None' = None) -> 'ChannelSelection'`
  - `channels(self, layer: 'int') -> 'tuple[int, ...]'`
  - `complement(self, model: 'nn.Module', *, expert: 'str | None' = None) -> 'ChannelSelection'` — All channels of the *affected* layers that are not selected.
  - `count(self) -> 'int'`
  - `difference(self, other: 'ChannelSelection') -> 'ChannelSelection'`
  - `from_dict(cls, d: 'Mapping[int | str, Iterable[int]]', *, intermediate_size: 'int | None' = None) -> 'ChannelSelection'` — Build from a (possibly JSON-loaded, string-keyed) dict.
  - `index_tensor(self, layer: 'int', device: 'torch.device | str | None' = None) -> 'torch.Tensor'`
  - `intersection(self, other: 'ChannelSelection') -> 'ChannelSelection'`
  - `items(self) -> 'Iterator[tuple[int, tuple[int, ...]]]'`
  - `layers(self) -> 'list[int]'`
  - `to_dict(self) -> 'dict[int, list[int]]'`
  - `top_k_from_scores(cls, scores: 'Mapping[int, torch.Tensor]', k: 'int', *, per_layer: 'bool' = False, largest: 'bool' = True) -> 'ChannelSelection'` — Select the ``k`` best channels globally (or ``k`` per layer).
  - `union(self, other: 'ChannelSelection') -> 'ChannelSelection'`
  - `validate(self, model: 'nn.Module', *, expert: 'str | None' = None) -> 'None'`
- **class `ChannelSelectionError`**
- **class `ChannelState`** — ChannelState(layers: 'dict[int, LayerChannels]')
  - `selection(self) -> 'ChannelSelection'`
- **class `ChannelTrainingHandle`** — Result of :func:`setup_channel_training`.
  - `remove(self) -> 'None'`
- **class `LayerChannels`** — Detached copies of the slices owned by ``channels`` in one layer (original dtype).
  - `select(self, channels: 'Iterable[int]') -> 'LayerChannels'` — Sub-state for ``channels`` (each must be present).
  - `vectors(self, dtype: 'torch.dtype' = torch.float64) -> 'torch.Tensor'` — ``[k, D]``: per channel the concatenation [gate row, up row, down col, (biases)].
- **class `RollbackReport`** — Per layer: rolled-back channels, how far they were from the reference, elements that changed.
  - `total_elements_changed`
- **class `ScopeReport`** — ScopeReport(violations: 'dict[str, ScopeViolation]', params_checked: 'int', elements_checked: 'int', changed_allowed: 'int')
  - `ok`
  - `summary(self) -> 'str'`
  - `total_violations`
- **class `ScopeViolation`** — ScopeViolation(count: 'int', max_abs: 'float', indices: 'torch.Tensor')
- **class `ScopeViolationError`** — Raised by :func:`verify_modification_scope` in strict mode.
- `ablate_channels(model: 'nn.Module', selection: 'ChannelSelection', *, expert: 'str | None' = None) -> 'ChannelState'` — Zero the selected channels' slices in place; returns their previous values (for rollback).
- `channel_diff(reference: 'ChannelState', current: 'ChannelState | nn.Module', layers: 'Iterable[int] | None' = None, *, expert: 'str | None' = None, allow_path_mismatch: 'bool' = False) -> 'dict[int, ChannelDiff]'` — Compare two channel states (or a state against a live model) channel by channel.
- `channel_mask(model: 'nn.Module', selection: 'ChannelSelection', *, trainable: 'bool' = True, expert: 'str | None' = None) -> 'ParameterMask'` — Mask over the gate/up/down weights (and biases) of every affected layer (of ``expert``).
- `channel_state(model: 'nn.Module', layers: 'Iterable[int] | None' = None, *, expert: 'str | None' = None) -> 'ChannelState'` — Every channel of ``layers`` (default: all FFN layers). Use as a rollback/diff reference.
- `extract_channels(model: 'nn.Module', selection: 'ChannelSelection', *, expert: 'str | None' = None) -> 'ChannelState'` — Detached CPU copies (original dtype) of the slices owned by the selected channels.
- `freeze_mask(model: 'nn.Module', selection: 'ChannelSelection', *, expert: 'str | None' = None) -> 'ParameterMask'` — Selected channels frozen; everything else in the affected FFNs may change.
- `ranked_channels(diff: 'Mapping[int, ChannelDiff]', metric: 'Metric' = 'l2') -> 'list[tuple[int, int, float]]'` — All ``(layer, channel, score)`` sorted by score descending, ties by (layer, channel).
- `rollback_channels(model: 'nn.Module', reference: 'ChannelState', selection: 'ChannelSelection | None' = None, *, allow_cast: 'bool' = False, expert: 'str | None' = None, allow_path_mismatch: 'bool' = False) -> 'RollbackReport'` — Restore ONLY the selected channels' slices (gate row, up row, down col, biases) to ``reference``.
- `setup_channel_training(model: 'nn.Module', selection: 'ChannelSelection', *, freeze_rest: 'bool' = True, disable_grad_outside: 'bool' = False, gradient_masks: 'bool' = True, expert: 'str | None' = None) -> 'ChannelTrainingHandle'` — Prepare ``model`` so that only the selected channel slices (of ``expert``) can change.
- `top_changed_channels(diff: 'Mapping[int, ChannelDiff]', k: 'int', *, metric: 'Metric' = 'l2', per_layer: 'bool' = False, min_score: 'float | None' = None) -> 'ChannelSelection'` — The ``k`` most-changed channels (globally or per layer); ``min_score`` drops scores <= it.
- `verify_modification_scope(before: 'Mapping[str, torch.Tensor]', after: 'Mapping[str, torch.Tensor] | nn.Module', allowed: 'ParameterMask | ChannelSelection', model: 'nn.Module', *, unlisted: "Literal['frozen', 'free']" = 'frozen', strict: 'bool' = False, expert: 'str | None' = None) -> 'ScopeReport'` — Report every element of ``before`` that differs in ``after`` outside ``allowed``.

## `research_rl.models.model_interface`

Model introspection shared by all parameter-level tools.

- **class `ModelInterfaceError`**
- **class `SwiGLURef`** — Location of one SwiGLU FFN inside a model.
  - `down`
  - `gate`
  - `intermediate_size`
  - `param_names(self) -> 'dict[str, str]'` — Fully-qualified weight names, keyed by projection name.
  - `up`
- `count_parameters(model: 'nn.Module', trainable_only: 'bool' = False) -> 'int'`
- `find_swiglu_ffns(model: 'nn.Module', layers: 'Iterable[int] | None' = None, *, expert: 'str | None' = None) -> 'list[SwiGLURef]'` — Locate the SwiGLU FFN (``layer.mlp``) of each decoder layer.
- `fingerprint(tensors: 'dict[str, torch.Tensor] | nn.Module', names: 'Iterable[str] | None' = None) -> 'str'` — SHA-256 over names, shapes, dtypes and raw bytes. Bit-exact identity check.
- `get_decoder_layers(model: 'nn.Module') -> 'tuple[str, nn.ModuleList]'` — Return ``(path, layers)`` for the decoder block list of an HF causal LM.
- `is_swiglu(module: 'nn.Module') -> 'bool'`
- `snapshot(model: 'nn.Module', names: 'Iterable[str] | None' = None) -> 'dict[str, torch.Tensor]'` — Detached CPU copy of parameters (and only parameters) for exact comparisons.

## `research_rl.models.parameter_mask`

Element-wise parameter masks.

- **class `MaskError`**
- **class `ParameterMask`**
  - `all_frozen(cls, model: 'nn.Module', names: 'Iterable[str]') -> 'ParameterMask'`
  - `all_trainable(cls, model: 'nn.Module', names: 'Iterable[str]') -> 'ParameterMask'`
  - `changed_outside(self, before: 'Mapping[str, torch.Tensor]', after: 'Mapping[str, torch.Tensor]') -> 'dict[str, int]'` — Number of masked-out elements that differ between two snapshots (should be all 0).
  - `from_state_dict(cls, state: 'Mapping[str, torch.Tensor]') -> 'ParameterMask'`
  - `get(self, name: 'str') -> 'torch.Tensor | None'`
  - `intersection(self, other: 'ParameterMask') -> 'ParameterMask'`
  - `items(self) -> 'Iterator[tuple[str, torch.Tensor]]'`
  - `names(self) -> 'list[str]'`
  - `num_frozen(self) -> 'int'`
  - `num_trainable(self) -> 'int'`
  - `restore_frozen(self, model: 'nn.Module', reference: 'Mapping[str, torch.Tensor]') -> 'None'` — Write ``reference`` back into every masked-out element of ``model``.
  - `set(self, name: 'str', mask: 'torch.Tensor') -> 'None'`
  - `state_dict(self) -> 'dict[str, torch.Tensor]'`
  - `to(self, device: 'torch.device | str') -> 'ParameterMask'`
  - `union(self, other: 'ParameterMask') -> 'ParameterMask'` — Element may change if it may change in either mask.
  - `validate(self, model: 'nn.Module') -> 'None'`

## `research_rl.models.tiny`

Tiny, fully offline models and tokenizers for tests, CI and smoke runs.

- `build_char_tokenizer(chars: 'str' = '0123456789+-*/=()^ abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ.,?!:;\'"\n<>{}[]\\_$#') -> 'PreTrainedTokenizerFast'`
- `build_tiny_llama(vocab_size: 'int', seed: 'int' = 0, **config_overrides: 'int') -> 'LlamaForCausalLM'`
- `save_tiny_model(path: 'str | Path', seed: 'int' = 0, **config_overrides: 'int') -> 'Path'` — Write a tiny Llama + char tokenizer to ``path`` (a normal HF model directory).
- `tiny_llama_config(vocab_size: 'int', hidden_size: 'int' = 32, intermediate_size: 'int' = 64, num_hidden_layers: 'int' = 2, num_attention_heads: 'int' = 4, num_key_value_heads: 'int' = 2, max_position_embeddings: 'int' = 256) -> 'LlamaConfig'`

## `research_rl.optim.gradient_hooks`

Backward-time gradient hooks.

- **class `GradientHookHandle`** — Holds the underlying torch handles; call :meth:`remove` to detach all hooks.
  - `remove(self) -> 'None'`
- `global_grad_norm(params: 'Iterable[torch.nn.Parameter]', norm_type: 'float' = 2.0) -> 'float'` — Reference global gradient norm (independent of ``clip_grad_norm_``).
- `register_gradient_masks(model: 'nn.Module', mask: 'ParameterMask') -> 'GradientHookHandle'` — Zero gradient elements that ``mask`` marks as frozen, at backward time.
- `register_gradient_transform(model: 'nn.Module', transform: 'GradTransform', names: 'Iterable[str] | None' = None) -> 'GradientHookHandle'` — Apply ``transform(name, grad) -> grad`` to the gradients of ``names`` (default: all trainable).

## `research_rl.optim.optimizer_wrapper`

``HookedOptimizer``: controls the *actual* parameter displacement of any torch optimizer.

- **class `HookedOptimizer`** — Wraps ``inner`` (any ``torch.optim.Optimizer``) and exposes the same interface.
  - `add_param_group(self, param_group: 'dict[str, Any]') -> 'None'`
  - `add_rule(self, rule: 'UpdateRule') -> 'None'`
  - `has_master(self, param: 'torch.Tensor') -> 'bool'` — True if ``param`` is backed by an fp32 master copy.
  - `load_state_dict(self, state_dict: 'dict[str, Any]') -> 'None'`
  - `master_of(self, param: 'torch.Tensor') -> 'torch.Tensor | None'` — The fp32 master of ``param`` (the live tensor, do not modify), or ``None``.
  - `named_parameters(self) -> 'list[tuple[str, torch.nn.Parameter]]'`
  - `param_groups`
  - `policy_version` — Number of parameter updates actually applied (skipped steps excluded).
  - `request_skip(self) -> 'None'` — Make a ``step()`` a no-op (counted in ``skipped_steps``, not in ``applied_updates``).
  - `resync_masters(self) -> 'None'` — Reset every master to ``param.float()`` (discards the sub-ulp information they hold).
  - `set_master(self, param: 'torch.nn.Parameter', value: 'torch.Tensor') -> 'None'` — Set the fp32 master of ``param`` to ``value`` and write ``round_to(value, nearest)`` into it.
  - `state`
  - `state_dict(self) -> 'dict[str, Any]'`
  - `step(self, closure: 'Callable[[], float] | None' = None) -> 'Any'`
  - `zero_grad(self, set_to_none: 'bool' = True) -> 'None'`
- `unwrap_hooked(optimizer: 'Any') -> 'HookedOptimizer | None'` — Find a HookedOptimizer inside accelerate/deepspeed wrappers (or return None).

## `research_rl.optim.parameter_prior`

Parameter priors built from task vectors (P²-MOPD-style parameter-direction fusion).

- **class `P2PriorAttachCallback`** — Attaches a :class:`P2PriorRule` to the (accelerate-wrapped) trainer optimizer at train begin.
  - `on_train_begin(self, args: 'Any', state: 'Any', control: 'Any', **kwargs: 'Any') -> 'None'`
- **class `P2PriorRule`** — Two-channel parameter-prior update of P²-MOPD (Algorithm 1) for :class:`HookedOptimizer`.
  - `attach(self, optimizer: 'Any') -> 'None'` — Bind to the optimizer and validate it (fails closed).
  - `callback(self) -> 'TrainerCallback'` — A ``TrainerCallback`` that calls :meth:`attach` with the trainer's optimizer at train begin.
  - `coefficient_gradient(self, named_params: 'Iterable[tuple[str, torch.nn.Parameter]]') -> 'torch.Tensor'` — g [F] = tied Σ <G_m, τ_{k,m}> from the current ``.grad`` (fp32, on CPU).
  - `coefficient_table(self) -> 'dict[str, torch.Tensor]'` — Effective α_{·,m} (shape [K]) for each prior module m.
  - `free_coefficients(self) -> 'dict[str, torch.Tensor]'` — Free coefficients per group (module / layer / model), shape [K] each.
  - `load_state_dict(self, state: 'Mapping[str, Any]') -> 'None'`
  - `lr_factor(self, step: 'int') -> 'float'` — Multiplier of ``lr`` at 1-based optimizer step ``step`` (``lr_lambda`` / warm-up / 1).
  - `materialize(self, model: 'nn.Module', base: 'Mapping[str, torch.Tensor]') -> 'None'` — Write W_m = B_m + Σ_k α_{k,m} τ_{k,m} (current α) into ``model`` for every prior module.
  - `on_attach(self, optimizer: 'Any') -> 'None'` — HookedOptimizer calls this when the rule is registered: validate immediately (fail closed).
  - `on_skip(self, ctx: 'StepContext') -> 'None'`
  - `post_grad(self, ctx: 'StepContext', named_params: 'list[tuple[str, torch.nn.Parameter]]') -> 'None'`
  - `pre_write(self, ctx: 'StepContext', name: 'str', param: 'torch.nn.Parameter', old: 'torch.Tensor', new: 'torch.Tensor') -> 'torch.Tensor | None'`
  - `state_dict(self) -> 'dict[str, Any]'`
  - `tracks(self, name: 'str', param: 'torch.nn.Parameter') -> 'bool'`
  - `trajectory_tensor(self) -> 'torch.Tensor'` — ``[T+1, F]`` free coefficients after each coefficient update (row 0 = initialisation).
- **class `PriorSubspaceRule`** — Project the candidate update Δ (after Adam/decay) onto or away from span{τ_k[p]} per parameter.
  - `basis(self, name: 'str', device: 'torch.device | None' = None) -> 'torch.Tensor'`
  - `load_state_dict(self, state: 'Mapping[str, Any]') -> 'None'`
  - `state_dict(self) -> 'dict[str, Any]'`
  - `tracks(self, name: 'str', param: 'torch.nn.Parameter') -> 'bool'`
  - `transform_update(self, ctx: 'StepContext', name: 'str', param: 'torch.nn.Parameter', old: 'torch.Tensor', delta: 'torch.Tensor') -> 'torch.Tensor | None'`
- **class `SpanProjection`** — Least-squares decomposition ``d = Σ_k c_k τ_k + residual``.
- **class `TaskVector`** — τ = θ_expert − θ_base over a set of named parameters (stored in ``dtype``, default fp32).
  - `cosine_similarity(self, other: 'TaskVector') -> 'float'` — Cosine of the angle between the two flattened task vectors (0.0 if either is zero).
  - `dot(self, other: 'TaskVector') -> 'float'`
  - `from_models(cls, expert: 'nn.Module', base: 'nn.Module', names: 'Iterable[str] | None' = None, dtype: 'torch.dtype' = torch.float32) -> 'TaskVector'`
  - `from_state_dict(cls, state: 'Mapping[str, torch.Tensor]') -> 'TaskVector'`
  - `from_state_dicts(cls, expert: 'Mapping[str, torch.Tensor]', base: 'Mapping[str, torch.Tensor]', names: 'Iterable[str] | None' = None, dtype: 'torch.dtype' = torch.float32) -> 'TaskVector'`
  - `items(self) -> 'Iterable[tuple[str, torch.Tensor]]'`
  - `names(self) -> 'list[str]'`
  - `norm(self) -> 'float'` — Global L2 norm over all covered parameters (computed in fp64).
  - `param_cosine_similarity(self, other: 'TaskVector') -> 'dict[str, float]'`
  - `param_norms(self) -> 'dict[str, float]'`
  - `state_dict(self) -> 'dict[str, torch.Tensor]'`
  - `to(self, *args: 'Any', **kwargs: 'Any') -> 'TaskVector'`
- **class `TaskVectorPrior`** — θ(α) = θ_base + Σ_i α_i τ_i over named parameters.
  - `apply_to(self, model: 'nn.Module', coefficients: 'Coefficients') -> 'dict[str, torch.Tensor]'` — Write ``combine(coefficients)`` into ``model`` bit-for-bit; returns the written tensors.
  - `combine(self, coefficients: 'Coefficients') -> 'dict[str, torch.Tensor]'` — Effective parameters θ_base + Σ_i α_i τ_i, returned in the base dtype.
  - `functional_call(self, model: 'nn.Module', coefficients: 'Coefficients', args: 'Any' = (), kwargs: 'Mapping[str, Any] | None' = None) -> 'Any'` — Run ``model`` with θ(α) substituted via ``torch.func.functional_call`` (differentiable in α).
  - `load_state_dict(self, state: 'Mapping[str, Any]') -> 'None'`
  - `project(self, name: 'str', displacement: 'torch.Tensor', rcond: 'float' = 1e-10) -> 'SpanProjection'` — Least-squares projection of a displacement of parameter ``name`` onto span{τ_k[name]}.
  - `register(self, name: 'str', task_vector: 'TaskVector') -> 'None'`
  - `state_dict(self) -> 'dict[str, Any]'`
  - `task_matrix(self, name: 'str') -> 'torch.Tensor'` — ``[K, *shape]`` stack of the task vectors at parameter ``name`` (zeros where uncovered).
  - `task_names`
- `default_prior_modules(names: 'Iterable[str]') -> 'list[str]'` — Parameter names of the linear projections q/k/v/o and gate/up/down (``*_proj.weight``).
- `orthonormal_span_basis(basis: 'torch.Tensor', rcond: 'float' = 1e-06) -> 'torch.Tensor'` — Orthonormal basis ``U`` [N, r] of the column span of ``basis`` [N, K] (SVD, rank-revealing).
- `project_onto_span(displacement: 'torch.Tensor', basis: 'Sequence[torch.Tensor] | torch.Tensor', *, rcond: 'float' = 1e-10, dtype: 'torch.dtype' = torch.float64) -> 'SpanProjection'` — Project ``displacement`` onto span{basis_k} by least squares (paper's analysis utility).

## `research_rl.optim.precision`

Low-precision weight updates: dtype checks, stochastic rounding and reproducible rounding RNG.

- `is_low_precision(dtype: 'torch.dtype') -> 'bool'` — True for bf16 and fp16 (the dtypes whose round-to-nearest writes drop small updates).
- `low_precision_trainable(named_params: 'Iterable[tuple[str, torch.Tensor]]') -> 'list[str]'` — Names of parameters with ``requires_grad=True`` stored in bf16/fp16 (in iteration order).
- `round_to(x: 'torch.Tensor', dtype: 'torch.dtype', mode: 'str' = 'nearest', generator: 'torch.Generator | None' = None) -> 'torch.Tensor'` — ``x.to(dtype)`` (``mode="nearest"``, round-to-nearest-even) or :func:`stochastic_round`.
- `rounding_generator(seed: 'int', step: 'int', index: 'int', device: 'torch.device | str' = 'cpu', *, generator: 'torch.Generator | None' = None) -> 'torch.Generator'` — Generator on ``device`` seeded with :func:`rounding_seed_for` ``(seed, step, index)``.
- `rounding_seed_for(seed: 'int', step: 'int', index: 'int') -> 'int'` — A well-mixed 63-bit seed for (seed, optimizer step, parameter index); pure function.
- `stochastic_round(x: 'torch.Tensor', dtype: 'torch.dtype', generator: 'torch.Generator | None' = None) -> 'torch.Tensor'` — Round ``x`` to ``dtype`` stochastically (unbiased); returns a new tensor of ``dtype``.

## `research_rl.optim.update_interface`

The update-rule contract of :class:`~research_rl.optim.optimizer_wrapper.HookedOptimizer`.

- **class `DisplacementLedger`** — Accumulates the *actual* per-parameter displacement norm. Useful for diagnostics/tests.
  - `load_state_dict(self, state: 'Mapping[str, Any]') -> 'None'`
  - `post_step(self, ctx: 'StepContext', displacements: 'Mapping[str, torch.Tensor]') -> 'None'`
  - `state_dict(self) -> 'dict[str, Any]'`
  - `tracks(self, name: 'str', param: 'torch.nn.Parameter') -> 'bool'`
- **class `MaskRule`** — Hard element-wise mask: masked-out elements are bit-identical before and after every step.
  - `load_state_dict(self, state: 'Mapping[str, Any]') -> 'None'`
  - `mask_digest(self) -> 'str'`
  - `post_grad(self, ctx: 'StepContext', named_params: 'list[tuple[str, torch.nn.Parameter]]') -> 'None'`
  - `pre_write(self, ctx: 'StepContext', name: 'str', param: 'torch.nn.Parameter', old: 'torch.Tensor', new: 'torch.Tensor') -> 'torch.Tensor | None'`
  - `state_dict(self) -> 'dict[str, Any]'`
  - `tracks(self, name: 'str', param: 'torch.nn.Parameter') -> 'bool'`
- **class `StepContext`** — Information handed to every rule hook.
- **class `UpdateRule`** — Base class; override any subset of the hooks.
  - `load_state_dict(self, state: 'Mapping[str, Any]') -> 'None'`
  - `on_attach(self, optimizer: 'Any') -> 'None'` — Called once when the rule is added to a HookedOptimizer (validate / bind here).
  - `on_skip(self, ctx: 'StepContext') -> 'None'` — Called instead of the other hooks when the step is skipped.
  - `post_grad(self, ctx: 'StepContext', named_params: 'list[tuple[str, torch.nn.Parameter]]') -> 'None'`
  - `post_step(self, ctx: 'StepContext', displacements: 'Mapping[str, torch.Tensor]') -> 'None'`
  - `pre_write(self, ctx: 'StepContext', name: 'str', param: 'torch.nn.Parameter', old: 'torch.Tensor', new: 'torch.Tensor') -> 'torch.Tensor | None'` — Return the tensor to write (same dtype as the parameter), or ``None`` to keep ``new``.
  - `state_dict(self) -> 'dict[str, Any]'`
  - `tracks(self, name: 'str', param: 'torch.nn.Parameter') -> 'bool'`
  - `transform_update(self, ctx: 'StepContext', name: 'str', param: 'torch.nn.Parameter', old: 'torch.Tensor', delta: 'torch.Tensor') -> 'torch.Tensor | None'` — Return a replacement fp32 Δ, or ``None`` to keep it.

## `research_rl.rewards.code_sandbox`

Code-execution rewards - an interface that is **disabled by default**.

- **class `CodeExecutionVerifier`** — stdout-matching code verifier on top of an isolated :class:`SandboxBackend` (disabled by default).
  - `verify(self, completion: 'str | None', reference: 'Any', **meta: 'Any') -> 'VerificationResult'`
- **class `ExecutionRequest`** — ExecutionRequest(code: 'str', language: 'str', stdin: 'str', time_limit_s: 'float', memory_limit_mb: 'int')
- **class `ExecutionResult`** — ExecutionResult(status: 'ExecStatus', stdout: 'str' = '', stderr: 'str' = '', return_code: 'int | None' = None, details: 'dict[str, Any]' = <factory>)
- **class `IsolationGuarantees`** — What a backend promises about where and how code runs. Defaults promise nothing.
- **class `SandboxBackend`** — An isolated code executor. Implementations must never run code in the caller's process.
  - `guarantees`
  - `run(self, request: 'ExecutionRequest') -> 'ExecutionResult'`
- **class `SandboxDisabledError`** — Code execution was requested but the sandbox is disabled or not safely configured.
- **class `SandboxFusionBackend`** — HTTP client for an external SandboxFusion ``run_code`` endpoint (see module docstring caveats).
  - `guarantees`
  - `payload(self, request: 'ExecutionRequest') -> 'dict[str, Any]'`
  - `run(self, request: 'ExecutionRequest') -> 'ExecutionResult'`
- **class `SandboxPolicy`** — Operator policy for executing generated code. Everything is off / tight by default.
- `check_sandbox(policy: 'SandboxPolicy', backend: 'SandboxBackend | None') -> 'None'` — Raise :class:`SandboxDisabledError` unless ``policy`` + ``backend`` form a safe configuration.
- `extract_code(text: 'str', language: 'str' = 'python') -> 'str | None'` — Body of the code block to run: the last block tagged with ``language`` (```` ```python ```` / ```` ```py ````), else the last bare ```` ``` ```` block, else ``None``.
- `parse_sandbox_fusion_response(resp: 'Any') -> 'ExecutionResult'` — Map a SandboxFusion ``run_code`` JSON response to an :class:`ExecutionResult` (defensively).
- `urllib_transport(url: 'str', body: 'bytes', headers: 'Mapping[str, str]', timeout: 'float') -> 'tuple[int, bytes]'` — Default HTTP POST transport (stdlib only).

## `research_rl.rewards.math_verifier`

Deterministic final-answer verifier for math RLVR (GSM8K / MATH style).

- **class `MathVerifier`** — Final-answer math verifier; see the module docstring for the exact pipeline.
  - `canonical(self, normalized: 'str') -> 'Fraction | str'`
  - `equivalent(self, answer: 'str', reference: 'str') -> 'bool'` — Equivalence of two already-normalised answers (exact rational, else exact string).
  - `normalize(self, answer: 'str') -> 'str'`
  - `reference_answer(self, reference: 'Any') -> 'str'` — Normalised reference. A raw GSM8K solution (``... #### 72``) or a ``\boxed{}`` solution is reduced to its final answer; ints / floats / Fractions are stringified.
  - `strip_terminators(self, text: 'str') -> 'tuple[str, bool]'` — Cut ``text`` at the first terminator; returns ``(text, terminated)``.
- `normalize_answer(answer: 'str', *, strip_percent: 'bool' = True, strip_degrees: 'bool' = True) -> 'str'` — Canonical string form of a final answer.
- `parse_number(s: 'str', *, max_digits: 'int' = 1000, max_exponent: 'int' = 400) -> 'Fraction | None'` — Exact rational value of a normalised answer, or ``None`` if it is not a plain number.

## `research_rl.rewards.verifier_interface`

Verifier contract for verifiable rewards (RLVR) and the adapter to TRL reward functions.

- **class `BaseVerifier`** — Convenience base: subclasses implement :meth:`_verify` returning ``(status, parsed, reference, details)``.
  - `make_result(self, status: 'Status', parsed: 'str | None' = None, reference: 'str | None' = None, details: 'dict[str, Any] | None' = None) -> 'VerificationResult'`
  - `verify(self, completion: 'str | None', reference: 'Any', **meta: 'Any') -> 'VerificationResult'`
- **class `FatalVerifierError`** — An error that must abort reward computation instead of being scored as ``status="error"``.
- **class `RewardScheme`** — Reward assigned to each verification status (defaults: binary 1/0 RLVR reward).
  - `reward_for(self, status: 'Status') -> 'float'`
- **class `TRLRewardFunction`** — Picklable TRL reward callable built by :func:`as_trl_reward_func` (``__name__`` drives TRL logging).
- **class `VerificationResult`** — Outcome of verifying one completion.
- **class `Verifier`** — Anything with ``verify(completion, reference, **meta) -> VerificationResult``.
  - `verify(self, completion: 'str | None', reference: 'Any', **meta: 'Any') -> 'VerificationResult'`
- `as_trl_reward_func(verifier: 'Verifier', answer_key: 'str' = 'answer', name: 'str | None' = None, *, eos_token_ids: 'Sequence[int] | None' = None, truncated_key: 'str | None' = None, none_on_error: 'bool' = False, log_status_metrics: 'bool' = True, max_error_frac: 'float | None' = None) -> 'TRLRewardFunction'` — Wrap ``verifier`` as a TRL ``reward_funcs`` entry.
- `completion_to_text(completion: 'Any') -> 'str | None'` — Text of a TRL completion.

## `research_rl.rollout.agent_env`

Agent–environment contract compatible with TRL ``GRPOTrainer(environment_factory=...)``.

- **class `CalculatorEnv`** — Deterministic toy: reach ``target`` using the ``add`` / ``multiply`` tools.
  - `add(self, a: 'float', b: 'float') -> 'float'` — Add two numbers.
  - `multiply(self, a: 'float', b: 'float') -> 'float'` — Multiply two numbers.
- **class `EpisodeLog`** — Turns of one episode with per-turn rewards and an optional terminal reward (added to the last turn).
  - `add(self, action: 'str', observation: 'str' = '', reward: 'float' = 0.0, **info: 'Any') -> 'Turn'`
  - `advantages(self, values: 'Sequence[float] | torch.Tensor', gamma: 'float' = 1.0, lam: 'float' = 0.95, last_value: 'float' = 0.0) -> 'torch.Tensor'`
  - `discounted_returns(self, gamma: 'float' = 1.0) -> 'torch.Tensor'`
  - `rewards(self) -> 'torch.Tensor'` — Per-turn rewards ``[T]`` (terminal reward added to the last turn; a lone terminal reward if T=0).
  - `set_final_reward(self, reward: 'float') -> 'None'`
  - `to_dict(self) -> 'dict[str, Any]'`
  - `total_reward(self) -> 'float'`
  - `turn_credit(self, gamma: 'float' = 1.0, baseline: 'float | Sequence[float] | torch.Tensor' = 0.0) -> 'torch.Tensor'` — Per-turn credit G_t − b_t (``baseline`` scalar or per turn).
- **class `ResearchEnvironment`** — Base for TRL ``environment_factory`` environments.
  - `episode`
  - `get_reward(self) -> 'float'` — Total episode reward (per-turn rewards + terminal reward); called once per rollout by TRL.
  - `num_resets`
  - `reset(self, **kwargs: 'Any') -> 'str | None'` — Start a new episode. ``kwargs`` is the dataset row TRL passes (``prompt`` and other columns).
- **class `Turn`** — Turn(action: 'str', observation: 'str', reward: 'float', info: 'dict[str, Any]' = <factory>)
- `discounted_returns(rewards: 'Sequence[float] | torch.Tensor', gamma: 'float') -> 'torch.Tensor'` — G_t = r_t + γ G_{t+1} (G_T = 0), float64.
- `generalized_advantages(rewards: 'Sequence[float] | torch.Tensor', values: 'Sequence[float] | torch.Tensor', gamma: 'float', lam: 'float', last_value: 'float' = 0.0) -> 'torch.Tensor'` — GAE: δ_t = r_t + γ V_{t+1} − V_t (V_T = ``last_value``), A_t = Σ_l (γλ)^l δ_{t+l}.
- `spread_to_tokens(per_turn: 'Sequence[float] | torch.Tensor', token_counts: 'Sequence[int]') -> 'torch.Tensor'` — Repeat each turn's value over that turn's tokens (``[Σ token_counts]``).

## `research_rl.rollout.policy_version`

Policy versions: which parameters generated a rollout.

- `policy_lag(current_version: 'int', batch: 'Mapping[str, Any]') -> 'int | None'` — Number of applied updates between generating ``batch`` and now (``None`` if unstamped).
- `policy_version_of(obj: 'Any') -> 'int'` — Current policy version of a trainer or optimizer.
- `read_policy_version(batch: 'Mapping[str, Any]') -> 'int | None'` — The stamped version of ``batch`` (``None`` if it was never stamped).
- `stamp_policy_version(batch: 'MutableMapping[str, Any]', version: 'int') -> 'MutableMapping[str, Any]'` — Store ``version`` in ``batch`` (in place) as a 0-dim ``long`` tensor; returns ``batch``.

## `research_rl.rollout.trajectory`

Rollout bookkeeping: what was generated, how it was scored, and by which policy version.

- **class `RolloutBatch`** — One generation batch (the local, per-process part) together with its scores.
  - `frac_zero_advantage` — Fraction of samples without learning signal (see :meth:`_per_sample_zero_adv`).
  - `from_tensors(cls, *, prompt_ids: 'torch.Tensor', prompt_mask: 'torch.Tensor', completion_ids: 'torch.Tensor', completion_mask: 'torch.Tensor', rewards: 'torch.Tensor', advantages: 'torch.Tensor', policy_version: 'int', global_step: 'int' = 0, mode: 'str' = 'train', signal: 'torch.Tensor | None' = None, expert_ids: 'Sequence[str] | None' = None) -> 'RolloutBatch'` — Build a batch from (possibly GPU, possibly grad-carrying) tensors; stores detached CPU copies.
  - `has_nonzero_advantage` — Does any sample carry a learning signal?
  - `mean_reward` — Mean over scorable samples (NaN if none is scorable).
  - `num_loss_tokens`
  - `stats(self) -> 'dict[str, float | int | bool]'`
  - `trajectories(self) -> 'list[Trajectory]'` — Unbatch into :class:`Trajectory` objects (prompt left-padding removed via ``prompt_mask``).
- **class `Trajectory`** — A single sample: prompt tokens (padding removed), completion row and its mask.
  - `num_loss_tokens`
- **class `TrajectoryBuffer`** — Bounded log of recent rollout batches plus running counters (the counters are checkpointed).
  - `append(self, batch: 'RolloutBatch') -> 'None'`
  - `clear(self) -> 'None'`
  - `counters(self) -> 'dict[str, Any]'` — Running counters (JSON- and ``weights_only``-safe: ints, ``None`` and a ``{str: int}`` dict).
  - `latest(self) -> 'RolloutBatch | None'`
  - `load_state_dict(self, state: 'Mapping[str, Any]') -> 'None'`
  - `policy_versions(self) -> 'list[int]'`
  - `record_window(self, *, has_signal: 'bool', skipped: 'bool') -> 'None'`
  - `state_dict(self) -> 'dict[str, Any]'`
  - `stats_history(self) -> 'list[dict[str, float | int | bool]]'`

## `research_rl.state.compute_controller`

Adaptive-compute contract for looped / recurrent-depth models (AIC, Best-of-Depth, ...).

- **class `BestOfDepthResult`** — BestOfDepthResult(output: 'torch.Tensor', best_depth: 'torch.Tensor', scores: 'torch.Tensor', entry: 'LedgerEntry | None' = None)
- **class `ComputeController`** — Base class. Override :meth:`halt`; :meth:`start` is called once per inference call.
  - `halt(self, depth: 'int', previous: 'torch.Tensor', current: 'torch.Tensor') -> 'torch.Tensor'` — Bool ``[rows]``: True = stop after ``depth`` applications (rows = currently active rows).
  - `load_state_dict(self, state: 'Mapping[str, Any]') -> 'None'`
  - `start(self, batch_size: 'int', device: 'torch.device') -> 'None'`
  - `state_dict(self) -> 'dict[str, Any]'`
- **class `ComputeLedger`** — Records actual iteration depth and estimated cost per call and per sample.
  - `load_state_dict(self, state: 'Mapping[str, Any]') -> 'None'`
  - `mean_depth(self) -> 'float'`
  - `per_sample(self) -> 'dict[str, dict[str, float]]'` — ``{sample_id: {"iterations": Σ depth, "cost": Σ cost, "calls": #calls}}`` over all calls.
  - `record(self, depths: 'torch.Tensor | Sequence[int]', cost_per_iteration: 'float | Sequence[float] | torch.Tensor' = 1.0, *, call_id: 'str | None' = None, sample_ids: 'Sequence[Hashable] | None' = None, kind: 'str' = 'looped') -> 'LedgerEntry'`
  - `state_dict(self) -> 'dict[str, Any]'`
  - `total_cost(self) -> 'float'`
  - `total_iterations(self) -> 'int'`
- **class `FixedDepth`** — Every sample receives exactly ``depth`` iterations (capped by ``max_depth``).
  - `halt(self, depth: 'int', previous: 'torch.Tensor', current: 'torch.Tensor') -> 'torch.Tensor'`
  - `load_state_dict(self, state: 'Mapping[str, Any]') -> 'None'`
  - `state_dict(self) -> 'dict[str, Any]'`
- **class `LedgerEntry`** — LedgerEntry(call_id: 'str', kind: 'str', sample_ids: 'list[str]', depths: 'list[int]', cost_per_iteration: 'list[float]', costs: 'list[float]')
  - `to_dict(self) -> 'dict[str, Any]'`
  - `total_cost`
  - `total_iterations`
- **class `LoopResult`** — LoopResult(output: 'torch.Tensor', depths: 'torch.Tensor', halted_by_controller: 'torch.Tensor', entry: 'LedgerEntry | None' = None)
- **class `ThresholdHalting`** — Halt a row once ‖h_t − h_{t−1}‖₂ (over all non-batch dims) < ``eps`` and depth ≥ ``min_depth``.
  - `halt(self, depth: 'int', previous: 'torch.Tensor', current: 'torch.Tensor') -> 'torch.Tensor'`
  - `load_state_dict(self, state: 'Mapping[str, Any]') -> 'None'`
  - `state_dict(self) -> 'dict[str, Any]'`
- `best_of_depth(outputs_per_depth: 'torch.Tensor | Sequence[torch.Tensor]', scorer: 'Callable[[torch.Tensor], torch.Tensor]', *, ledger: 'ComputeLedger | None' = None, cost_per_iteration: 'float | Sequence[float] | torch.Tensor' = 1.0, call_id: 'str | None' = None, sample_ids: 'Sequence[Hashable] | None' = None, iterations_computed: 'torch.Tensor | Sequence[int] | None' = None) -> 'BestOfDepthResult'` — Pick, per sample, the depth whose output scores highest (ties → earliest depth).
- `collect_depth_outputs(block: 'Callable[[torch.Tensor], torch.Tensor]', h0: 'torch.Tensor', max_depth: 'int') -> 'torch.Tensor'` — ``[D, B, ...]`` stack of h_1..h_D (all rows run the full ``max_depth``).
- `run_looped(block: 'Callable[[torch.Tensor], torch.Tensor]', h0: 'torch.Tensor', controller: 'ComputeController', max_depth: 'int', *, ledger: 'ComputeLedger | None' = None, cost_per_iteration: 'float | Sequence[float] | torch.Tensor' = 1.0, call_id: 'str | None' = None, sample_ids: 'Sequence[Hashable] | None' = None) -> 'LoopResult'` — Apply the weight-shared ``block`` repeatedly with per-sample halting (see module docstring).

## `research_rl.state.recurrent_interface`

Per-sample recurrent state contract (memory carried across steps / turns / chunks).

- **class `FastSlowMemory`** — Two-timescale memory. Per row, with input x_t ∈ R^d and step counter n::
  - `init(self, batch_size: 'int', device: 'torch.device | str | None' = None, dtype: 'torch.dtype' = torch.float32) -> 'State'` — Fresh state: ``fast`` in ``dtype``, ``slow`` in :meth:`slow_storage_dtype`, ``steps`` long.
  - `load_state_dict(self, state: 'Mapping[str, Any]') -> 'None'`
  - `read(self, state: 'State') -> 'torch.Tensor'` — ``concat(fast, slow)`` in ``promote_types(fast dtype, slow dtype)``.
  - `slow_storage_dtype(self, dtype: 'torch.dtype') -> 'torch.dtype'` — Dtype in which the slow state is stored for a requested state ``dtype``.
  - `state_dict(self) -> 'dict[str, Any]'`
  - `state_dtype(self, state: 'State') -> 'torch.dtype'`
  - `update(self, state: 'State', inputs: 'torch.Tensor', active: 'torch.Tensor | None' = None) -> 'State'`
- **class `RecurrentState`** — Abstract per-sample recurrent state (see module docstring for the contract).
  - `init(self, batch_size: 'int', device: 'torch.device | str | None' = None, dtype: 'torch.dtype' = torch.float32) -> 'State'`
  - `load_state_dict(self, state: 'Mapping[str, Any]') -> 'None'`
  - `read(self, state: 'State') -> 'torch.Tensor'`
  - `reset(self, state: 'State', row_mask: 'torch.Tensor') -> 'State'` — Re-initialise rows where ``row_mask`` is True; other rows are returned unchanged.
  - `state_dict(self) -> 'dict[str, Any]'`
  - `state_dtype(self, state: 'State') -> 'torch.dtype'` — The dtype that was requested from :meth:`init` for ``state`` (used by :meth:`reset`).
  - `update(self, state: 'State', inputs: 'torch.Tensor', active: 'torch.Tensor | None' = None) -> 'State'`
- **class `StateStore`** — Per-sample-id storage of :class:`RecurrentState` rows (no implicit cross-sample sharing).
  - `forget(self, ids: 'Sequence[Hashable]') -> 'None'` — Drop the state of ``ids`` (they start fresh next time).
  - `gather(self, ids: 'Sequence[Hashable]') -> 'State'` — Batch the states of ``ids`` (fresh state for unseen ids). Must be followed by ``scatter``.
  - `get(self, sample_id: 'Hashable') -> 'State'`
  - `ids(self) -> 'list[int | str]'`
  - `load_state_dict(self, state: 'Mapping[str, Any]') -> 'None'`
  - `release(self) -> 'None'` — Abandon the gathered batch: nothing is stored, and ids first seen in that ``gather`` stay unknown.
  - `scatter(self, ids: 'Sequence[Hashable]', state: 'State') -> 'None'` — Store row i of ``state`` under ``ids[i]``; ids must equal those of the last ``gather``.
  - `state_dict(self) -> 'dict[str, Any]'`
  - `step(self, ids: 'Sequence[Hashable]', inputs: 'torch.Tensor') -> 'torch.Tensor'` — gather → update → scatter; returns the readout after the update.
- `batch_size_of(state: 'State') -> 'int'`

## `research_rl.trainers.base`

Glue between ResearchRL and TRL trainers.

- **class `ResearchCallback`** — Feeds the experiment recorder and writes ResearchRL extras into each checkpoint.
  - `bind(self, trainer: 'Any') -> 'None'`
  - `on_log(self, args: 'Any', state: 'Any', control: 'Any', logs: 'dict[str, Any] | None' = None, **kw: 'Any') -> 'None'`
  - `on_save(self, args: 'Any', state: 'Any', control: 'Any', **kw: 'Any') -> 'None'`
  - `on_train_begin(self, args: 'Any', state: 'Any', control: 'Any', **kw: 'Any') -> 'None'`
  - `on_train_end(self, args: 'Any', state: 'Any', control: 'Any', **kw: 'Any') -> 'None'`
- **class `ResearchRun`** — A configured run: the TRL trainer plus its recorder.
  - `hooked_optimizer`
  - `train(self, resume_from_checkpoint: 'str | bool | None' = None) -> 'dict[str, Any]'`
- **class `ResearchTrainerMixin`** — Mix *before* a TRL trainer class: ``class X(ResearchTrainerMixin, GRPOTrainer)``.
  - `create_optimizer(self, model: 'Any' = None) -> 'Any'`
  - `hooked_optimizer`
  - `train(self, resume_from_checkpoint: 'str | bool | None' = None, **kwargs: 'Any') -> 'Any'`
- **class `TrainerAdapter`** — Base adapter. Subclasses set the class variables and may override ``trainer_kwargs``.
  - `build(self, exp: 'ExperimentConfig', *, train_dataset: 'Any', model: 'PreTrainedModel | str | None' = None, processing_class: 'Any' = None, eval_dataset: 'Any' = None, update_rules: 'Sequence[UpdateRule]' = (), components: 'Mapping[str, StatefulComponent] | None' = None, callbacks: 'Sequence[TrainerCallback]' = (), record: 'bool' = True, **algo_kwargs: 'Any') -> 'ResearchRun'`
  - `build_trl_config(self, exp: 'ExperimentConfig') -> 'Any'`
  - `trainer_class(self) -> 'type'`
  - `trainer_kwargs(self, exp: 'ExperimentConfig', **algo_kwargs: 'Any') -> 'dict[str, Any]'` — Algorithm-specific keyword arguments for the TRL trainer constructor.
- `load_causal_lm(spec: 'ModelSpec') -> 'PreTrainedModel'`
- `load_tokenizer(spec: 'ModelSpec') -> 'Any'`
- `precision_info(model: 'torch.nn.Module', optim: 'OptimSpec') -> 'dict[str, Any]'` — ``{"param_dtypes": {dtype: #trainable tensors}, "master_weights", "write_rounding"}`` for the record.
- `research_trainer_class(trl_cls: 'type') -> 'type'` — Return ``ResearchTrainerMixin`` combined with ``trl_cls`` (cached).
- `resolve_dtype(name: 'str') -> 'torch.dtype | str'`

## `research_rl.trainers.distillation_adapter`

On-policy knowledge distillation adapter around TRL's ``DistillationTrainer``.

- **class `DistillationAdapter`** — ``TRAINERS.build("distillation")`` -> adapter for on-policy distillation.
  - `build(self, exp: 'ExperimentConfig', **kwargs: 'Any') -> 'ResearchRun'`
  - `trainer_kwargs(self, exp: 'ExperimentConfig', **algo_kwargs: 'Any') -> 'dict[str, Any]'`
  - `trl_config_cls(output_dir: str | None = None, per_device_train_batch_size: int = 8, num_train_epochs: float = 3.0, max_steps: int = -1, learning_rate: float = 1e-06, lr_scheduler_type: transformers.trainer_utils.SchedulerType | str = 'linear', lr_scheduler_kwargs: dict | str | None = None, warmup_steps: float = 0, optim: transformers.training_args.OptimizerNames | str = 'adamw_torch_fused', optim_args: str | None = None, weight_decay: float = 0.0, adam_beta1: float = 0.9, adam_beta2: float = 0.999, adam_epsilon: float = 1e-08, optim_target_modules: None | str | list[str] = None, gradient_accumulation_steps: int = 1, average_tokens_across_devices: bool = True, max_grad_norm: float = 1.0, label_smoothing_factor: float = 0.0, bf16: bool | None = None, fp16: bool = False, bf16_full_eval: bool = False, fp16_full_eval: bool = False, tf32: bool | None = None, activation_checkpointing: bool | None = None, activation_checkpointing_kwargs: dict[str, typing.Any] | str | None = None, gradient_checkpointing: bool = True, gradient_checkpointing_kwargs: dict[str, typing.Any] | str | None = None, torch_compile: bool = False, torch_compile_backend: str | None = None, torch_compile_mode: str | None = None, use_liger_kernel: bool = False, liger_kernel_config: dict[str, bool] | None = None, use_cache: bool = False, neftune_noise_alpha: float | None = None, torch_empty_cache_steps: int | None = None, auto_find_batch_size: bool = False, logging_strategy: transformers.trainer_utils.IntervalStrategy | str = 'steps', logging_steps: float = 10, logging_first_step: bool = False, log_on_each_node: bool = True, logging_nan_inf_filter: bool = True, include_num_input_tokens_seen: str | bool = 'no', log_level: str = 'passive', log_level_replica: str = 'warning', disable_tqdm: bool | None = None, report_to: None | str | list[str] = 'none', run_name: str | None = None, project: str = 'huggingface', trackio_space_id: str | None = None, trackio_bucket_id: str | None = None, trackio_static_space_id: Union[str, NoneType, Literal[False]] = None, eval_strategy: transformers.trainer_utils.IntervalStrategy | str = 'no', eval_steps: float | None = None, eval_delay: float = 0, per_device_eval_batch_size: int = 8, prediction_loss_only: bool = False, eval_on_start: bool = False, eval_do_concat_batches: bool = True, eval_use_gather_object: bool = False, eval_accumulation_steps: int | None = None, include_for_metrics: list[str] = <factory>, batch_eval_metrics: bool = False, save_only_model: bool = False, save_strategy: transformers.trainer_utils.SaveStrategy | str = 'steps', save_steps: float = 500, save_on_each_node: bool = False, save_total_limit: int | None = None, enable_jit_checkpoint: bool = False, push_to_hub: bool = False, hub_token: str | None = None, hub_private_repo: bool | None = None, hub_model_id: str | None = None, hub_strategy: transformers.trainer_utils.HubStrategy | str = 'every_save', hub_always_push: bool = False, hub_revision: str | None = None, load_best_model_at_end: bool = False, metric_for_best_model: str | None = None, greater_is_better: bool | None = None, ignore_data_skip: bool = False, restore_callback_states_from_checkpoint: bool = False, full_determinism: bool = False, seed: int = 42, data_seed: int | None = None, use_cpu: bool = False, accelerator_config: dict | str | None = None, parallelism_config: accelerate.parallelism_config.ParallelismConfig | None = None, dataloader_drop_last: bool = False, dataloader_num_workers: int = 0, dataloader_pin_memory: bool = True, dataloader_persistent_workers: bool = False, dataloader_prefetch_factor: int | None = None, dataloader_multiprocessing_context: str | None = None, dataloader_in_order: bool = True, remove_unused_columns: bool | None = False, label_names: list[str] | None = None, train_sampling_strategy: str = 'random', length_column_name: str = 'length', ddp_find_unused_parameters: bool | None = None, ddp_bucket_cap_mb: int | None = None, ddp_broadcast_buffers: bool | None = None, ddp_static_graph: bool | None = None, ddp_backend: str | None = None, ddp_timeout: int = 1800, fsdp: str | None = None, fsdp_config: dict[str, typing.Any] | str | None = None, deepspeed: dict | str | None = None, debug: str | list[transformers.debug_utils.DebugOption] = '', skip_memory_metrics: bool = True, do_train: bool = False, do_eval: bool = False, do_predict: bool = False, resume_from_checkpoint: str | None = None, local_rank: int = -1, model_init_kwargs: dict[str, typing.Any] | str | None = None, trust_remote_code: bool = False, teacher_model_name_or_path: str | None = None, teacher_model_revision: str | None = None, teacher_model_init_kwargs: dict[str, typing.Any] | str | None = None, disable_dropout: bool = False, max_completion_length: int | None = 512, ds3_gather_for_generation: bool = True, shuffle_dataset: bool | None = True, pad_to_multiple_of: int | str | None = 'auto', temperature: float = 1.0, top_p: float = 1.0, top_k: int = 0, min_p: float | None = None, generation_kwargs: dict | str | None = None, chat_template_kwargs: dict | str | None = None, repetition_penalty: float = 1.0, cache_implementation: str | None = None, use_vllm: bool = False, vllm_mode: str = 'colocate', vllm_model_impl: str = 'vllm', vllm_enable_sleep_mode: bool = False, vllm_structured_outputs_regex: str | None = None, vllm_server_base_url: str | None = None, vllm_server_host: str = '0.0.0.0', vllm_server_port: int = 8000, vllm_server_timeout: float = 240.0, vllm_group_port: int = 51216, vllm_gpu_memory_utilization: float = 0.3, vllm_max_model_length: int | None = None, vllm_tensor_parallel_size: int = 1, beta: float = 1.0, max_tool_calling_iterations: int | None = None, log_completions: bool = False, num_completions_to_print: int | None = None, log_unique_prompts: bool = False) -> None` — Configuration class for the [`DistillationTrainer`].
  - `trl_trainer_cls(model: 'str | PreTrainedModel | PeftModel', teacher_model: str | transformers.modeling_utils.PreTrainedModel | None = None, args: trl.trainer.distillation_config.DistillationConfig | None = None, train_dataset: datasets.arrow_dataset.Dataset | None = None, eval_dataset: datasets.arrow_dataset.Dataset | dict[str, datasets.arrow_dataset.Dataset] | None = None, processing_class: transformers.tokenization_utils_base.PreTrainedTokenizerBase | transformers.processing_utils.ProcessorMixin | None = None, callbacks: list[transformers.trainer_callback.TrainerCallback] | None = None, optimizers: tuple[torch.optim.optimizer.Optimizer | None, torch.optim.lr_scheduler.LambdaLR | None] = (None, None), quantization_config: 'BitsAndBytesConfig | None' = None, peft_config: 'PeftConfig | None' = None, tools: list[collections.abc.Callable] | None = None)` — Trainer for knowledge distillation. The student is trained on-policy — it generates the completions itself — to match the teacher's next-token distribution under a generalized Jensen-Shannon divergence (interpolating forward KL, reverse KL, and JSD via `beta`), as introduced in [On-Policy Distillation of Language Models](https://huggingface.co/papers/2306.13649).

## `research_rl.trainers.dpo_adapter`

Direct Preference Optimisation (DPO) adapter around TRL's ``DPOTrainer``.

- **class `DPOAdapter`** — ``TRAINERS.build("dpo")`` -> adapter for offline preference optimisation with ``DPOTrainer``.
  - `build(self, exp: 'ExperimentConfig', *, train_dataset: 'Any', model: 'PreTrainedModel | str | None' = None, ref_model: 'PreTrainedModel | str | None' = None, **kwargs: 'Any') -> 'ResearchRun'`
  - `trainer_class(self) -> 'type'`
  - `trl_config_cls(output_dir: str | None = None, per_device_train_batch_size: int = 8, num_train_epochs: float = 3.0, max_steps: int = -1, learning_rate: float = 1e-06, lr_scheduler_type: transformers.trainer_utils.SchedulerType | str = 'linear', lr_scheduler_kwargs: dict | str | None = None, warmup_steps: float = 0, optim: transformers.training_args.OptimizerNames | str = 'adamw_torch_fused', optim_args: str | None = None, weight_decay: float = 0.0, adam_beta1: float = 0.9, adam_beta2: float = 0.999, adam_epsilon: float = 1e-08, optim_target_modules: None | str | list[str] = None, gradient_accumulation_steps: int = 1, average_tokens_across_devices: bool = True, max_grad_norm: float = 1.0, label_smoothing_factor: float = 0.0, bf16: bool | None = None, fp16: bool = False, bf16_full_eval: bool = False, fp16_full_eval: bool = False, tf32: bool | None = None, activation_checkpointing: bool | None = None, activation_checkpointing_kwargs: dict[str, typing.Any] | str | None = None, gradient_checkpointing: bool = True, gradient_checkpointing_kwargs: dict[str, typing.Any] | str | None = None, torch_compile: bool = False, torch_compile_backend: str | None = None, torch_compile_mode: str | None = None, use_liger_kernel: bool = False, liger_kernel_config: dict[str, bool] | None = None, use_cache: bool = False, neftune_noise_alpha: float | None = None, torch_empty_cache_steps: int | None = None, auto_find_batch_size: bool = False, logging_strategy: transformers.trainer_utils.IntervalStrategy | str = 'steps', logging_steps: float = 10, logging_first_step: bool = False, log_on_each_node: bool = True, logging_nan_inf_filter: bool = True, include_num_input_tokens_seen: str | bool = 'no', log_level: str = 'passive', log_level_replica: str = 'warning', disable_tqdm: bool | None = None, report_to: None | str | list[str] = 'none', run_name: str | None = None, project: str = 'huggingface', trackio_space_id: str | None = None, trackio_bucket_id: str | None = None, trackio_static_space_id: Union[str, NoneType, Literal[False]] = None, eval_strategy: transformers.trainer_utils.IntervalStrategy | str = 'no', eval_steps: float | None = None, eval_delay: float = 0, per_device_eval_batch_size: int = 8, prediction_loss_only: bool = False, eval_on_start: bool = False, eval_do_concat_batches: bool = True, eval_use_gather_object: bool = False, eval_accumulation_steps: int | None = None, include_for_metrics: list[str] = <factory>, batch_eval_metrics: bool = False, save_only_model: bool = False, save_strategy: transformers.trainer_utils.SaveStrategy | str = 'steps', save_steps: float = 500, save_on_each_node: bool = False, save_total_limit: int | None = None, enable_jit_checkpoint: bool = False, push_to_hub: bool = False, hub_token: str | None = None, hub_private_repo: bool | None = None, hub_model_id: str | None = None, hub_strategy: transformers.trainer_utils.HubStrategy | str = 'every_save', hub_always_push: bool = False, hub_revision: str | None = None, load_best_model_at_end: bool = False, metric_for_best_model: str | None = None, greater_is_better: bool | None = None, ignore_data_skip: bool = False, restore_callback_states_from_checkpoint: bool = False, full_determinism: bool = False, seed: int = 42, data_seed: int | None = None, use_cpu: bool = False, accelerator_config: dict | str | None = None, parallelism_config: accelerate.parallelism_config.ParallelismConfig | None = None, dataloader_drop_last: bool = False, dataloader_num_workers: int = 0, dataloader_pin_memory: bool = True, dataloader_persistent_workers: bool = False, dataloader_prefetch_factor: int | None = None, dataloader_multiprocessing_context: str | None = None, dataloader_in_order: bool = True, remove_unused_columns: bool = True, label_names: list[str] | None = None, train_sampling_strategy: str = 'random', length_column_name: str = 'length', ddp_find_unused_parameters: bool | None = None, ddp_bucket_cap_mb: int | None = None, ddp_broadcast_buffers: bool | None = None, ddp_static_graph: bool | None = None, ddp_backend: str | None = None, ddp_timeout: int = 1800, fsdp: str | None = None, fsdp_config: dict[str, typing.Any] | str | None = None, deepspeed: dict | str | None = None, debug: str | list[transformers.debug_utils.DebugOption] = '', skip_memory_metrics: bool = True, do_train: bool = False, do_eval: bool = False, do_predict: bool = False, resume_from_checkpoint: str | None = None, local_rank: int = -1, model_init_kwargs: dict[str, typing.Any] | str | None = None, trust_remote_code: bool = False, router_aux_loss_coef: float | None = None, disable_dropout: bool = True, dataset_num_proc: int | None = None, max_length: int | None = 1024, truncation_mode: str = 'keep_start', padding_free: bool = False, pad_to_multiple_of: int | str | None = 'auto', precompute_ref_log_probs: bool = False, precompute_ref_batch_size: int | None = None, loss_type: list[str] = <factory>, loss_weights: list[float] | None = None, ld_alpha: float | None = None, f_divergence_type: str = 'reverse_kl', f_alpha_divergence_coef: float = 0.5, label_smoothing: float = 0.0, beta: float = 0.1, use_weighting: bool = False, discopop_tau: float = 0.05, activation_offloading: bool = False, sync_ref_model: bool = False, ref_model_mixup_alpha: float = 0.6, ref_model_sync_steps: int = 512, pad_token: str | None = None) -> None` — Configuration class for the [`DPOTrainer`].
  - `trl_trainer_cls(model: 'str | PreTrainedModel | PeftModel', ref_model: transformers.modeling_utils.PreTrainedModel | None = None, args: trl.trainer.dpo_config.DPOConfig | None = None, data_collator: collections.abc.Callable[[list[typing.Any]], dict[str, typing.Any]] | None = None, train_dataset: datasets.arrow_dataset.Dataset | datasets.iterable_dataset.IterableDataset | None = None, eval_dataset: datasets.arrow_dataset.Dataset | datasets.iterable_dataset.IterableDataset | datasets.dataset_dict.DatasetDict | datasets.dataset_dict.IterableDatasetDict | dict[str, datasets.arrow_dataset.Dataset | datasets.iterable_dataset.IterableDataset] | None = None, processing_class: transformers.tokenization_utils_base.PreTrainedTokenizerBase | transformers.processing_utils.ProcessorMixin | None = None, compute_metrics: collections.abc.Callable[[transformers.trainer_utils.EvalPrediction], dict] | None = None, callbacks: list[transformers.trainer_callback.TrainerCallback] | None = None, optimizers: tuple[torch.optim.optimizer.Optimizer | None, torch.optim.lr_scheduler.LambdaLR | None] = (None, None), quantization_config: 'BitsAndBytesConfig | None' = None, peft_config: 'PeftConfig | None' = None)` — Trainer for Direct Preference Optimization (DPO) method. This algorithm was initially proposed in the paper [Direct Preference Optimization: Your Language Model is Secretly a Reward Model](https://huggingface.co/papers/2305.18290). This class is a wrapper around the [`~transformers.Trainer`] class and inherits all of its attributes and methods.
- `frozen_copy(model: 'PreTrainedModel') -> 'PreTrainedModel'` — Deep copy with gradients disabled and eval mode (used as a fixed reference policy).

## `research_rl.trainers.grpo_adapter`

GRPO adapter on top of TRL's ``GRPOTrainer`` (reinforcement learning from verifiable/learned rewards).

- **class `GRPOAdapter`** — ``TRAINERS.build("grpo")``: maps an ``ExperimentConfig`` onto :class:`ResearchGRPOTrainer`.
  - `research_trainer_cls(*args: 'Any', skip_zero_advantage_steps: 'bool' = False, trajectory_capacity: 'int' = 16, advantage_shapers: 'Sequence[AdvantageShaper | str]' = (), expert_manager: 'ExpertManager | None' = None, expert_column: 'str' = 'expert', **kwargs: 'Any') -> 'None'` — ``GRPOTrainer`` + HookedOptimizer + policy-version stamps, trajectory log, zero-advantage skip.
  - `trl_config_cls(output_dir: str | None = None, per_device_train_batch_size: int = 8, num_train_epochs: float = 3.0, max_steps: int = -1, learning_rate: float = 1e-06, lr_scheduler_type: transformers.trainer_utils.SchedulerType | str = 'linear', lr_scheduler_kwargs: dict | str | None = None, warmup_steps: float = 0, optim: transformers.training_args.OptimizerNames | str = 'adamw_torch_fused', optim_args: str | None = None, weight_decay: float = 0.0, adam_beta1: float = 0.9, adam_beta2: float = 0.999, adam_epsilon: float = 1e-08, optim_target_modules: None | str | list[str] = None, gradient_accumulation_steps: int = 1, average_tokens_across_devices: bool = True, max_grad_norm: float = 1.0, label_smoothing_factor: float = 0.0, bf16: bool | None = None, fp16: bool = False, bf16_full_eval: bool = False, fp16_full_eval: bool = False, tf32: bool | None = None, activation_checkpointing: bool | None = None, activation_checkpointing_kwargs: dict[str, typing.Any] | str | None = None, gradient_checkpointing: bool = True, gradient_checkpointing_kwargs: dict[str, typing.Any] | str | None = None, torch_compile: bool = False, torch_compile_backend: str | None = None, torch_compile_mode: str | None = None, use_liger_kernel: bool = False, liger_kernel_config: dict[str, bool] | None = None, use_cache: bool = False, neftune_noise_alpha: float | None = None, torch_empty_cache_steps: int | None = None, auto_find_batch_size: bool = False, logging_strategy: transformers.trainer_utils.IntervalStrategy | str = 'steps', logging_steps: float = 10, logging_first_step: bool = False, log_on_each_node: bool = True, logging_nan_inf_filter: bool = True, include_num_input_tokens_seen: str | bool = 'no', log_level: str = 'passive', log_level_replica: str = 'warning', disable_tqdm: bool | None = None, report_to: None | str | list[str] = 'none', run_name: str | None = None, project: str = 'huggingface', trackio_space_id: str | None = None, trackio_bucket_id: str | None = None, trackio_static_space_id: Union[str, NoneType, Literal[False]] = None, eval_strategy: transformers.trainer_utils.IntervalStrategy | str = 'no', eval_steps: float | None = None, eval_delay: float = 0, per_device_eval_batch_size: int = 8, prediction_loss_only: bool = False, eval_on_start: bool = False, eval_do_concat_batches: bool = True, eval_use_gather_object: bool = False, eval_accumulation_steps: int | None = None, include_for_metrics: list[str] = <factory>, batch_eval_metrics: bool = False, save_only_model: bool = False, save_strategy: transformers.trainer_utils.SaveStrategy | str = 'steps', save_steps: float = 500, save_on_each_node: bool = False, save_total_limit: int | None = None, enable_jit_checkpoint: bool = False, push_to_hub: bool = False, hub_token: str | None = None, hub_private_repo: bool | None = None, hub_model_id: str | None = None, hub_strategy: transformers.trainer_utils.HubStrategy | str = 'every_save', hub_always_push: bool = False, hub_revision: str | None = None, load_best_model_at_end: bool = False, metric_for_best_model: str | None = None, greater_is_better: bool | None = None, ignore_data_skip: bool = False, restore_callback_states_from_checkpoint: bool = False, full_determinism: bool = False, seed: int = 42, data_seed: int | None = None, use_cpu: bool = False, accelerator_config: dict | str | None = None, parallelism_config: accelerate.parallelism_config.ParallelismConfig | None = None, dataloader_drop_last: bool = False, dataloader_num_workers: int = 0, dataloader_pin_memory: bool = True, dataloader_persistent_workers: bool = False, dataloader_prefetch_factor: int | None = None, dataloader_multiprocessing_context: str | None = None, dataloader_in_order: bool = True, remove_unused_columns: bool | None = False, label_names: list[str] | None = None, train_sampling_strategy: str = 'random', length_column_name: str = 'length', ddp_find_unused_parameters: bool | None = None, ddp_bucket_cap_mb: int | None = None, ddp_broadcast_buffers: bool | None = None, ddp_static_graph: bool | None = None, ddp_backend: str | None = None, ddp_timeout: int = 1800, fsdp: str | None = None, fsdp_config: dict[str, typing.Any] | str | None = None, deepspeed: dict | str | None = None, debug: str | list[transformers.debug_utils.DebugOption] = '', skip_memory_metrics: bool = True, do_train: bool = False, do_eval: bool = False, do_predict: bool = False, resume_from_checkpoint: str | None = None, local_rank: int = -1, model_init_kwargs: dict[str, typing.Any] | str | None = None, trust_remote_code: bool = False, router_aux_loss_coef: float | None = None, disable_dropout: bool = False, cast_lm_head_to_fp32: bool = False, num_generations: int | None = 8, num_generations_eval: int | None = None, max_completion_length: int | None = 512, ds3_gather_for_generation: bool = True, shuffle_dataset: bool | None = True, pad_to_multiple_of: int | str | None = 'auto', generation_batch_size: int | None = None, steps_per_generation: int | None = None, temperature: float = 1.0, top_p: float = 1.0, top_k: int = 0, min_p: float | None = None, generation_kwargs: dict | str | None = None, chat_template_kwargs: dict | str | None = None, repetition_penalty: float = 1.0, cache_implementation: str | None = None, use_vllm: bool = False, vllm_mode: str = 'colocate', vllm_model_impl: str = 'vllm', vllm_enable_sleep_mode: bool = False, vllm_structured_outputs_regex: str | None = None, vllm_server_base_url: str | None = None, vllm_server_host: str = '0.0.0.0', vllm_server_port: int = 8000, vllm_server_timeout: float = 240.0, vllm_group_port: int = 51216, vllm_gpu_memory_utilization: float = 0.3, vllm_max_model_length: int | None = None, vllm_tensor_parallel_size: int = 1, beta: float = 0.0, num_iterations: int = 1, epsilon: float = 0.2, delta: float | None = None, epsilon_high: float | None = None, sapo_temperature_neg: float = 1.05, sapo_temperature_pos: float = 1.0, vespo_k_pos: float = 2.0, vespo_lambda_pos: float = 3.0, vespo_k_neg: float = 3.0, vespo_lambda_neg: float = 2.0, importance_sampling_level: str = 'token', reward_weights: list[float] | None = None, multi_objective_aggregation: str = 'sum_then_normalize', scale_rewards: str = 'group', loss_type: str = 'dapo', mask_truncated_completions: bool = False, sync_ref_model: bool = False, ref_model_mixup_alpha: float = 0.6, ref_model_sync_steps: int = 512, top_entropy_quantile: float = 1.0, entropy_coef: float = 0.0, use_adaptive_entropy: bool = False, entropy_coef_min: float = 0.0, entropy_coef_max: float = 1.0, entropy_coef_delta: float = 0.005, entropy_target: float = 0.2, max_tool_calling_iterations: int | None = None, vllm_importance_sampling_correction: bool = True, vllm_importance_sampling_mode: str = 'sequence_mask', vllm_importance_sampling_clip_max: float | None = 3.0, vllm_importance_sampling_clip_min: float | None = None, off_policy_mask_threshold: float | None = None, use_bias_correction_kl: bool = True, log_completions: bool = False, log_multimodal: bool = True, num_completions_to_print: int | None = None, log_unique_prompts: bool = False, log_completions_hub_repo: str | None = None, use_transformers_continuous_batching: bool = False, transformers_continuous_batching_config: dict | str | None = None, use_transformers_paged: bool = False, vllm_importance_sampling_cap: float | None = None) -> None` — Configuration class for the [`GRPOTrainer`].
  - `trl_trainer_cls(model: 'str | PreTrainedModel | PeftModel', reward_funcs: str | transformers.modeling_utils.PreTrainedModel | collections.abc.Callable[..., list[float | None]] | list[str | transformers.modeling_utils.PreTrainedModel | collections.abc.Callable[..., list[float | None]]] | None = None, args: trl.trainer.grpo_config.GRPOConfig | None = None, train_dataset: datasets.arrow_dataset.Dataset | datasets.iterable_dataset.IterableDataset | None = None, eval_dataset: datasets.arrow_dataset.Dataset | datasets.iterable_dataset.IterableDataset | datasets.dataset_dict.DatasetDict | datasets.dataset_dict.IterableDatasetDict | dict[str, datasets.arrow_dataset.Dataset | datasets.iterable_dataset.IterableDataset] | None = None, processing_class: transformers.tokenization_utils_base.PreTrainedTokenizerBase | transformers.processing_utils.ProcessorMixin | None = None, reward_processing_classes: transformers.tokenization_utils_base.PreTrainedTokenizerBase | list[transformers.tokenization_utils_base.PreTrainedTokenizerBase] | None = None, callbacks: list[transformers.trainer_callback.TrainerCallback] | None = None, optimizers: tuple[torch.optim.optimizer.Optimizer | None, torch.optim.lr_scheduler.LambdaLR | None] = (None, None), quantization_config: 'BitsAndBytesConfig | None' = None, peft_config: 'PeftConfig | None' = None, tools: list[collections.abc.Callable] | None = None, rollout_func: collections.abc.Callable[[list[str], 'GRPOTrainer'], dict[str, typing.Any]] | None = None, environment_factory: collections.abc.Callable[[], trl.trainer.grpo_trainer._SupportsReset] | dict[str, collections.abc.Callable[[], trl.trainer.grpo_trainer._SupportsReset]] | None = None)` — Trainer for the Group Relative Policy Optimization (GRPO) method. This algorithm was initially proposed in the paper [DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models](https://huggingface.co/papers/2402.03300).
- **class `ResearchGRPOTrainer`** — ``GRPOTrainer`` + HookedOptimizer + policy-version stamps, trajectory log, zero-advantage skip.

## `research_rl.trainers.multi_teacher`

Multi-teacher on-policy distillation building blocks.

- **class `MultiTeacherBatch`** — Student completions with K teachers' next-token log-probs.
  - `aggregated(self, mode: 'str' = 'mixture') -> 'torch.Tensor'`
  - `forward_kl(self, student_logprobs: 'torch.Tensor', mode: 'str' = 'mixture') -> 'torch.Tensor'`
  - `num_teachers`
  - `reverse_kl(self, student_logprobs: 'torch.Tensor', mode: 'str' = 'mixture') -> 'torch.Tensor'`
  - `teacher_token_logprobs(self) -> 'torch.Tensor'` — ``[K, B, T]`` log-prob each teacher assigns to the sampled completion tokens.
- **class `TeacherEnsemble`** — Several frozen HF causal LMs (shared vocabulary) acting as one teacher with static weights.
  - `base_model`
  - `forward(self, input_ids: 'torch.Tensor', attention_mask: 'torch.Tensor | None' = None, **kwargs: 'Any') -> 'CausalLMOutputWithPast'`
  - `from_specs(cls, specs: 'Sequence[TeacherSpec]', mode: 'str' = 'mixture', dtype: 'torch.dtype | str | None' = None) -> 'TeacherEnsemble'`
  - `get_input_embeddings(self) -> 'nn.Module'`
  - `get_output_embeddings(self) -> '_EnsembleHead'`
  - `teacher_logprobs(self, input_ids: 'torch.Tensor', attention_mask: 'torch.Tensor | None' = None, **kwargs: 'Any') -> 'torch.Tensor'` — ``[K, B, L, V]`` fp32 log-probs of every teacher.
  - `trl_hidden_states(self, input_ids: 'torch.Tensor', attention_mask: 'torch.Tensor | None' = None, logits_to_keep: 'int | None' = None, **kwargs: 'Any') -> 'torch.Tensor'` — Hidden states that, through :meth:`get_output_embeddings`, give the aggregated logits.
- **class `TeacherSpec`** — A teacher: a model id/path or an instantiated causal LM, plus its aggregation weight.
  - `load(self, dtype: 'torch.dtype | str | None' = None) -> 'nn.Module'`
- `aggregate_teacher_logprobs(teacher_logprobs: 'torch.Tensor', weights: 'torch.Tensor | Sequence[float]', mode: 'str' = 'mixture', *, normalize: 'bool' = True) -> 'torch.Tensor'` — Combine K teachers' log-probabilities ``[K, ..., V]`` into one log-distribution ``[..., V]``.
- `masked_forward_kl(teacher_logprobs: 'torch.Tensor', student_logprobs: 'torch.Tensor', mask: 'torch.Tensor | None' = None, reduction: 'str' = 'mean') -> 'torch.Tensor'` — KL(p_teacher ‖ p_student) per token over the last (vocab) dim; ``mean`` averages valid tokens.
- `masked_reverse_kl(student_logprobs: 'torch.Tensor', teacher_logprobs: 'torch.Tensor', mask: 'torch.Tensor | None' = None, reduction: 'str' = 'mean') -> 'torch.Tensor'` — KL(p_student ‖ p_teacher) per token (the on-policy distillation objective).
- `normalize_weights(weights: 'torch.Tensor | Sequence[float]') -> 'torch.Tensor'` — Non-negative weights normalised to sum to 1 along dim 0 (the teacher axis), in float64.

## `research_rl.trainers.rl_common`

Shared machinery of the RL adapters (GRPO, RLOO).

- **class `AdvantageShaper`** — Rewrites the advantages of a generation batch (turn/step credit, sign rebalancing, per-token KL rewards, ...). See the module docstring for where it runs and how its ``signal`` is used.
- **class `RLRolloutMixin`** — Policy-version stamps, trajectory log, advantage shaping, zero-advantage skipping and expert routing for TRL GRPO/RLOO trainers.
  - `compute_loss(self, model: 'Any', inputs: 'Any', *args: 'Any', **kwargs: 'Any') -> 'Any'` — TRL's loss with the micro-batch's rows routed through their experts (from :data:`EXPERT_KEY`).
  - `expert_routing(self, batch: 'Mapping[str, Any]') -> 'Iterator[None]'` — Select the experts of ``batch``'s rows (a generation batch or micro-batch of this trainer).
  - `policy_version` — Applied parameter updates so far (see :func:`research_rl.rollout.policy_version_of`).
  - `training_step(self, model: 'Any', inputs: 'Any', *args: 'Any', **kwargs: 'Any') -> 'Any'` — Keeps the micro-batch's expert selection active through ``compute_loss`` *and* backward.
- **class `RLTrainerAdapter`** — Base for RL adapters: ``trainer_class()`` returns :attr:`research_trainer_cls`.
  - `build(self, exp: 'ExperimentConfig', **kwargs: 'Any') -> 'ResearchRun'`
  - `trainer_class(self) -> 'type'`
  - `trainer_kwargs(self, exp: 'ExperimentConfig', **algo_kwargs: 'Any') -> 'dict[str, Any]'`
- **class `RolloutRecordCallback`** — Copies the trajectory-buffer counters into ``record.json`` under ``"rollout"``.
  - `on_step_end(self, args: 'Any', state: 'Any', control: 'Any', **kwargs: 'Any') -> 'None'`
- **class `ShapedAdvantages`** — Result of an :class:`AdvantageShaper`.
- `eos_cut_length(ids: 'Sequence[int]', eos_token_ids: 'Sequence[int]', pad_token_id: 'int | None') -> 'int'` — Length of ``ids`` after cutting at the first EOS (kept) or stripping trailing pad tokens.
- `eos_truncating_rollout(rollout_func: 'RolloutFunc', eos_token_ids: 'Sequence[int] | None' = None, pad_token_id: 'int | None' = None) -> 'RolloutFunc'` — Wrap a GRPO ``rollout_func`` so each completion ends at its first EOS (see module docstring).
- `group_deviation_signal(rewards: 'torch.Tensor', num_generations: 'int') -> 'torch.Tensor'` — Per sample: does its reward differ from the mean of its group? (NaN = unscorable -> ``False``).
- `resume_is_generation_aligned(global_step: 'int', gradient_accumulation_steps: 'int', steps_per_generation: 'int', num_iterations: 'int') -> 'bool'` — Is ``global_step`` a generation boundary of TRL's GRPO/RLOO buffering?

## `research_rl.trainers.rloo_adapter`

RLOO adapter on top of TRL's ``RLOOTrainer`` (REINFORCE with a leave-one-out baseline).

- **class `RLOOAdapter`** — ``TRAINERS.build("rloo")``: maps an ``ExperimentConfig`` onto :class:`ResearchRLOOTrainer`.
  - `research_trainer_cls(*args: 'Any', skip_zero_advantage_steps: 'bool' = False, trajectory_capacity: 'int' = 16, advantage_shapers: 'Sequence[AdvantageShaper | str]' = (), expert_manager: 'ExpertManager | None' = None, expert_column: 'str' = 'expert', **kwargs: 'Any') -> 'None'` — ``RLOOTrainer`` + HookedOptimizer + policy-version stamps, trajectory log, zero-advantage skip.
  - `trl_config_cls(output_dir: str | None = None, per_device_train_batch_size: int = 8, num_train_epochs: float = 3.0, max_steps: int = -1, learning_rate: float = 1e-06, lr_scheduler_type: transformers.trainer_utils.SchedulerType | str = 'linear', lr_scheduler_kwargs: dict | str | None = None, warmup_steps: float = 0, optim: transformers.training_args.OptimizerNames | str = 'adamw_torch_fused', optim_args: str | None = None, weight_decay: float = 0.0, adam_beta1: float = 0.9, adam_beta2: float = 0.999, adam_epsilon: float = 1e-08, optim_target_modules: None | str | list[str] = None, gradient_accumulation_steps: int = 1, average_tokens_across_devices: bool = True, max_grad_norm: float = 1.0, label_smoothing_factor: float = 0.0, bf16: bool | None = None, fp16: bool = False, bf16_full_eval: bool = False, fp16_full_eval: bool = False, tf32: bool | None = None, activation_checkpointing: bool | None = None, activation_checkpointing_kwargs: dict[str, typing.Any] | str | None = None, gradient_checkpointing: bool = True, gradient_checkpointing_kwargs: dict[str, typing.Any] | str | None = None, torch_compile: bool = False, torch_compile_backend: str | None = None, torch_compile_mode: str | None = None, use_liger_kernel: bool = False, liger_kernel_config: dict[str, bool] | None = None, use_cache: bool = False, neftune_noise_alpha: float | None = None, torch_empty_cache_steps: int | None = None, auto_find_batch_size: bool = False, logging_strategy: transformers.trainer_utils.IntervalStrategy | str = 'steps', logging_steps: float = 10, logging_first_step: bool = False, log_on_each_node: bool = True, logging_nan_inf_filter: bool = True, include_num_input_tokens_seen: str | bool = 'no', log_level: str = 'passive', log_level_replica: str = 'warning', disable_tqdm: bool | None = None, report_to: None | str | list[str] = 'none', run_name: str | None = None, project: str = 'huggingface', trackio_space_id: str | None = None, trackio_bucket_id: str | None = None, trackio_static_space_id: Union[str, NoneType, Literal[False]] = None, eval_strategy: transformers.trainer_utils.IntervalStrategy | str = 'no', eval_steps: float | None = None, eval_delay: float = 0, per_device_eval_batch_size: int = 8, prediction_loss_only: bool = False, eval_on_start: bool = False, eval_do_concat_batches: bool = True, eval_use_gather_object: bool = False, eval_accumulation_steps: int | None = None, include_for_metrics: list[str] = <factory>, batch_eval_metrics: bool = False, save_only_model: bool = False, save_strategy: transformers.trainer_utils.SaveStrategy | str = 'steps', save_steps: float = 500, save_on_each_node: bool = False, save_total_limit: int | None = None, enable_jit_checkpoint: bool = False, push_to_hub: bool = False, hub_token: str | None = None, hub_private_repo: bool | None = None, hub_model_id: str | None = None, hub_strategy: transformers.trainer_utils.HubStrategy | str = 'every_save', hub_always_push: bool = False, hub_revision: str | None = None, load_best_model_at_end: bool = False, metric_for_best_model: str | None = None, greater_is_better: bool | None = None, ignore_data_skip: bool = False, restore_callback_states_from_checkpoint: bool = False, full_determinism: bool = False, seed: int = 42, data_seed: int | None = None, use_cpu: bool = False, accelerator_config: dict | str | None = None, parallelism_config: accelerate.parallelism_config.ParallelismConfig | None = None, dataloader_drop_last: bool = False, dataloader_num_workers: int = 0, dataloader_pin_memory: bool = True, dataloader_persistent_workers: bool = False, dataloader_prefetch_factor: int | None = None, dataloader_multiprocessing_context: str | None = None, dataloader_in_order: bool = True, remove_unused_columns: bool | None = False, label_names: list[str] | None = None, train_sampling_strategy: str = 'random', length_column_name: str = 'length', ddp_find_unused_parameters: bool | None = None, ddp_bucket_cap_mb: int | None = None, ddp_broadcast_buffers: bool | None = None, ddp_static_graph: bool | None = None, ddp_backend: str | None = None, ddp_timeout: int = 1800, fsdp: str | None = None, fsdp_config: dict[str, typing.Any] | str | None = None, deepspeed: dict | str | None = None, debug: str | list[transformers.debug_utils.DebugOption] = '', skip_memory_metrics: bool = True, do_train: bool = False, do_eval: bool = False, do_predict: bool = False, resume_from_checkpoint: str | None = None, local_rank: int = -1, model_init_kwargs: dict[str, typing.Any] | str | None = None, trust_remote_code: bool = False, router_aux_loss_coef: float | None = None, disable_dropout: bool = False, num_generations: int | None = 2, num_generations_eval: int | None = None, max_completion_length: int | None = 512, ds3_gather_for_generation: bool = True, shuffle_dataset: bool | None = True, pad_to_multiple_of: int | str | None = 'auto', generation_batch_size: int | None = None, steps_per_generation: int | None = None, temperature: float = 1.0, top_p: float = 1.0, top_k: int = 0, min_p: float | None = None, generation_kwargs: dict | str | None = None, chat_template_kwargs: dict | str | None = None, repetition_penalty: float = 1.0, cache_implementation: str | None = None, use_vllm: bool = False, vllm_mode: str = 'colocate', vllm_model_impl: str = 'vllm', vllm_enable_sleep_mode: bool = False, vllm_structured_outputs_regex: str | None = None, vllm_server_base_url: str | None = None, vllm_server_host: str = '0.0.0.0', vllm_server_port: int = 8000, vllm_server_timeout: float = 240.0, vllm_group_port: int = 51216, vllm_gpu_memory_utilization: float = 0.3, vllm_max_model_length: int | None = None, vllm_tensor_parallel_size: int = 1, beta: float = 0.05, num_iterations: int = 1, epsilon: float = 0.2, epsilon_high: float | None = None, reward_weights: list[float] | None = None, normalize_advantages: bool = False, reward_clip_range: tuple[float, float] | None = None, mask_truncated_completions: bool = False, sync_ref_model: bool = False, ref_model_mixup_alpha: float = 0.6, ref_model_sync_steps: int = 512, log_completions: bool = False, log_multimodal: bool = True, num_completions_to_print: int | None = None, log_unique_prompts: bool = False, use_transformers_continuous_batching: bool = False, transformers_continuous_batching_config: dict | str | None = None, use_transformers_paged: bool = False) -> None` — Configuration class for the [`RLOOTrainer`].
  - `trl_trainer_cls(model: 'str | PreTrainedModel | PeftModel', reward_funcs: str | transformers.modeling_utils.PreTrainedModel | collections.abc.Callable[..., list[float | None]] | list[str | transformers.modeling_utils.PreTrainedModel | collections.abc.Callable[..., list[float | None]]], args: trl.trainer.rloo_config.RLOOConfig | None = None, train_dataset: datasets.arrow_dataset.Dataset | datasets.iterable_dataset.IterableDataset | None = None, eval_dataset: datasets.arrow_dataset.Dataset | datasets.iterable_dataset.IterableDataset | datasets.dataset_dict.DatasetDict | datasets.dataset_dict.IterableDatasetDict | dict[str, datasets.arrow_dataset.Dataset | datasets.iterable_dataset.IterableDataset] | None = None, processing_class: transformers.tokenization_utils_base.PreTrainedTokenizerBase | transformers.processing_utils.ProcessorMixin | None = None, reward_processing_classes: transformers.tokenization_utils_base.PreTrainedTokenizerBase | list[transformers.tokenization_utils_base.PreTrainedTokenizerBase] | None = None, callbacks: list[transformers.trainer_callback.TrainerCallback] | None = None, optimizers: tuple[torch.optim.optimizer.Optimizer | None, torch.optim.lr_scheduler.LambdaLR | None] = (None, None), quantization_config: 'BitsAndBytesConfig | None' = None, peft_config: 'PeftConfig | None' = None)` — Trainer for the Reinforce Leave One Out (RLOO) method. This algorithm was initially proposed in the paper [Back to Basics: Revisiting REINFORCE Style Optimization for Learning from Human Feedback in LLMs](https://huggingface.co/papers/2402.14740).
- **class `ResearchRLOOTrainer`** — ``RLOOTrainer`` + HookedOptimizer + policy-version stamps, trajectory log, zero-advantage skip.

## `research_rl.trainers.sft_adapter`

Supervised fine-tuning (SFT) adapter around TRL's ``SFTTrainer``.

- **class `SFTAdapter`** — ``TRAINERS.build("sft")`` -> adapter mapping an ``ExperimentConfig`` onto ``SFTTrainer``.
  - `build(self, exp: 'ExperimentConfig', **kwargs: 'Any') -> 'ResearchRun'`
  - `trainer_class(self) -> 'type'`
  - `trl_config_cls(output_dir: str | None = None, per_device_train_batch_size: int = 8, num_train_epochs: float = 3.0, max_steps: int = -1, learning_rate: float = 2e-05, lr_scheduler_type: transformers.trainer_utils.SchedulerType | str = 'linear', lr_scheduler_kwargs: dict | str | None = None, warmup_steps: float = 0, optim: transformers.training_args.OptimizerNames | str = 'adamw_torch_fused', optim_args: str | None = None, weight_decay: float = 0.0, adam_beta1: float = 0.9, adam_beta2: float = 0.999, adam_epsilon: float = 1e-08, optim_target_modules: None | str | list[str] = None, gradient_accumulation_steps: int = 1, average_tokens_across_devices: bool = True, max_grad_norm: float = 1.0, label_smoothing_factor: float = 0.0, bf16: bool | None = None, fp16: bool = False, bf16_full_eval: bool = False, fp16_full_eval: bool = False, tf32: bool | None = None, activation_checkpointing: bool | None = None, activation_checkpointing_kwargs: dict[str, typing.Any] | str | None = None, gradient_checkpointing: bool = True, gradient_checkpointing_kwargs: dict[str, typing.Any] | str | None = None, torch_compile: bool = False, torch_compile_backend: str | None = None, torch_compile_mode: str | None = None, use_liger_kernel: bool = False, liger_kernel_config: dict[str, bool] | None = None, use_cache: bool = False, neftune_noise_alpha: float | None = None, torch_empty_cache_steps: int | None = None, auto_find_batch_size: bool = False, logging_strategy: transformers.trainer_utils.IntervalStrategy | str = 'steps', logging_steps: float = 10, logging_first_step: bool = False, log_on_each_node: bool = True, logging_nan_inf_filter: bool = True, include_num_input_tokens_seen: str | bool = 'no', log_level: str = 'passive', log_level_replica: str = 'warning', disable_tqdm: bool | None = None, report_to: None | str | list[str] = 'none', run_name: str | None = None, project: str = 'huggingface', trackio_space_id: str | None = None, trackio_bucket_id: str | None = None, trackio_static_space_id: Union[str, NoneType, Literal[False]] = None, eval_strategy: transformers.trainer_utils.IntervalStrategy | str = 'no', eval_steps: float | None = None, eval_delay: float = 0, per_device_eval_batch_size: int = 8, prediction_loss_only: bool = False, eval_on_start: bool = False, eval_do_concat_batches: bool = True, eval_use_gather_object: bool = False, eval_accumulation_steps: int | None = None, include_for_metrics: list[str] = <factory>, batch_eval_metrics: bool = False, save_only_model: bool = False, save_strategy: transformers.trainer_utils.SaveStrategy | str = 'steps', save_steps: float = 500, save_on_each_node: bool = False, save_total_limit: int | None = None, enable_jit_checkpoint: bool = False, push_to_hub: bool = False, hub_token: str | None = None, hub_private_repo: bool | None = None, hub_model_id: str | None = None, hub_strategy: transformers.trainer_utils.HubStrategy | str = 'every_save', hub_always_push: bool = False, hub_revision: str | None = None, load_best_model_at_end: bool = False, metric_for_best_model: str | None = None, greater_is_better: bool | None = None, ignore_data_skip: bool = False, restore_callback_states_from_checkpoint: bool = False, full_determinism: bool = False, seed: int = 42, data_seed: int | None = None, use_cpu: bool = False, accelerator_config: dict | str | None = None, parallelism_config: accelerate.parallelism_config.ParallelismConfig | None = None, dataloader_drop_last: bool = False, dataloader_num_workers: int = 0, dataloader_pin_memory: bool = True, dataloader_persistent_workers: bool = False, dataloader_prefetch_factor: int | None = None, dataloader_multiprocessing_context: str | None = None, dataloader_in_order: bool = True, remove_unused_columns: bool = True, label_names: list[str] | None = None, train_sampling_strategy: str = 'random', length_column_name: str = 'length', ddp_find_unused_parameters: bool | None = None, ddp_bucket_cap_mb: int | None = None, ddp_broadcast_buffers: bool | None = None, ddp_static_graph: bool | None = None, ddp_backend: str | None = None, ddp_timeout: int = 1800, fsdp: str | None = None, fsdp_config: dict[str, typing.Any] | str | None = None, deepspeed: dict | str | None = None, debug: str | list[transformers.debug_utils.DebugOption] = '', skip_memory_metrics: bool = True, do_train: bool = False, do_eval: bool = False, do_predict: bool = False, resume_from_checkpoint: str | None = None, local_rank: int = -1, model_init_kwargs: dict[str, typing.Any] | str | None = None, trust_remote_code: bool = False, router_aux_loss_coef: float | None = None, chat_template_path: str | None = None, dataset_text_field: str = 'text', dataset_kwargs: dict[str, typing.Any] | str | None = None, dataset_num_proc: int | None = None, eos_token: str | None = None, max_length: int | None = 1024, truncation_mode: str = 'keep_start', shuffle_dataset: bool = False, packing: bool = False, packing_strategy: str = 'bfd', padding_free: bool = False, pad_to_multiple_of: int | str | None = 'auto', eval_packing: bool | None = None, completion_only_loss: bool | None = None, assistant_only_loss: bool = False, loss_type: str = 'nll', activation_offloading: bool = False, pad_token: str | None = None) -> None` — Configuration class for the [`SFTTrainer`].
  - `trl_trainer_cls(model: 'str | PreTrainedModel | PeftModel', args: trl.trainer.sft_config.SFTConfig | transformers.training_args.TrainingArguments | None = None, data_collator: collections.abc.Callable[[list[typing.Any]], dict[str, typing.Any]] | None = None, train_dataset: datasets.arrow_dataset.Dataset | datasets.iterable_dataset.IterableDataset | None = None, eval_dataset: datasets.arrow_dataset.Dataset | datasets.iterable_dataset.IterableDataset | datasets.dataset_dict.DatasetDict | datasets.dataset_dict.IterableDatasetDict | dict[str, datasets.arrow_dataset.Dataset | datasets.iterable_dataset.IterableDataset] | None = None, processing_class: transformers.tokenization_utils_base.PreTrainedTokenizerBase | transformers.processing_utils.ProcessorMixin | None = None, compute_loss_func: collections.abc.Callable | None = None, compute_metrics: collections.abc.Callable[[transformers.trainer_utils.EvalPrediction], dict] | None = None, callbacks: list[transformers.trainer_callback.TrainerCallback] | None = None, optimizers: tuple[torch.optim.optimizer.Optimizer | None, torch.optim.lr_scheduler.LambdaLR | None] = (None, None), optimizer_cls_and_kwargs: tuple[type[torch.optim.optimizer.Optimizer], dict[str, typing.Any]] | None = None, preprocess_logits_for_metrics: collections.abc.Callable[[torch.Tensor, torch.Tensor], torch.Tensor] | None = None, quantization_config: 'BitsAndBytesConfig | None' = None, peft_config: 'PeftConfig | None' = None, formatting_func: collections.abc.Callable[[dict], str] | None = None)` — Trainer for Supervised Fine-Tuning (SFT) method.
- **class `TrainTokensCallback`** — Copies ``trainer._total_train_tokens`` into the ResearchRL checkpoint state after every step.
  - `on_step_end(self, args: 'Any', state: 'Any', control: 'Any', **kw: 'Any') -> 'None'`
- **class `TrainTokensResumeMixin`** — Mixed in directly above a TRL trainer that counts tokens in ``_total_train_tokens``.
  - `train(self, resume_from_checkpoint: 'str | bool | None' = None, **kwargs: 'Any') -> 'Any'`
- `token_resume_trainer_class(trl_cls: 'type') -> 'type'` — ``research_trainer_class`` of ``trl_cls`` with :class:`TrainTokensResumeMixin` (cached).
- `track_train_tokens(run: 'ResearchRun') -> 'ResearchRun'`

## `research_rl.utils.logging`

Consistent loggers for ResearchRL code, with an optional rank-0-only mode for multi-process runs.

- `get_logger(name: 'str', level: 'int | str' = 20, *, rank_zero_only: 'bool' = False, stream: 'TextIO | None' = None) -> 'logging.Logger'` — Return ``logging.getLogger(name)`` with exactly one ResearchRL handler attached.
- `get_rank() -> 'int'` — Global process rank: ``torch.distributed`` if initialised, else ``$RANK``, else 0.

## `research_rl.utils.profiling`

Timing, throughput and hardware/memory reports.

- **class `StepTimer`** — Records wall-clock durations of steps (and optionally tokens processed per step).
  - `count`
  - `last`
  - `start(self) -> 'None'`
  - `step(self, tokens: 'int | None' = None) -> 'Iterator[StepTimer]'`
  - `stop(self, tokens: 'int | None' = None) -> 'float'`
  - `summary(self) -> 'dict[str, float | int | None]'` — Aggregate statistics; ``tokens_per_second`` only when every step reported tokens.
  - `total`
- `gpu_memory_report(device: 'int | None' = None) -> 'dict[str, Any]'` — Measured CUDA memory statistics, or ``{"status": "NOT_TESTED", ...}`` without CUDA.
- `hardware_report() -> 'dict[str, Any]'` — What this process runs on. GPU entries are NOT_TESTED unless CUDA is available.
- `measure_throughput(fn: 'Callable[[], Any]', n_steps: 'int', warmup: 'int' = 1, tokens_per_step: 'int | None' = None, *, clock: 'Clock' = <built-in function perf_counter>, synchronize: 'Sync' = 'auto') -> 'dict[str, Any]'` — Call ``fn`` ``warmup`` times untimed, then ``n_steps`` times timed as one block.

## `research_rl.utils.smoke`

End-to-end smoke check: a real TRL trainer step with ResearchRL update rules installed.

- `trl_rules_smoke(output_dir: 'str | Path', steps: 'int' = 2, bf16: 'bool' = False) -> 'dict[str, Any]'` — Run ``steps`` SFT steps through TRL with MaskRule + P2PriorRule; return the checks.
