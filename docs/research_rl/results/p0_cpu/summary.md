# Capability pilot P0 summary

**CPU tiny-model pilot (P0). Not evidence about 7-8B models; no GPU numbers.**

Every number below comes from a tiny Llama on synthetic digit-string tasks on CPU. GPU metrics: NOT_TESTED. Smoke runs (`--smoke`) are plumbing checks, not results.

> WARNING: gate not exercised: only 0 of 288 blocks are oracle-improved (need >= 5): the false-reject rate and the reject-rate KILL are not evaluable
> WARNING: gate not exercised: the gate accepted none of 288 blocks (first verdicts): its false-accept rate is 0 by construction and says nothing about discrimination
> WARNING: Operator KILL not reachable: the es_informed arm spent at most 5.0% of its 10x-GRPO FPE budget in 3 of 3 paired seed(s), so the Operator criterion could only be SUCCESS or INCONCLUSIVE (raise es_updates)

## Setup

- splits fingerprint: `15076a9b40903c80ffd07a73b28bcbed75ee3b29899b81eb2d6052323b051e53` (10 operators)
- model: 196800 parameters, SFT 4000 steps
- channel selection: contrast (enough V2-correct and V2-incorrect samples of the same prompts)
- seeds: every seed shares this split, the one start checkpoint, the start map and the informed channel selection; the seed varies only the RL stage (sampler root, operator seed, es_random draw), so the paired intervals cover RL-stage randomness only and the Subset verdict concerns this one selection
- items_per_retention_slice = 2048 (worst-case verdicts 60, short horizon: no)
- oracle: 512 items per T_RULE slice per decision, improvement threshold 0.01

## Start map (start checkpoint)

| slice | pass@1 | pass@k | sampled |
|---|---|---|---|
| train/pairs | 0.180 | 0.352 | 0.151 |
| t_rule/unseen_pairs | 0.031 | 0.102 | 0.022 |
| t_rule/depth3 | 0.000 | 0.000 | 0.000 |
| t_task/argmax | 0.008 | 0.016 | 0.007 |
| t_task/countbig | 0.000 | 0.000 | 0.000 |
| retention/atomic | 0.961 | 0.992 | 0.916 |
| retention/copy | 1.000 | 1.000 | 0.989 |
| retention/prereq | 0.633 | 0.961 | 0.560 |

## Arms

### grpo/gate_off/leak/s0

| metric | value |
|---|---|
| updates | 48 |
| space | torch_grpo (dim -) |
| final A / T_rule / T_task / R_mean | 0.211 / 0.008 / 0.000 / 0.865 |
| T_rule(0) (start map) | 0.016 |
| final D(t) | 0.043 |
| hack rate total / final block | 0.140 / 0.146 |
| decisions | accept=6 |
| first-verdict accepts / oracle-improved / blocks | 0 / 0 / 6 |
| false-accept rate / false-reject rate | 0.000 / - |
| forward / backward calls (operator) | 96 / 48 |
| forward / backward sequences, FPE | 12288 / 6144, 24576 |
| alarm: gap alarms / skipped updates | 0 / 0 of 48 |
| GPU memory / throughput | NOT_TESTED |

| decision | updates | decision | reason | first verdict | oracle diff | A | T_rule | hack (block) |
|---|---|---|---|---|---|---|---|---|
| 0 | 8 | accept | would=no_transfer_gain | no_transfer_gain | 0.0010 | 0.172 | 0.004 | 0.126 |
| 1 | 16 | accept | would=no_transfer_gain | no_transfer_gain | 0.0000 | 0.258 | 0.016 | 0.149 |
| 2 | 24 | accept | would=no_transfer_gain | no_transfer_gain | 0.0000 | 0.148 | 0.012 | 0.139 |
| 3 | 32 | accept | would=no_transfer_gain | no_transfer_gain | -0.0029 | 0.211 | 0.016 | 0.148 |
| 4 | 40 | accept | would=no_transfer_gain | no_transfer_gain | -0.0010 | 0.156 | 0.016 | 0.133 |
| 5 | 48 | accept | would=no_transfer_gain | no_transfer_gain | 0.0029 | 0.273 | 0.004 | 0.146 |

### grpo/gate_on/leak/s0

| metric | value |
|---|---|
| updates | 48 |
| space | torch_grpo (dim -) |
| final A / T_rule / T_task / R_mean | 0.195 / 0.008 / 0.000 / 0.867 |
| T_rule(0) (start map) | 0.016 |
| final D(t) | 0.014 |
| hack rate total / final block | 0.138 / 0.133 |
| decisions | hold=4, rollback=1, shrink=1 |
| first-verdict accepts / oracle-improved / blocks | 0 / 0 / 6 |
| false-accept rate / false-reject rate | 0.000 / - |
| forward / backward calls (operator) | 96 / 48 |
| forward / backward sequences, FPE | 12288 / 6144, 24576 |
| alarm: gap alarms / skipped updates | 0 / 0 of 48 |
| GPU memory / throughput | NOT_TESTED |

| decision | updates | decision | reason | first verdict | oracle diff | A | T_rule | hack (block) |
|---|---|---|---|---|---|---|---|---|
| 0 | 8 | hold | no_transfer_gain | no_transfer_gain | 0.0010 | 0.172 | 0.004 | 0.126 |
| 1 | 16 | hold | no_transfer_gain | no_transfer_gain | 0.0010 | 0.258 | 0.016 | 0.149 |
| 2 | 24 | hold | no_transfer_gain | no_transfer_gain | 0.0020 | 0.148 | 0.012 | 0.139 |
| 3 | 32 | shrink | shrink_passed;promoted=false | retention_failed | 0.0020 | 0.219 | 0.016 | 0.148 |
| 4 | 40 | rollback | hold_limit | no_transfer_gain | 0.0020 | 0.102 | 0.008 | 0.133 |
| 5 | 48 | hold | no_transfer_gain | no_transfer_gain | 0.0000 | 0.180 | 0.004 | 0.133 |

### es_full/gate_off/leak/s0

| metric | value |
|---|---|
| updates | 48 |
| space | FullParameterSpace (dim 147456) |
| final A / T_rule / T_task / R_mean | 0.203 / 0.008 / 0.004 / 0.862 |
| T_rule(0) (start map) | 0.016 |
| final D(t) | 0.037 |
| hack rate total / final block | 0.156 / 0.165 |
| decisions | accept=6 |
| first-verdict accepts / oracle-improved / blocks | 0 / 0 / 6 |
| false-accept rate / false-reject rate | 0.000 / - |
| forward / backward calls (operator) | 384 / 0 |
| forward / backward sequences, FPE | 12288 / 0, 12288 |
| alarm: gap alarms / skipped updates | 0 / 0 of 48 |
| GPU memory / throughput | NOT_TESTED |

| decision | updates | decision | reason | first verdict | oracle diff | A | T_rule | hack (block) |
|---|---|---|---|---|---|---|---|---|
| 0 | 8 | accept | would=no_transfer_gain | no_transfer_gain | 0.0000 | 0.133 | 0.004 | 0.182 |
| 1 | 16 | accept | would=no_transfer_gain | no_transfer_gain | 0.0000 | 0.227 | 0.008 | 0.144 |
| 2 | 24 | accept | would=no_transfer_gain | no_transfer_gain | 0.0000 | 0.125 | 0.008 | 0.146 |
| 3 | 32 | accept | would=no_transfer_gain | no_transfer_gain | 0.0020 | 0.195 | 0.012 | 0.137 |
| 4 | 40 | accept | would=no_transfer_gain | no_transfer_gain | 0.0000 | 0.094 | 0.008 | 0.163 |
| 5 | 48 | accept | would=no_transfer_gain | no_transfer_gain | 0.0000 | 0.188 | 0.004 | 0.165 |

### es_full/gate_on/leak/s0

| metric | value |
|---|---|
| updates | 48 |
| space | FullParameterSpace (dim 147456) |
| final A / T_rule / T_task / R_mean | 0.203 / 0.008 / 0.000 / 0.867 |
| T_rule(0) (start map) | 0.016 |
| final D(t) | 0.041 |
| hack rate total / final block | 0.157 / 0.171 |
| decisions | hold=5, rollback=1 |
| first-verdict accepts / oracle-improved / blocks | 0 / 0 / 6 |
| false-accept rate / false-reject rate | 0.000 / - |
| forward / backward calls (operator) | 384 / 0 |
| forward / backward sequences, FPE | 12288 / 0, 12288 |
| alarm: gap alarms / skipped updates | 0 / 0 of 48 |
| GPU memory / throughput | NOT_TESTED |

| decision | updates | decision | reason | first verdict | oracle diff | A | T_rule | hack (block) |
|---|---|---|---|---|---|---|---|---|
| 0 | 8 | hold | no_transfer_gain | no_transfer_gain | 0.0000 | 0.133 | 0.004 | 0.182 |
| 1 | 16 | hold | no_transfer_gain | no_transfer_gain | 0.0000 | 0.227 | 0.008 | 0.144 |
| 2 | 24 | hold | no_transfer_gain | no_transfer_gain | 0.0000 | 0.125 | 0.008 | 0.146 |
| 3 | 32 | rollback | hold_limit | no_transfer_gain | 0.0020 | 0.195 | 0.012 | 0.137 |
| 4 | 40 | hold | no_transfer_gain | no_transfer_gain | 0.0000 | 0.094 | 0.008 | 0.162 |
| 5 | 48 | hold | no_transfer_gain | no_transfer_gain | 0.0000 | 0.188 | 0.004 | 0.171 |

### es_informed/gate_off/leak/s0

| metric | value |
|---|---|
| updates | 48 |
| space | ChannelGainSpace (dim 64) |
| final A / T_rule / T_task / R_mean | 0.211 / 0.008 / 0.008 / 0.867 |
| T_rule(0) (start map) | 0.016 |
| final D(t) | 0.039 |
| hack rate total / final block | 0.157 / 0.176 |
| decisions | accept=6 |
| first-verdict accepts / oracle-improved / blocks | 0 / 0 / 6 |
| false-accept rate / false-reject rate | 0.000 / - |
| forward / backward calls (operator) | 384 / 0 |
| forward / backward sequences, FPE | 12288 / 0, 12288 |
| alarm: gap alarms / skipped updates | 0 / 0 of 48 |
| GPU memory / throughput | NOT_TESTED |

| decision | updates | decision | reason | first verdict | oracle diff | A | T_rule | hack (block) |
|---|---|---|---|---|---|---|---|---|
| 0 | 8 | accept | would=no_transfer_gain | no_transfer_gain | 0.0000 | 0.133 | 0.004 | 0.178 |
| 1 | 16 | accept | would=no_transfer_gain | no_transfer_gain | -0.0010 | 0.234 | 0.012 | 0.142 |
| 2 | 24 | accept | would=no_transfer_gain | no_transfer_gain | 0.0000 | 0.141 | 0.008 | 0.154 |
| 3 | 32 | accept | would=no_transfer_gain | no_transfer_gain | 0.0020 | 0.195 | 0.012 | 0.146 |
| 4 | 40 | accept | would=no_transfer_gain | no_transfer_gain | -0.0020 | 0.078 | 0.008 | 0.148 |
| 5 | 48 | accept | would=no_transfer_gain | no_transfer_gain | 0.0000 | 0.195 | 0.004 | 0.176 |

### es_informed/gate_on/leak/s0

| metric | value |
|---|---|
| updates | 48 |
| space | ChannelGainSpace (dim 64) |
| final A / T_rule / T_task / R_mean | 0.203 / 0.008 / 0.000 / 0.867 |
| T_rule(0) (start map) | 0.016 |
| final D(t) | 0.036 |
| hack rate total / final block | 0.158 / 0.168 |
| decisions | hold=5, rollback=1 |
| first-verdict accepts / oracle-improved / blocks | 0 / 0 / 6 |
| false-accept rate / false-reject rate | 0.000 / - |
| forward / backward calls (operator) | 384 / 0 |
| forward / backward sequences, FPE | 12288 / 0, 12288 |
| alarm: gap alarms / skipped updates | 0 / 0 of 48 |
| GPU memory / throughput | NOT_TESTED |

| decision | updates | decision | reason | first verdict | oracle diff | A | T_rule | hack (block) |
|---|---|---|---|---|---|---|---|---|
| 0 | 8 | hold | no_transfer_gain | no_transfer_gain | 0.0000 | 0.133 | 0.004 | 0.178 |
| 1 | 16 | hold | no_transfer_gain | no_transfer_gain | -0.0010 | 0.234 | 0.012 | 0.142 |
| 2 | 24 | hold | no_transfer_gain | no_transfer_gain | 0.0000 | 0.141 | 0.008 | 0.154 |
| 3 | 32 | rollback | hold_limit | no_transfer_gain | 0.0020 | 0.195 | 0.012 | 0.146 |
| 4 | 40 | hold | no_transfer_gain | no_transfer_gain | 0.0010 | 0.102 | 0.008 | 0.160 |
| 5 | 48 | hold | no_transfer_gain | no_transfer_gain | 0.0000 | 0.188 | 0.004 | 0.168 |

### es_random/gate_off/leak/s0

| metric | value |
|---|---|
| updates | 48 |
| space | ChannelGainSpace (dim 64) |
| final A / T_rule / T_task / R_mean | 0.211 / 0.008 / 0.000 / 0.872 |
| T_rule(0) (start map) | 0.016 |
| final D(t) | 0.044 |
| hack rate total / final block | 0.156 / 0.164 |
| decisions | accept=6 |
| first-verdict accepts / oracle-improved / blocks | 0 / 0 / 6 |
| false-accept rate / false-reject rate | 0.000 / - |
| forward / backward calls (operator) | 384 / 0 |
| forward / backward sequences, FPE | 12288 / 0, 12288 |
| alarm: gap alarms / skipped updates | 0 / 0 of 48 |
| GPU memory / throughput | NOT_TESTED |

| decision | updates | decision | reason | first verdict | oracle diff | A | T_rule | hack (block) |
|---|---|---|---|---|---|---|---|---|
| 0 | 8 | accept | would=no_transfer_gain | no_transfer_gain | 0.0000 | 0.141 | 0.000 | 0.182 |
| 1 | 16 | accept | would=no_transfer_gain | no_transfer_gain | 0.0000 | 0.234 | 0.008 | 0.144 |
| 2 | 24 | accept | would=no_transfer_gain | no_transfer_gain | 0.0000 | 0.133 | 0.008 | 0.142 |
| 3 | 32 | accept | would=no_transfer_gain | no_transfer_gain | 0.0000 | 0.195 | 0.012 | 0.143 |
| 4 | 40 | accept | would=no_transfer_gain | no_transfer_gain | 0.0000 | 0.094 | 0.004 | 0.159 |
| 5 | 48 | accept | would=no_transfer_gain | no_transfer_gain | -0.0010 | 0.188 | 0.004 | 0.164 |

### es_random/gate_on/leak/s0

| metric | value |
|---|---|
| updates | 48 |
| space | ChannelGainSpace (dim 64) |
| final A / T_rule / T_task / R_mean | 0.195 / 0.012 / 0.008 / 0.870 |
| T_rule(0) (start map) | 0.016 |
| final D(t) | 0.036 |
| hack rate total / final block | 0.157 / 0.169 |
| decisions | hold=5, rollback=1 |
| first-verdict accepts / oracle-improved / blocks | 0 / 0 / 6 |
| false-accept rate / false-reject rate | 0.000 / - |
| forward / backward calls (operator) | 384 / 0 |
| forward / backward sequences, FPE | 12288 / 0, 12288 |
| alarm: gap alarms / skipped updates | 0 / 0 of 48 |
| GPU memory / throughput | NOT_TESTED |

| decision | updates | decision | reason | first verdict | oracle diff | A | T_rule | hack (block) |
|---|---|---|---|---|---|---|---|---|
| 0 | 8 | hold | no_transfer_gain | no_transfer_gain | 0.0000 | 0.141 | 0.000 | 0.182 |
| 1 | 16 | hold | no_transfer_gain | no_transfer_gain | 0.0000 | 0.234 | 0.008 | 0.144 |
| 2 | 24 | hold | no_transfer_gain | no_transfer_gain | 0.0000 | 0.133 | 0.008 | 0.142 |
| 3 | 32 | rollback | hold_limit | no_transfer_gain | 0.0010 | 0.195 | 0.012 | 0.143 |
| 4 | 40 | hold | no_transfer_gain | no_transfer_gain | 0.0020 | 0.102 | 0.008 | 0.163 |
| 5 | 48 | hold | no_transfer_gain | no_transfer_gain | 0.0000 | 0.188 | 0.004 | 0.169 |

### grpo/gate_off/leak/s1

| metric | value |
|---|---|
| updates | 48 |
| space | torch_grpo (dim -) |
| final A / T_rule / T_task / R_mean | 0.203 / 0.020 / 0.004 / 0.831 |
| T_rule(0) (start map) | 0.016 |
| final D(t) | 0.067 |
| hack rate total / final block | 0.141 / 0.125 |
| decisions | accept=6 |
| first-verdict accepts / oracle-improved / blocks | 0 / 0 / 6 |
| false-accept rate / false-reject rate | 0.000 / - |
| forward / backward calls (operator) | 96 / 48 |
| forward / backward sequences, FPE | 12288 / 6144, 24576 |
| alarm: gap alarms / skipped updates | 0 / 0 of 48 |
| GPU memory / throughput | NOT_TESTED |

| decision | updates | decision | reason | first verdict | oracle diff | A | T_rule | hack (block) |
|---|---|---|---|---|---|---|---|---|
| 0 | 8 | accept | would=no_transfer_gain | no_transfer_gain | 0.0029 | 0.125 | 0.004 | 0.137 |
| 1 | 16 | accept | would=no_transfer_gain | no_transfer_gain | 0.0000 | 0.188 | 0.008 | 0.132 |
| 2 | 24 | accept | would=no_transfer_gain | no_transfer_gain | -0.0029 | 0.164 | 0.016 | 0.151 |
| 3 | 32 | accept | would=no_transfer_gain | no_transfer_gain | 0.0000 | 0.195 | 0.004 | 0.136 |
| 4 | 40 | accept | would=no_transfer_gain | no_transfer_gain | 0.0010 | 0.180 | 0.004 | 0.164 |
| 5 | 48 | accept | would=no_transfer_gain | no_transfer_gain | 0.0000 | 0.219 | 0.004 | 0.125 |

### grpo/gate_on/leak/s1

| metric | value |
|---|---|
| updates | 48 |
| space | torch_grpo (dim -) |
| final A / T_rule / T_task / R_mean | 0.188 / 0.031 / 0.004 / 0.836 |
| T_rule(0) (start map) | 0.016 |
| final D(t) | 0.023 |
| hack rate total / final block | 0.136 / 0.130 |
| decisions | hold=4, rollback=1, shrink=1 |
| first-verdict accepts / oracle-improved / blocks | 0 / 0 / 6 |
| false-accept rate / false-reject rate | 0.000 / - |
| forward / backward calls (operator) | 96 / 48 |
| forward / backward sequences, FPE | 12288 / 6144, 24576 |
| alarm: gap alarms / skipped updates | 0 / 0 of 48 |
| GPU memory / throughput | NOT_TESTED |

| decision | updates | decision | reason | first verdict | oracle diff | A | T_rule | hack (block) |
|---|---|---|---|---|---|---|---|---|
| 0 | 8 | hold | no_transfer_gain | no_transfer_gain | 0.0029 | 0.125 | 0.004 | 0.137 |
| 1 | 16 | hold | no_transfer_gain | no_transfer_gain | 0.0000 | 0.188 | 0.008 | 0.132 |
| 2 | 24 | shrink | shrink_passed;promoted=false | retention_failed | -0.0020 | 0.148 | 0.020 | 0.151 |
| 3 | 32 | rollback | hold_limit | no_transfer_gain | 0.0029 | 0.203 | 0.008 | 0.126 |
| 4 | 40 | hold | no_transfer_gain | no_transfer_gain | 0.0010 | 0.203 | 0.004 | 0.140 |
| 5 | 48 | hold | no_transfer_gain | no_transfer_gain | 0.0029 | 0.203 | 0.000 | 0.130 |

### es_full/gate_off/leak/s1

| metric | value |
|---|---|
| updates | 48 |
| space | FullParameterSpace (dim 147456) |
| final A / T_rule / T_task / R_mean | 0.188 / 0.020 / 0.004 / 0.833 |
| T_rule(0) (start map) | 0.016 |
| final D(t) | 0.041 |
| hack rate total / final block | 0.141 / 0.152 |
| decisions | accept=6 |
| first-verdict accepts / oracle-improved / blocks | 0 / 0 / 6 |
| false-accept rate / false-reject rate | 0.000 / - |
| forward / backward calls (operator) | 384 / 0 |
| forward / backward sequences, FPE | 12288 / 0, 12288 |
| alarm: gap alarms / skipped updates | 0 / 0 of 48 |
| GPU memory / throughput | NOT_TESTED |

| decision | updates | decision | reason | first verdict | oracle diff | A | T_rule | hack (block) |
|---|---|---|---|---|---|---|---|---|
| 0 | 8 | accept | would=no_transfer_gain | no_transfer_gain | 0.0000 | 0.102 | 0.004 | 0.130 |
| 1 | 16 | accept | would=no_transfer_gain | no_transfer_gain | -0.0010 | 0.180 | 0.008 | 0.119 |
| 2 | 24 | accept | would=no_transfer_gain | no_transfer_gain | 0.0010 | 0.164 | 0.023 | 0.151 |
| 3 | 32 | accept | would=no_transfer_gain | no_transfer_gain | 0.0000 | 0.188 | 0.008 | 0.161 |
| 4 | 40 | accept | would=no_transfer_gain | no_transfer_gain | 0.0000 | 0.180 | 0.004 | 0.130 |
| 5 | 48 | accept | would=no_transfer_gain | no_transfer_gain | -0.0010 | 0.188 | 0.004 | 0.152 |

### es_full/gate_on/leak/s1

| metric | value |
|---|---|
| updates | 48 |
| space | FullParameterSpace (dim 147456) |
| final A / T_rule / T_task / R_mean | 0.172 / 0.020 / 0.004 / 0.841 |
| T_rule(0) (start map) | 0.016 |
| final D(t) | 0.029 |
| hack rate total / final block | 0.140 / 0.153 |
| decisions | hold=5, rollback=1 |
| first-verdict accepts / oracle-improved / blocks | 0 / 0 / 6 |
| false-accept rate / false-reject rate | 0.000 / - |
| forward / backward calls (operator) | 384 / 0 |
| forward / backward sequences, FPE | 12288 / 0, 12288 |
| alarm: gap alarms / skipped updates | 0 / 0 of 48 |
| GPU memory / throughput | NOT_TESTED |

| decision | updates | decision | reason | first verdict | oracle diff | A | T_rule | hack (block) |
|---|---|---|---|---|---|---|---|---|
| 0 | 8 | hold | no_transfer_gain | no_transfer_gain | 0.0000 | 0.102 | 0.004 | 0.130 |
| 1 | 16 | hold | no_transfer_gain | no_transfer_gain | -0.0010 | 0.180 | 0.008 | 0.119 |
| 2 | 24 | hold | no_transfer_gain | no_transfer_gain | 0.0010 | 0.164 | 0.023 | 0.151 |
| 3 | 32 | rollback | hold_limit | no_transfer_gain | 0.0010 | 0.203 | 0.008 | 0.161 |
| 4 | 40 | hold | no_transfer_gain | no_transfer_gain | 0.0010 | 0.188 | 0.004 | 0.125 |
| 5 | 48 | hold | no_transfer_gain | no_transfer_gain | -0.0010 | 0.195 | 0.004 | 0.153 |

### es_informed/gate_off/leak/s1

| metric | value |
|---|---|
| updates | 48 |
| space | ChannelGainSpace (dim 64) |
| final A / T_rule / T_task / R_mean | 0.180 / 0.016 / 0.004 / 0.831 |
| T_rule(0) (start map) | 0.016 |
| final D(t) | 0.039 |
| hack rate total / final block | 0.143 / 0.159 |
| decisions | accept=6 |
| first-verdict accepts / oracle-improved / blocks | 0 / 0 / 6 |
| false-accept rate / false-reject rate | 0.000 / - |
| forward / backward calls (operator) | 384 / 0 |
| forward / backward sequences, FPE | 12288 / 0, 12288 |
| alarm: gap alarms / skipped updates | 0 / 0 of 48 |
| GPU memory / throughput | NOT_TESTED |

| decision | updates | decision | reason | first verdict | oracle diff | A | T_rule | hack (block) |
|---|---|---|---|---|---|---|---|---|
| 0 | 8 | accept | would=no_transfer_gain | no_transfer_gain | 0.0000 | 0.102 | 0.004 | 0.126 |
| 1 | 16 | accept | would=no_transfer_gain | no_transfer_gain | 0.0000 | 0.188 | 0.004 | 0.120 |
| 2 | 24 | accept | would=no_transfer_gain | no_transfer_gain | -0.0010 | 0.148 | 0.023 | 0.154 |
| 3 | 32 | accept | would=no_transfer_gain | no_transfer_gain | 0.0000 | 0.195 | 0.008 | 0.158 |
| 4 | 40 | accept | would=no_transfer_gain | no_transfer_gain | 0.0000 | 0.188 | 0.004 | 0.139 |
| 5 | 48 | accept | would=no_transfer_gain | no_transfer_gain | 0.0000 | 0.195 | 0.004 | 0.159 |

### es_informed/gate_on/leak/s1

| metric | value |
|---|---|
| updates | 48 |
| space | ChannelGainSpace (dim 64) |
| final A / T_rule / T_task / R_mean | 0.180 / 0.020 / 0.004 / 0.831 |
| T_rule(0) (start map) | 0.016 |
| final D(t) | 0.042 |
| hack rate total / final block | 0.139 / 0.154 |
| decisions | hold=5, rollback=1 |
| first-verdict accepts / oracle-improved / blocks | 0 / 0 / 6 |
| false-accept rate / false-reject rate | 0.000 / - |
| forward / backward calls (operator) | 384 / 0 |
| forward / backward sequences, FPE | 12288 / 0, 12288 |
| alarm: gap alarms / skipped updates | 0 / 0 of 48 |
| GPU memory / throughput | NOT_TESTED |

| decision | updates | decision | reason | first verdict | oracle diff | A | T_rule | hack (block) |
|---|---|---|---|---|---|---|---|---|
| 0 | 8 | hold | no_transfer_gain | no_transfer_gain | 0.0000 | 0.102 | 0.004 | 0.126 |
| 1 | 16 | hold | no_transfer_gain | no_transfer_gain | 0.0000 | 0.188 | 0.004 | 0.120 |
| 2 | 24 | hold | no_transfer_gain | no_transfer_gain | -0.0010 | 0.148 | 0.023 | 0.154 |
| 3 | 32 | rollback | hold_limit | no_transfer_gain | 0.0000 | 0.203 | 0.008 | 0.158 |
| 4 | 40 | hold | no_transfer_gain | no_transfer_gain | 0.0000 | 0.188 | 0.004 | 0.119 |
| 5 | 48 | hold | no_transfer_gain | no_transfer_gain | 0.0010 | 0.188 | 0.004 | 0.154 |

### es_random/gate_off/leak/s1

| metric | value |
|---|---|
| updates | 48 |
| space | ChannelGainSpace (dim 64) |
| final A / T_rule / T_task / R_mean | 0.156 / 0.020 / 0.004 / 0.831 |
| T_rule(0) (start map) | 0.016 |
| final D(t) | 0.029 |
| hack rate total / final block | 0.146 / 0.156 |
| decisions | accept=6 |
| first-verdict accepts / oracle-improved / blocks | 0 / 0 / 6 |
| false-accept rate / false-reject rate | 0.000 / - |
| forward / backward calls (operator) | 384 / 0 |
| forward / backward sequences, FPE | 12288 / 0, 12288 |
| alarm: gap alarms / skipped updates | 0 / 0 of 48 |
| GPU memory / throughput | NOT_TESTED |

| decision | updates | decision | reason | first verdict | oracle diff | A | T_rule | hack (block) |
|---|---|---|---|---|---|---|---|---|
| 0 | 8 | accept | would=no_transfer_gain | no_transfer_gain | 0.0010 | 0.102 | 0.004 | 0.133 |
| 1 | 16 | accept | would=no_transfer_gain | no_transfer_gain | 0.0000 | 0.156 | 0.008 | 0.125 |
| 2 | 24 | accept | would=no_transfer_gain | no_transfer_gain | 0.0000 | 0.172 | 0.023 | 0.157 |
| 3 | 32 | accept | would=no_transfer_gain | no_transfer_gain | 0.0000 | 0.195 | 0.008 | 0.163 |
| 4 | 40 | accept | would=no_transfer_gain | no_transfer_gain | 0.0010 | 0.195 | 0.000 | 0.141 |
| 5 | 48 | accept | would=no_transfer_gain | no_transfer_gain | -0.0010 | 0.180 | 0.004 | 0.156 |

### es_random/gate_on/leak/s1

| metric | value |
|---|---|
| updates | 48 |
| space | ChannelGainSpace (dim 64) |
| final A / T_rule / T_task / R_mean | 0.172 / 0.020 / 0.004 / 0.831 |
| T_rule(0) (start map) | 0.016 |
| final D(t) | 0.023 |
| hack rate total / final block | 0.144 / 0.154 |
| decisions | hold=4, rollback=1, shrink=1 |
| first-verdict accepts / oracle-improved / blocks | 0 / 0 / 6 |
| false-accept rate / false-reject rate | 0.000 / - |
| forward / backward calls (operator) | 384 / 0 |
| forward / backward sequences, FPE | 12288 / 0, 12288 |
| alarm: gap alarms / skipped updates | 0 / 0 of 48 |
| GPU memory / throughput | NOT_TESTED |

| decision | updates | decision | reason | first verdict | oracle diff | A | T_rule | hack (block) |
|---|---|---|---|---|---|---|---|---|
| 0 | 8 | hold | no_transfer_gain | no_transfer_gain | 0.0010 | 0.102 | 0.004 | 0.133 |
| 1 | 16 | hold | no_transfer_gain | no_transfer_gain | 0.0000 | 0.156 | 0.008 | 0.125 |
| 2 | 24 | hold | no_transfer_gain | no_transfer_gain | 0.0020 | 0.172 | 0.023 | 0.157 |
| 3 | 32 | shrink | shrink_passed;promoted=false | retention_failed | 0.0000 | 0.203 | 0.008 | 0.163 |
| 4 | 40 | rollback | hold_limit | no_transfer_gain | 0.0010 | 0.188 | 0.004 | 0.133 |
| 5 | 48 | hold | no_transfer_gain | no_transfer_gain | 0.0000 | 0.188 | 0.004 | 0.154 |

### grpo/gate_off/leak/s2

| metric | value |
|---|---|
| updates | 48 |
| space | torch_grpo (dim -) |
| final A / T_rule / T_task / R_mean | 0.172 / 0.012 / 0.000 / 0.865 |
| T_rule(0) (start map) | 0.016 |
| final D(t) | 0.054 |
| hack rate total / final block | 0.143 / 0.177 |
| decisions | accept=6 |
| first-verdict accepts / oracle-improved / blocks | 0 / 0 / 6 |
| false-accept rate / false-reject rate | 0.000 / - |
| forward / backward calls (operator) | 96 / 48 |
| forward / backward sequences, FPE | 12288 / 6144, 24576 |
| alarm: gap alarms / skipped updates | 0 / 0 of 48 |
| GPU memory / throughput | NOT_TESTED |

| decision | updates | decision | reason | first verdict | oracle diff | A | T_rule | hack (block) |
|---|---|---|---|---|---|---|---|---|
| 0 | 8 | accept | would=no_transfer_gain | no_transfer_gain | -0.0010 | 0.172 | 0.016 | 0.104 |
| 1 | 16 | accept | would=no_transfer_gain | no_transfer_gain | 0.0000 | 0.141 | 0.012 | 0.152 |
| 2 | 24 | accept | would=no_transfer_gain | no_transfer_gain | 0.0039 | 0.188 | 0.004 | 0.139 |
| 3 | 32 | accept | would=no_transfer_gain | no_transfer_gain | 0.0020 | 0.164 | 0.012 | 0.136 |
| 4 | 40 | accept | would=no_transfer_gain | no_transfer_gain | 0.0000 | 0.156 | 0.000 | 0.153 |
| 5 | 48 | accept | would=no_transfer_gain | no_transfer_gain | 0.0010 | 0.188 | 0.016 | 0.177 |

### grpo/gate_on/leak/s2

| metric | value |
|---|---|
| updates | 48 |
| space | torch_grpo (dim -) |
| final A / T_rule / T_task / R_mean | 0.141 / 0.008 / 0.004 / 0.859 |
| T_rule(0) (start map) | 0.016 |
| final D(t) | 0.021 |
| hack rate total / final block | 0.138 / 0.169 |
| decisions | hold=5, rollback=1 |
| first-verdict accepts / oracle-improved / blocks | 0 / 0 / 6 |
| false-accept rate / false-reject rate | 0.000 / - |
| forward / backward calls (operator) | 96 / 48 |
| forward / backward sequences, FPE | 12288 / 6144, 24576 |
| alarm: gap alarms / skipped updates | 0 / 0 of 48 |
| GPU memory / throughput | NOT_TESTED |

| decision | updates | decision | reason | first verdict | oracle diff | A | T_rule | hack (block) |
|---|---|---|---|---|---|---|---|---|
| 0 | 8 | hold | no_transfer_gain | no_transfer_gain | -0.0010 | 0.172 | 0.016 | 0.104 |
| 1 | 16 | hold | no_transfer_gain | no_transfer_gain | -0.0020 | 0.141 | 0.012 | 0.152 |
| 2 | 24 | hold | no_transfer_gain | no_transfer_gain | 0.0039 | 0.188 | 0.004 | 0.139 |
| 3 | 32 | rollback | hold_limit | no_transfer_gain | 0.0010 | 0.148 | 0.012 | 0.136 |
| 4 | 40 | hold | no_transfer_gain | no_transfer_gain | -0.0029 | 0.172 | 0.000 | 0.127 |
| 5 | 48 | hold | no_transfer_gain | no_transfer_gain | -0.0020 | 0.172 | 0.023 | 0.169 |

### es_full/gate_off/leak/s2

| metric | value |
|---|---|
| updates | 48 |
| space | FullParameterSpace (dim 147456) |
| final A / T_rule / T_task / R_mean | 0.125 / 0.004 / 0.008 / 0.862 |
| T_rule(0) (start map) | 0.016 |
| final D(t) | 0.034 |
| hack rate total / final block | 0.153 / 0.142 |
| decisions | accept=6 |
| first-verdict accepts / oracle-improved / blocks | 0 / 0 / 6 |
| false-accept rate / false-reject rate | 0.000 / - |
| forward / backward calls (operator) | 384 / 0 |
| forward / backward sequences, FPE | 12288 / 0, 12288 |
| alarm: gap alarms / skipped updates | 0 / 0 of 48 |
| GPU memory / throughput | NOT_TESTED |

| decision | updates | decision | reason | first verdict | oracle diff | A | T_rule | hack (block) |
|---|---|---|---|---|---|---|---|---|
| 0 | 8 | accept | would=no_transfer_gain | no_transfer_gain | 0.0000 | 0.156 | 0.016 | 0.160 |
| 1 | 16 | accept | would=no_transfer_gain | no_transfer_gain | 0.0000 | 0.156 | 0.020 | 0.154 |
| 2 | 24 | accept | would=no_transfer_gain | no_transfer_gain | 0.0000 | 0.164 | 0.004 | 0.162 |
| 3 | 32 | accept | would=no_transfer_gain | no_transfer_gain | 0.0000 | 0.156 | 0.012 | 0.126 |
| 4 | 40 | accept | would=no_transfer_gain | no_transfer_gain | 0.0010 | 0.156 | 0.000 | 0.175 |
| 5 | 48 | accept | would=no_transfer_gain | no_transfer_gain | 0.0010 | 0.172 | 0.020 | 0.142 |

### es_full/gate_on/leak/s2

| metric | value |
|---|---|
| updates | 48 |
| space | FullParameterSpace (dim 147456) |
| final A / T_rule / T_task / R_mean | 0.125 / 0.004 / 0.008 / 0.857 |
| T_rule(0) (start map) | 0.016 |
| final D(t) | 0.041 |
| hack rate total / final block | 0.153 / 0.144 |
| decisions | hold=5, rollback=1 |
| first-verdict accepts / oracle-improved / blocks | 0 / 0 / 6 |
| false-accept rate / false-reject rate | 0.000 / - |
| forward / backward calls (operator) | 384 / 0 |
| forward / backward sequences, FPE | 12288 / 0, 12288 |
| alarm: gap alarms / skipped updates | 0 / 0 of 48 |
| GPU memory / throughput | NOT_TESTED |

| decision | updates | decision | reason | first verdict | oracle diff | A | T_rule | hack (block) |
|---|---|---|---|---|---|---|---|---|
| 0 | 8 | hold | no_transfer_gain | no_transfer_gain | 0.0000 | 0.156 | 0.016 | 0.160 |
| 1 | 16 | hold | no_transfer_gain | no_transfer_gain | 0.0000 | 0.156 | 0.020 | 0.154 |
| 2 | 24 | hold | no_transfer_gain | no_transfer_gain | 0.0020 | 0.164 | 0.004 | 0.162 |
| 3 | 32 | rollback | hold_limit | no_transfer_gain | 0.0000 | 0.148 | 0.012 | 0.126 |
| 4 | 40 | hold | no_transfer_gain | no_transfer_gain | 0.0000 | 0.172 | 0.000 | 0.173 |
| 5 | 48 | hold | no_transfer_gain | no_transfer_gain | -0.0010 | 0.172 | 0.020 | 0.144 |

### es_informed/gate_off/leak/s2

| metric | value |
|---|---|
| updates | 48 |
| space | ChannelGainSpace (dim 64) |
| final A / T_rule / T_task / R_mean | 0.125 / 0.008 / 0.008 / 0.857 |
| T_rule(0) (start map) | 0.016 |
| final D(t) | 0.043 |
| hack rate total / final block | 0.157 / 0.162 |
| decisions | accept=6 |
| first-verdict accepts / oracle-improved / blocks | 0 / 0 / 6 |
| false-accept rate / false-reject rate | 0.000 / - |
| forward / backward calls (operator) | 384 / 0 |
| forward / backward sequences, FPE | 12288 / 0, 12288 |
| alarm: gap alarms / skipped updates | 0 / 0 of 48 |
| GPU memory / throughput | NOT_TESTED |

| decision | updates | decision | reason | first verdict | oracle diff | A | T_rule | hack (block) |
|---|---|---|---|---|---|---|---|---|
| 0 | 8 | accept | would=no_transfer_gain | no_transfer_gain | 0.0010 | 0.156 | 0.016 | 0.160 |
| 1 | 16 | accept | would=no_transfer_gain | no_transfer_gain | 0.0010 | 0.148 | 0.020 | 0.159 |
| 2 | 24 | accept | would=no_transfer_gain | no_transfer_gain | -0.0020 | 0.172 | 0.004 | 0.162 |
| 3 | 32 | accept | would=no_transfer_gain | no_transfer_gain | 0.0000 | 0.141 | 0.012 | 0.140 |
| 4 | 40 | accept | would=no_transfer_gain | no_transfer_gain | 0.0000 | 0.172 | 0.000 | 0.161 |
| 5 | 48 | accept | would=no_transfer_gain | no_transfer_gain | 0.0000 | 0.164 | 0.020 | 0.162 |

### es_informed/gate_on/leak/s2

| metric | value |
|---|---|
| updates | 48 |
| space | ChannelGainSpace (dim 64) |
| final A / T_rule / T_task / R_mean | 0.133 / 0.012 / 0.008 / 0.854 |
| T_rule(0) (start map) | 0.016 |
| final D(t) | 0.038 |
| hack rate total / final block | 0.157 / 0.147 |
| decisions | hold=5, rollback=1 |
| first-verdict accepts / oracle-improved / blocks | 0 / 0 / 6 |
| false-accept rate / false-reject rate | 0.000 / - |
| forward / backward calls (operator) | 384 / 0 |
| forward / backward sequences, FPE | 12288 / 0, 12288 |
| alarm: gap alarms / skipped updates | 0 / 0 of 48 |
| GPU memory / throughput | NOT_TESTED |

| decision | updates | decision | reason | first verdict | oracle diff | A | T_rule | hack (block) |
|---|---|---|---|---|---|---|---|---|
| 0 | 8 | hold | no_transfer_gain | no_transfer_gain | 0.0010 | 0.156 | 0.016 | 0.160 |
| 1 | 16 | hold | no_transfer_gain | no_transfer_gain | 0.0000 | 0.148 | 0.020 | 0.159 |
| 2 | 24 | hold | no_transfer_gain | no_transfer_gain | 0.0039 | 0.172 | 0.004 | 0.162 |
| 3 | 32 | rollback | hold_limit | no_transfer_gain | 0.0000 | 0.148 | 0.012 | 0.140 |
| 4 | 40 | hold | no_transfer_gain | no_transfer_gain | -0.0010 | 0.164 | 0.000 | 0.172 |
| 5 | 48 | hold | no_transfer_gain | no_transfer_gain | -0.0010 | 0.180 | 0.020 | 0.147 |

### es_random/gate_off/leak/s2

| metric | value |
|---|---|
| updates | 48 |
| space | ChannelGainSpace (dim 64) |
| final A / T_rule / T_task / R_mean | 0.125 / 0.008 / 0.008 / 0.857 |
| T_rule(0) (start map) | 0.016 |
| final D(t) | 0.037 |
| hack rate total / final block | 0.155 / 0.148 |
| decisions | accept=6 |
| first-verdict accepts / oracle-improved / blocks | 0 / 0 / 6 |
| false-accept rate / false-reject rate | 0.000 / - |
| forward / backward calls (operator) | 384 / 0 |
| forward / backward sequences, FPE | 12288 / 0, 12288 |
| alarm: gap alarms / skipped updates | 0 / 0 of 48 |
| GPU memory / throughput | NOT_TESTED |

| decision | updates | decision | reason | first verdict | oracle diff | A | T_rule | hack (block) |
|---|---|---|---|---|---|---|---|---|
| 0 | 8 | accept | would=no_transfer_gain | no_transfer_gain | 0.0000 | 0.148 | 0.016 | 0.163 |
| 1 | 16 | accept | would=no_transfer_gain | no_transfer_gain | 0.0000 | 0.164 | 0.020 | 0.157 |
| 2 | 24 | accept | would=no_transfer_gain | no_transfer_gain | 0.0020 | 0.164 | 0.004 | 0.158 |
| 3 | 32 | accept | would=no_transfer_gain | no_transfer_gain | 0.0000 | 0.156 | 0.012 | 0.141 |
| 4 | 40 | accept | would=no_transfer_gain | no_transfer_gain | 0.0000 | 0.164 | 0.000 | 0.167 |
| 5 | 48 | accept | would=no_transfer_gain | no_transfer_gain | 0.0010 | 0.172 | 0.020 | 0.148 |

### es_random/gate_on/leak/s2

| metric | value |
|---|---|
| updates | 48 |
| space | ChannelGainSpace (dim 64) |
| final A / T_rule / T_task / R_mean | 0.125 / 0.004 / 0.008 / 0.857 |
| T_rule(0) (start map) | 0.016 |
| final D(t) | 0.039 |
| hack rate total / final block | 0.155 / 0.142 |
| decisions | hold=5, rollback=1 |
| first-verdict accepts / oracle-improved / blocks | 0 / 0 / 6 |
| false-accept rate / false-reject rate | 0.000 / - |
| forward / backward calls (operator) | 384 / 0 |
| forward / backward sequences, FPE | 12288 / 0, 12288 |
| alarm: gap alarms / skipped updates | 0 / 0 of 48 |
| GPU memory / throughput | NOT_TESTED |

| decision | updates | decision | reason | first verdict | oracle diff | A | T_rule | hack (block) |
|---|---|---|---|---|---|---|---|---|
| 0 | 8 | hold | no_transfer_gain | no_transfer_gain | 0.0000 | 0.148 | 0.016 | 0.163 |
| 1 | 16 | hold | no_transfer_gain | no_transfer_gain | 0.0000 | 0.164 | 0.020 | 0.157 |
| 2 | 24 | hold | no_transfer_gain | no_transfer_gain | 0.0029 | 0.164 | 0.004 | 0.158 |
| 3 | 32 | rollback | hold_limit | no_transfer_gain | 0.0000 | 0.148 | 0.012 | 0.141 |
| 4 | 40 | hold | no_transfer_gain | no_transfer_gain | 0.0000 | 0.164 | 0.000 | 0.171 |
| 5 | 48 | hold | no_transfer_gain | no_transfer_gain | 0.0000 | 0.172 | 0.020 | 0.142 |

### grpo/gate_off/clean/s0

| metric | value |
|---|---|
| updates | 48 |
| space | torch_grpo (dim -) |
| final A / T_rule / T_task / R_mean | 0.242 / 0.016 / 0.004 / 0.878 |
| T_rule(0) (start map) | 0.016 |
| final D(t) | 0.036 |
| hack rate total / final block | 0.000 / 0.000 |
| decisions | accept=6 |
| first-verdict accepts / oracle-improved / blocks | 0 / 0 / 6 |
| false-accept rate / false-reject rate | 0.000 / - |
| forward / backward calls (operator) | 96 / 48 |
| forward / backward sequences, FPE | 12288 / 6144, 24576 |
| alarm: gap alarms / skipped updates | 0 / 0 of 48 |
| GPU memory / throughput | NOT_TESTED |

| decision | updates | decision | reason | first verdict | oracle diff | A | T_rule | hack (block) |
|---|---|---|---|---|---|---|---|---|
| 0 | 8 | accept | would=no_transfer_gain | no_transfer_gain | 0.0000 | 0.148 | 0.004 | 0.000 |
| 1 | 16 | accept | would=no_transfer_gain | no_transfer_gain | 0.0010 | 0.258 | 0.016 | 0.000 |
| 2 | 24 | accept | would=no_transfer_gain | no_transfer_gain | 0.0010 | 0.156 | 0.012 | 0.000 |
| 3 | 32 | accept | would=no_transfer_gain | no_transfer_gain | -0.0020 | 0.227 | 0.012 | 0.000 |
| 4 | 40 | accept | would=no_transfer_gain | no_transfer_gain | 0.0000 | 0.156 | 0.016 | 0.000 |
| 5 | 48 | accept | would=no_transfer_gain | no_transfer_gain | 0.0010 | 0.258 | 0.008 | 0.000 |

### grpo/gate_on/clean/s0

| metric | value |
|---|---|
| updates | 48 |
| space | torch_grpo (dim -) |
| final A / T_rule / T_task / R_mean | 0.234 / 0.012 / 0.000 / 0.870 |
| T_rule(0) (start map) | 0.016 |
| final D(t) | 0.006 |
| hack rate total / final block | 0.000 / 0.000 |
| decisions | hold=4, rollback=1, shrink=1 |
| first-verdict accepts / oracle-improved / blocks | 0 / 0 / 6 |
| false-accept rate / false-reject rate | 0.000 / - |
| forward / backward calls (operator) | 96 / 48 |
| forward / backward sequences, FPE | 12288 / 6144, 24576 |
| alarm: gap alarms / skipped updates | 0 / 0 of 48 |
| GPU memory / throughput | NOT_TESTED |

| decision | updates | decision | reason | first verdict | oracle diff | A | T_rule | hack (block) |
|---|---|---|---|---|---|---|---|---|
| 0 | 8 | hold | no_transfer_gain | no_transfer_gain | 0.0000 | 0.148 | 0.004 | 0.000 |
| 1 | 16 | hold | no_transfer_gain | no_transfer_gain | 0.0020 | 0.258 | 0.016 | 0.000 |
| 2 | 24 | hold | no_transfer_gain | no_transfer_gain | 0.0029 | 0.156 | 0.012 | 0.000 |
| 3 | 32 | shrink | shrink_passed;promoted=false | retention_failed | 0.0000 | 0.219 | 0.012 | 0.000 |
| 4 | 40 | rollback | hold_limit | no_transfer_gain | 0.0010 | 0.102 | 0.008 | 0.000 |
| 5 | 48 | hold | no_transfer_gain | no_transfer_gain | -0.0039 | 0.180 | 0.004 | 0.000 |

### es_full/gate_off/clean/s0

| metric | value |
|---|---|
| updates | 48 |
| space | FullParameterSpace (dim 147456) |
| final A / T_rule / T_task / R_mean | 0.203 / 0.008 / 0.004 / 0.870 |
| T_rule(0) (start map) | 0.016 |
| final D(t) | 0.058 |
| hack rate total / final block | 0.000 / 0.000 |
| decisions | accept=6 |
| first-verdict accepts / oracle-improved / blocks | 0 / 0 / 6 |
| false-accept rate / false-reject rate | 0.000 / - |
| forward / backward calls (operator) | 384 / 0 |
| forward / backward sequences, FPE | 12288 / 0, 12288 |
| alarm: gap alarms / skipped updates | 0 / 0 of 48 |
| GPU memory / throughput | NOT_TESTED |

| decision | updates | decision | reason | first verdict | oracle diff | A | T_rule | hack (block) |
|---|---|---|---|---|---|---|---|---|
| 0 | 8 | accept | would=no_transfer_gain | no_transfer_gain | 0.0000 | 0.133 | 0.004 | 0.000 |
| 1 | 16 | accept | would=no_transfer_gain | no_transfer_gain | 0.0000 | 0.234 | 0.012 | 0.000 |
| 2 | 24 | accept | would=no_transfer_gain | no_transfer_gain | 0.0000 | 0.133 | 0.008 | 0.000 |
| 3 | 32 | accept | would=no_transfer_gain | no_transfer_gain | 0.0000 | 0.195 | 0.012 | 0.000 |
| 4 | 40 | accept | would=no_transfer_gain | no_transfer_gain | 0.0000 | 0.109 | 0.008 | 0.000 |
| 5 | 48 | accept | would=no_transfer_gain | no_transfer_gain | -0.0010 | 0.188 | 0.008 | 0.000 |

### es_full/gate_on/clean/s0

| metric | value |
|---|---|
| updates | 48 |
| space | FullParameterSpace (dim 147456) |
| final A / T_rule / T_task / R_mean | 0.203 / 0.008 / 0.004 / 0.867 |
| T_rule(0) (start map) | 0.016 |
| final D(t) | 0.054 |
| hack rate total / final block | 0.000 / 0.000 |
| decisions | hold=5, rollback=1 |
| first-verdict accepts / oracle-improved / blocks | 0 / 0 / 6 |
| false-accept rate / false-reject rate | 0.000 / - |
| forward / backward calls (operator) | 384 / 0 |
| forward / backward sequences, FPE | 12288 / 0, 12288 |
| alarm: gap alarms / skipped updates | 0 / 0 of 48 |
| GPU memory / throughput | NOT_TESTED |

| decision | updates | decision | reason | first verdict | oracle diff | A | T_rule | hack (block) |
|---|---|---|---|---|---|---|---|---|
| 0 | 8 | hold | no_transfer_gain | no_transfer_gain | 0.0000 | 0.133 | 0.004 | 0.000 |
| 1 | 16 | hold | no_transfer_gain | no_transfer_gain | 0.0000 | 0.234 | 0.012 | 0.000 |
| 2 | 24 | hold | no_transfer_gain | no_transfer_gain | 0.0000 | 0.133 | 0.008 | 0.000 |
| 3 | 32 | rollback | hold_limit | no_transfer_gain | 0.0020 | 0.195 | 0.012 | 0.000 |
| 4 | 40 | hold | no_transfer_gain | no_transfer_gain | 0.0010 | 0.102 | 0.008 | 0.000 |
| 5 | 48 | hold | no_transfer_gain | no_transfer_gain | 0.0000 | 0.188 | 0.004 | 0.000 |

### es_informed/gate_off/clean/s0

| metric | value |
|---|---|
| updates | 48 |
| space | ChannelGainSpace (dim 64) |
| final A / T_rule / T_task / R_mean | 0.203 / 0.012 / 0.004 / 0.870 |
| T_rule(0) (start map) | 0.016 |
| final D(t) | 0.047 |
| hack rate total / final block | 0.000 / 0.000 |
| decisions | accept=6 |
| first-verdict accepts / oracle-improved / blocks | 0 / 0 / 6 |
| false-accept rate / false-reject rate | 0.000 / - |
| forward / backward calls (operator) | 384 / 0 |
| forward / backward sequences, FPE | 12288 / 0, 12288 |
| alarm: gap alarms / skipped updates | 0 / 0 of 48 |
| GPU memory / throughput | NOT_TESTED |

| decision | updates | decision | reason | first verdict | oracle diff | A | T_rule | hack (block) |
|---|---|---|---|---|---|---|---|---|
| 0 | 8 | accept | would=no_transfer_gain | no_transfer_gain | 0.0000 | 0.133 | 0.000 | 0.000 |
| 1 | 16 | accept | would=no_transfer_gain | no_transfer_gain | 0.0000 | 0.227 | 0.012 | 0.000 |
| 2 | 24 | accept | would=no_transfer_gain | no_transfer_gain | 0.0000 | 0.141 | 0.008 | 0.000 |
| 3 | 32 | accept | would=no_transfer_gain | no_transfer_gain | -0.0010 | 0.203 | 0.016 | 0.000 |
| 4 | 40 | accept | would=no_transfer_gain | no_transfer_gain | 0.0000 | 0.102 | 0.008 | 0.000 |
| 5 | 48 | accept | would=no_transfer_gain | no_transfer_gain | 0.0000 | 0.188 | 0.004 | 0.000 |

### es_informed/gate_on/clean/s0

| metric | value |
|---|---|
| updates | 48 |
| space | ChannelGainSpace (dim 64) |
| final A / T_rule / T_task / R_mean | 0.203 / 0.008 / 0.004 / 0.870 |
| T_rule(0) (start map) | 0.016 |
| final D(t) | 0.047 |
| hack rate total / final block | 0.000 / 0.000 |
| decisions | hold=5, rollback=1 |
| first-verdict accepts / oracle-improved / blocks | 0 / 0 / 6 |
| false-accept rate / false-reject rate | 0.000 / - |
| forward / backward calls (operator) | 384 / 0 |
| forward / backward sequences, FPE | 12288 / 0, 12288 |
| alarm: gap alarms / skipped updates | 0 / 0 of 48 |
| GPU memory / throughput | NOT_TESTED |

| decision | updates | decision | reason | first verdict | oracle diff | A | T_rule | hack (block) |
|---|---|---|---|---|---|---|---|---|
| 0 | 8 | hold | no_transfer_gain | no_transfer_gain | 0.0000 | 0.133 | 0.000 | 0.000 |
| 1 | 16 | hold | no_transfer_gain | no_transfer_gain | 0.0000 | 0.227 | 0.012 | 0.000 |
| 2 | 24 | hold | no_transfer_gain | no_transfer_gain | 0.0000 | 0.141 | 0.008 | 0.000 |
| 3 | 32 | rollback | hold_limit | no_transfer_gain | -0.0010 | 0.195 | 0.012 | 0.000 |
| 4 | 40 | hold | no_transfer_gain | no_transfer_gain | 0.0000 | 0.109 | 0.008 | 0.000 |
| 5 | 48 | hold | no_transfer_gain | no_transfer_gain | -0.0010 | 0.188 | 0.004 | 0.000 |

### es_random/gate_off/clean/s0

| metric | value |
|---|---|
| updates | 48 |
| space | ChannelGainSpace (dim 64) |
| final A / T_rule / T_task / R_mean | 0.203 / 0.008 / 0.004 / 0.870 |
| T_rule(0) (start map) | 0.016 |
| final D(t) | 0.060 |
| hack rate total / final block | 0.000 / 0.000 |
| decisions | accept=6 |
| first-verdict accepts / oracle-improved / blocks | 0 / 0 / 6 |
| false-accept rate / false-reject rate | 0.000 / - |
| forward / backward calls (operator) | 384 / 0 |
| forward / backward sequences, FPE | 12288 / 0, 12288 |
| alarm: gap alarms / skipped updates | 0 / 0 of 48 |
| GPU memory / throughput | NOT_TESTED |

| decision | updates | decision | reason | first verdict | oracle diff | A | T_rule | hack (block) |
|---|---|---|---|---|---|---|---|---|
| 0 | 8 | accept | would=no_transfer_gain | no_transfer_gain | 0.0000 | 0.148 | 0.004 | 0.000 |
| 1 | 16 | accept | would=no_transfer_gain | no_transfer_gain | 0.0000 | 0.227 | 0.012 | 0.000 |
| 2 | 24 | accept | would=no_transfer_gain | no_transfer_gain | 0.0010 | 0.133 | 0.012 | 0.000 |
| 3 | 32 | accept | would=no_transfer_gain | no_transfer_gain | 0.0000 | 0.195 | 0.012 | 0.000 |
| 4 | 40 | accept | would=no_transfer_gain | no_transfer_gain | 0.0000 | 0.102 | 0.004 | 0.000 |
| 5 | 48 | accept | would=no_transfer_gain | no_transfer_gain | 0.0010 | 0.188 | 0.008 | 0.000 |

### es_random/gate_on/clean/s0

| metric | value |
|---|---|
| updates | 48 |
| space | ChannelGainSpace (dim 64) |
| final A / T_rule / T_task / R_mean | 0.203 / 0.012 / 0.004 / 0.867 |
| T_rule(0) (start map) | 0.016 |
| final D(t) | 0.039 |
| hack rate total / final block | 0.000 / 0.000 |
| decisions | hold=5, rollback=1 |
| first-verdict accepts / oracle-improved / blocks | 0 / 0 / 6 |
| false-accept rate / false-reject rate | 0.000 / - |
| forward / backward calls (operator) | 384 / 0 |
| forward / backward sequences, FPE | 12288 / 0, 12288 |
| alarm: gap alarms / skipped updates | 0 / 0 of 48 |
| GPU memory / throughput | NOT_TESTED |

| decision | updates | decision | reason | first verdict | oracle diff | A | T_rule | hack (block) |
|---|---|---|---|---|---|---|---|---|
| 0 | 8 | hold | no_transfer_gain | no_transfer_gain | 0.0000 | 0.148 | 0.004 | 0.000 |
| 1 | 16 | hold | no_transfer_gain | no_transfer_gain | 0.0000 | 0.227 | 0.012 | 0.000 |
| 2 | 24 | hold | no_transfer_gain | no_transfer_gain | 0.0010 | 0.133 | 0.012 | 0.000 |
| 3 | 32 | rollback | hold_limit | no_transfer_gain | 0.0010 | 0.195 | 0.012 | 0.000 |
| 4 | 40 | hold | no_transfer_gain | no_transfer_gain | 0.0000 | 0.109 | 0.008 | 0.000 |
| 5 | 48 | hold | no_transfer_gain | no_transfer_gain | 0.0000 | 0.188 | 0.004 | 0.000 |

### grpo/gate_off/clean/s1

| metric | value |
|---|---|
| updates | 48 |
| space | torch_grpo (dim -) |
| final A / T_rule / T_task / R_mean | 0.219 / 0.023 / 0.004 / 0.839 |
| T_rule(0) (start map) | 0.016 |
| final D(t) | 0.079 |
| hack rate total / final block | 0.000 / 0.000 |
| decisions | accept=6 |
| first-verdict accepts / oracle-improved / blocks | 0 / 0 / 6 |
| false-accept rate / false-reject rate | 0.000 / - |
| forward / backward calls (operator) | 96 / 48 |
| forward / backward sequences, FPE | 12288 / 6144, 24576 |
| alarm: gap alarms / skipped updates | 0 / 0 of 48 |
| GPU memory / throughput | NOT_TESTED |

| decision | updates | decision | reason | first verdict | oracle diff | A | T_rule | hack (block) |
|---|---|---|---|---|---|---|---|---|
| 0 | 8 | accept | would=no_transfer_gain | no_transfer_gain | 0.0010 | 0.133 | 0.004 | 0.000 |
| 1 | 16 | accept | would=no_transfer_gain | no_transfer_gain | 0.0029 | 0.188 | 0.008 | 0.000 |
| 2 | 24 | accept | would=no_transfer_gain | no_transfer_gain | 0.0000 | 0.125 | 0.020 | 0.000 |
| 3 | 32 | accept | would=no_transfer_gain | no_transfer_gain | -0.0020 | 0.172 | 0.004 | 0.000 |
| 4 | 40 | accept | would=no_transfer_gain | no_transfer_gain | -0.0010 | 0.250 | 0.000 | 0.000 |
| 5 | 48 | accept | would=no_transfer_gain | no_transfer_gain | 0.0010 | 0.211 | 0.008 | 0.000 |

### grpo/gate_on/clean/s1

| metric | value |
|---|---|
| updates | 48 |
| space | torch_grpo (dim -) |
| final A / T_rule / T_task / R_mean | 0.195 / 0.023 / 0.004 / 0.852 |
| T_rule(0) (start map) | 0.016 |
| final D(t) | 0.053 |
| hack rate total / final block | 0.000 / 0.000 |
| decisions | hold=5, rollback=1 |
| first-verdict accepts / oracle-improved / blocks | 0 / 0 / 6 |
| false-accept rate / false-reject rate | 0.000 / - |
| forward / backward calls (operator) | 96 / 48 |
| forward / backward sequences, FPE | 12288 / 6144, 24576 |
| alarm: gap alarms / skipped updates | 0 / 0 of 48 |
| GPU memory / throughput | NOT_TESTED |

| decision | updates | decision | reason | first verdict | oracle diff | A | T_rule | hack (block) |
|---|---|---|---|---|---|---|---|---|
| 0 | 8 | hold | no_transfer_gain | no_transfer_gain | 0.0010 | 0.133 | 0.004 | 0.000 |
| 1 | 16 | hold | no_transfer_gain | no_transfer_gain | 0.0020 | 0.188 | 0.008 | 0.000 |
| 2 | 24 | hold | no_transfer_gain | no_transfer_gain | 0.0039 | 0.125 | 0.020 | 0.000 |
| 3 | 32 | rollback | hold_limit | no_transfer_gain | 0.0000 | 0.203 | 0.008 | 0.000 |
| 4 | 40 | hold | no_transfer_gain | no_transfer_gain | 0.0010 | 0.227 | 0.004 | 0.000 |
| 5 | 48 | hold | no_transfer_gain | no_transfer_gain | 0.0000 | 0.211 | 0.000 | 0.000 |

### es_full/gate_off/clean/s1

| metric | value |
|---|---|
| updates | 48 |
| space | FullParameterSpace (dim 147456) |
| final A / T_rule / T_task / R_mean | 0.172 / 0.020 / 0.004 / 0.833 |
| T_rule(0) (start map) | 0.016 |
| final D(t) | 0.014 |
| hack rate total / final block | 0.000 / 0.000 |
| decisions | accept=6 |
| first-verdict accepts / oracle-improved / blocks | 0 / 0 / 6 |
| false-accept rate / false-reject rate | 0.000 / - |
| forward / backward calls (operator) | 384 / 0 |
| forward / backward sequences, FPE | 12288 / 0, 12288 |
| alarm: gap alarms / skipped updates | 0 / 0 of 48 |
| GPU memory / throughput | NOT_TESTED |

| decision | updates | decision | reason | first verdict | oracle diff | A | T_rule | hack (block) |
|---|---|---|---|---|---|---|---|---|
| 0 | 8 | accept | would=no_transfer_gain | no_transfer_gain | 0.0000 | 0.102 | 0.004 | 0.000 |
| 1 | 16 | accept | would=no_transfer_gain | no_transfer_gain | 0.0000 | 0.172 | 0.008 | 0.000 |
| 2 | 24 | accept | would=no_transfer_gain | no_transfer_gain | -0.0010 | 0.156 | 0.023 | 0.000 |
| 3 | 32 | accept | would=no_transfer_gain | no_transfer_gain | 0.0010 | 0.203 | 0.004 | 0.000 |
| 4 | 40 | accept | would=no_transfer_gain | no_transfer_gain | 0.0010 | 0.195 | 0.004 | 0.000 |
| 5 | 48 | accept | would=no_transfer_gain | no_transfer_gain | 0.0000 | 0.188 | 0.004 | 0.000 |

### es_full/gate_on/clean/s1

| metric | value |
|---|---|
| updates | 48 |
| space | FullParameterSpace (dim 147456) |
| final A / T_rule / T_task / R_mean | 0.172 / 0.020 / 0.004 / 0.833 |
| T_rule(0) (start map) | 0.016 |
| final D(t) | 0.001 |
| hack rate total / final block | 0.000 / 0.000 |
| decisions | hold=5, rollback=1 |
| first-verdict accepts / oracle-improved / blocks | 0 / 0 / 6 |
| false-accept rate / false-reject rate | 0.000 / - |
| forward / backward calls (operator) | 384 / 0 |
| forward / backward sequences, FPE | 12288 / 0, 12288 |
| alarm: gap alarms / skipped updates | 0 / 0 of 48 |
| GPU memory / throughput | NOT_TESTED |

| decision | updates | decision | reason | first verdict | oracle diff | A | T_rule | hack (block) |
|---|---|---|---|---|---|---|---|---|
| 0 | 8 | hold | no_transfer_gain | no_transfer_gain | 0.0000 | 0.102 | 0.004 | 0.000 |
| 1 | 16 | hold | no_transfer_gain | no_transfer_gain | 0.0000 | 0.172 | 0.008 | 0.000 |
| 2 | 24 | hold | no_transfer_gain | no_transfer_gain | 0.0000 | 0.156 | 0.023 | 0.000 |
| 3 | 32 | rollback | hold_limit | no_transfer_gain | 0.0010 | 0.203 | 0.008 | 0.000 |
| 4 | 40 | hold | no_transfer_gain | no_transfer_gain | 0.0010 | 0.188 | 0.004 | 0.000 |
| 5 | 48 | hold | no_transfer_gain | no_transfer_gain | 0.0000 | 0.188 | 0.004 | 0.000 |

### es_informed/gate_off/clean/s1

| metric | value |
|---|---|
| updates | 48 |
| space | ChannelGainSpace (dim 64) |
| final A / T_rule / T_task / R_mean | 0.180 / 0.020 / 0.004 / 0.833 |
| T_rule(0) (start map) | 0.016 |
| final D(t) | 0.007 |
| hack rate total / final block | 0.000 / 0.000 |
| decisions | accept=6 |
| first-verdict accepts / oracle-improved / blocks | 0 / 0 / 6 |
| false-accept rate / false-reject rate | 0.000 / - |
| forward / backward calls (operator) | 384 / 0 |
| forward / backward sequences, FPE | 12288 / 0, 12288 |
| alarm: gap alarms / skipped updates | 0 / 0 of 48 |
| GPU memory / throughput | NOT_TESTED |

| decision | updates | decision | reason | first verdict | oracle diff | A | T_rule | hack (block) |
|---|---|---|---|---|---|---|---|---|
| 0 | 8 | accept | would=no_transfer_gain | no_transfer_gain | 0.0000 | 0.102 | 0.004 | 0.000 |
| 1 | 16 | accept | would=no_transfer_gain | no_transfer_gain | 0.0000 | 0.172 | 0.008 | 0.000 |
| 2 | 24 | accept | would=no_transfer_gain | no_transfer_gain | 0.0000 | 0.156 | 0.023 | 0.000 |
| 3 | 32 | accept | would=no_transfer_gain | no_transfer_gain | 0.0000 | 0.195 | 0.008 | 0.000 |
| 4 | 40 | accept | would=no_transfer_gain | no_transfer_gain | 0.0010 | 0.188 | 0.004 | 0.000 |
| 5 | 48 | accept | would=no_transfer_gain | no_transfer_gain | -0.0010 | 0.188 | 0.004 | 0.000 |

### es_informed/gate_on/clean/s1

| metric | value |
|---|---|
| updates | 48 |
| space | ChannelGainSpace (dim 64) |
| final A / T_rule / T_task / R_mean | 0.172 / 0.020 / 0.004 / 0.831 |
| T_rule(0) (start map) | 0.016 |
| final D(t) | 0.006 |
| hack rate total / final block | 0.000 / 0.000 |
| decisions | hold=5, rollback=1 |
| first-verdict accepts / oracle-improved / blocks | 0 / 0 / 6 |
| false-accept rate / false-reject rate | 0.000 / - |
| forward / backward calls (operator) | 384 / 0 |
| forward / backward sequences, FPE | 12288 / 0, 12288 |
| alarm: gap alarms / skipped updates | 0 / 0 of 48 |
| GPU memory / throughput | NOT_TESTED |

| decision | updates | decision | reason | first verdict | oracle diff | A | T_rule | hack (block) |
|---|---|---|---|---|---|---|---|---|
| 0 | 8 | hold | no_transfer_gain | no_transfer_gain | 0.0000 | 0.102 | 0.004 | 0.000 |
| 1 | 16 | hold | no_transfer_gain | no_transfer_gain | 0.0000 | 0.172 | 0.008 | 0.000 |
| 2 | 24 | hold | no_transfer_gain | no_transfer_gain | -0.0010 | 0.156 | 0.023 | 0.000 |
| 3 | 32 | rollback | hold_limit | no_transfer_gain | -0.0010 | 0.203 | 0.008 | 0.000 |
| 4 | 40 | hold | no_transfer_gain | no_transfer_gain | 0.0000 | 0.188 | 0.004 | 0.000 |
| 5 | 48 | hold | no_transfer_gain | no_transfer_gain | 0.0010 | 0.180 | 0.004 | 0.000 |

### es_random/gate_off/clean/s1

| metric | value |
|---|---|
| updates | 48 |
| space | ChannelGainSpace (dim 64) |
| final A / T_rule / T_task / R_mean | 0.172 / 0.023 / 0.004 / 0.833 |
| T_rule(0) (start map) | 0.016 |
| final D(t) | -0.007 |
| hack rate total / final block | 0.000 / 0.000 |
| decisions | accept=6 |
| first-verdict accepts / oracle-improved / blocks | 0 / 0 / 6 |
| false-accept rate / false-reject rate | 0.000 / - |
| forward / backward calls (operator) | 384 / 0 |
| forward / backward sequences, FPE | 12288 / 0, 12288 |
| alarm: gap alarms / skipped updates | 0 / 0 of 48 |
| GPU memory / throughput | NOT_TESTED |

| decision | updates | decision | reason | first verdict | oracle diff | A | T_rule | hack (block) |
|---|---|---|---|---|---|---|---|---|
| 0 | 8 | accept | would=no_transfer_gain | no_transfer_gain | 0.0010 | 0.102 | 0.004 | 0.000 |
| 1 | 16 | accept | would=no_transfer_gain | no_transfer_gain | 0.0000 | 0.172 | 0.008 | 0.000 |
| 2 | 24 | accept | would=no_transfer_gain | no_transfer_gain | 0.0000 | 0.148 | 0.023 | 0.000 |
| 3 | 32 | accept | would=no_transfer_gain | no_transfer_gain | 0.0010 | 0.203 | 0.008 | 0.000 |
| 4 | 40 | accept | would=no_transfer_gain | no_transfer_gain | 0.0000 | 0.188 | 0.004 | 0.000 |
| 5 | 48 | accept | would=no_transfer_gain | no_transfer_gain | -0.0010 | 0.180 | 0.004 | 0.000 |

### es_random/gate_on/clean/s1

| metric | value |
|---|---|
| updates | 48 |
| space | ChannelGainSpace (dim 64) |
| final A / T_rule / T_task / R_mean | 0.172 / 0.020 / 0.004 / 0.833 |
| T_rule(0) (start map) | 0.016 |
| final D(t) | 0.001 |
| hack rate total / final block | 0.000 / 0.000 |
| decisions | hold=5, rollback=1 |
| first-verdict accepts / oracle-improved / blocks | 0 / 0 / 6 |
| false-accept rate / false-reject rate | 0.000 / - |
| forward / backward calls (operator) | 384 / 0 |
| forward / backward sequences, FPE | 12288 / 0, 12288 |
| alarm: gap alarms / skipped updates | 0 / 0 of 48 |
| GPU memory / throughput | NOT_TESTED |

| decision | updates | decision | reason | first verdict | oracle diff | A | T_rule | hack (block) |
|---|---|---|---|---|---|---|---|---|
| 0 | 8 | hold | no_transfer_gain | no_transfer_gain | 0.0010 | 0.102 | 0.004 | 0.000 |
| 1 | 16 | hold | no_transfer_gain | no_transfer_gain | 0.0000 | 0.172 | 0.008 | 0.000 |
| 2 | 24 | hold | no_transfer_gain | no_transfer_gain | 0.0010 | 0.148 | 0.023 | 0.000 |
| 3 | 32 | rollback | hold_limit | no_transfer_gain | 0.0010 | 0.203 | 0.008 | 0.000 |
| 4 | 40 | hold | no_transfer_gain | no_transfer_gain | 0.0010 | 0.188 | 0.004 | 0.000 |
| 5 | 48 | hold | no_transfer_gain | no_transfer_gain | 0.0010 | 0.180 | 0.004 | 0.000 |

### grpo/gate_off/clean/s2

| metric | value |
|---|---|
| updates | 48 |
| space | torch_grpo (dim -) |
| final A / T_rule / T_task / R_mean | 0.164 / 0.012 / 0.004 / 0.862 |
| T_rule(0) (start map) | 0.016 |
| final D(t) | -0.004 |
| hack rate total / final block | 0.000 / 0.000 |
| decisions | accept=6 |
| first-verdict accepts / oracle-improved / blocks | 0 / 0 / 6 |
| false-accept rate / false-reject rate | 0.000 / - |
| forward / backward calls (operator) | 96 / 48 |
| forward / backward sequences, FPE | 12288 / 6144, 24576 |
| alarm: gap alarms / skipped updates | 0 / 0 of 48 |
| GPU memory / throughput | NOT_TESTED |

| decision | updates | decision | reason | first verdict | oracle diff | A | T_rule | hack (block) |
|---|---|---|---|---|---|---|---|---|
| 0 | 8 | accept | would=no_transfer_gain | no_transfer_gain | 0.0000 | 0.164 | 0.012 | 0.000 |
| 1 | 16 | accept | would=no_transfer_gain | no_transfer_gain | 0.0020 | 0.148 | 0.016 | 0.000 |
| 2 | 24 | accept | would=no_transfer_gain | no_transfer_gain | 0.0000 | 0.188 | 0.004 | 0.000 |
| 3 | 32 | accept | would=no_transfer_gain | no_transfer_gain | -0.0020 | 0.180 | 0.016 | 0.000 |
| 4 | 40 | accept | would=no_transfer_gain | no_transfer_gain | 0.0010 | 0.172 | 0.000 | 0.000 |
| 5 | 48 | accept | would=no_transfer_gain | no_transfer_gain | 0.0010 | 0.188 | 0.016 | 0.000 |

### grpo/gate_on/clean/s2

| metric | value |
|---|---|
| updates | 48 |
| space | torch_grpo (dim -) |
| final A / T_rule / T_task / R_mean | 0.148 / 0.008 / 0.000 / 0.867 |
| T_rule(0) (start map) | 0.016 |
| final D(t) | -0.028 |
| hack rate total / final block | 0.000 / 0.000 |
| decisions | hold=5, rollback=1 |
| first-verdict accepts / oracle-improved / blocks | 0 / 0 / 6 |
| false-accept rate / false-reject rate | 0.000 / - |
| forward / backward calls (operator) | 96 / 48 |
| forward / backward sequences, FPE | 12288 / 6144, 24576 |
| alarm: gap alarms / skipped updates | 0 / 0 of 48 |
| GPU memory / throughput | NOT_TESTED |

| decision | updates | decision | reason | first verdict | oracle diff | A | T_rule | hack (block) |
|---|---|---|---|---|---|---|---|---|
| 0 | 8 | hold | no_transfer_gain | no_transfer_gain | 0.0000 | 0.164 | 0.012 | 0.000 |
| 1 | 16 | hold | no_transfer_gain | no_transfer_gain | 0.0000 | 0.148 | 0.016 | 0.000 |
| 2 | 24 | hold | no_transfer_gain | no_transfer_gain | 0.0039 | 0.188 | 0.004 | 0.000 |
| 3 | 32 | rollback | hold_limit | no_transfer_gain | 0.0000 | 0.148 | 0.012 | 0.000 |
| 4 | 40 | hold | no_transfer_gain | no_transfer_gain | -0.0010 | 0.172 | 0.000 | 0.000 |
| 5 | 48 | hold | no_transfer_gain | no_transfer_gain | 0.0020 | 0.172 | 0.023 | 0.000 |

### es_full/gate_off/clean/s2

| metric | value |
|---|---|
| updates | 48 |
| space | FullParameterSpace (dim 147456) |
| final A / T_rule / T_task / R_mean | 0.125 / 0.008 / 0.008 / 0.857 |
| T_rule(0) (start map) | 0.016 |
| final D(t) | 0.060 |
| hack rate total / final block | 0.000 / 0.000 |
| decisions | accept=6 |
| first-verdict accepts / oracle-improved / blocks | 0 / 0 / 6 |
| false-accept rate / false-reject rate | 0.000 / - |
| forward / backward calls (operator) | 384 / 0 |
| forward / backward sequences, FPE | 12288 / 0, 12288 |
| alarm: gap alarms / skipped updates | 0 / 0 of 48 |
| GPU memory / throughput | NOT_TESTED |

| decision | updates | decision | reason | first verdict | oracle diff | A | T_rule | hack (block) |
|---|---|---|---|---|---|---|---|---|
| 0 | 8 | accept | would=no_transfer_gain | no_transfer_gain | 0.0000 | 0.156 | 0.016 | 0.000 |
| 1 | 16 | accept | would=no_transfer_gain | no_transfer_gain | 0.0000 | 0.156 | 0.020 | 0.000 |
| 2 | 24 | accept | would=no_transfer_gain | no_transfer_gain | 0.0000 | 0.172 | 0.004 | 0.000 |
| 3 | 32 | accept | would=no_transfer_gain | no_transfer_gain | 0.0000 | 0.148 | 0.012 | 0.000 |
| 4 | 40 | accept | would=no_transfer_gain | no_transfer_gain | 0.0010 | 0.172 | 0.000 | 0.000 |
| 5 | 48 | accept | would=no_transfer_gain | no_transfer_gain | 0.0000 | 0.164 | 0.020 | 0.000 |

### es_full/gate_on/clean/s2

| metric | value |
|---|---|
| updates | 48 |
| space | FullParameterSpace (dim 147456) |
| final A / T_rule / T_task / R_mean | 0.125 / 0.008 / 0.008 / 0.852 |
| T_rule(0) (start map) | 0.016 |
| final D(t) | 0.061 |
| hack rate total / final block | 0.000 / 0.000 |
| decisions | hold=5, rollback=1 |
| first-verdict accepts / oracle-improved / blocks | 0 / 0 / 6 |
| false-accept rate / false-reject rate | 0.000 / - |
| forward / backward calls (operator) | 384 / 0 |
| forward / backward sequences, FPE | 12288 / 0, 12288 |
| alarm: gap alarms / skipped updates | 0 / 0 of 48 |
| GPU memory / throughput | NOT_TESTED |

| decision | updates | decision | reason | first verdict | oracle diff | A | T_rule | hack (block) |
|---|---|---|---|---|---|---|---|---|
| 0 | 8 | hold | no_transfer_gain | no_transfer_gain | 0.0000 | 0.156 | 0.016 | 0.000 |
| 1 | 16 | hold | no_transfer_gain | no_transfer_gain | 0.0000 | 0.156 | 0.020 | 0.000 |
| 2 | 24 | hold | no_transfer_gain | no_transfer_gain | 0.0010 | 0.172 | 0.004 | 0.000 |
| 3 | 32 | rollback | hold_limit | no_transfer_gain | 0.0000 | 0.148 | 0.012 | 0.000 |
| 4 | 40 | hold | no_transfer_gain | no_transfer_gain | -0.0010 | 0.172 | 0.000 | 0.000 |
| 5 | 48 | hold | no_transfer_gain | no_transfer_gain | 0.0000 | 0.172 | 0.020 | 0.000 |

### es_informed/gate_off/clean/s2

| metric | value |
|---|---|
| updates | 48 |
| space | ChannelGainSpace (dim 64) |
| final A / T_rule / T_task / R_mean | 0.125 / 0.004 / 0.008 / 0.857 |
| T_rule(0) (start map) | 0.016 |
| final D(t) | 0.050 |
| hack rate total / final block | 0.000 / 0.000 |
| decisions | accept=6 |
| first-verdict accepts / oracle-improved / blocks | 0 / 0 / 6 |
| false-accept rate / false-reject rate | 0.000 / - |
| forward / backward calls (operator) | 384 / 0 |
| forward / backward sequences, FPE | 12288 / 0, 12288 |
| alarm: gap alarms / skipped updates | 0 / 0 of 48 |
| GPU memory / throughput | NOT_TESTED |

| decision | updates | decision | reason | first verdict | oracle diff | A | T_rule | hack (block) |
|---|---|---|---|---|---|---|---|---|
| 0 | 8 | accept | would=no_transfer_gain | no_transfer_gain | 0.0010 | 0.164 | 0.016 | 0.000 |
| 1 | 16 | accept | would=no_transfer_gain | no_transfer_gain | 0.0000 | 0.156 | 0.020 | 0.000 |
| 2 | 24 | accept | would=no_transfer_gain | no_transfer_gain | -0.0010 | 0.172 | 0.004 | 0.000 |
| 3 | 32 | accept | would=no_transfer_gain | no_transfer_gain | 0.0000 | 0.148 | 0.012 | 0.000 |
| 4 | 40 | accept | would=no_transfer_gain | no_transfer_gain | -0.0020 | 0.172 | 0.000 | 0.000 |
| 5 | 48 | accept | would=no_transfer_gain | no_transfer_gain | 0.0000 | 0.164 | 0.020 | 0.000 |

### es_informed/gate_on/clean/s2

| metric | value |
|---|---|
| updates | 48 |
| space | ChannelGainSpace (dim 64) |
| final A / T_rule / T_task / R_mean | 0.125 / 0.008 / 0.008 / 0.857 |
| T_rule(0) (start map) | 0.016 |
| final D(t) | 0.052 |
| hack rate total / final block | 0.000 / 0.000 |
| decisions | hold=5, rollback=1 |
| first-verdict accepts / oracle-improved / blocks | 0 / 0 / 6 |
| false-accept rate / false-reject rate | 0.000 / - |
| forward / backward calls (operator) | 384 / 0 |
| forward / backward sequences, FPE | 12288 / 0, 12288 |
| alarm: gap alarms / skipped updates | 0 / 0 of 48 |
| GPU memory / throughput | NOT_TESTED |

| decision | updates | decision | reason | first verdict | oracle diff | A | T_rule | hack (block) |
|---|---|---|---|---|---|---|---|---|
| 0 | 8 | hold | no_transfer_gain | no_transfer_gain | 0.0010 | 0.164 | 0.016 | 0.000 |
| 1 | 16 | hold | no_transfer_gain | no_transfer_gain | 0.0000 | 0.156 | 0.020 | 0.000 |
| 2 | 24 | hold | no_transfer_gain | no_transfer_gain | 0.0029 | 0.172 | 0.004 | 0.000 |
| 3 | 32 | rollback | hold_limit | no_transfer_gain | 0.0000 | 0.148 | 0.012 | 0.000 |
| 4 | 40 | hold | no_transfer_gain | no_transfer_gain | 0.0000 | 0.164 | 0.000 | 0.000 |
| 5 | 48 | hold | no_transfer_gain | no_transfer_gain | 0.0000 | 0.180 | 0.020 | 0.000 |

### es_random/gate_off/clean/s2

| metric | value |
|---|---|
| updates | 48 |
| space | ChannelGainSpace (dim 64) |
| final A / T_rule / T_task / R_mean | 0.125 / 0.008 / 0.008 / 0.857 |
| T_rule(0) (start map) | 0.016 |
| final D(t) | 0.048 |
| hack rate total / final block | 0.000 / 0.000 |
| decisions | accept=6 |
| first-verdict accepts / oracle-improved / blocks | 0 / 0 / 6 |
| false-accept rate / false-reject rate | 0.000 / - |
| forward / backward calls (operator) | 384 / 0 |
| forward / backward sequences, FPE | 12288 / 0, 12288 |
| alarm: gap alarms / skipped updates | 0 / 0 of 48 |
| GPU memory / throughput | NOT_TESTED |

| decision | updates | decision | reason | first verdict | oracle diff | A | T_rule | hack (block) |
|---|---|---|---|---|---|---|---|---|
| 0 | 8 | accept | would=no_transfer_gain | no_transfer_gain | 0.0010 | 0.156 | 0.016 | 0.000 |
| 1 | 16 | accept | would=no_transfer_gain | no_transfer_gain | 0.0000 | 0.164 | 0.020 | 0.000 |
| 2 | 24 | accept | would=no_transfer_gain | no_transfer_gain | 0.0000 | 0.172 | 0.004 | 0.000 |
| 3 | 32 | accept | would=no_transfer_gain | no_transfer_gain | 0.0000 | 0.156 | 0.012 | 0.000 |
| 4 | 40 | accept | would=no_transfer_gain | no_transfer_gain | 0.0000 | 0.172 | 0.000 | 0.000 |
| 5 | 48 | accept | would=no_transfer_gain | no_transfer_gain | -0.0010 | 0.172 | 0.023 | 0.000 |

### es_random/gate_on/clean/s2

| metric | value |
|---|---|
| updates | 48 |
| space | ChannelGainSpace (dim 64) |
| final A / T_rule / T_task / R_mean | 0.125 / 0.008 / 0.008 / 0.854 |
| T_rule(0) (start map) | 0.016 |
| final D(t) | 0.055 |
| hack rate total / final block | 0.000 / 0.000 |
| decisions | hold=5, rollback=1 |
| first-verdict accepts / oracle-improved / blocks | 0 / 0 / 6 |
| false-accept rate / false-reject rate | 0.000 / - |
| forward / backward calls (operator) | 384 / 0 |
| forward / backward sequences, FPE | 12288 / 0, 12288 |
| alarm: gap alarms / skipped updates | 0 / 0 of 48 |
| GPU memory / throughput | NOT_TESTED |

| decision | updates | decision | reason | first verdict | oracle diff | A | T_rule | hack (block) |
|---|---|---|---|---|---|---|---|---|
| 0 | 8 | hold | no_transfer_gain | no_transfer_gain | 0.0010 | 0.156 | 0.016 | 0.000 |
| 1 | 16 | hold | no_transfer_gain | no_transfer_gain | 0.0000 | 0.164 | 0.020 | 0.000 |
| 2 | 24 | hold | no_transfer_gain | no_transfer_gain | 0.0010 | 0.172 | 0.004 | 0.000 |
| 3 | 32 | rollback | hold_limit | no_transfer_gain | 0.0000 | 0.148 | 0.012 | 0.000 |
| 4 | 40 | hold | no_transfer_gain | no_transfer_gain | 0.0000 | 0.172 | 0.000 | 0.000 |
| 5 | 48 | hold | no_transfer_gain | no_transfer_gain | -0.0010 | 0.172 | 0.020 | 0.000 |

## Gate exercise (are the error rates meaningful?)

- blocks 288: first-verdict accepts 0, oracle-improved 0 (need >= 5), max oracle diff 0.0039 (threshold 0.01)
- false-reject rate evaluable: no; false-accept rate informative: no; exercised: **no**
- only 0 of 288 blocks are oracle-improved (need >= 5): the false-reject rate and the reject-rate KILL are not evaluable
- the gate accepted none of 288 blocks (first verdicts): its false-accept rate is 0 by construction and says nothing about discrimination

## Section-3 criteria (automatic evaluation)

| criterion | verdict | reason |
|---|---|---|
| gate | **INCONCLUSIVE** | reject-rate KILL not evaluable: 0 oracle-improved block(s), need >= 5 (rejected 0, rate -); the T_rule / hack intervals do not establish success either [gate not exercised: only 0 of 288 blocks are oracle-improved (need >= 5): the false-reject rate and the reject-rate KILL are not evaluable; the gate accepted none of 288 blocks (first verdicts): its false-accept rate is 0 by construction and says nothing about discrimination] |
| subset | **KILL** | no separation in favour of the informed subset: keep channel groups only as the rollback unit |
| operator | **INCONCLUSIVE** | interval of r contains zero [Operator KILL not reachable: the es_informed arm stopped before spending the 10x budget in 3 of 3 paired seed(s) (at most 5.0% of it spent); raise es_updates] |
| low_start | **INCONCLUSIVE** | no arm significantly exceeds the start's pass@k, but not every arm is below it |

### gate

- d_hack: mean 0.0031 [-0.0053, 0.0114] (n=3)
- d_T_rule_leak: mean 0.0013 [-0.0048, 0.0074] (n=3)
- c_T_rule_clean: mean -0.0007 [-0.0021, 0.0007] (n=3)
- k_cost_minus_gain: mean -0.0007 [-0.0057, 0.0044] (n=3)
- definition: Per operator and seed: leak variant d_hack = hack_rate_final(gate_off) - hack_rate_final(gate_on) and d_T = T_rule_final(gate_on) - T_rule_final(gate_off); clean variant c = T_rule_final(gate_on) - T_rule_final(gate_off); each averaged over the operators that have all four arms for that seed. SUCCESS: interval(d_hack).lo > 0 and interval(d_T).lo > 0 and mean(c) >= -se(c) (clean T_rule within one standard error of gate-off). KILL: the clean-run T_rule cost exceeds the planted-run gain, tested paired: k = -c - d_T per seed, interval(k).lo > 0; or the first verdicts (all arms, both gate settings) reject more than half of the blocks the oracle marks as improved (evaluable only with >= min_improved_blocks such blocks). SUCCESS requires both KILL conditions to be evaluable: with fewer than min_improved_blocks oracle-improved blocks the verdict is INCONCLUSIVE ('reject-rate KILL not evaluable'), whatever the intervals show (the k KILL still applies). Otherwise INCONCLUSIVE. T_rule_final = tracker T_rule of the weights after the last update, measured at the tracker step oracle_exclusion_steps(config), the same for every arm (so same-seed arms' finals are scored on identical tracker items whatever their decision counts); hack_rate_final = V1-pass & V2-fail share of the last block_updates updates' training rollouts. Every gate verdict carries gate_exercise: when fewer than min_improved_blocks blocks are oracle-improved or the gate accepted no block, the reason says 'gate not exercised' and the error rates are reported as not evaluable / not informative.

### subset

- d_R: mean -0.0007 [-0.0023, 0.0010] (n=3)
- d_T_rule: mean -0.0007 [-0.0044, 0.0031] (n=3)
- d_A: mean 0.0046 [-0.0066, 0.0158] (n=3)
- A_gain_informed: mean 0.0345 [0.0273, 0.0417] (n=12)
- A_gain_random: mean 0.0299 [0.0213, 0.0386] (n=12)
- definition: Per seed (averaged over the gate settings and variants present for both arms): d_R = R_mean(es_informed) - R_mean(es_random), d_T = T_rule(es_informed) - T_rule(es_random), d_A = A(es_informed) - A(es_random) at the end of training. 'Equal A': |mean(d_A)| <= equal_a_tolerance. SUCCESS: equal A and (interval(d_R).lo > 0 or interval(d_T).lo > 0). KILL: equal A and neither interval excludes zero in favour of the informed subset (no separation). INCONCLUSIVE: A not equal, or < 3 seeds. The informed subset is one fixed selection shared by every seed (only the density-matched random control is redrawn per seed), so a verdict is about this selection, not about the selection procedure. Reported beside it: each subset's A gain since its step-0 tracker point (a no-separation KILL where neither subset acquired anything is vacuous and should be read as such).

### operator

- grpo_gain: mean 0.0078 [-0.0019, 0.0175] (n=3)
- r: mean -0.0039 [-0.0436, 0.0358] (n=3)
- definition: Per seed, GRPO arm (clean variant and gate_off if present, else the first available): G = T_rule_final - T_rule at the arm's own step-0 tracker point (not the start map: one fixed sample shared by every seed would add a bias the interval cannot see) and budget B = operator_budget_factor x its total forward-pass equivalents (sequence level: each generated or training-forward sequence = 1, each backward sequence = 2). ES arm (operator_es_arm, same variant and gate): T_rule gain (relative to its own step-0 point) at its last tracker point with cumulative FPE <= B (one point, like GRPO's final point; not a maximum over noisy points, which would favour ES by selection); r = gain - operator_gain_fraction x G. SUCCESS: interval(r).lo > 0 and mean(G) > 0. KILL: interval(r).hi < 0, mean(G) > 0 and the ES arm spent at least B in every seed. INCONCLUSIVE otherwise (in particular when GRPO gained nothing or ES stopped before spending B). kill_reachable (statistic) = the ES arm spent B in every paired seed (None without a paired seed); when it is False every reason says 'Operator KILL not reachable' and the run warns: only SUCCESS or INCONCLUSIVE were possible (raise es_updates).

### low_start

- definition: Start pass@k (k = start_k, sampled) on t_rule/unseen_pairs versus each arm's final greedy pass@1 on that slice (tracker), per seed. SUCCESS: some arm (operator x gate x variant) has interval(final - start_pass_k).lo > 0. KILL: every arm has interval.hi <= 0 (none exceeds the start's pass@k: the task is acquisition-limited; reported, the task is not weakened). Otherwise INCONCLUSIVE. The start pass@k is one sample shared by every arm and seed, so its sampling error is common to all paired differences: each interval adds se_start = baseline_standard_error(pass_k, start items) (binomial bound, never 0); the interval half-width is the quadrature sum of the seeds' t half-width and the baseline's half-width (interval_with_baseline_error). Several arms are compared without a multiplicity correction (stated, not corrected).

**CPU tiny-model pilot (P0). Not evidence about 7-8B models; no GPU numbers.**
