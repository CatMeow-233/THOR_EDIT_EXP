import json, sys
from pathlib import Path
import torch
from safetensors.torch import load_file
from research_rl.capability import pilot as P
from research_rl.capability.tasks import ItemSampler, ExactAnswerVerifier
from research_rl.capability.probe import StartMap
from research_rl.capability.tracker import CapabilityTracker
out = Path(sys.argv[1])
res = json.loads((out/"results.json").read_text())
cfg = P.PilotConfig.from_dict(json.loads((out/"pilot_config.json").read_text()))
torch.set_num_threads(4)
splits = P.build_splits(cfg); tok = P.build_char_tokenizer()
model = P.build_model(cfg, len(tok)); model.load_state_dict(load_file(out/"start_model.safetensors")); model.eval()
budget = P._budget(cfg, splits)
start = StartMap.from_dict(res["start_map"])
fs = P.oracle_exclusion_steps(cfg)
arms = res["arms"]
print("final_step", fs)
for seed in cfg.seeds:
    sampler = ItemSampler(splits, root_seed=P._sampler_root(cfg, seed))
    fn = P._CountingScoreFn(tok, ExactAnswerVerifier(), max_new_tokens=budget, batch_size=cfg.eval_batch_size)
    tr = CapabilityTracker(sampler, fn, start, items_per_slice=cfg.track_items_per_slice)
    pt = tr.measure(model, fs, None).to_dict()
    print(f"seed {seed} START  A={pt['A']:.3f} T_rule={pt['T_rule']:.3f} R={sum(pt['R'].values())/len(pt['R']):.3f}")
    for name, a in sorted(arms.items()) if isinstance(arms, dict) else []:
        pass
    for a in (arms.values() if isinstance(arms, dict) else arms):
        if a.get("seed", a.get("spec",{}).get("seed")) == seed:
            f = a["final"]
            print(f"   {a['key']}: A={f['A']:.3f} T_rule={f['T_rule']:.3f} R={f['R_mean']:.3f} step={a.get('final_step')}")
