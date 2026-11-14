"""The TRL + update-rules smoke check used by the CUDA smoke test also runs (and passes) on CPU."""

import pytest

from research_rl.utils.smoke import trl_rules_smoke


@pytest.mark.integration
def test_trl_rules_smoke_on_cpu(tmp_path):
    out = trl_rules_smoke(tmp_path, steps=2)
    assert out["applied_updates"] == 2 and out["p2_step_count"] == 2
    assert out["mask_bit_exact"] and out["p2_coefficients_moved"]
    assert out["device"] == "cpu"
