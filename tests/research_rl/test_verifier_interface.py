"""Verifier contract, TRL reward adapter (incl. a real GRPOTrainer step) and the disabled-by-default sandbox."""

from __future__ import annotations

import json
import math
import pickle
from typing import Any

import pytest
import torch
from datasets import Dataset
from trl import GRPOConfig, GRPOTrainer
from trl.trainer.utils import get_callable_name, is_async_callable

from research_rl.core.registry import VERIFIERS
from research_rl.rewards import (
    BaseVerifier,
    CodeExecutionVerifier,
    ExecutionRequest,
    ExecutionResult,
    FatalVerifierError,
    IsolationGuarantees,
    MathVerifier,
    RewardScheme,
    SandboxDisabledError,
    SandboxFusionBackend,
    SandboxPolicy,
    VerificationResult,
    Verifier,
    as_trl_reward_func,
    check_sandbox,
    completion_to_text,
    extract_code,
)
from research_rl.rewards.code_sandbox import parse_sandbox_fusion_response, urllib_transport

FULL = IsolationGuarantees(
    isolated_from_host=True, network_disabled=True, enforces_time_limit=True, enforces_memory_limit=True
)
ENABLED = SandboxPolicy(enabled=True)

# ----------------------------------------------------------------------------- VerificationResult / scheme


def test_verification_result_validation():
    VerificationResult(reward=1.0, correct=True, status="correct")
    with pytest.raises(ValueError):
        VerificationResult(reward=1.0, correct=True, status="right")  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        VerificationResult(reward=1.0, correct=True, status="incorrect")
    with pytest.raises(ValueError):
        VerificationResult(reward=0.0, correct=False, status="correct")
    with pytest.raises(ValueError):
        VerificationResult(reward=math.nan, correct=False, status="incorrect")
    with pytest.raises(ValueError):
        RewardScheme(correct=math.inf)


class _Fixed(BaseVerifier):
    name = "fixed"

    def __init__(self, status, exc=None, rewards=None):
        super().__init__(rewards)
        self.status, self.exc, self.calls = status, exc, []

    def _verify(self, completion, reference, **meta):
        self.calls.append((completion, reference, meta))
        if self.exc is not None:
            raise self.exc
        return self.status, completion, str(reference), {}


@pytest.mark.parametrize(
    ("status", "expected"),
    [
        ("correct", 3.0),
        ("incorrect", -1.0),
        ("format_error", -2.0),
        ("no_answer", -3.0),
        ("truncated", -4.0),
        ("error", -5.0),
    ],
)
def test_reward_scheme_maps_every_status(status, expected):
    scheme = RewardScheme(3.0, -1.0, -2.0, -3.0, -4.0, -5.0)
    result = _Fixed(status, rewards=scheme).verify("x", "y")
    assert (result.status, result.reward, result.correct) == (status, expected, status == "correct")


def test_base_verifier_turns_bugs_into_error_but_propagates_fatal():
    result = _Fixed("correct", exc=ZeroDivisionError("boom")).verify("x", "y")
    assert result.status == "error" and "ZeroDivisionError" in result.details["exception"]
    with pytest.raises(FatalVerifierError):
        _Fixed("correct", exc=FatalVerifierError("stop")).verify("x", "y")
    assert isinstance(MathVerifier(), Verifier)


# ----------------------------------------------------------------------------- completion_to_text


@pytest.mark.parametrize(
    ("completion", "expected"),
    [
        ("plain", "plain"),
        (None, None),
        ([{"role": "assistant", "content": "\\boxed{1}"}], "\\boxed{1}"),
        ({"role": "assistant", "content": "single"}, "single"),
        (
            [
                {"role": "assistant", "content": "call tool"},
                {"role": "tool", "content": "42"},
                {"role": "assistant", "content": "final"},
            ],
            "final",
        ),
        ([{"role": "assistant", "content": [{"type": "text", "text": "a"}, {"type": "image"}, "b"]}], "ab"),
        ([{"role": "tool", "content": "x"}], None),
        ([], None),
    ],
)
def test_completion_to_text(completion, expected):
    assert completion_to_text(completion) == expected


# ----------------------------------------------------------------------------- as_trl_reward_func

COMPLETIONS = ["\\boxed{4}", "#### 5", "no idea", "\\boxed{}", "The answer is 0.5"]
ANSWERS = ["4", "6", "1", "2", "1/2"]
EXPECTED_STATUS = ["correct", "incorrect", "no_answer", "format_error", "correct"]
SCHEME = RewardScheme(
    correct=1.0, incorrect=0.0, format_error=-0.5, no_answer=-0.25, truncated=-1.0, error=-2.0
)
EXPECTED_REWARD = [1.0, 0.0, -0.25, -0.5, 1.0]


def _trl_kwargs(**extra):
    logged = {}
    kw = {"trainer_state": object(), "log_extra": lambda c, v: None, "log_metric": logged.__setitem__}
    kw.update(extra)
    return kw, logged


def test_reward_func_string_completions():
    fn = as_trl_reward_func(MathVerifier(rewards=SCHEME))
    kw, logged = _trl_kwargs(answer=ANSWERS)
    out = fn(prompts=["p"] * 5, completions=COMPLETIONS, completion_ids=[[1]] * 5, **kw)
    assert out == EXPECTED_REWARD
    assert [r.status for r in fn.last_results] == EXPECTED_STATUS
    assert logged["verifier/math_reward/correct_frac"] == pytest.approx(2 / 5)
    assert logged["verifier/math_reward/no_answer_frac"] == pytest.approx(1 / 5)
    assert sum(logged.values()) == pytest.approx(1.0)


def test_reward_func_conversational_completions_and_custom_key():
    fn = as_trl_reward_func(MathVerifier(rewards=SCHEME), answer_key="solution", name="gsm8k_acc")
    convs = [[{"role": "assistant", "content": c}] for c in COMPLETIONS]
    prompts = [[{"role": "user", "content": "q"}]] * 5
    kw, _ = _trl_kwargs(solution=ANSWERS, answer=["wrong"] * 5)
    assert fn(prompts=prompts, completions=convs, **kw) == EXPECTED_REWARD
    assert fn.__name__ == "gsm8k_acc" and get_callable_name(fn) == "gsm8k_acc"
    assert not is_async_callable(fn)


def test_reward_func_truncation_from_completion_ids():
    fn = as_trl_reward_func(MathVerifier(rewards=SCHEME), eos_token_ids=[2, 7])
    comps = ["\\boxed{4}", "\\boxed{4}", "\\boxed{4}", "", "\\boxed{4}</s>", "\\boxed{4}</s>"]
    ids = [[5, 6, 2], [5, 6, 7], [5, 6, 9], [], [5, 6, 9], [5, 6, 2]]
    kw, _ = _trl_kwargs(answer=["4"] * 6)
    out = fn(prompts=None, completions=comps, completion_ids=ids, **kw)
    # ends in eos -> graded; ends in other id -> truncated; empty ids -> truncated; a textual </s> does
    # NOT clear the token-level truncation flag (with skip_special_tokens it came from ordinary tokens)
    statuses = ["correct", "correct", "truncated", "truncated", "truncated", "correct"]
    assert [r.status for r in fn.last_results] == statuses
    assert out == [1.0, 1.0, -1.0, -1.0, -1.0, 1.0]
    # without eos ids nothing is flagged
    plain = as_trl_reward_func(MathVerifier(rewards=SCHEME))
    assert plain(completions=comps[:3], completion_ids=ids[:3], answer=["4"] * 3) == [1.0, 1.0, 1.0]


def test_reward_func_truncated_key_column():
    fn = as_trl_reward_func(MathVerifier(rewards=SCHEME), truncated_key="cut")
    assert fn(completions=["\\boxed{4}"] * 2, answer=["4", "4"], cut=[True, False]) == [-1.0, 1.0]
    with pytest.raises(KeyError):
        fn(completions=["\\boxed{4}"], answer=["4"])


def test_reward_func_fails_loudly_on_bad_inputs():
    fn = as_trl_reward_func(MathVerifier())
    with pytest.raises(KeyError, match="answer"):
        fn(prompts=["p"], completions=["\\boxed{1}"], ground_truth=["1"])
    with pytest.raises(ValueError):
        fn(completions=["\\boxed{1}", "\\boxed{2}"], answer=["1"])
    with pytest.raises(ValueError):
        as_trl_reward_func(MathVerifier(), eos_token_ids=[2])(
            completions=["a"], completion_ids=[[1], [2]], answer=["1"]
        )
    with pytest.raises(TypeError):
        as_trl_reward_func(object())  # type: ignore[arg-type]


def test_reward_func_none_on_error():
    fn = as_trl_reward_func(MathVerifier(rewards=SCHEME), none_on_error=True)
    assert fn(completions=["\\boxed{1}", "\\boxed{1}"], answer=[None, "1"]) == [None, 1.0]
    strict = as_trl_reward_func(MathVerifier(rewards=SCHEME))
    assert strict(completions=["\\boxed{1}"], answer=[None]) == [-2.0]


def test_reward_func_max_error_frac():
    # default: per-row bad references are scored, not raised
    assert as_trl_reward_func(MathVerifier())(completions=["\\boxed{1}"] * 2, answer=[None, None]) == [
        0.0,
        0.0,
    ]
    fn = as_trl_reward_func(MathVerifier(), max_error_frac=0.5)
    assert fn(completions=["\\boxed{1}"] * 2, answer=[None, "1"]) == [0.0, 1.0]  # 1/2 is not > 0.5
    with pytest.raises(ValueError, match="2/3 completions"):
        fn(completions=["\\boxed{1}"] * 3, answer=[None, "", "1"])
    assert [r.status for r in fn.last_results] == ["error", "error", "correct"]  # still inspectable
    assert pickle.loads(pickle.dumps(fn)).max_error_frac == 0.5
    for bad in (1.0, -0.1):
        with pytest.raises(ValueError):
            as_trl_reward_func(MathVerifier(), max_error_frac=bad)


def test_reward_func_passes_through_meta_and_is_picklable():
    probe = _Fixed("correct")
    fn = as_trl_reward_func(probe, eos_token_ids=[2])
    assert fn.__name__ == "fixed_reward"
    fn(completions=["a", "b"], completion_ids=[[2], [3]], answer=[1, 2])
    assert probe.calls == [("a", 1, {"truncated": False}), ("b", 2, {"truncated": True})]
    clone = pickle.loads(pickle.dumps(as_trl_reward_func(MathVerifier(), name="m")))
    assert clone.__name__ == "m" and clone(completions=["\\boxed{3}"], answer=["3"]) == [1.0]


def test_registry_entries():
    assert VERIFIERS.get("math") is MathVerifier
    assert VERIFIERS.get("code_execution") is CodeExecutionVerifier
    built = VERIFIERS.build("math", strict_boxed=True)
    assert built.verify("5", "5").status == "format_error"
    with pytest.raises(SandboxDisabledError):
        VERIFIERS.build("code_execution")


# ----------------------------------------------------------------------------- real TRL GRPOTrainer step


class _Recorder:
    """Records what TRL passes to / gets from the wrapped reward function."""

    def __init__(self, fn):
        self.fn, self.__name__, self.calls = fn, fn.__name__, []

    def __call__(self, prompts=None, completions=None, completion_ids=None, **kwargs):
        out = self.fn(prompts=prompts, completions=completions, completion_ids=completion_ids, **kwargs)
        self.calls.append(
            {"prompts": prompts, "completions": completions, "ids": completion_ids, "kw": kwargs}
        )
        self.calls[-1]["out"] = out
        return out


@pytest.mark.integration
def test_grpo_trainer_one_step_with_math_reward(tiny_model_dir, tmp_path):
    rows = [
        {"prompt": [{"role": "user", "content": f"{a}+{b}"}], "answer": str(a + b)}
        for a in range(2)
        for b in range(2)
    ]
    scheme = RewardScheme(correct=1.0, incorrect=0.0, format_error=-0.25, no_answer=-0.5, truncated=-1.0)
    reward = _Recorder(as_trl_reward_func(MathVerifier(rewards=scheme), eos_token_ids=[2]))
    args = GRPOConfig(
        output_dir=str(tmp_path),
        per_device_train_batch_size=4,
        num_generations=4,
        max_completion_length=8,
        max_steps=1,
        learning_rate=1e-2,
        logging_steps=1,
        report_to=[],
        use_cpu=True,
        seed=0,
        save_strategy="no",
    )
    trainer = GRPOTrainer(
        model=tiny_model_dir, reward_funcs=[reward], args=args, train_dataset=Dataset.from_list(rows)
    )
    before = {n: p.detach().clone() for n, p in trainer.model.named_parameters()}
    trainer.train()

    assert len(reward.calls) == 1
    call = reward.calls[0]
    assert len(call["completions"]) == 4 and "trainer_state" in call["kw"]
    # references are the dataset answers of the prompts TRL actually sampled
    by_prompt = {r["prompt"][0]["content"]: r["answer"] for r in rows}
    assert call["kw"]["answer"] == [by_prompt[p[-1]["content"]] for p in call["prompts"]]
    # independent recomputation: conversational text + truncation from ids (no EOS id 2 at the end)
    verifier = MathVerifier(rewards=scheme)
    expected = []
    for comp, ids, ref in zip(call["completions"], call["ids"], call["kw"]["answer"], strict=True):
        assert comp[0]["role"] == "assistant" and isinstance(comp[0]["content"], str)
        truncated = len(ids) == 0 or ids[-1] != 2
        expected.append(verifier.verify(comp[0]["content"], ref, truncated=truncated).reward)
    assert call["out"] == expected
    # hand-written status rule, independent of MathVerifier: no final EOS id -> truncated; terminated and
    # without any answer marker -> no_answer
    for comp, ids, result in zip(call["completions"], call["ids"], reward.fn.last_results, strict=True):
        text = comp[0]["content"]
        if len(ids) == 0 or ids[-1] != 2:
            assert result.status == "truncated", (text, result)
        elif not any(m in text.lower() for m in ("\\boxed", "\\fbox", "####", "answer")):
            assert result.status == "no_answer", (text, result)
    assert len(set(expected)) > 1, "rewards constant -> zero advantages; pick another seed"
    logs = trainer.state.log_history[0]
    assert logs["rewards/math_reward/mean"] == pytest.approx(sum(expected) / 4, abs=1e-6)
    status_fracs = [logs[f"verifier/math_reward/{s}_frac"] for s in ("truncated", "no_answer")]
    assert status_fracs[0] == pytest.approx(sum(r == -1.0 for r in expected) / 4)
    assert sum(status_fracs) > 0
    changed = any(not torch.equal(before[n], p.detach()) for n, p in trainer.model.named_parameters())
    assert changed, "non-zero advantages must move the policy"


# ----------------------------------------------------------------------------- code sandbox


class FakeBackend:
    """Canned results keyed by stdin. Never executes anything."""

    def __init__(self, guarantees=FULL, outputs=None):
        self._g = guarantees
        self.outputs: dict[str, ExecutionResult] = outputs or {}
        self.requests: list[ExecutionRequest] = []

    @property
    def guarantees(self):
        return self._g

    def run(self, request: ExecutionRequest) -> ExecutionResult:
        self.requests.append(request)
        return self.outputs[request.stdin]


def test_sandbox_disabled_by_default():
    assert SandboxPolicy().enabled is False and SandboxPolicy().network_access is False
    with pytest.raises(SandboxDisabledError, match="disabled"):
        CodeExecutionVerifier()
    with pytest.raises(SandboxDisabledError, match="disabled"):
        CodeExecutionVerifier(backend=FakeBackend())  # a fully isolated backend alone does not enable it
    with pytest.raises(SandboxDisabledError, match="no sandbox backend"):
        CodeExecutionVerifier(ENABLED)


@pytest.mark.parametrize(
    "missing", ["isolated_from_host", "network_disabled", "enforces_time_limit", "enforces_memory_limit"]
)
def test_sandbox_refuses_backend_missing_a_guarantee(missing):
    partial = IsolationGuarantees(**{**FULL.__dict__, missing: False})
    with pytest.raises(SandboxDisabledError, match=missing):
        CodeExecutionVerifier(ENABLED, FakeBackend(partial))


def test_sandbox_refuses_other_unsafe_configurations():
    with pytest.raises(SandboxDisabledError, match="network"):
        CodeExecutionVerifier(SandboxPolicy(enabled=True, network_access=True), FakeBackend())
    with pytest.raises(SandboxDisabledError, match="does not implement"):
        CodeExecutionVerifier(ENABLED, object())  # type: ignore[arg-type]
    with pytest.raises(SandboxDisabledError, match="IsolationGuarantees"):
        CodeExecutionVerifier(ENABLED, FakeBackend(guarantees={"isolated_from_host": True}))
    with pytest.raises(SandboxDisabledError, match="does not guarantee"):  # 1 is not True
        check_sandbox(ENABLED, FakeBackend(IsolationGuarantees(1, 1, 1, 1)))  # type: ignore[arg-type]
    with pytest.raises(SandboxDisabledError, match="language"):
        CodeExecutionVerifier(ENABLED, FakeBackend(), language="bash")
    for bad in ({"time_limit_s": 0}, {"memory_limit_mb": -1}, {"languages": ()}, {"time_limit_s": 1e9}):
        with pytest.raises(ValueError):
            SandboxPolicy(**bad)


CODE = "Here:\n```python\nprint(input())\n```\n"


def test_code_verifier_with_fake_backend():
    outputs = {
        "1": ExecutionResult("ok", "1\n", return_code=0),
        "2": ExecutionResult("ok", "3  \n\n", return_code=0),
        "3": ExecutionResult("timeout"),
        "4": ExecutionResult("sandbox_error", details={"why": "down"}),
    }
    backend = FakeBackend(outputs=outputs)
    v = CodeExecutionVerifier(SandboxPolicy(enabled=True, time_limit_s=2.0, memory_limit_mb=256), backend)
    r = v.verify(CODE, [{"input": "1", "output": "1"}, {"input": "2", "output": "3"}])
    assert (r.status, r.reward, r.details["passed"], r.details["total"]) == ("correct", 1.0, 2, 2)
    assert backend.requests[0] == ExecutionRequest("print(input())\n", "python", "1", 2.0, 256)
    r = v.verify(CODE, {"tests": [{"input": "1", "output": "1"}, {"input": "3", "output": "x"}]})
    assert (r.status, r.details["outcomes"]) == ("incorrect", ["pass", "timeout"])
    r = v.verify(CODE, {"input": "2", "output": "2"})
    assert (r.status, r.details["outcomes"]) == ("incorrect", ["wrong_output"])
    assert v.verify(CODE, {"input": "4", "output": "1"}).status == "error"
    assert v.verify("no code", "1").status == "no_answer"
    assert v.verify("```python\n\n```", "1").status == "format_error"
    assert v.verify(CODE, "1", truncated=True).status == "truncated"
    assert v.verify(CODE, 42).status == "error"
    n_requests = len(backend.requests)
    tiny = CodeExecutionVerifier(SandboxPolicy(enabled=True, max_code_chars=5), backend)
    assert tiny.verify(CODE, "1").details["reason"] == "code_too_long" and len(backend.requests) == n_requests


def test_code_verifier_output_longer_than_cap_fails():
    policy = SandboxPolicy(enabled=True, max_output_chars=5)
    # overlong output whose prefix equals the expected output must not pass
    backend = FakeBackend(outputs={"": ExecutionResult("ok", "12345WRONGEXTRA", return_code=0)})
    r = CodeExecutionVerifier(policy, backend).verify("```python\nprint(1)\n```", "12345")
    assert (r.status, r.details["outcomes"]) == ("incorrect", ["output_too_long"])
    # output of exactly the cap still passes, also with a trailing newline (compared after normalisation)
    backend = FakeBackend(outputs={"": ExecutionResult("ok", "12345\n", return_code=0)})
    assert (
        CodeExecutionVerifier(policy, backend).verify("```python\nprint(1)\n```", "12345").status == "correct"
    )
    # an expected output longer than the cap cannot be checked: bad reference, nothing is executed
    backend = FakeBackend(outputs={})
    r = CodeExecutionVerifier(policy, backend).verify("```python\nprint(1)\n```", "123456")
    assert (r.status, r.details["reason"], backend.requests) == ("error", "bad_reference", [])


def test_sandbox_recheck_propagates_through_trl_adapter():
    backend = FakeBackend(outputs={"": ExecutionResult("ok", "7", return_code=0)})
    v = CodeExecutionVerifier(ENABLED, backend)
    fn = as_trl_reward_func(v, none_on_error=True)
    assert fn(completions=[CODE], answer=["7"]) == [1.0]
    backend._g = IsolationGuarantees()  # backend withdraws its guarantees at runtime
    with pytest.raises(SandboxDisabledError):
        fn(completions=[CODE], answer=["7"])


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("```python\na=1\n```", "a=1\n"),
        ("```py\nx\n```\n```python\ny\n```", "y\n"),
        ("```python\nx\n```\n```bash\nrm -rf /\n```", "x\n"),
        ("```\nbare\n```", "bare\n"),
        # a tagged block wins over a later untagged example-output block, and vice versa
        ("```python\nprint(41+1)\n```\nOutput:\n```\n42\n```", "print(41+1)\n"),
        ("```\n42\n```\n```python\nprint(42)\n```", "print(42)\n"),
        ("```\na\n```\n```\nb\n```", "b\n"),
        ("```\nbare\n```\n```bash\nrm -rf /\n```", "bare\n"),
        ("```python\nunclosed", None),
        ("no fences", None),
        ("```python", None),
    ],
)
def test_extract_code(text, expected):
    assert extract_code(text) == expected


class FakeTransport:
    def __init__(self, status=200, body: Any = b"{}", exc=None):
        self.status, self.body, self.exc, self.calls = status, body, exc, []

    def __call__(self, url, body, headers, timeout):
        self.calls.append((url, json.loads(body), dict(headers), timeout))
        if self.exc is not None:
            raise self.exc
        return self.status, self.body if isinstance(self.body, bytes) else json.dumps(self.body).encode()


def test_sandbox_fusion_backend_requires_explicit_url_and_declaration():
    for url in ("", "127.0.0.1:8080/run_code", "ftp://host/run_code", "file:///etc/passwd", "http://"):
        with pytest.raises(ValueError):
            SandboxFusionBackend(url, FULL)
    with pytest.raises(TypeError):
        SandboxFusionBackend("http://127.0.0.1:8080/run_code")  # type: ignore[call-arg]
    with pytest.raises(TypeError):
        SandboxFusionBackend("http://127.0.0.1:8080/run_code", {"network_disabled": True})  # type: ignore[arg-type]
    undeclared = SandboxFusionBackend("http://127.0.0.1:8080/run_code", IsolationGuarantees())
    with pytest.raises(SandboxDisabledError):
        CodeExecutionVerifier(ENABLED, undeclared)
    assert SandboxFusionBackend("http://h:1/run_code", FULL).transport is urllib_transport


def test_sandbox_fusion_backend_with_fake_transport():
    resp = {
        "status": "Success",
        "run_result": {"status": "Finished", "return_code": 0, "stdout": "7\n", "stderr": ""},
    }
    transport = FakeTransport(body=resp)
    backend = SandboxFusionBackend("http://sandbox.internal:8080/run_code", FULL, transport=transport)
    v = CodeExecutionVerifier(SandboxPolicy(enabled=True, time_limit_s=3.0, memory_limit_mb=512), backend)
    result = v.verify("```python\nprint(7)\n```", {"input": "x", "output": "7"})
    assert result.status == "correct"
    url, payload, headers, _ = transport.calls[0]
    assert url == "http://sandbox.internal:8080/run_code" and headers["Content-Type"] == "application/json"
    assert payload == {
        "code": "print(7)\n",
        "language": "python",
        "stdin": "x",
        "run_timeout": 3.0,
        "compile_timeout": 3.0,
        "memory_limit_MB": 512,
    }
    for t in (
        FakeTransport(status=500),
        FakeTransport(body=b"not json"),
        FakeTransport(exc=ConnectionRefusedError()),
    ):
        r = SandboxFusionBackend("http://h:1/run_code", FULL, transport=t).run(
            ExecutionRequest("c", "python", "", 1, 1)
        )
        assert r.status == "sandbox_error"


@pytest.mark.parametrize(
    ("resp", "status", "stdout", "rc"),
    [
        (
            {"status": "Success", "run_result": {"status": "Finished", "return_code": 0, "stdout": "a"}},
            "ok",
            "a",
            0,
        ),
        ({"run_result": {"return_code": "0", "stdout": "a"}}, "ok", "a", 0),
        (
            {"status": "Failed", "run_result": {"status": "Finished", "return_code": 1, "stderr": "E"}},
            "runtime_error",
            "",
            1,
        ),
        (
            {"status": "Failed", "run_result": {"status": "TimeLimitExceeded", "return_code": None}},
            "timeout",
            "",
            None,
        ),
        (
            {"status": "Failed", "compile_result": {"return_code": 2, "stderr": "syntax"}},
            "compile_error",
            "",
            2,
        ),
        ({"status": "SandboxError", "message": "oops"}, "sandbox_error", "", None),
        ({"status": "Success", "run_result": {"stdout": "a"}}, "sandbox_error", "a", None),
        ({"run_result": {"return_code": True}}, "sandbox_error", "", None),
        ({"run_result": {"return_code": "²"}}, "sandbox_error", "", None),
        ({}, "sandbox_error", "", None),
        ([1, 2], "sandbox_error", "", None),
    ],
)
def test_parse_sandbox_fusion_response(resp, status, stdout, rc):
    r = parse_sandbox_fusion_response(resp)
    assert (r.status, r.stdout, r.return_code) == (status, stdout, rc)
