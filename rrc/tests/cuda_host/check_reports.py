"""Checks the JSON written by the CUDA mains of rrc_selftest and rrc_bench on the fake runtime
(HOST-ONLY, `make cuda-host-test`): the sections only the CUDA build writes (build, device,
model GEMM tally and census, the benchmark's GEMM census) must be present and well-formed, and
neither file may claim a GPU measurement.

    python3 check_reports.py SELFTEST_REPORT.json BENCH.json
"""

from __future__ import annotations

import json
import sys


def fail(msg: str) -> None:
    print(f"check_reports: {msg}")
    sys.exit(1)


def need(d: dict, keys: list[str], where: str) -> None:
    missing = [k for k in keys if k not in d]
    if missing:
        fail(f"{where}: missing keys {missing}")


def main() -> None:
    rep = json.load(open(sys.argv[1]))
    need(rep, ["schema", "tool", "complete", "result", "build", "device", "op_passes", "e2e"], "selftest report")
    if rep["tool"] != "rrc_selftest" or not rep["complete"] or rep["result"] != "PASS":
        fail(f"selftest report: tool {rep['tool']}, complete {rep['complete']}, result {rep['result']}")
    b = rep["build"]
    need(b, ["git_rev", "host_shim", "nvcc_version", "cudart_header_version", "cublas_header_version", "cuda_arch_list"], "build")
    if b["host_shim"] is not True:
        fail("build.host_shim must be true in the HOST-ONLY build")
    dv = rep["device"]
    need(dv, ["name", "compute_capability", "sm_count", "total_mem_bytes", "driver_api_version", "runtime_version",
              "cublaslt_version", "kernel_binary_version", "kernel_ptx_version"], "device")
    if "fake" not in dv["name"]:
        fail(f"device name {dv['name']!r} does not say it is the fake runtime")
    if "no GPU" not in rep["status_note"]:
        fail("status_note must say that no GPU was used")
    e = rep["e2e"]
    need(e, ["result", "items", "model_gemm_tally", "model_gemm_census", "model_gemm_census_dropped_calls"], "e2e")
    t = e["model_gemm_tally"]
    need(t, ["per_dtype_combination", "calls_total", "kernel_fallback_calls", "kernel_fallback_flop_share",
             "h2d_calls", "h2d_staged_chunks", "stream_syncs"], "model_gemm_tally")
    if t["calls_total"] <= 0 or t["h2d_staged_chunks"] <= 0:
        fail(f"model_gemm_tally: {t['calls_total']} GEMM calls, {t['h2d_staged_chunks']} staged chunks")
    calls = sum(sum(v["calls_per_level"]) for v in t["per_dtype_combination"].values())
    if calls != t["calls_total"]:
        fail(f"model_gemm_tally: per-combination calls {calls} != calls_total {t['calls_total']}")
    cen = e["model_gemm_census"]
    if not cen:
        fail("model_gemm_census is empty")
    for g in cen:
        need(g, ["op", "m", "n", "k", "batch", "ld", "calls", "level", "path"], "census entry")
    if sum(g["calls"] for g in cen) + e["model_gemm_census_dropped_calls"] != t["calls_total"]:
        fail("census calls do not add up to the tally")

    bench = json.load(open(sys.argv[2]))
    need(bench, ["schema", "tool", "status", "measured_on", "timing_source", "versions", "device", "config", "results",
                 "gemm"], "bench")
    if bench["measured_on"] is not None or not bench["status"].startswith("smoke"):
        fail(f"bench: measured_on {bench['measured_on']!r}, status {bench['status']!r} (no GPU here)")
    r = bench["results"]
    if r["tokens_per_s"] is not None or r["step_time_s"] is not None:
        fail("bench: timing reported on the fake runtime")
    if bench["versions"].get("host_shim") is not True:
        fail("bench: versions.host_shim must be true")
    g = bench["gemm"]
    need(g, ["per_dtype_combination", "calls_per_step", "flops_per_step_computed", "kernel_fallback_calls_per_step",
             "kernel_fallback_flop_share", "h2d_calls_per_step", "stream_syncs_per_step", "census"], "bench gemm")
    if g["calls_per_step"] <= 0 or not g["census"] or sum(c["calls"] for c in g["census"]) != g["calls_per_step"]:
        fail("bench gemm: census does not add up to the calls of the step")
    pm = r["peak_memory"]
    need(pm, ["allocator_peak_bytes", "backend_total_peak_bytes", "device_used_max_sampled_bytes"], "bench peak_memory")
    if not (0 < pm["allocator_peak_bytes"] <= pm["backend_total_peak_bytes"]):
        fail(f"bench peak_memory: allocator {pm['allocator_peak_bytes']}, total {pm['backend_total_peak_bytes']}")
    print(f"check_reports: OK (selftest report: {len(cen)} census entries, {t['calls_total']} GEMM calls; "
          f"bench: {g['calls_per_step']} GEMM calls, {g['h2d_calls_per_step']} uploads, "
          f"{g['stream_syncs_per_step']} stream synchronizes per step)")


if __name__ == "__main__":
    main()
