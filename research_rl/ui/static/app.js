// Channel Edit Lab front end. Plain JavaScript, no external assets, no inline code (CSP).
// Every string that comes from the server (tokens of the vocabulary, notes, override reasons,
// error messages) is written with textContent only, never as HTML.
"use strict";

(function () {
  let token = null;
  const state = { status: null, model: null, history: null, lastVerdict: null };

  // ------------------------------------------------------------------ DOM helpers
  function $(id) { return document.getElementById(id); }

  function el(tag, opts, ...children) {
    const node = document.createElement(tag);
    if (opts) {
      if (opts.cls) node.className = opts.cls;
      if (opts.text !== undefined && opts.text !== null) node.textContent = String(opts.text);
      if (opts.title) node.title = String(opts.title);
      if (opts.attrs) for (const [k, v] of Object.entries(opts.attrs)) node.setAttribute(k, String(v));
      if (opts.on) for (const [k, v] of Object.entries(opts.on)) node.addEventListener(k, v);
    }
    for (const c of children) {
      if (c === null || c === undefined) continue;
      node.appendChild(typeof c === "string" ? document.createTextNode(c) : c);
    }
    return node;
  }

  function svg(tag, attrs) {
    const node = document.createElementNS("http://www.w3.org/2000/svg", tag);
    if (attrs) for (const [k, v] of Object.entries(attrs)) node.setAttribute(k, String(v));
    return node;
  }

  function clear(node) { while (node.firstChild) node.removeChild(node.firstChild); }

  function fmt(v, digits) {
    if (v === null || v === undefined) return "n/a";
    if (typeof v === "number") {
      if (!isFinite(v)) return "n/a";
      const d = digits === undefined ? 4 : digits;
      return Math.abs(v) >= 1e4 || (Math.abs(v) < 1e-3 && v !== 0) ? v.toExponential(2) : v.toFixed(d);
    }
    if (typeof v === "boolean") return v ? "yes" : "no";
    return String(v);
  }

  function short(id) { return typeof id === "string" && id.length > 12 ? id.slice(0, 12) : String(id); }

  function fillTable(table, rows) {
    const body = table.tBodies[0];
    clear(body);
    for (const cells of rows) {
      const tr = el("tr");
      for (const c of cells) {
        if (c instanceof Node) { const td = el("td"); td.appendChild(c); tr.appendChild(td); }
        else if (typeof c === "number") tr.appendChild(el("td", { cls: "num", text: fmt(c) }));
        else tr.appendChild(el("td", { text: c === null || c === undefined ? "" : String(c) }));
      }
      body.appendChild(tr);
    }
  }

  function showError(node, err) {
    clear(node);
    node.appendChild(el("div", { cls: "error", text: err && err.message ? err.message : String(err) }));
  }

  // ------------------------------------------------------------------ API
  class ApiFailure extends Error {
    constructor(status, message) { super(message); this.status = status; }
  }

  async function api(method, path, body) {
    if (!token) throw new ApiFailure(401, "no session token");
    const opts = { method, headers: { "X-Session-Token": token }, cache: "no-store", credentials: "omit" };
    if (body !== undefined) {
      opts.headers["Content-Type"] = "application/json";
      opts.body = JSON.stringify(body);
    }
    const res = await fetch("/api/" + path, opts);
    let data = null;
    try { data = await res.json(); } catch (e) { data = null; }
    if (res.status === 401) { askToken(); }
    if (!res.ok && res.status !== 202) {
      const msg = data && data.error ? data.error.message : res.statusText;
      throw new ApiFailure(res.status, "HTTP " + res.status + ": " + msg);
    }
    return { status: res.status, data };
  }

  function sleep(ms) { return new Promise((r) => setTimeout(r, ms)); }

  // Runs a request that may answer 200 {result} or 202 {job}; polls the job and shows progress.
  async function runJob(method, path, body, jobNode) {
    const first = await api(method, path, body);
    if (first.status === 200) return first.data.result !== undefined ? first.data.result : first.data;
    let job = first.data.job;
    if (jobNode) clear(jobNode);
    const label = el("div");
    const fill = el("div", { cls: "fill" });
    const bar = el("div", { cls: "bar" }, fill);
    if (jobNode) { jobNode.appendChild(label); jobNode.appendChild(bar); }
    for (;;) {
      label.textContent = "job " + job.id + " [" + job.kind + "] " + job.status + " " +
        Math.round((job.progress || 0) * 100) + "% " + (job.message || "");
      fill.style.width = Math.round((job.progress || 0) * 100) + "%";
      if (job.status === "done") return job.result;
      if (job.status === "failed") {
        throw new ApiFailure(job.error ? job.error.status : 500, "job failed: " + (job.error ? job.error.message : "unknown error"));
      }
      await sleep(400);
      job = (await api("GET", "jobs/" + encodeURIComponent(job.id))).data.job;
    }
  }

  // ------------------------------------------------------------------ token
  function readTokenFromFragment() {
    const hash = window.location.hash || "";
    const m = hash.match(/(?:^#|&)token=([A-Za-z0-9_-]{16,256})/);
    if (m) {
      token = m[1];
      // Drop the fragment so the token does not stay in the address bar or history entry.
      history.replaceState(null, "", window.location.pathname + window.location.search);
    }
  }

  function askToken() {
    token = null;
    $("token-box").classList.remove("hidden");
  }

  $("token-form").addEventListener("submit", (ev) => {
    ev.preventDefault();
    const v = $("token-input").value.trim();
    if (/^[A-Za-z0-9_-]{16,256}$/.test(v)) {
      token = v;
      $("token-input").value = "";
      $("token-box").classList.add("hidden");
      refreshAll();
    }
  });

  // ------------------------------------------------------------------ tabs
  for (const btn of document.querySelectorAll(".tab")) {
    btn.addEventListener("click", () => selectTab(btn.dataset.tab));
  }

  function selectTab(name) {
    for (const b of document.querySelectorAll(".tab")) b.classList.toggle("active", b.dataset.tab === name);
    for (const p of document.querySelectorAll(".panel")) p.classList.toggle("active", p.id === "tab-" + name);
    if (name === "versions") loadHistory();
    if (name === "training") loadRuns();
  }

  // ------------------------------------------------------------------ status
  async function loadStatus() {
    const s = (await api("GET", "status")).data;
    state.status = s;
    $("honesty-text").textContent = s.honesty;
    const tiny = $("tiny-note");
    if (s.tiny_note) { tiny.textContent = s.tiny_note; tiny.classList.remove("hidden"); }
    else tiny.classList.add("hidden");
    const worker = s.worker && s.worker.current_job ? "busy (" + s.worker.current_job + ")" : "idle";
    $("status-line").textContent = "model: " + s.source + " | head " + short(s.head) +
      (s.staged ? " | edit staged" : "") + " | evaluations " + s.evaluation_count + " | worker " + worker;
    const rows = Object.values(s.scores || {}).map((sc) => [sc.name, sc.available ? "yes" : "no: " + sc.reason,
      sc.stale ? "STALE (weights changed)" : "current", sc.source]);
    fillTable($("scores-table"), rows);
  }

  async function loadModel() {
    state.model = (await api("GET", "model")).data;
  }

  async function refreshAll() {
    try {
      await loadModel();
      await loadStatus();
    } catch (e) {
      $("status-line").textContent = e.message;
    }
  }

  // ------------------------------------------------------------------ Search tab
  $("stats-form").addEventListener("submit", async (ev) => {
    ev.preventDefault();
    const node = $("stats-job");
    try {
      await runJob("POST", "stats", {
        split: $("stats-split").value,
        n: Number($("stats-n").value),
        threshold: Number($("stats-threshold").value),
        contrast: $("stats-contrast").checked,
      }, node);
      node.appendChild(el("div", { cls: "ok", text: "statistics stored" }));
      await loadStatus();
    } catch (e) { showError(node, e); }
  });

  function rankQuery(score, split) {
    const q = new URLSearchParams();
    q.set("score", score);
    if (score === "wanda" || score === "contrast") q.set("split", split);
    const layer = $("rank-layer").value.trim();
    if (layer !== "") q.set("layer", layer);
    const group = $("rank-group").value.trim();
    if (group !== "") q.set("group", group);
    q.set("bands", $("rank-bands").value || "4");
    q.set("top", $("rank-top").value || "30");
    return q.toString();
  }

  $("rank-form").addEventListener("submit", async (ev) => {
    ev.preventDefault();
    const meta = $("rank-meta");
    try {
      const r = (await api("GET", "channels?" + rankQuery($("rank-score").value, $("rank-split").value))).data;
      meta.textContent = r.score.name + " (" + r.score.source + ")" + (r.score.stale ? " - STALE: weights changed since collection" : "") +
        " | " + r.total + " channels match | groups: " + r.groups.map((g) => g.name + "=" + g.channels).join(", ");
      fillTable($("rank-table"), r.channels.map((c) => [c.rank, c.layer, c.channel, c.score, c.group,
        el("button", { cls: "link", text: "inspect", on: { click: () => inspect(c.layer, c.channel) } })]));
    } catch (e) { showError(meta, e); fillTable($("rank-table"), []); }
  });

  function parseList(text) {
    const t = text.trim();
    if (!t) return null;
    return t.split(",").map((x) => Number(x.trim())).filter((x) => Number.isInteger(x));
  }

  $("search-form").addEventListener("submit", async (ev) => {
    ev.preventDefault();
    const node = $("search-job");
    const out = $("search-result");
    clear(out);
    const controls = [];
    if ($("search-c-random").checked) controls.push("random");
    if ($("search-c-whole").checked) controls.push("whole_layer");
    if ($("search-c-unres").checked) controls.push("unrestricted");
    const body = {
      score: $("search-score").value,
      bands: Number($("search-bands").value),
      rounds: Number($("search-rounds").value),
      burst_updates: Number($("search-burst").value),
      repeats_final: Number($("search-repeats").value),
      items_per_transfer_slice: Number($("search-it").value),
      items_per_retention_slice: Number($("search-ir").value),
      population: Number($("search-pop").value),
      sigma: Number($("search-sigma").value),
      lr: Number($("search-lr").value),
      fitness_items: Number($("search-fit").value),
      seed: Number($("search-seed").value),
      controls,
    };
    if (body.score === "wanda" || body.score === "contrast") body.split = $("search-split").value;
    const layers = parseList($("search-layers").value);
    if (layers) body.layers = layers;
    try {
      const res = await runJob("POST", "search", body, node);
      renderSearch(res, out);
    } catch (e) { showError(out, e); }
  });

  function finalRows(list) {
    return list.map((f) => [f.candidate || f.name, f.kind || "candidate", f.channels,
      f.mean_gain, f.mean_worst_drop, f.mean_objective,
      f.winner_overlap === undefined ? "" : String(f.winner_overlap), f.same_as_winner ? "yes (vacuous)" : ""]);
  }

  function renderSearch(res, out) {
    const rep = res.report;
    out.appendChild(el("p", { cls: "meta", text: res.honesty }));
    out.appendChild(el("p", { cls: "meta", text: rep.statement }));
    out.appendChild(el("p", {}, el("strong", { text: "Winner: " }), rep.winner + " " + JSON.stringify(rep.winner_selection)));
    const head = ["subset", "kind", "channels", "mean transfer gain", "mean worst retention drop", "mean objective", "overlap with winner", "same as winner"];
    const table = el("table", { cls: "data" }, el("thead", {}, el("tr", {}, ...head.map((h) => el("th", { text: h })))), el("tbody"));
    fillTable(table, finalRows(rep.final).concat(finalRows(Object.values(rep.controls))));
    out.appendChild(table);
    const totals = el("p", { cls: "meta", text: "totals: " + JSON.stringify(rep.totals) });
    out.appendChild(totals);
    out.appendChild(el("button", { cls: "secondary", text: "Stage an edit on the winner's channels", on: { click: () => {
      $("edit-selection").value = selectionToText(rep.winner_selection);
      selectTab("edit");
    } } }));
  }

  // ------------------------------------------------------------------ Inspect tab
  $("inspect-form").addEventListener("submit", (ev) => {
    ev.preventDefault();
    inspect(Number($("inspect-layer").value), Number($("inspect-channel").value));
  });

  $("inspect-to-edit").addEventListener("click", () => {
    $("edit-selection").value = $("inspect-layer").value + ": " + $("inspect-channel").value;
    selectTab("edit");
  });

  function histogram(h) {
    const wrap = el("div", { cls: "hist" });
    const s = svg("svg", { viewBox: "0 0 200 60", preserveAspectRatio: "none" });
    const max = Math.max(1, ...h.counts);
    const w = 200 / h.counts.length;
    h.counts.forEach((c, i) => {
      const height = (c / max) * 50;
      const r = svg("rect", { x: i * w + 0.5, y: 52 - height, width: Math.max(0.5, w - 1), height: Math.max(0, height) });
      if (i === h.channel_bin) r.setAttribute("class", "mark");
      s.appendChild(r);
    });
    const lo = svg("text", { x: 0, y: 60 }); lo.textContent = fmt(h.edges[0], 3);
    const hi = svg("text", { x: 200, y: 60, "text-anchor": "end" }); hi.textContent = fmt(h.edges[h.edges.length - 1], 3);
    s.appendChild(lo); s.appendChild(hi);
    wrap.appendChild(s);
    return wrap;
  }

  async function inspect(layer, channel) {
    selectTab("inspect");
    $("inspect-layer").value = layer;
    $("inspect-channel").value = channel;
    const meta = $("inspect-meta");
    meta.textContent = "loading...";
    try {
      const d = await runJob("GET", "channel/" + layer + "/" + channel + "?k=" + Number($("inspect-k").value || 10), undefined, meta);
      clear(meta);
      meta.textContent = d.module_path + " channel " + d.channel + " of " + d.intermediate_size +
        " | weights fingerprint " + short(d.fingerprint) + " | computed by the model loaded on this machine";
      $("proj-method").textContent = d.projection.method + " (|column| = " + fmt(d.projection.column_norm) + ")";
      const tokRow = (t) => [JSON.stringify(t.text), t.id, t.logit];
      fillTable($("proj-top"), d.projection.top.map(tokRow));
      fillTable($("proj-bottom"), d.projection.bottom.map(tokRow));
      $("act-note").textContent = d.activation.note;
      const box = $("act-splits");
      clear(box);
      if (!d.activation.splits.length) box.appendChild(el("p", { cls: "hint", text: "No statistics stored yet (Search tab, step 1)." }));
      for (const s of d.activation.splits) {
        const dl = el("dl", { cls: "kv" },
          el("dt", { text: "mean |h|" }), el("dd", { text: fmt(s.mean_abs) }),
          el("dt", { text: "RMS" }), el("dd", { text: fmt(s.rms) }),
          el("dt", { text: "active rate (|h| > " + s.threshold + ")" }), el("dd", { text: fmt(s.active_rate) }),
          el("dt", { text: "percentile in layer" }), el("dd", { text: fmt(s.percentile_in_layer, 1) }),
          el("dt", { text: "prompts / tokens" }), el("dd", { text: s.n_prompts + " / " + s.tokens }));
        box.appendChild(el("h4", { text: s.split + (s.stale ? " (STALE: weights changed)" : "") }));
        box.appendChild(dl);
        box.appendChild(el("div", { cls: "hint", text: "layer distribution of mean |h| (this channel highlighted)" }));
        box.appendChild(histogram(s.layer_mean_abs_histogram));
      }
      fillTable($("chan-versions"), d.versions.map((v) => [v.version === "live" ? "live weights" : short(v.version) + (v.head ? " (head)" : ""),
        v.edits.join(", "), v.gate_norm, v.up_norm, v.down_norm, v.change_from_parent, v.distance_to_live]));
      fillTable($("chan-scores"), Object.entries(d.scores).map(([k, v]) => [k, v.score, v.rank_in_layer, v.stale ? "stale" : ""]));
    } catch (e) { showError(meta, e); }
  }

  // ------------------------------------------------------------------ Edit tab
  function parseSelection(text) {
    const out = {};
    for (const part of text.split(";")) {
      const p = part.trim();
      if (!p) continue;
      const [layer, chans] = p.split(":");
      if (chans === undefined) throw new Error("selection format: 'layer: c1,c2; layer: c3'");
      const l = layer.trim();
      if (!/^\d+$/.test(l)) throw new Error("layer must be an integer: " + l);
      out[l] = chans.split(",").map((c) => c.trim()).filter((c) => c !== "").map((c) => {
        if (!/^\d+$/.test(c)) throw new Error("channel must be an integer: " + c);
        return Number(c);
      });
    }
    return out;
  }

  function selectionToText(sel) {
    return Object.entries(sel).map(([l, cs]) => l + ": " + cs.join(",")).join("; ");
  }

  function buildEdit() {
    const op = $("edit-op").value;
    const params = {};
    if (op === "scale") { params.factor = Number($("edit-factor").value); params.up = $("edit-up").checked; }
    if (op === "revert") params.version = $("edit-version").value.trim();
    if (op === "interpolate") { params.version = $("edit-version").value.trim(); params.lam = Number($("edit-lam").value); }
    if (op === "apply_delta") { params.delta = $("edit-delta").value.trim(); params.factor = Number($("edit-factor").value); }
    return { op, selection: parseSelection($("edit-selection").value), params, note: $("edit-note").value };
  }

  function syncOpParams() {
    const op = $("edit-op").value;
    for (const n of document.querySelectorAll(".op-param")) {
      n.classList.toggle("hidden", !n.dataset.op.split(" ").includes(op));
    }
    try { $("edit-preview").textContent = JSON.stringify(buildEdit(), null, 2); }
    catch (e) { $("edit-preview").textContent = e.message; }
  }
  for (const id of ["edit-op", "edit-selection", "edit-factor", "edit-up", "edit-version", "edit-lam", "edit-delta", "edit-note"]) {
    $(id).addEventListener("input", syncOpParams);
    $(id).addEventListener("change", syncOpParams);
  }

  $("edit-form").addEventListener("submit", async (ev) => {
    ev.preventDefault();
    const node = $("stage-result");
    try {
      const r = await runJob("POST", "stage", buildEdit(), node);
      clear(node);
      node.appendChild(el("div", { cls: "ok", text: "staged on parent " + short(r.parent) + ": " + JSON.stringify(r.selection) }));
      clear($("verdict"));
      await loadStatus();
    } catch (e) { showError(node, e); }
  });

  function testRow(t) {
    return [t.slice, t.n, t.point, t.bound, t.margin === null ? "" : fmt(t.margin), t.p_value, t.alpha, t.method,
      el("span", { cls: t.passed ? "ok" : "bad", text: t.passed ? "pass" : "fail" })];
  }

  function renderVerdict(res) {
    const box = $("verdict");
    clear(box);
    const v = res.verdict;
    const wrap = el("div", { cls: "verdict-box " + (res.passed ? "pass" : "fail") });
    wrap.appendChild(el("p", {}, el("strong", { text: res.passed ? "Gate passed" : "Gate did not pass" }),
      " (evaluation " + res.evaluation_index + "; transfer " + (v.transfer_ok ? "confirmed" : "not confirmed") +
      ", retention " + (v.retention_ok ? "held" : "not shown to hold") + (v.alarm ? ", ALARM" : "") + ")"));
    const head = ["slice", "n", "point", "bound", "margin", "p", "alpha", "method", "result"];
    const mk = () => el("table", { cls: "data" }, el("thead", {}, el("tr", {}, ...head.map((h) => el("th", { text: h })))), el("tbody"));
    wrap.appendChild(el("h4", { text: "Transfer (pooled; point = mean gain, bound = one-sided lower bound)" }));
    const tt = mk(); fillTable(tt, [testRow(v.transfer)]); wrap.appendChild(tt);
    wrap.appendChild(el("h4", { text: "Retention per slice (point = mean drop, bound = one-sided upper bound; pass iff bound < margin)" }));
    const rt = mk(); fillTable(rt, v.retention.map(testRow)); wrap.appendChild(rt);
    const r = res.reward;
    const d = res.divergence;
    wrap.appendChild(el("h4", { text: "Reward beside capability" }));
    if (!r || !d) {
      wrap.appendChild(el("p", { cls: "error", text: res.reward_error || "the training reward is not available for this evaluation" }));
      wrap.appendChild(el("p", { cls: "hint", text: res.honesty }));
      box.appendChild(wrap);
      $("commit-override").placeholder = res.passed ? "optional" : "required: why commit without a passing verdict?";
      return;
    }
    wrap.appendChild(el("dl", { cls: "kv" },
      el("dt", { text: "training reward V1 (" + r.verifier_v1 + ", " + r.items + " " + r.slice + " items)" }),
      el("dd", { text: fmt(r.incumbent_v1) + " -> " + fmt(r.candidate_v1) + " (change " + fmt(r.change_v1) + ")" }),
      el("dt", { text: "strict V2 on the same items" }),
      el("dd", { text: fmt(r.incumbent_v2) + " -> " + fmt(r.candidate_v2) + " (change " + fmt(r.change_v2) + ")" }),
      el("dt", { text: "held-out transfer gain (point / lower bound)" }),
      el("dd", { text: fmt(d.transfer_gain_point) + " / " + fmt(d.transfer_lower_bound) }),
      el("dt", { text: "divergence d" }), el("dd", { text: fmt(d.d) + " - " + d.flag }),
    ));
    wrap.appendChild(el("p", { cls: "hint", text: d.definition + ". " + res.honesty }));
    box.appendChild(wrap);
    $("commit-override").placeholder = res.passed ? "optional" : "required: why commit without a passing verdict?";
  }

  $("btn-evaluate").addEventListener("click", async () => {
    const node = $("eval-job");
    try {
      const res = await runJob("POST", "evaluate", {}, node);
      state.lastVerdict = res;
      renderVerdict(res);
      await loadStatus();
    } catch (e) { showError(node, e); }
  });

  $("btn-discard").addEventListener("click", async () => {
    const node = $("commit-result");
    try {
      await runJob("POST", "discard", {}, node);
      clear(node); node.appendChild(el("div", { cls: "ok", text: "staged edit discarded; weights back to the head" }));
      clear($("verdict"));
      await loadStatus();
    } catch (e) { showError(node, e); }
  });

  $("commit-form").addEventListener("submit", async (ev) => {
    ev.preventDefault();
    const node = $("commit-result");
    const override = $("commit-override").value.trim();
    const body = { note: $("commit-note").value };
    if (override) body.override = override;
    try {
      const r = await runJob("POST", "commit", body, node);
      clear(node);
      node.appendChild(el("div", { cls: "ok", text: "committed version " + short(r.version.id) + (r.version.override ? " (override recorded)" : " (gate passed)") }));
      $("commit-override").value = "";
      clear($("verdict"));
      await loadStatus();
    } catch (e) { showError(node, e); }
  });

  // ------------------------------------------------------------------ Versions tab
  async function loadHistory() {
    const meta = $("versions-meta");
    try {
      const h = (await api("GET", "history")).data;
      state.history = h;
      meta.textContent = "head " + short(h.head) + " | " + h.versions.length + " versions | store evaluations " + h.evaluation_count +
        (h.staged ? " | an edit is staged (commit or discard before restoring)" : "");
      const lineage = new Set(h.lineage);
      fillTable($("versions-table"), h.versions.map((v) => [
        short(v.id) + (v.id === h.head ? " (head)" : lineage.has(v.id) ? " (lineage)" : ""),
        v.parent ? short(v.parent) : "root",
        v.edits.map((e) => e.op + " " + JSON.stringify(e.selection) + (Object.keys(e.params).length ? " " + JSON.stringify(e.params) : "")).join("; "),
        v.parent === null ? "" : v.passed ? "passed" : "not passed",
        v.override || "",
        v.note,
        v.created,
        el("button", { cls: "link", text: "restore", on: { click: () => restore(v.id) } }),
      ]));
    } catch (e) { showError(meta, e); }
  }

  async function restore(id) {
    const meta = $("versions-meta");
    try {
      await runJob("POST", "restore", { version: id }, meta);
      await loadHistory();
      await loadStatus();
    } catch (e) { showError(meta, e); }
  }

  // ------------------------------------------------------------------ Training tab
  async function loadRuns() {
    const sel = $("runs-select");
    try {
      const r = (await api("GET", "runs")).data;
      clear(sel);
      for (const run of r.runs) sel.appendChild(el("option", { text: run.name, attrs: { value: run.name } }));
      $("runs-meta").textContent = r.runs.length ? r.runs.length + " run(s)" : (r.note || "no decision logs found");
    } catch (e) { showError($("runs-meta"), e); }
  }

  $("runs-refresh").addEventListener("click", loadRuns);

  function chart(records) {
    const box = $("runs-chart");
    clear(box);
    const pts = records.filter((r) => r.decision_index !== null);
    if (!pts.length) return;
    const W = 600, H = 200, P = 30;
    const s = svg("svg", { viewBox: "0 0 " + W + " " + H, preserveAspectRatio: "none" });
    const xs = pts.map((r) => r.decision_index);
    const series = [
      { key: "reward_change_v1", cls: "s1", label: "training reward change (V1)" },
      { key: "accepted_transfer_gain", cls: "s2", label: "accepted transfer gain (gate evidence)" },
    ];
    const vals = pts.flatMap((r) => series.map((sr) => r[sr.key]).filter((v) => typeof v === "number"));
    const lo = Math.min(0, ...vals), hi = Math.max(1e-9, ...vals);
    const xmin = Math.min(...xs), xmax = Math.max(xmin + 1, ...xs);
    const X = (x) => P + ((x - xmin) / (xmax - xmin)) * (W - 2 * P);
    const Y = (y) => H - P - ((y - lo) / (hi - lo || 1)) * (H - 2 * P);
    s.appendChild(svg("line", { x1: P, y1: Y(0), x2: W - P, y2: Y(0), class: "axis" }));
    series.forEach((sr, i) => {
      const d = pts.filter((r) => typeof r[sr.key] === "number").map((r, j) => (j ? "L" : "M") + X(r.decision_index) + " " + Y(r[sr.key])).join(" ");
      if (d) s.appendChild(svg("path", { d, class: sr.cls }));
      const t = svg("text", { x: P + i * 260, y: 14 }); t.textContent = sr.label; s.appendChild(t);
    });
    const tl = svg("text", { x: 2, y: Y(hi) + 4 }); tl.textContent = fmt(hi, 2); s.appendChild(tl);
    const tb = svg("text", { x: 2, y: Y(lo) }); tb.textContent = fmt(lo, 2); s.appendChild(tb);
    box.appendChild(s);
  }

  $("runs-form").addEventListener("submit", async (ev) => {
    ev.preventDefault();
    const name = $("runs-select").value;
    if (!name) return;
    const meta = $("runs-meta");
    try {
      const r = (await api("GET", "runs/" + encodeURIComponent(name) + "/decisions")).data;
      meta.textContent = r.name + ": " + r.total_records + " decisions" + (r.skipped_lines ? ", " + r.skipped_lines + " unreadable lines" : "") + ". " + r.note;
      chart(r.records);
      fillTable($("runs-table"), r.records.map((x) => {
        const fv = x.first_verdict;
        const t = fv && fv.transfer ? fv.transfer : {};
        let worst = "";
        if (fv && fv.retention.length) {
          const w = fv.retention.reduce((a, b) => ((b.bound ?? -1) - (b.margin ?? 0) > (a.bound ?? -1) - (a.margin ?? 0) ? b : a));
          worst = w.slice + ": " + fmt(w.bound) + " / " + fmt(w.margin) + (w.passed ? " pass" : " fail");
        }
        return [x.decision_index, x.update_index, x.decision, x.reason, x.reward_v1, x.reward_change_v1, t.point, t.bound, worst, x.alarm ? "ALARM" : ""];
      }));
    } catch (e) { showError(meta, e); }
  });

  // ------------------------------------------------------------------ start
  readTokenFromFragment();
  syncOpParams();
  if (!token) askToken(); else refreshAll();
  setInterval(() => { if (token) loadStatus().catch(() => {}); }, 5000);
})();
