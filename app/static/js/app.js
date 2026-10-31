/* Fraud Desk front end. Plain JavaScript, no build step. */
(() => {
  "use strict";

  const $ = (id) => document.getElementById(id);
  const state = { card: null, samples: [], sample: null, batch: null };
  const VERDICT = { high: "Likely fraud", medium: "Needs a second look", low: "Looks legitimate" };

  /* ---------- helpers ---------- */
  function el(tag, props = {}, ...kids) {
    const node = document.createElement(tag);
    for (const [k, v] of Object.entries(props)) {
      if (k === "class") node.className = v;
      else if (k === "text") node.textContent = v;
      else node.setAttribute(k, v);
    }
    kids.forEach((kid) => node.append(kid));
    return node;
  }
  const money = (n) => Number(n).toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 });
  const pct = (n) => (Number(n) * 100).toFixed(1) + "%";
  const num3 = (n) => Number(n).toFixed(3);

  async function api(path, options) {
    let res;
    try {
      res = await fetch(path, options);
    } catch {
      throw new Error("Could not reach the server. Check that it is running and try again.");
    }
    let body = null;
    try { body = await res.json(); } catch { /* not JSON */ }
    if (!res.ok) {
      const d = body && body.detail;
      throw new Error(typeof d === "string" ? d : "The request was not valid. Check the values and try again.");
    }
    return body;
  }

  function showError(node, message) {
    node.textContent = message || "";
    node.hidden = !message;
  }

  /* ---------- the form ---------- */
  const form = $("tx-form");
  const vNames = [];

  function buildVInputs(features) {
    const grid = $("v-grid");
    features.filter((f) => f !== "Amount" && f !== "Hour").forEach((f) => {
      vNames.push(f);
      const input = el("input", { id: "f-" + f, name: f, type: "number", inputmode: "decimal", step: "any" });
      grid.append(el("label", { for: "f-" + f, text: f }, input));
    });
  }
  const allInputs = () => [$("f-Amount"), $("f-Hour"), ...vNames.map((n) => $("f-" + n))];

  function updateCount() {
    const filled = vNames.filter((n) => $("f-" + n).value.trim() !== "").length;
    $("v-count").textContent = `(${filled} of ${vNames.length} filled)`;
  }

  function setValues(features) {
    allInputs().forEach((i) => { i.value = features[i.name] ?? ""; i.removeAttribute("aria-invalid"); });
    updateCount();
  }

  function readValues() {
    const features = {}, missing = [];
    allInputs().forEach((i) => {
      const raw = i.value.trim();
      const n = raw === "" ? NaN : Number(raw);
      if (Number.isFinite(n)) { features[i.name] = n; i.removeAttribute("aria-invalid"); }
      else { missing.push(i.name); i.setAttribute("aria-invalid", "true"); }
    });
    return { features, missing };
  }

  function clearSampleSelection() {
    state.sample = null;
    document.querySelectorAll("#sample-buttons button").forEach((b) => b.setAttribute("aria-pressed", "false"));
  }

  function renderSampleButtons() {
    const row = $("sample-buttons");
    row.replaceChildren();
    state.samples.forEach((s, idx) => {
      const b = el("button", { type: "button", "aria-pressed": "false", text: s.label });
      b.addEventListener("click", () => {
        clearSampleSelection();
        b.setAttribute("aria-pressed", "true");
        state.sample = s;
        setValues(s.features);
        showError($("form-error"), "");
        scoreCurrent();
      });
      row.append(b);
    });
  }

  form.addEventListener("input", (e) => {
    if (e.target.matches("input")) { updateCount(); if (state.sample) clearSampleSelection(); }
  });

  $("clear-btn").addEventListener("click", () => {
    clearSampleSelection();
    setValues({});
    showError($("form-error"), "");
    resetSlip();
  });

  $("json-apply").addEventListener("click", () => {
    const err = $("form-error");
    try {
      const obj = JSON.parse($("json-in").value);
      const features = obj && typeof obj.features === "object" ? obj.features : obj;
      if (!features || typeof features !== "object" || Array.isArray(features)) throw new Error();
      clearSampleSelection();
      setValues(features);
      showError(err, "");
    } catch {
      showError(err, "That is not valid JSON. Paste an object like {\"V1\": -1.36, ..., \"Amount\": 149.62, \"Hour\": 0}.");
    }
  });

  form.addEventListener("submit", (e) => { e.preventDefault(); scoreCurrent(); });

  async function scoreCurrent() {
    const err = $("form-error");
    const { features, missing } = readValues();
    if (missing.length) {
      const shown = missing.slice(0, 6).join(", ") + (missing.length > 6 ? ` and ${missing.length - 6} more` : "");
      showError(err, `Fill in every field before scoring. Missing: ${shown}. A sample button fills them all.`);
      return;
    }
    if (features.Amount < 0) { showError(err, "Amount cannot be negative."); $("f-Amount").setAttribute("aria-invalid", "true"); return; }
    if (!Number.isInteger(features.Hour) || features.Hour < 0 || features.Hour > 23) {
      showError(err, "Hour must be a whole number from 0 to 23."); $("f-Hour").setAttribute("aria-invalid", "true"); return;
    }
    showError(err, "");
    const btn = $("score-btn");
    btn.disabled = true; btn.textContent = "Scoring…";
    try {
      const result = await api("/api/predict", {
        method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ features }),
      });
      renderResult(features, result);
    } catch (e2) {
      showError(err, e2.message);
    } finally {
      btn.disabled = false; btn.textContent = "Score transaction";
    }
  }

  /* ---------- the result slip ---------- */
  function resetSlip() {
    $("slip").dataset.state = "empty";
    $("slip-empty").hidden = false;
    $("slip-body").hidden = true;
  }

  function renderResult(features, r) {
    $("slip").dataset.state = r.risk_level;
    $("slip-empty").hidden = true;
    $("slip-body").hidden = false;
    $("r-amount").textContent = money(features.Amount);
    $("r-hour").textContent = String(features.Hour).padStart(2, "0") + ":00";
    $("r-verdict").textContent = VERDICT[r.risk_level];
    $("r-score").textContent = r.fraud_score.toFixed(2);
    $("r-fill").style.width = Math.max(1, r.fraud_score * 100) + "%";
    $("r-tick").style.left = r.threshold * 100 + "%";
    $("r-tick-label").textContent = "cut-off " + r.threshold.toFixed(2);
    $("r-scale").setAttribute("aria-label",
      `Fraud score ${r.fraud_score.toFixed(2)} on a scale from 0 to 1. Cut-off for flagging is ${r.threshold.toFixed(2)}.`);

    const list = $("r-signals");
    list.replaceChildren();
    if (!r.signals.length) {
      list.append(el("li", {}, el("span", { class: "none", text: "No single feature stands out from a typical legitimate transaction." })));
    } else {
      r.signals.forEach((s) => list.append(el("li", {},
        el("span", { class: "sig-name", text: s.feature }),
        el("span", { class: "sig-detail", text: `${s.value} (typical ${s.typical}, ${Math.abs(s.z)} spreads ${s.z > 0 ? "above" : "below"})` }))));
    }

    const actual = $("r-actual");
    if (state.sample) {
      const fraud = state.sample.actual === "fraud";
      const msg = fraud
        ? (r.is_fraud ? "Actual label in the dataset: fraud. The model caught it." : "Actual label in the dataset: fraud. The model missed this one.")
        : (r.is_fraud ? "Actual label in the dataset: legitimate. This is a false alarm." : "Actual label in the dataset: legitimate. The model agrees.");
      actual.textContent = msg;
      actual.hidden = false;
    } else {
      actual.hidden = true;
    }
  }

  /* ---------- batch scoring ---------- */
  const csvCell = (v) => { const s = String(v); return /[",\n]/.test(s) ? `"${s.replace(/"/g, '""')}"` : s; };
  function download(filename, text) {
    const url = URL.createObjectURL(new Blob([text], { type: "text/csv" }));
    const a = el("a", { href: url, download: filename });
    document.body.append(a); a.click(); a.remove();
    setTimeout(() => URL.revokeObjectURL(url), 1000);
  }

  $("sample-csv").addEventListener("click", () => {
    if (!state.samples.length) return;
    const cols = [...state.card.features, "Class"];
    const lines = [cols.join(",")];
    state.samples.forEach((s) => lines.push([...state.card.features.map((f) => s.features[f]), s.actual === "fraud" ? 1 : 0].join(",")));
    download("sample_transactions.csv", lines.join("\n") + "\n");
  });

  async function scoreFile(file) {
    const err = $("batch-error");
    showError(err, "");
    $("batch-out").hidden = true;
    if (!file) return;
    $("drop-name").textContent = file.name;
    if (file.size > 5 * 1024 * 1024) { showError(err, "That file is larger than 5 MB. Split it and upload the parts."); return; }
    const fd = new FormData();
    fd.append("file", file);
    $("drop-name").textContent = file.name + " (scoring…)";
    try {
      const data = await api("/api/predict/batch", { method: "POST", body: fd });
      state.batch = data;
      renderBatch(data);
      $("drop-name").textContent = file.name;
    } catch (e) {
      $("drop-name").textContent = file.name;
      showError(err, e.message);
    }
  }

  function renderBatch(d) {
    const s = d.summary;
    let text = `${s.flagged.toLocaleString()} of ${s.rows.toLocaleString()} transactions flagged (${s.flagged_pct}%).`;
    if (s.actual_fraud !== undefined) {
      text += ` The file labels ${s.actual_fraud} as fraud; the model caught ${s.caught} and raised ${s.false_alarms} false alarm${s.false_alarms === 1 ? "" : "s"}.`;
    }
    $("batch-summary").textContent = text;
    const hasActual = d.results.length && d.results[0].actual !== undefined;
    $("th-actual").hidden = !hasActual;
    const body = $("batch-rows");
    body.replaceChildren();
    const top = [...d.results].sort((a, b) => b.fraud_score - a.fraud_score).slice(0, 50);
    top.forEach((r) => {
      const cls = r.risk_level === "high" ? "flag" : r.risk_level === "medium" ? "review" : "ok";
      const label = r.risk_level === "high" ? "Flagged" : r.risk_level === "medium" ? "Second look" : "Clear";
      const tr = el("tr", {},
        el("td", { text: r.row }),
        el("td", { class: "num", text: money(r.amount) }),
        el("td", { class: "num", text: r.fraud_score.toFixed(3) }),
        el("td", { class: cls, text: label }));
      if (hasActual) tr.append(el("td", { text: r.actual === 1 ? "Fraud" : "Legit" }));
      body.append(tr);
    });
    $("batch-note").textContent = d.results.length > 50
      ? `Showing the 50 highest scores. Download the file for all ${d.results.length.toLocaleString()} rows.`
      : "Sorted by fraud score, highest first.";
    $("batch-out").hidden = false;
  }

  $("csv-in").addEventListener("change", (e) => scoreFile(e.target.files[0]));
  const drop = $("drop");
  ["dragenter", "dragover"].forEach((ev) => drop.addEventListener(ev, (e) => { e.preventDefault(); drop.classList.add("over"); }));
  ["dragleave", "drop"].forEach((ev) => drop.addEventListener(ev, (e) => { e.preventDefault(); drop.classList.remove("over"); }));
  drop.addEventListener("drop", (e) => { const f = e.dataTransfer.files[0]; if (f) scoreFile(f); });

  $("dl-results").addEventListener("click", () => {
    if (!state.batch) return;
    const rows = state.batch.results;
    const cols = ["row", "amount", "fraud_score", "is_fraud", "risk_level", ...(rows[0].actual !== undefined ? ["actual"] : [])];
    download("fraud_scores.csv", [cols.join(","), ...rows.map((r) => cols.map((c) => csvCell(r[c])).join(","))].join("\n") + "\n");
  });

  /* ---------- model section ---------- */
  function renderModel(card) {
    const m = card.metrics, d = card.dataset;
    const names = Object.keys(m);
    const keys = ["precision", "recall", "f1", "pr_auc", "roc_auc"];
    const deployed = card.deployed_model;
    const others = names.filter((n) => n !== deployed);
    const winsAll = others.every((o) => keys.every((k) => m[deployed][k] >= m[o][k]));
    $("model-lede").textContent =
      `Two approaches were tested on ${d.test_rows.toLocaleString()} transactions the models never saw, ${d.test_fraud_cases} of them fraud: ` +
      `an unsupervised anomaly detector (Isolation Forest) and a supervised ${deployed}. ` +
      (winsAll ? `The ${deployed} scored higher on all five measures, so it powers this page.` : `The ${deployed} powers this page.`);
    $("metrics-caption").textContent = `Held-out test set. Flagging cut-off for ${deployed}: ${card.threshold.toFixed(3)}. Higher is better in every column.`;
    const tbody = document.querySelector("#metrics-table tbody");
    tbody.replaceChildren();
    names.forEach((n) => {
      const tr = el("tr", { class: n === deployed ? "best" : "" }, el("td", { text: n === deployed ? n + " (deployed)" : n }));
      keys.forEach((k) => tr.append(el("td", { class: "num", text: num3(m[n][k]) })));
      tbody.append(tr);
    });
  }

  /* ---------- API example ---------- */
  function renderCurl() {
    const sample = state.samples[0];
    const body = JSON.stringify({ features: sample ? sample.features : {} });
    $("curl").textContent = `curl -X POST ${location.origin}/api/predict \\\n  -H "Content-Type: application/json" \\\n  -d '${body}'`;
  }
  $("copy-curl").addEventListener("click", async (e) => {
    const btn = e.currentTarget;
    try { await navigator.clipboard.writeText($("curl").textContent); btn.textContent = "Copied"; }
    catch { btn.textContent = "Select the text and copy it"; }
    setTimeout(() => (btn.textContent = "Copy the command"), 1800);
  });

  /* ---------- start ---------- */
  async function init() {
    try {
      const [card, samples] = await Promise.all([api("/api/model"), api("/api/samples")]);
      state.card = card; state.samples = samples;
      buildVInputs(card.features);
      updateCount();
      renderSampleButtons();
      renderModel(card);
      renderCurl();
    } catch (e) {
      $("sample-buttons").replaceChildren(el("span", { class: "error", text: e.message }));
    }
  }
  init();
})();
