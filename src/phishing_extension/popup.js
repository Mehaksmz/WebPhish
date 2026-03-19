const BACKEND_URL = "http://127.0.0.1:8000";

const urlEl = document.getElementById("url");
const modelEl = document.getElementById("model");
const scanBtn = document.getElementById("scan");
const statusEl = document.getElementById("status");
const resultEl = document.getElementById("result");

const falseAlarmBtn = document.getElementById("reportFalseAlarm");
const missedPhishingBtn = document.getElementById("reportMissedPhishing");
const feedbackStatusEl = document.getElementById("feedbackStatus");

let last = null;

function setStatus(el, kind, text) {
  el.classList.remove("ok", "warn", "err");
  if (!text) {
    el.textContent = "";
    return;
  }
  if (kind) el.classList.add(kind);
  el.textContent = text;
}

function setFeedbackEnabled(enabled) {
  falseAlarmBtn.disabled = !enabled;
  missedPhishingBtn.disabled = !enabled;
}

function renderResult(data) {
  resultEl.classList.remove("hidden");
  const pred = data?.prediction ?? "—";
  const conf = data?.confidence ?? "—";
  const modelUsed = data?.model_used ?? "—";

  resultEl.innerHTML = `
    <div class="label">Results</div>
    <div class="kv"><span class="k">Prediction</span><span class="v ${pred === "Phishing" ? "pill-red" : "pill-green"}">${pred}</span></div>
    <div class="kv"><span class="k">Confidence</span><span class="v mono">${conf}</span></div>
    <div class="kv"><span class="k">Model Used</span><span class="v mono">${modelUsed}</span></div>
  `;
}

chrome.tabs.query({ active: true, currentWindow: true }, (tabs) => {
  const tabUrl = tabs?.[0]?.url || "";
  urlEl.textContent = tabUrl;
});

scanBtn.addEventListener("click", () => {
  setStatus(statusEl, null, "");
  setStatus(feedbackStatusEl, null, "");
  setFeedbackEnabled(false);

  chrome.tabs.query({ active: true, currentWindow: true }, (tabs) => {
    const url = tabs?.[0]?.url;
    const model = modelEl.value;

    if (!url || !url.startsWith("http")) {
      setStatus(statusEl, "err", "Open a valid http(s) page first.");
      return;
    }

    urlEl.textContent = url;
    setStatus(statusEl, "warn", "Analyzing...");

    fetch(`${BACKEND_URL}/predict`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ url, model_name: model }),
    })
      .then(async (res) => {
        const data = await res.json().catch(() => ({}));
        if (!res.ok) throw new Error(data?.detail || `HTTP ${res.status}`);
        return data;
      })
      .then((data) => {
        last = data;
        renderResult(data);
        setStatus(statusEl, "ok", "Done.");
        setFeedbackEnabled(true);
      })
      .catch((e) => {
        setStatus(statusEl, "err", `Error: ${e.message || e}`);
      });
  });
});

falseAlarmBtn.addEventListener("click", () => {
  if (!last) return;
  setStatus(feedbackStatusEl, null, "");
  falseAlarmBtn.disabled = true;

  fetch(`${BACKEND_URL}/report_false_alarm`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ url: last.url, model_name: last.model_used }),
  })
    .then((res) => {
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      setStatus(feedbackStatusEl, "ok", "Feedback submitted. Thank you!");
    })
    .catch((e) => {
      falseAlarmBtn.disabled = false;
      setStatus(feedbackStatusEl, "err", `Failed to submit feedback: ${e.message || e}`);
    });
});

missedPhishingBtn.addEventListener("click", () => {
  if (!last) return;
  setStatus(feedbackStatusEl, null, "");
  missedPhishingBtn.disabled = true;

  fetch(`${BACKEND_URL}/report_missed_phishing`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ url: last.url, model_name: last.model_used }),
  })
    .then((res) => {
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      setStatus(feedbackStatusEl, "ok", "Feedback submitted. Thank you!");
    })
    .catch((e) => {
      missedPhishingBtn.disabled = false;
      setStatus(feedbackStatusEl, "err", `Failed to submit feedback: ${e.message || e}`);
    });
});