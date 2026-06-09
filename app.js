"use strict";

const messages = document.getElementById("messages");
const form = document.getElementById("chatForm");
const input = document.getElementById("questionInput");
const clearBtn = document.getElementById("clearBtn");
const questionList = document.getElementById("questionList");
const supportedCount = document.getElementById("supportedCount");

function splitQuestions(text) {
  const trimmed = text.trim();
  if (!trimmed) return [];
  const parts = trimmed.split(/\n(?=\s*(?:\d+[\.\)]|-)\s+)/)
    .map(x => x.replace(/^\s*(?:\d+[\.\)]|-)\s*/, "").trim())
    .filter(Boolean);
  return parts.length > 1 ? parts : [trimmed];
}

function addMessage(role, content, meta = {}) {
  const wrapper = document.createElement("article");
  wrapper.className = `message ${role}`;

  const bubble = document.createElement("div");
  bubble.className = "bubble";
  bubble.textContent = content;
  wrapper.appendChild(bubble);

  if (meta.backend || meta.trace) {
    const metaRow = document.createElement("div");
    metaRow.className = "meta";
    const status = document.createElement("span");
    status.className = `badge ${meta.abstain ? "warn" : "ok"}`;
    status.textContent = meta.abstain ? "Abstained" : "Verified";
    metaRow.appendChild(status);
    if (meta.backend) {
      const backend = document.createElement("span");
      backend.className = "badge";
      backend.textContent = meta.backend;
      metaRow.appendChild(backend);
    }
    wrapper.appendChild(metaRow);
  }

  if (meta.dsl) {
    const dsl = document.createElement("pre");
    dsl.className = "trace";
    dsl.textContent = `Parsed DSL\n${JSON.stringify(meta.dsl, null, 2)}`;
    wrapper.appendChild(dsl);
  }

  if (meta.trace && meta.trace.length) {
    const trace = document.createElement("pre");
    trace.className = "trace";
    trace.textContent = `Reasoning steps\n${meta.trace.map((line, index) => `${index + 1}. ${line}`).join("\n")}`;
    wrapper.appendChild(trace);
  }

  messages.appendChild(wrapper);
  messages.scrollTop = messages.scrollHeight;
}

function ask(text) {
  const questions = splitQuestions(text);
  questions.forEach(question => {
    addMessage("user", question);
    const result = window.AHRCore.routeQuestion(question);
    addMessage("assistant", result.text, result);
  });
}

function renderQuestions() {
  supportedCount.textContent = `${window.AHRCore.examples.length} examples`;
  window.AHRCore.examples.forEach(sample => {
    const button = document.createElement("button");
    button.type = "button";
    button.className = "question-button";
    button.textContent = sample;
    button.addEventListener("click", () => {
      input.value = sample;
      input.focus();
    });
    questionList.appendChild(button);
  });
}

function drawCanvas() {
  const canvas = document.getElementById("reasoningCanvas");
  const ctx = canvas.getContext("2d");
  const w = canvas.width;
  const h = canvas.height;
  ctx.clearRect(0, 0, w, h);
  ctx.fillStyle = "#f5f8fb";
  ctx.fillRect(0, 0, w, h);

  const nodes = [
    { x: 72, y: 150, label: "Question", c: "#096b72" },
    { x: 210, y: 82, label: "Parser", c: "#4c6f8a" },
    { x: 382, y: 82, label: "Exact\nbackend", c: "#1d7f4f" },
    { x: 382, y: 218, label: "Abstain", c: "#b64d2a" },
    { x: 548, y: 150, label: "Trace", c: "#096b72" },
  ];

  function line(a, b, color) {
    ctx.strokeStyle = color;
    ctx.lineWidth = 3;
    ctx.beginPath();
    ctx.moveTo(a.x, a.y);
    ctx.lineTo(b.x, b.y);
    ctx.stroke();
  }

  line(nodes[0], nodes[1], "#9bb4bf");
  line(nodes[1], nodes[2], "#8fc7a5");
  line(nodes[1], nodes[3], "#d4a184");
  line(nodes[2], nodes[4], "#8fc7a5");
  line(nodes[3], nodes[4], "#d4a184");

  nodes.forEach(node => {
    ctx.fillStyle = node.c;
    ctx.beginPath();
    ctx.arc(node.x, node.y, 40, 0, Math.PI * 2);
    ctx.fill();
    ctx.fillStyle = "white";
    ctx.font = "700 17px system-ui";
    ctx.textAlign = "center";
    ctx.textBaseline = "middle";
    node.label.split("\n").forEach((lineText, i, arr) => {
      ctx.fillText(lineText, node.x, node.y + (i - (arr.length - 1) / 2) * 18);
    });
  });
}

form.addEventListener("submit", event => {
  event.preventDefault();
  const text = input.value.trim();
  if (!text) return;
  input.value = "";
  ask(text);
});

clearBtn.addEventListener("click", () => {
  messages.innerHTML = "";
  addMessage("assistant", "Ask a supported probability/AHR question, or paste an unsupported one to see the abstention behavior.", {
    backend: "demo policy",
    dsl: { family: "demo_policy", behavior: "verified_or_abstain" },
    trace: ["supported: parse to DSL", "verified: exact backend computes answer", "unsupported: abstain instead of guessing"],
  });
});

renderQuestions();
drawCanvas();
addMessage("assistant", "Ask a supported probability/AHR question, or paste an unsupported one to see the abstention behavior.", {
  backend: "demo policy",
  dsl: { family: "demo_policy", behavior: "verified_or_abstain" },
  trace: ["supported: parse to DSL", "verified: exact backend computes answer", "unsupported: abstain instead of guessing"],
});
