"use strict";

const SUPPORTED = [
  {
    id: "coin_two_heads",
    title: "Coin toss",
    sample: "A fair coin is tossed 3 times. What is the probability of getting exactly 2 heads?",
    match: q => /coin|toss|flip/.test(q) && /exactly|precisely/.test(q) && /2|two/.test(q) && /head/.test(q) && /3|three/.test(q),
    answer: () => ({
      text: "The probability is 3/8 = 0.375000.",
      backend: "exact probability calculator",
      trace: ["favorable outcomes = C(3,2) = 3", "total outcomes = 2^3 = 8", "probability = 3/8"],
    }),
  },
  {
    id: "dice_sum_seven",
    title: "Dice sum",
    sample: "A fair six-sided die is rolled twice. What is the probability that the sum is 7?",
    match: q => /die|dice|d6/.test(q) && /sum|total|add/.test(q) && /7|seven/.test(q),
    answer: () => ({
      text: "The probability is 1/6 = 0.166667.",
      backend: "exact probability calculator",
      trace: ["favorable ordered pairs = 6", "total ordered pairs = 36", "probability = 6/36 = 1/6"],
    }),
  },
  {
    id: "king_or_heart",
    title: "Cards",
    sample: "A card is drawn from a standard 52-card deck. What is the probability of drawing a king or a heart?",
    match: q => /card|deck/.test(q) && /king/.test(q) && /heart/.test(q),
    answer: () => ({
      text: "The probability is 4/13 = 0.307692.",
      backend: "exact inclusion-exclusion",
      trace: ["kings = 4", "hearts = 13", "overlap = king of hearts = 1", "(4 + 13 - 1) / 52 = 16/52 = 4/13"],
    }),
  },
  {
    id: "balls_red",
    title: "Balls in a bag",
    sample: "A bag contains 5 red, 3 blue, and 2 green balls. Two balls are drawn without replacement. What is the probability that both are red?",
    match: q => /5 red/.test(q) && /3 blue/.test(q) && /2 green/.test(q) && /red/.test(q) && /without replacement|both/.test(q),
    answer: () => ({
      text: "The probability is 2/9 = 0.222222.",
      backend: "exact probability calculator",
      trace: ["first red = 5/10", "second red given first red = 4/9", "(5/10) * (4/9) = 2/9"],
    }),
  },
  {
    id: "second_ace",
    title: "Conditional cards",
    sample: "Two cards are drawn from a deck without replacement. Given that the first card is an ace, what is the probability that the second card is also an ace?",
    match: q => /ace/.test(q) && /first/.test(q) && /second|next/.test(q),
    answer: () => ({
      text: "The probability is 1/17 = 0.058824.",
      backend: "exact conditional probability",
      trace: ["after first ace, remaining aces = 3", "remaining cards = 51", "probability = 3/51 = 1/17"],
    }),
  },
  {
    id: "bayes_disease",
    title: "Bayes disease test",
    sample: "A disease affects 1% of a population. A test has 99% sensitivity and 95% specificity. If a person tests positive, what is the probability they actually have the disease?",
    match: q => /disease|sick|test/.test(q) && /positive/.test(q) && /99/.test(q) && /95/.test(q),
    answer: () => ({
      text: "The probability is 1/6 = 16.6667%.",
      backend: "exact Bayes calculator",
      trace: ["true positive mass = 0.99 * 0.01 = 0.0099", "false positive mass = 0.05 * 0.99 = 0.0495", "posterior = 0.0099 / (0.0099 + 0.0495) = 1/6"],
    }),
  },
  {
    id: "birthday_23",
    title: "Birthday problem",
    sample: "In a room of 23 people, what is the probability that at least two people share the same birthday?",
    match: q => /birthday/.test(q) && /23|twenty-three/.test(q) && /share|same|match/.test(q),
    answer: () => ({
      text: "The probability is approximately 0.507297.",
      backend: "exact combinatorics calculator",
      trace: ["P(match) = 1 - P(no match)", "P(no match) = product from i=0 to 22 of (365-i)/365", "P(match) ~= 0.507297"],
    }),
  },
  {
    id: "monty_hall",
    title: "Monty Hall",
    sample: "There are 3 doors. One contains a car and two contain goats. You pick one door. The host opens another door showing a goat. Should you switch?",
    match: q => /monty hall/.test(q) || (/3 doors|three doors/.test(q) && /goat|switch|car/.test(q)),
    answer: () => ({
      text: "Yes, switch. The probability of winning by switching is 2/3 = 0.666667.",
      backend: "proof template",
      trace: ["initial chosen door wins with probability 1/3", "the other two doors carry probability 2/3", "host reveals a goat, so switching keeps the 2/3 mass"],
    }),
  },
  {
    id: "coin_random_variable",
    title: "Binomial variable",
    sample: "Let X be the number of heads obtained when flipping 5 fair coins. Find P(X=3), E[X], and Var(X).",
    match: q => /5|five/.test(q) && /coin|heads|binomial/.test(q) && /var|variance/.test(q) && /expect|e\[|mean/.test(q),
    answer: () => ({
      text: "P(X=3)=5/16 = 0.312500; E[X]=2.5; Var(X)=1.25.",
      backend: "exact binomial calculator",
      trace: ["X ~ Binomial(5, 1/2)", "P(X=3) = C(5,3)/2^5 = 10/32 = 5/16", "E[X] = np = 2.5", "Var(X) = np(1-p) = 1.25"],
    }),
  },
  {
    id: "three_boxes",
    title: "Three boxes",
    sample: "Box A contains 2 gold coins. Box B contains 2 silver coins. Box C contains 1 gold and 1 silver coin. A box is chosen at random, and one coin drawn is gold. What is the probability the other coin is also gold?",
    match: q => /box|boxes|gg|gold/.test(q) && /silver|ss/.test(q) && /other|remaining|partner/.test(q),
    answer: () => ({
      text: "The probability is 2/3 = 0.666667.",
      backend: "exact conditional probability",
      trace: ["conditioned gold observations: GG coin 1, GG coin 2, GS gold coin", "success cases where other coin is gold = 2", "total conditioned cases = 3"],
    }),
  },
  {
    id: "prisoners",
    title: "100 prisoners",
    sample: "100 prisoners and 100 numbered boxes. Each prisoner may open 50 boxes. What strategy maximizes their survival probability?",
    match: q => /100 prisoners/.test(q) && /box|boxes/.test(q),
    answer: () => ({
      text: "Use the cycle-following strategy. The group survival probability is about 0.311828.",
      backend: "combinatorial proof template",
      trace: ["each prisoner starts at the box with their own number", "then follows the number found in each opened box", "success iff the permutation has no cycle longer than 50", "probability ~= 0.311828"],
    }),
  },
  {
    id: "die_until_six",
    title: "Expected die rolls",
    sample: "You repeatedly roll a fair die until a 6 appears. What is the expected number of rolls?",
    match: q => /die|d6/.test(q) && /until|first|expected|expectation/.test(q) && /6|six/.test(q),
    answer: () => ({
      text: "The expected number of rolls is 6.",
      backend: "exact geometric expectation",
      trace: ["success probability p = 1/6", "geometric expected waiting time = 1/p", "1 / (1/6) = 6"],
    }),
  },
  {
    id: "operation_synthesis",
    title: "AHR operation synthesis",
    sample: "Using REV, DELTA, DOUBLE-EVENS, and ZIP-SELF, is there exactly five operations that maps [1,4,9,16,25,36] to [1,6,5,14,9]?",
    match: q => /rev/.test(q) && /delta/.test(q) && /double/.test(q) && /zip/.test(q),
    answer: () => ({
      text: "No five-operation composition produces the target under the declared operation semantics.",
      backend: "AHR exhaustive finite program search",
      trace: ["operation choices = 4", "program length = 5", "programs checked = 4^5 = 1024", "solutions found = 0"],
    }),
  },
  {
    id: "underdetermined_marbles",
    title: "Unsupported by design",
    sample: "A bag has red and blue balls in an unknown ratio. After seeing one red, give the exact probability of red next without any prior.",
    match: q => /unknown/.test(q) && /probability|prior|ratio|loaded|unspecified/.test(q),
    answer: () => ({
      text: "This demo abstains: no single numerical probability is determined without a prior or specified model.",
      backend: "router abstention",
      trace: ["required probability model is missing", "different hidden models give different answers", "safe behavior = abstain"],
      abstain: true,
    }),
  },
];

const messages = document.getElementById("messages");
const form = document.getElementById("chatForm");
const input = document.getElementById("questionInput");
const clearBtn = document.getElementById("clearBtn");
const questionList = document.getElementById("questionList");
const supportedCount = document.getElementById("supportedCount");

function normalize(text) {
  return text.toLowerCase()
    .replaceAll("\u2212", "-")
    .replaceAll("\u207b\u00b9", "^-1")
    .replace(/\s+/g, " ")
    .trim();
}

function splitQuestions(text) {
  const trimmed = text.trim();
  if (!trimmed) return [];
  const parts = trimmed.split(/\n(?=\s*(?:\d+[\.\)]|-)\s+)/).map(x => x.replace(/^\s*(?:\d+[\.\)]|-)\s*/, "").trim()).filter(Boolean);
  return parts.length > 1 ? parts : [trimmed];
}

function routeQuestion(text) {
  const q = normalize(text);
  const route = SUPPORTED.find(item => item.match(q));
  if (!route) {
    return {
      id: "unsupported_question",
      title: "Not yet supported",
      text: "This demo does not have a verified backend for that question yet, so it will not guess.",
      backend: "router abstention",
      trace: ["no supported verifier matched", "safe behavior = abstain", "try one of the supported examples on the right"],
      abstain: true,
    };
  }
  return { id: route.id, title: route.title, ...route.answer() };
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

  if (meta.trace && meta.trace.length) {
    const trace = document.createElement("pre");
    trace.className = "trace";
    trace.textContent = meta.trace.map((line, index) => `${index + 1}. ${line}`).join("\n");
    wrapper.appendChild(trace);
  }

  messages.appendChild(wrapper);
  messages.scrollTop = messages.scrollHeight;
}

function ask(text) {
  const questions = splitQuestions(text);
  questions.forEach(question => {
    addMessage("user", question);
    const result = routeQuestion(question);
    addMessage("assistant", result.text, result);
  });
}

function renderQuestions() {
  supportedCount.textContent = `${SUPPORTED.length} examples`;
  SUPPORTED.forEach(item => {
    const button = document.createElement("button");
    button.type = "button";
    button.className = "question-button";
    button.textContent = item.sample;
    button.addEventListener("click", () => {
      input.value = item.sample;
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
    { x: 210, y: 82, label: "Router", c: "#4c6f8a" },
    { x: 382, y: 82, label: "Exact\nbackend", c: "#1d7f4f" },
    { x: 382, y: 218, label: "Abstain", c: "#b64d2a" },
    { x: 548, y: 150, label: "Answer", c: "#096b72" },
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
    trace: ["supported: exact verified families", "unsupported: abstain instead of guessing"],
  });
});

renderQuestions();
drawCanvas();
addMessage("assistant", "Ask a supported probability/AHR question, or paste an unsupported one to see the abstention behavior.", {
  backend: "demo policy",
  trace: ["supported: exact verified families", "unsupported: abstain instead of guessing"],
});
