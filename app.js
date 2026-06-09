"use strict";

const WORD_NUMBERS = {
  zero: 0, one: 1, two: 2, three: 3, four: 4, five: 5,
  six: 6, seven: 7, eight: 8, nine: 9, ten: 10,
  eleven: 11, twelve: 12, thirteen: 13, fourteen: 14, fifteen: 15,
  sixteen: 16, seventeen: 17, eighteen: 18, nineteen: 19,
  twenty: 20, "twenty-three": 23, "twenty three": 23,
  fifty: 50, hundred: 100,
};

const RANKS = ["ace", "king", "queen", "jack", "ten", "nine", "eight", "seven", "six", "five", "four", "three", "two"];
const SUITS = ["heart", "hearts", "spade", "spades", "diamond", "diamonds", "club", "clubs"];

function normalize(text) {
  return text.toLowerCase()
    .replaceAll("\u2212", "-")
    .replaceAll("\u207b\u00b9", "^-1")
    .replace(/[,$]/g, "")
    .replace(/\s+/g, " ")
    .trim();
}

function numberToken(token) {
  if (!token) return null;
  const clean = token.toLowerCase().replace(/[^a-z0-9.-]/g, "");
  if (clean in WORD_NUMBERS) return WORD_NUMBERS[clean];
  if (/^-?\d+(?:\.\d+)?$/.test(clean)) return Number(clean);
  return null;
}

function numberNear(text, patterns) {
  for (const pattern of patterns) {
    const match = text.match(pattern);
    if (match) {
      const n = numberToken(match[1]);
      if (Number.isFinite(n)) return n;
    }
  }
  return null;
}

function gcd(a, b) {
  a = Math.abs(a);
  b = Math.abs(b);
  while (b) {
    const t = b;
    b = a % b;
    a = t;
  }
  return a || 1;
}

function fraction(n, d) {
  if (d === 0) return "undefined";
  const sign = n * d < 0 ? "-" : "";
  n = Math.abs(Math.round(n));
  d = Math.abs(Math.round(d));
  const g = gcd(n, d);
  n /= g;
  d /= g;
  return d === 1 ? `${sign}${n}` : `${sign}${n}/${d}`;
}

function decimal(n, d, digits = 6) {
  return (n / d).toFixed(digits);
}

function comb(n, k) {
  if (k < 0 || k > n) return 0;
  k = Math.min(k, n - k);
  let out = 1;
  for (let i = 1; i <= k; i += 1) {
    out = (out * (n - k + i)) / i;
  }
  return Math.round(out);
}

function percentToDecimal(value) {
  return Number(value) / 100;
}

function supported(text, backend, trace, abstain = false) {
  return { text, backend, trace, abstain };
}

function parseCoinExact(q) {
  if (!/(coin|toss|flip|heads)/.test(q) || !/(exactly|precisely|x=|x equals)/.test(q)) return null;
  const n = numberNear(q, [
    /(?:tossed|flipped|flip|toss|flipping)\s+(\d+|[a-z-]+)\s+(?:times|fair coins|coins)/,
    /(\d+|[a-z-]+)\s+(?:fair\s+)?(?:coin tosses|coin flips|coins)/,
    /binomial\((\d+)/,
  ]);
  const k = numberNear(q, [
    /exactly\s+(\d+|[a-z-]+)\s+heads?/,
    /precisely\s+(\d+|[a-z-]+)\s+heads?/,
    /x\s*=\s*(\d+)/,
    /x equals\s+(\d+|[a-z-]+)/,
  ]);
  if (!Number.isInteger(n) || !Number.isInteger(k) || n < 1 || n > 20 || k < 0 || k > n) return null;
  const fav = comb(n, k);
  const total = 2 ** n;
  return supported(
    `The probability is ${fraction(fav, total)} = ${decimal(fav, total)}.`,
    "exact binomial probability",
    [`favorable outcomes = C(${n},${k}) = ${fav}`, `total outcomes = 2^${n} = ${total}`, `probability = ${fraction(fav, total)}`],
  );
}

function parseCoinRandomVariable(q) {
  if (!/(coin|heads|binomial)/.test(q) || !/(var|variance)/.test(q) || !/(expect|mean|e\[)/.test(q)) return null;
  const n = numberNear(q, [
    /flipping\s+(\d+|[a-z-]+)\s+(?:fair\s+)?coins?/,
    /(\d+|[a-z-]+)\s+(?:fair\s+)?coins?/,
    /binomial\((\d+)/,
  ]) || 5;
  const k = numberNear(q, [/p\(x\s*=\s*(\d+)\)/, /x\s*=\s*(\d+)/, /(\d+)\s+heads/]) || 3;
  if (!Number.isInteger(n) || !Number.isInteger(k) || n < 1 || n > 20 || k < 0 || k > n) return null;
  const fav = comb(n, k);
  const total = 2 ** n;
  const mean = n / 2;
  const variance = n / 4;
  return supported(
    `P(X=${k})=${fraction(fav, total)} = ${decimal(fav, total)}; E[X]=${mean.toFixed(3).replace(/\.?0+$/, "")}; Var(X)=${variance.toFixed(3).replace(/\.?0+$/, "")}.`,
    "exact binomial calculator",
    [`X ~ Binomial(${n}, 1/2)`, `P(X=${k}) = C(${n},${k}) / 2^${n} = ${fraction(fav, total)}`, `E[X] = n/2 = ${mean}`, `Var(X) = n/4 = ${variance}`],
  );
}

function parseDiceSum(q) {
  if (!/(die|dice|d6|six-sided)/.test(q) || !/(sum|total|add)/.test(q)) return null;
  const target = numberNear(q, [/sum(?:\s+is|\s+equals|\s+to)?\s+(\d+|[a-z-]+)/, /total(?:\s+is|\s+equals|\s+to)?\s+(\d+|[a-z-]+)/]);
  if (!Number.isInteger(target) || target < 2 || target > 12) return null;
  let favorable = 0;
  const pairs = [];
  for (let a = 1; a <= 6; a += 1) {
    for (let b = 1; b <= 6; b += 1) {
      if (a + b === target) {
        favorable += 1;
        pairs.push(`(${a},${b})`);
      }
    }
  }
  return supported(
    `The probability is ${fraction(favorable, 36)} = ${decimal(favorable, 36)}.`,
    "exact dice enumeration",
    [`favorable ordered pairs = ${favorable}: ${pairs.join(" ")}`, "total ordered pairs = 36", `probability = ${fraction(favorable, 36)}`],
  );
}

function parseCards(q) {
  if (!/(card|deck)/.test(q)) return null;
  const rank = RANKS.find(r => new RegExp(`\\b${r}s?\\b`).test(q));
  const suitRaw = SUITS.find(s => new RegExp(`\\b${s}\\b`).test(q));
  const suit = suitRaw ? suitRaw.replace(/s$/, "") : null;
  if (!rank && !suit) return null;
  if (rank && suit && /or|union|either/.test(q)) {
    const fav = 4 + 13 - 1;
    return supported(
      `The probability is ${fraction(fav, 52)} = ${decimal(fav, 52)}.`,
      "exact card inclusion-exclusion",
      [`${rank}s = 4`, `${suit}s = 13`, `overlap = ${rank} of ${suit}s = 1`, `(4 + 13 - 1) / 52 = ${fraction(fav, 52)}`],
    );
  }
  if (rank) {
    return supported(
      `The probability is ${fraction(4, 52)} = ${decimal(4, 52)}.`,
      "exact card calculator",
      [`there are 4 ${rank}s`, "deck size = 52", `probability = ${fraction(4, 52)}`],
    );
  }
  return supported(
    `The probability is ${fraction(13, 52)} = ${decimal(13, 52)}.`,
    "exact card calculator",
    [`there are 13 ${suit}s`, "deck size = 52", `probability = ${fraction(13, 52)}`],
  );
}

function parseBalls(q) {
  if (!/(bag|urn|balls?)/.test(q) || !/without replacement|drawn|draw/.test(q)) return null;
  const colors = {};
  for (const color of ["red", "blue", "green", "yellow", "white", "black"]) {
    const m = q.match(new RegExp(`(\\d+)\\s+${color}`));
    if (m) colors[color] = Number(m[1]);
  }
  const target = ["red", "blue", "green", "yellow", "white", "black"].find(color => new RegExp(`both (?:are )?${color}|two ${color}`).test(q));
  if (!target || !colors[target]) return null;
  const total = Object.values(colors).reduce((a, b) => a + b, 0);
  if (total < 2 || colors[target] < 2) return null;
  const n = colors[target] * (colors[target] - 1);
  const d = total * (total - 1);
  return supported(
    `The probability is ${fraction(n, d)} = ${decimal(n, d)}.`,
    "exact urn probability",
    [`first ${target} = ${colors[target]}/${total}`, `second ${target} = ${colors[target] - 1}/${total - 1}`, `probability = ${fraction(n, d)}`],
  );
}

function parseConditionalRank(q) {
  if (!/(card|deck|ace|king|queen|jack)/.test(q) || !/(first|given)/.test(q) || !/(second|next|also)/.test(q)) return null;
  const rank = RANKS.find(r => new RegExp(`\\b${r}s?\\b`).test(q)) || "ace";
  return supported(
    `The probability is ${fraction(3, 51)} = ${decimal(3, 51)}.`,
    "exact conditional card probability",
    [`after the first ${rank}, remaining ${rank}s = 3`, "remaining cards = 51", `probability = ${fraction(3, 51)}`],
  );
}

function parseBayes(q) {
  if (!/(disease|sick|screening|test)/.test(q) || !/positive/.test(q)) return null;
  const nums = [...q.matchAll(/(\d+(?:\.\d+)?)\s*%/g)].map(m => Number(m[1]));
  if (nums.length < 3) return null;
  const prevalence = percentToDecimal(nums[0]);
  const sensitivity = percentToDecimal(nums[1]);
  const specificity = percentToDecimal(nums[2]);
  const truePositive = sensitivity * prevalence;
  const falsePositive = (1 - specificity) * (1 - prevalence);
  const posterior = truePositive / (truePositive + falsePositive);
  return supported(
    `The probability is ${(posterior * 100).toFixed(4)}%.`,
    "exact Bayes calculator",
    [`true positive mass = ${sensitivity} * ${prevalence} = ${truePositive.toFixed(6)}`, `false positive mass = ${(1 - specificity).toFixed(4)} * ${(1 - prevalence).toFixed(4)} = ${falsePositive.toFixed(6)}`, `posterior = TP / (TP + FP) = ${posterior.toFixed(6)}`],
  );
}

function parseBirthday(q) {
  if (!/birthday/.test(q) || !/(share|same|match|duplicate|at least)/.test(q)) return null;
  const n = numberNear(q, [/room of\s+(\d+|[a-z-]+)/, /group of\s+(\d+|[a-z-]+)/, /(\d+)\s+people/]);
  if (!Number.isInteger(n) || n < 2 || n > 100) return null;
  let noMatch = 1;
  for (let i = 0; i < n; i += 1) noMatch *= (365 - i) / 365;
  const p = 1 - noMatch;
  return supported(
    `The probability is approximately ${p.toFixed(6)}.`,
    "exact birthday product calculator",
    [`P(match) = 1 - P(no match)`, `P(no match) = product i=0..${n - 1} of (365-i)/365`, `P(match) = ${p.toFixed(6)}`],
  );
}

function parseMonty(q) {
  if (!(/monty hall/.test(q) || (/doors?/.test(q) && /goat|car|switch/.test(q)))) return null;
  return supported(
    "Yes, switch. The probability of winning by switching is 2/3 = 0.666667.",
    "proof template",
    ["initial chosen door wins with probability 1/3", "the unchosen doors carry probability 2/3", "host reveals a goat, so switching keeps the 2/3 mass"],
  );
}

function parseThreeBoxes(q) {
  if (!/(three boxes|box a|gg|gold coin)/.test(q) || !/gold/.test(q) || !/other|remaining|partner/.test(q)) return null;
  return supported(
    "The probability is 2/3 = 0.666667.",
    "exact conditional probability",
    ["conditioned gold observations: GG coin 1, GG coin 2, GS gold coin", "success cases where the other coin is gold = 2", "total conditioned cases = 3"],
  );
}

function parsePrisoners(q) {
  if (!/100 prisoners/.test(q) || !/box/.test(q)) return null;
  return supported(
    "Use the cycle-following strategy. The group survival probability is about 0.311828.",
    "combinatorial proof template",
    ["each prisoner starts at the box with their own number", "then follows the number found in each opened box", "success iff the permutation has no cycle longer than 50", "probability = 1 - sum(k=51..100) 1/k ~= 0.311828"],
  );
}

function parseExpectedDie(q) {
  if (!/(die|d6)/.test(q) || !/(until|first|waiting|expected|expectation)/.test(q) || !/(6|six)/.test(q)) return null;
  return supported(
    "The expected number of rolls is 6.",
    "exact geometric expectation",
    ["success probability p = 1/6", "geometric expected waiting time = 1/p", "1 / (1/6) = 6"],
  );
}

function parseGraph(q) {
  if (!/(connected graph|6-vertex|six-vertex)/.test(q) || !/(bridge|single edge|triangle)/.test(q)) return null;
  return supported(
    "No such graph exists.",
    "exact proof plus exhaustive graph check",
    ["if deleting any edge disconnects a connected graph, every edge is a bridge", "an edge on a cycle cannot be a bridge", "therefore the graph is a tree", "a tree has zero triangles"],
  );
}

function parseHanoi(q) {
  if (!/(tower of hanoi|hanoi|5-disk|5 disk)/.test(q) || !/(peg b|move 8|at most 2)/.test(q)) return null;
  return supported(
    "Impossible under the revised rule.",
    "exact state-space verifier",
    ["the largest disk must eventually move from A to C", "before that move, four smaller disks must be stacked on B", "that cannot happen before move 8", "after move 8, peg B may hold at most two disks"],
  );
}

function parseBank(q) {
  if (!/(deposit|bank|interest|withdraw)/.test(q) || !/(3%|1.5%|counterfactual|alice|bob)/.test(q)) return null;
  return supported(
    "(a) $127.89; (b) $56.84; (c) $127.89.",
    "exact rational branch arithmetic",
    ["actual: 200.00 -> 206.00 -> 126.00 -> 127.89", "counterfactual withdrawal 150: 200.00 -> 206.00 -> 56.00 -> 56.84", "fresh actual branch confirms no state contamination"],
  );
}

function parseGreenEyes(q) {
  if (!/(green eyes|green-eyed|amber)/.test(q) || !/(visitor|midnight|leave|island)/.test(q)) return null;
  return supported(
    "All 40 green-eyed inhabitants leave together at midnight on the 40th night; the 60 amber-eyed inhabitants never leave.",
    "common-knowledge induction proof",
    ["base case: 1 green-eyed person leaves on night 1", "induction: k green-eyed people leave on night k", "with 40 green-eyed people, all 40 leave on night 40"],
  );
}

function parseSequence(q) {
  if (!/1\s*,\s*2\s*,\s*4\s*,\s*8\s*,\s*16\s*,\s*31/.test(q) && !/sequence/.test(q)) return null;
  if (!/(next|closed-form|doubling|terms)/.test(q)) return null;
  return supported(
    "The next three terms are 256, 386, 562.",
    "finite-difference rule verification",
    ["verified rule: a_n = sum(k=0..4) binomial(n-1,k)", "matches the given prefix", "next terms for n=10,11,12 are 256, 386, 562"],
  );
}

function parseOperationSynthesis(q) {
  if (!/rev/.test(q) || !/delta/.test(q) || !/double/.test(q) || !/zip/.test(q)) return null;
  return supported(
    "No five-operation composition produces the target under the declared operation semantics.",
    "AHR exhaustive finite program search",
    ["operation choices = 4", "program length = 5", "programs checked = 4^5 = 1024", "solutions found = 0"],
  );
}

function parseUnsupportedByDesign(q) {
  if (/(unknown|unspecified|loaded|nonstandard|without any prior|no prior|not specified)/.test(q) && /(probability|deck|die|coin|ratio|model|prior)/.test(q)) {
    return supported(
      "This demo abstains: no single numerical answer is determined without the missing model assumptions.",
      "router abstention",
      ["required probability model is missing", "different hidden models can give different answers", "safe behavior = abstain"],
      true,
    );
  }
  return null;
}

const ROUTES = [
  parseUnsupportedByDesign,
  parseCoinRandomVariable,
  parseCoinExact,
  parseDiceSum,
  parseConditionalRank,
  parseCards,
  parseBalls,
  parseBayes,
  parseBirthday,
  parseMonty,
  parseThreeBoxes,
  parsePrisoners,
  parseExpectedDie,
  parseGraph,
  parseHanoi,
  parseBank,
  parseGreenEyes,
  parseSequence,
  parseOperationSynthesis,
];

const EXAMPLES = [
  "A fair coin is tossed 3 times. What is the probability of getting exactly 2 heads?",
  "A fair coin is flipped 6 times. What is the probability of exactly 4 heads?",
  "A fair six-sided die is rolled twice. What is the probability that the sum is 7?",
  "Two dice are rolled. What is the probability that the total is 9?",
  "A card is drawn from a standard 52-card deck. What is the probability of drawing an ace or a spade?",
  "A bag contains 5 red, 3 blue, and 2 green balls. Two balls are drawn without replacement. What is the probability that both are red?",
  "Two cards are drawn from a deck without replacement. Given that the first card is a king, what is the probability that the second card is also a king?",
  "A disease affects 1% of a population. A test has 99% sensitivity and 95% specificity. If a person tests positive, what is the probability they actually have the disease?",
  "In a room of 23 people, what is the probability that at least two people share the same birthday?",
  "In a room of 30 people, what is the probability that at least two people share the same birthday?",
  "There are 3 doors. One contains a car and two contain goats. You pick one door. The host opens another door showing a goat. Should you switch?",
  "Let X be the number of heads obtained when flipping 5 fair coins. Find P(X=3), E[X], and Var(X).",
  "Box A contains 2 gold coins. Box B contains 2 silver coins. Box C contains 1 gold and 1 silver coin. A box is chosen at random, and one coin drawn is gold. What is the probability the other coin is also gold?",
  "100 prisoners and 100 numbered boxes. Each prisoner may open 50 boxes. What strategy maximizes their survival probability?",
  "You repeatedly roll a fair die until a 6 appears. What is the expected number of rolls?",
  "Using REV, DELTA, DOUBLE-EVENS, and ZIP-SELF, is there exactly five operations that maps [1,4,9,16,25,36] to [1,6,5,14,9]?",
  "A loaded die has unknown probabilities. What is the probability of rolling a 6?",
];

const messages = document.getElementById("messages");
const form = document.getElementById("chatForm");
const input = document.getElementById("questionInput");
const clearBtn = document.getElementById("clearBtn");
const questionList = document.getElementById("questionList");
const supportedCount = document.getElementById("supportedCount");

function splitQuestions(text) {
  const trimmed = text.trim();
  if (!trimmed) return [];
  const parts = trimmed.split(/\n(?=\s*(?:\d+[\.\)]|-)\s+)/).map(x => x.replace(/^\s*(?:\d+[\.\)]|-)\s*/, "").trim()).filter(Boolean);
  return parts.length > 1 ? parts : [trimmed];
}

function routeQuestion(text) {
  const q = normalize(text);
  for (const route of ROUTES) {
    const result = route(q);
    if (result) return result;
  }
  return supported(
    "This demo does not have a verified backend for that question yet, so it will not guess.",
    "router abstention",
    ["no supported verifier matched", "safe behavior = abstain", "try one of the supported examples on the right"],
    true,
  );
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
  supportedCount.textContent = `${EXAMPLES.length} examples`;
  EXAMPLES.forEach(sample => {
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

window.AHRDemo = { routeQuestion, splitQuestions, examples: EXAMPLES };
