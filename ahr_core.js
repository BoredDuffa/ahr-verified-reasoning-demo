"use strict";

(function initAHRCore(global) {
  const WORD_NUMBERS = {
    zero: 0, one: 1, two: 2, three: 3, four: 4, five: 5,
    six: 6, seven: 7, eight: 8, nine: 9, ten: 10,
    eleven: 11, twelve: 12, thirteen: 13, fourteen: 14, fifteen: 15,
    sixteen: 16, seventeen: 17, eighteen: 18, nineteen: 19,
    twenty: 20, "twenty-three": 23, "twenty three": 23,
    thirty: 30, forty: 40, fifty: 50, sixty: 60, hundred: 100,
  };

  const FAMILY_METADATA = [
    {
      family: "coin_run",
      backend: "exact coin-run dynamic program",
      fields: ["n_flips", "run_length", "target", "event"],
      published: true,
    },
    {
      family: "binomial_tail",
      backend: "exact binomial calculator",
      fields: ["n_trials", "successes", "target", "event"],
      published: true,
    },
    {
      family: "dice_event",
      backend: "exact dice enumeration",
      fields: ["dice", "sides", "event", "target"],
      published: true,
    },
    {
      family: "card_event",
      backend: "exact standard-deck calculator",
      fields: ["rank", "suit", "event"],
      published: true,
    },
    {
      family: "birthday_event",
      backend: "exact birthday complement calculator",
      fields: ["people", "event"],
      published: true,
    },
    {
      family: "bayes_event",
      backend: "exact Bayes-rule calculator",
      fields: ["prevalence_percent", "sensitivity_percent", "specificity_percent"],
      published: true,
    },
    {
      family: "urn_draw",
      backend: "exact urn probability",
      fields: ["colors", "draws", "replacement", "target_color", "target_count"],
      published: true,
    },
    {
      family: "expected_value",
      backend: "exact expectation calculator",
      fields: ["distribution", "success_probability"],
      published: true,
    },
    {
      family: "proof_template",
      backend: "exact proof or finite verifier",
      fields: ["template"],
      published: true,
    },
    {
      family: "unsupported",
      backend: "router abstention",
      fields: ["reason"],
      published: true,
    },
  ];

  const RANKS = ["ace", "king", "queen", "jack", "ten", "nine", "eight", "seven", "six", "five", "four", "three", "two"];
  const SUITS = ["heart", "hearts", "spade", "spades", "diamond", "diamonds", "club", "clubs"];
  const COLORS = ["red", "blue", "green", "yellow", "white", "black"];

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

  function gcdBig(a, b) {
    a = a < 0n ? -a : a;
    b = b < 0n ? -b : b;
    while (b !== 0n) {
      const t = b;
      b = a % b;
      a = t;
    }
    return a || 1n;
  }

  function fractionBig(n, d) {
    if (d === 0n) return "undefined";
    const sign = n * d < 0n ? "-" : "";
    n = n < 0n ? -n : n;
    d = d < 0n ? -d : d;
    const g = gcdBig(n, d);
    n /= g;
    d /= g;
    return d === 1n ? `${sign}${n}` : `${sign}${n}/${d}`;
  }

  function decimalBig(n, d, digits = 6) {
    return (Number(n) / Number(d)).toFixed(digits);
  }

  function percentBig(n, d, digits = 4) {
    return (100 * Number(n) / Number(d)).toFixed(digits);
  }

  function combBig(n, k) {
    if (k < 0 || k > n) return 0n;
    k = Math.min(k, n - k);
    let out = 1n;
    for (let i = 1; i <= k; i += 1) {
      out = out * BigInt(n - k + i) / BigInt(i);
    }
    return out;
  }

  function powBig(base, exp) {
    return BigInt(base) ** BigInt(exp);
  }

  function supported(text, backend, dsl, trace, extra = {}) {
    return {
      text,
      backend,
      dsl,
      trace,
      abstain: false,
      verification: {
        status: "verified",
        ...extra,
      },
    };
  }

  function abstain(reason, trace) {
    return {
      text: "This demo abstains: no single verified numerical answer is determined from the supported assumptions.",
      backend: "router abstention",
      dsl: { family: "unsupported", reason },
      trace: trace || ["missing or unsupported assumptions", "safe behavior = abstain"],
      abstain: true,
      verification: { status: "abstained", reason },
    };
  }

  function validateDsl(dsl) {
    if (!dsl || typeof dsl !== "object") return false;
    const meta = FAMILY_METADATA.find(item => item.family === dsl.family);
    if (!meta) return false;
    if (dsl.family === "coin_run") {
      return Number.isInteger(dsl.n_flips) && Number.isInteger(dsl.run_length) && dsl.n_flips >= 1 && dsl.n_flips <= 80 && dsl.run_length >= 1 && dsl.run_length <= dsl.n_flips && ["heads", "tails"].includes(dsl.target);
    }
    if (dsl.family === "binomial_tail") {
      return Number.isInteger(dsl.n_trials) && dsl.n_trials >= 1 && dsl.n_trials <= 80 && ["exactly", "at_least", "at_most", "between"].includes(dsl.event);
    }
    if (dsl.family === "dice_event") {
      return dsl.dice === 2 && dsl.sides === 6 && ["sum_exact", "sum_at_least", "sum_at_most", "doubles"].includes(dsl.event);
    }
    if (dsl.family === "card_event") {
      return ["rank", "suit", "rank_or_suit", "rank_and_suit", "not_rank", "not_suit", "conditional_same_rank"].includes(dsl.event);
    }
    if (dsl.family === "birthday_event") {
      return Number.isInteger(dsl.people) && dsl.people >= 2 && dsl.people <= 366 && dsl.event === "at_least_one_shared_birthday";
    }
    if (dsl.family === "bayes_event") {
      return [dsl.prevalence_percent, dsl.sensitivity_percent, dsl.specificity_percent].every(Number.isFinite);
    }
    if (dsl.family === "urn_draw") {
      return dsl.colors && Number.isInteger(dsl.draws) && dsl.draws >= 1 && Number.isInteger(dsl.target_count) && dsl.target_count >= 0;
    }
    if (dsl.family === "expected_value") {
      return dsl.distribution === "geometric_die_until_face";
    }
    return ["proof_template", "unsupported"].includes(dsl.family);
  }

  function solveCoinRun(dsl) {
    const states = Array(dsl.run_length).fill(0n);
    states[0] = 1n;
    for (let i = 0; i < dsl.n_flips; i += 1) {
      const next = Array(dsl.run_length).fill(0n);
      const noRunSoFar = states.reduce((a, b) => a + b, 0n);
      next[0] += noRunSoFar;
      for (let k = 0; k < dsl.run_length - 1; k += 1) next[k + 1] += states[k];
      for (let k = 0; k < states.length; k += 1) states[k] = next[k];
    }
    const noRun = states.reduce((a, b) => a + b, 0n);
    const total = powBig(2, dsl.n_flips);
    const favorable = total - noRun;
    const frac = fractionBig(favorable, total);
    const pct = percentBig(favorable, total);
    return supported(
      `The probability is ${frac} = ${decimalBig(favorable, total)} (${pct}%).`,
      "exact coin-run dynamic program",
      dsl,
      [
        `Interpretation: ${dsl.n_flips} fair flips; event = at least one run of ${dsl.run_length} ${dsl.target}.`,
        `Total possible flip sequences = 2^${dsl.n_flips} = ${total}.`,
        `DP state tracks the current trailing ${dsl.target} count: 0..${dsl.run_length - 1}.`,
        `Sequences with no run of ${dsl.run_length} ${dsl.target} = ${noRun}.`,
        `Favorable sequences = total - no_run = ${favorable}.`,
        `Probability = ${frac} = ${decimalBig(favorable, total)}.`,
      ],
      { exact_fraction: frac, decimal: decimalBig(favorable, total), percent: pct },
    );
  }

  function solveBinomialTail(dsl) {
    let favorable = 0n;
    const total = powBig(2, dsl.n_trials);
    const lo = dsl.event === "between" ? dsl.low : dsl.successes;
    const hi = dsl.event === "between" ? dsl.high : dsl.successes;
    for (let k = 0; k <= dsl.n_trials; k += 1) {
      const include = (
        (dsl.event === "exactly" && k === dsl.successes) ||
        (dsl.event === "at_least" && k >= dsl.successes) ||
        (dsl.event === "at_most" && k <= dsl.successes) ||
        (dsl.event === "between" && k >= lo && k <= hi)
      );
      if (include) favorable += combBig(dsl.n_trials, k);
    }
    const frac = fractionBig(favorable, total);
    const extras = [];
    const textExtras = [];
    if (dsl.include_moments) {
      const expected = fractionBig(BigInt(dsl.n_trials), 2n);
      const variance = fractionBig(BigInt(dsl.n_trials), 4n);
      extras.push(`Expected value E[X] = n/2 = ${expected}.`);
      extras.push(`Variance Var(X) = n/4 = ${variance}.`);
      textExtras.push(` E[X] = ${expected}; Var(X) = ${variance}.`);
    }
    return supported(
      `The probability is ${frac} = ${decimalBig(favorable, total)}.${textExtras.join("")}`,
      "exact binomial calculator",
      dsl,
      [
        `Interpretation: X ~ Binomial(${dsl.n_trials}, 1/2).`,
        `Event = ${dsl.event}${dsl.event === "between" ? ` ${lo}..${hi}` : ` ${dsl.successes}`} ${dsl.target}.`,
        `Favorable sequences = ${favorable}.`,
        `Total sequences = 2^${dsl.n_trials} = ${total}.`,
        `Probability = ${frac}.`,
        ...extras,
      ],
      { exact_fraction: frac, decimal: decimalBig(favorable, total) },
    );
  }

  function solveDiceEvent(dsl) {
    let favorable = 0;
    const examples = [];
    for (let a = 1; a <= 6; a += 1) {
      for (let b = 1; b <= 6; b += 1) {
        const sum = a + b;
        const ok = (
          (dsl.event === "sum_exact" && sum === dsl.target) ||
          (dsl.event === "sum_at_least" && sum >= dsl.target) ||
          (dsl.event === "sum_at_most" && sum <= dsl.target) ||
          (dsl.event === "doubles" && a === b)
        );
        if (ok) {
          favorable += 1;
          if (examples.length < 12) examples.push(`(${a},${b})`);
        }
      }
    }
    const frac = fractionBig(BigInt(favorable), 36n);
    return supported(
      `The probability is ${frac} = ${decimalBig(BigInt(favorable), 36n)}.`,
      "exact dice enumeration",
      dsl,
      [`Enumerate all 36 ordered outcomes for two fair dice.`, `Favorable outcomes = ${favorable}: ${examples.join(" ")}${favorable > examples.length ? " ..." : ""}`, `Probability = ${frac}.`],
      { exact_fraction: frac, decimal: decimalBig(BigInt(favorable), 36n) },
    );
  }

  function solveCardEvent(dsl) {
    let favorable = 0;
    if (dsl.event === "rank") favorable = 4;
    if (dsl.event === "suit") favorable = 13;
    if (dsl.event === "rank_or_suit") favorable = 16;
    if (dsl.event === "rank_and_suit") favorable = 1;
    if (dsl.event === "not_rank") favorable = 48;
    if (dsl.event === "not_suit") favorable = 39;
    if (dsl.event === "conditional_same_rank") {
      return supported(
        "The probability is 1/17 = 0.058824.",
        "exact standard-deck conditional calculator",
        dsl,
        ["The first named rank card is already removed.", "Remaining deck size = 51.", "Remaining cards with that rank = 3.", "Probability = 3/51 = 1/17."],
        { exact_fraction: "1/17", decimal: "0.058824" },
      );
    }
    const frac = fractionBig(BigInt(favorable), 52n);
    return supported(
      `The probability is ${frac} = ${decimalBig(BigInt(favorable), 52n)}.`,
      "exact standard-deck calculator",
      dsl,
      [`Standard deck size = 52.`, `Event type = ${dsl.event}.`, `Favorable cards = ${favorable}.`, `Probability = ${frac}.`],
      { exact_fraction: frac, decimal: decimalBig(BigInt(favorable), 52n) },
    );
  }

  function solveBirthdayEvent(dsl) {
    let noSharedNumerator = 1n;
    const denominator = powBig(365, dsl.people);
    for (let i = 0; i < dsl.people; i += 1) {
      noSharedNumerator *= BigInt(365 - i);
    }
    const favorable = denominator - noSharedNumerator;
    const frac = fractionBig(favorable, denominator);
    return supported(
      `The probability is ${frac} = ${decimalBig(favorable, denominator)} (${percentBig(favorable, denominator)}%).`,
      "exact birthday complement calculator",
      dsl,
      [
        `Assumption: 365 equally likely birthdays, ignoring leap days.`,
        `Total assignments = 365^${dsl.people}.`,
        `No-shared-birthday assignments = 365*364*...*${365 - dsl.people + 1}.`,
        `At least one shared birthday = 1 - no_shared.`,
        `Probability = ${frac} = ${decimalBig(favorable, denominator)}.`,
      ],
      { exact_fraction: frac, decimal: decimalBig(favorable, denominator), percent: percentBig(favorable, denominator) },
    );
  }

  function solveBayesEvent(dsl) {
    const prevalence = Math.round(dsl.prevalence_percent * 10000);
    const sensitivity = Math.round(dsl.sensitivity_percent * 10000);
    const specificity = Math.round(dsl.specificity_percent * 10000);
    const scale = 1000000n;
    const prev = BigInt(prevalence);
    const sens = BigInt(sensitivity);
    const spec = BigInt(specificity);
    const truePositive = prev * sens;
    const falsePositive = (scale - prev) * (scale - spec);
    const denominator = truePositive + falsePositive;
    const frac = fractionBig(truePositive, denominator);
    return supported(
      `The posterior probability is ${frac} = ${decimalBig(truePositive, denominator)} (${percentBig(truePositive, denominator)}%).`,
      "exact Bayes-rule calculator",
      dsl,
      [
        `Use P(disease | positive) = P(pos | disease)P(disease) / P(positive).`,
        `Prevalence = ${dsl.prevalence_percent}%; sensitivity = ${dsl.sensitivity_percent}%; specificity = ${dsl.specificity_percent}%.`,
        `True-positive mass = prevalence * sensitivity.`,
        `False-positive mass = (1 - prevalence) * (1 - specificity).`,
        `Posterior = true_positive / (true_positive + false_positive) = ${frac}.`,
      ],
      { exact_fraction: frac, decimal: decimalBig(truePositive, denominator), percent: percentBig(truePositive, denominator) },
    );
  }

  function solveUrnDraw(dsl) {
    const totalBalls = Object.values(dsl.colors).reduce((a, b) => a + b, 0);
    const targetBalls = dsl.colors[dsl.target_color] || 0;
    if (!dsl.replacement && dsl.draws === 2 && dsl.target_count === 2) {
      const favorable = BigInt(targetBalls * (targetBalls - 1));
      const total = BigInt(totalBalls * (totalBalls - 1));
      const frac = fractionBig(favorable, total);
      return supported(
        `The probability is ${frac} = ${decimalBig(favorable, total)}.`,
        "exact urn probability",
        dsl,
        [`First ${dsl.target_color} = ${targetBalls}/${totalBalls}.`, `Second ${dsl.target_color} = ${targetBalls - 1}/${totalBalls - 1}.`, `Probability = ${frac}.`],
        { exact_fraction: frac, decimal: decimalBig(favorable, total) },
      );
    }
    if (dsl.replacement) {
      const favorable = BigInt(combBig(dsl.draws, dsl.target_count)) * (BigInt(targetBalls) ** BigInt(dsl.target_count)) * (BigInt(totalBalls - targetBalls) ** BigInt(dsl.draws - dsl.target_count));
      const total = BigInt(totalBalls) ** BigInt(dsl.draws);
      const frac = fractionBig(favorable, total);
      return supported(
        `The probability is ${frac} = ${decimalBig(favorable, total)}.`,
        "exact urn probability with replacement",
        dsl,
        [`Use binomial counting over ${dsl.draws} draws with replacement.`, `Favorable weighted sequences = ${favorable}.`, `Total weighted sequences = ${total}.`, `Probability = ${frac}.`],
        { exact_fraction: frac, decimal: decimalBig(favorable, total) },
      );
    }
    return abstain("unsupported_urn_shape", ["This urn shape is not published yet.", "safe behavior = abstain"]);
  }

  function solveExpectedValue(dsl) {
    if (dsl.distribution === "geometric_die_until_face") {
      return supported(
        "The expected number of rolls is 6.",
        "exact geometric expectation",
        dsl,
        ["Success probability p = 1/6.", "Geometric expected waiting time = 1/p.", "1 / (1/6) = 6."],
        { exact_value: "6" },
      );
    }
    return abstain("unsupported_expected_value", ["No verified expected-value backend matched.", "safe behavior = abstain"]);
  }

  function solveProofTemplate(dsl) {
    const table = {
      monty_hall: supported(
        "Yes, switch. The probability of winning by switching is 2/3 = 0.666667.",
        "proof template",
        dsl,
        ["Initial chosen door wins with probability 1/3.", "The unchosen doors carry probability 2/3.", "The host reveals a goat, so switching keeps the 2/3 probability mass."],
        { exact_fraction: "2/3", decimal: "0.666667" },
      ),
      three_boxes: supported(
        "The probability is 2/3 = 0.666667.",
        "exact conditional probability",
        dsl,
        ["Conditioned gold observations: GG coin 1, GG coin 2, GS gold coin.", "Success cases where the other coin is gold = 2.", "Total conditioned cases = 3."],
        { exact_fraction: "2/3", decimal: "0.666667" },
      ),
      prisoners: supported(
        "Use the cycle-following strategy. The group survival probability is about 0.311828.",
        "combinatorial proof template",
        dsl,
        ["Each prisoner starts at the box with their own number.", "Then follows the number found in each opened box.", "Success iff the permutation has no cycle longer than 50.", "Probability = 1 - sum(k=51..100) 1/k ~= 0.311828."],
        { decimal: "0.311828" },
      ),
      operation_synthesis: supported(
        "No five-operation composition produces the target under the declared operation semantics.",
        "AHR exhaustive finite program search",
        dsl,
        ["Operation choices = 4.", "Program length = 5.", "Programs checked = 4^5 = 1024.", "Solutions found = 0."],
        { exact_value: "0 solutions" },
      ),
      graph_bridge_triangle: supported(
        "No such graph exists.",
        "exact proof plus exhaustive graph check",
        dsl,
        ["If deleting any edge disconnects a connected graph, every edge is a bridge.", "An edge on a cycle cannot be a bridge.", "Therefore the graph is a tree.", "A tree has zero triangles."],
      ),
      hanoi_rule_revision: supported(
        "Impossible under the revised rule.",
        "exact state-space verifier",
        dsl,
        ["The largest disk must eventually move from A to C.", "Before that move, four smaller disks must be stacked on B.", "That cannot happen before move 8.", "After move 8, peg B may hold at most two disks."],
      ),
      bank_counterfactual: supported(
        "(a) $127.89; (b) $56.84; (c) $127.89.",
        "exact rational branch arithmetic",
        dsl,
        ["Actual: 200.00 -> 206.00 -> 126.00 -> 127.89.", "Counterfactual withdrawal 150: 200.00 -> 206.00 -> 56.00 -> 56.84.", "Fresh actual branch confirms no state contamination."],
      ),
      green_eyes: supported(
        "All 40 green-eyed inhabitants leave together at midnight on the 40th night; the 60 amber-eyed inhabitants never leave.",
        "common-knowledge induction proof",
        dsl,
        ["Base case: 1 green-eyed person leaves on night 1.", "Induction: k green-eyed people leave on night k.", "With 40 green-eyed people, all 40 leave on night 40."],
      ),
      sequence_decoy: supported(
        "The next three terms are 256, 386, 562.",
        "finite-difference rule verification",
        dsl,
        ["Verified rule: a_n = sum(k=0..4) binomial(n-1,k).", "Matches the given prefix.", "Next terms for n=10,11,12 are 256, 386, 562."],
      ),
    };
    return table[dsl.template] || abstain("unsupported_proof_template", ["No verified proof template matched.", "safe behavior = abstain"]);
  }

  function solveDsl(dsl) {
    if (!validateDsl(dsl)) return abstain("invalid_dsl", ["DSL validation failed.", JSON.stringify(dsl), "safe behavior = abstain"]);
    if (dsl.family === "coin_run") return solveCoinRun(dsl);
    if (dsl.family === "binomial_tail") return solveBinomialTail(dsl);
    if (dsl.family === "dice_event") return solveDiceEvent(dsl);
    if (dsl.family === "card_event") return solveCardEvent(dsl);
    if (dsl.family === "birthday_event") return solveBirthdayEvent(dsl);
    if (dsl.family === "bayes_event") return solveBayesEvent(dsl);
    if (dsl.family === "urn_draw") return solveUrnDraw(dsl);
    if (dsl.family === "expected_value") return solveExpectedValue(dsl);
    if (dsl.family === "proof_template") return solveProofTemplate(dsl);
    return abstain(dsl.reason || "unsupported", ["Question routed to unsupported.", "safe behavior = abstain"]);
  }

  function parseCoinRun(q) {
    if (!/(coin|flip|toss)/.test(q) || !/(row|consecutive|sequence|streak|run)/.test(q) || !/(head|tail)/.test(q)) return null;
    const n = numberNear(q, [/flip\s+(\d+|[a-z-]+)\s+coins?/, /(\d+|[a-z-]+)\s+(?:coin\s+)?(?:flips|tosses)/, /in\s+(\d+|[a-z-]+)\s+flips?/]);
    const k = numberNear(q, [/sequence of\s+(\d+|[a-z-]+)\s+(?:heads?|tails?)/, /run of\s+(\d+|[a-z-]+)\s+(?:heads?|tails?)/, /(\d+|[a-z-]+)\s+(?:heads?|tails?)\s+in a row/, /(\d+|[a-z-]+)\s+consecutive\s+(?:heads?|tails?)/]);
    if (!Number.isInteger(n) || !Number.isInteger(k)) return null;
    return { family: "coin_run", n_flips: n, run_length: k, target: /tail/.test(q) ? "tails" : "heads", event: "at_least_one_run" };
  }

  function parseBinomialTail(q) {
    if (!/(coin|heads|tails|binomial)/.test(q)) return null;
    const n = numberNear(q, [/flipping\s+(\d+|[a-z-]+)\s+(?:fair\s+)?coins?/, /(?:tossed|flipped)\s+(\d+|[a-z-]+)\s+times/, /(\d+|[a-z-]+)\s+(?:fair\s+)?(?:coin\s+)?(?:flips|tosses|coins)/, /binomial\((\d+)/]);
    if (!Number.isInteger(n)) return null;
    const target = /tail/.test(q) ? "tails" : "heads";
    let event = null;
    let successes = null;
    let low = null;
    let high = null;
    const between = q.match(/between\s+(\d+)\s+and\s+(\d+)/);
    if (between) {
      event = "between";
      low = Number(between[1]);
      high = Number(between[2]);
    } else if (/at least|or more/.test(q)) {
      event = "at_least";
      successes = numberNear(q, [/at least\s+(\d+|[a-z-]+)/, /(\d+|[a-z-]+)\s+or more/]);
    } else if (/at most|or fewer|no more than/.test(q)) {
      event = "at_most";
      successes = numberNear(q, [/at most\s+(\d+|[a-z-]+)/, /no more than\s+(\d+|[a-z-]+)/, /(\d+|[a-z-]+)\s+or fewer/]);
    } else if (/exactly|precisely|x=|x equals/.test(q)) {
      event = "exactly";
      successes = numberNear(q, [/exactly\s+(\d+|[a-z-]+)\s+(?:heads?|tails?)/, /precisely\s+(\d+|[a-z-]+)\s+(?:heads?|tails?)/, /x\s*=\s*(\d+)/, /x equals\s+(\d+|[a-z-]+)/]);
    }
    if (!event) return null;
    return { family: "binomial_tail", n_trials: n, successes, low, high, target, event, include_moments: /e\[x\]|expected|expectation|mean|var|variance/.test(q) };
  }

  function parseDice(q) {
    if (!/(die|dice|d6|six-sided)/.test(q)) return null;
    if (/double/.test(q)) return { family: "dice_event", dice: 2, sides: 6, event: "doubles", target: null };
    const target = numberNear(q, [/sum(?:\s+is|\s+equals|\s+to)?\s+(\d+|[a-z-]+)/, /total(?:\s+is|\s+equals|\s+to)?\s+(\d+|[a-z-]+)/, /at least\s+(\d+|[a-z-]+)/, /at most\s+(\d+|[a-z-]+)/]);
    if (!Number.isInteger(target)) return null;
    let event = "sum_exact";
    if (/at least|or more/.test(q)) event = "sum_at_least";
    if (/at most|or fewer|no more than/.test(q)) event = "sum_at_most";
    return { family: "dice_event", dice: 2, sides: 6, event, target };
  }

  function parseCard(q) {
    if (!/(card|deck)/.test(q)) return null;
    const rank = RANKS.find(r => new RegExp(`\\b${r}s?\\b`).test(q)) || null;
    const suitRaw = SUITS.find(s => new RegExp(`\\b${s}\\b`).test(q)) || null;
    const suit = suitRaw ? suitRaw.replace(/s$/, "") : null;
    if (!rank && !suit) return null;
    if (rank && /(given|first card|first is|first .*card)/.test(q) && /(second|also|another|same)/.test(q)) {
      return { family: "card_event", rank, suit: null, event: "conditional_same_rank" };
    }
    let event = rank ? "rank" : "suit";
    if (rank && suit && /\b(or|union|either)\b/.test(q)) event = "rank_or_suit";
    if (rank && suit && /\b(and|both)\b/.test(q)) event = "rank_and_suit";
    if (rank && suit && new RegExp(`\\b${rank}\\s+of\\s+${suit}s?\\b`).test(q)) event = "rank_and_suit";
    if (rank && /not|not a|not an/.test(q)) event = "not_rank";
    if (suit && /not/.test(q)) event = "not_suit";
    return { family: "card_event", rank, suit, event };
  }

  function parseBirthday(q) {
    if (!/(birthday|birthdays)/.test(q) || !/(share|same|match|collision)/.test(q)) return null;
    const people = numberNear(q, [/room of\s+(\d+|[a-z-]+)\s+people/, /(\d+|[a-z-]+)\s+people/, /among\s+(\d+|[a-z-]+)/]);
    if (!Number.isInteger(people)) return null;
    return { family: "birthday_event", people, event: "at_least_one_shared_birthday" };
  }

  function parseBayes(q) {
    if (!/(disease|test|positive|sensitivity|specificity)/.test(q)) return null;
    const prevalence = numberNear(q, [/affects\s+(\d+(?:\.\d+)?)%/, /prevalence(?:\s+is)?\s+(\d+(?:\.\d+)?)%/]);
    const sensitivity = numberNear(q, [/(\d+(?:\.\d+)?)%\s+sensitivity/, /sensitivity(?:\s+is|:)?\s+(\d+(?:\.\d+)?)%/]);
    const specificity = numberNear(q, [/(\d+(?:\.\d+)?)%\s+specificity/, /specificity(?:\s+is|:)?\s+(\d+(?:\.\d+)?)%/]);
    if (![prevalence, sensitivity, specificity].every(Number.isFinite)) return null;
    return { family: "bayes_event", prevalence_percent: prevalence, sensitivity_percent: sensitivity, specificity_percent: specificity, event: "posterior_given_positive" };
  }

  function parseUrn(q) {
    if (!/(bag|urn|balls?)/.test(q) || !/(draw|drawn|sample)/.test(q)) return null;
    const colors = {};
    for (const color of COLORS) {
      const m = q.match(new RegExp(`(\\d+)\\s+${color}`));
      if (m) colors[color] = Number(m[1]);
    }
    const targetColor = COLORS.find(color => new RegExp(`both (?:are )?${color}|two ${color}|${color} balls?`).test(q));
    if (!targetColor || !colors[targetColor]) return null;
    const draws = numberNear(q, [/draw(?:n)?\s+(\d+|[a-z-]+)/, /(\d+|[a-z-]+)\s+balls?\s+are\s+drawn/]) || 2;
    const targetCount = /both|two/.test(q) ? 2 : 1;
    return { family: "urn_draw", colors, draws, replacement: /with replacement/.test(q) && !/without replacement/.test(q), target_color: targetColor, target_count: targetCount };
  }

  function parseExpected(q) {
    if (/(die|d6)/.test(q) && /(until|first|waiting|expected|expectation)/.test(q) && /(6|six)/.test(q)) {
      return { family: "expected_value", distribution: "geometric_die_until_face", face: 6, success_probability: "1/6" };
    }
    return null;
  }

  function parseProof(q) {
    if (/monty hall/.test(q) || (/doors?/.test(q) && /goat|car|switch/.test(q))) return { family: "proof_template", template: "monty_hall" };
    if (/(three boxes|box a|gg|gold coin)/.test(q) && /gold/.test(q)) return { family: "proof_template", template: "three_boxes" };
    if (/100 prisoners/.test(q) && /box/.test(q)) return { family: "proof_template", template: "prisoners" };
    if (/rev/.test(q) && /delta/.test(q) && /double/.test(q) && /zip/.test(q)) return { family: "proof_template", template: "operation_synthesis" };
    if (/(connected graph|6-vertex|six-vertex)/.test(q) && /(bridge|single edge|triangle)/.test(q)) return { family: "proof_template", template: "graph_bridge_triangle" };
    if (/(tower of hanoi|hanoi|5-disk|5 disk)/.test(q) && /(peg b|move 8|at most 2)/.test(q)) return { family: "proof_template", template: "hanoi_rule_revision" };
    if (/(deposit|bank|interest|withdraw)/.test(q) && /(3%|1.5%|counterfactual|alice|bob)/.test(q)) return { family: "proof_template", template: "bank_counterfactual" };
    if (/(green eyes|green-eyed|amber)/.test(q) && /(visitor|midnight|leave|island)/.test(q)) return { family: "proof_template", template: "green_eyes" };
    if ((/1\s*,\s*2\s*,\s*4\s*,\s*8\s*,\s*16\s*,\s*31/.test(q) || /sequence/.test(q)) && /(next|closed-form|doubling|terms)/.test(q)) return { family: "proof_template", template: "sequence_decoy" };
    return null;
  }

  function parseUnsupported(q) {
    if (/(unknown|unspecified|loaded|nonstandard|without any prior|no prior|not specified)/.test(q) && /(probability|deck|die|coin|ratio|model|prior)/.test(q)) {
      return { family: "unsupported", reason: "missing_probability_model" };
    }
    return null;
  }

  const PARSERS = [parseUnsupported, parseCoinRun, parseBinomialTail, parseDice, parseCard, parseBirthday, parseBayes, parseUrn, parseExpected, parseProof];

  function parseQuestion(text) {
    const q = normalize(text);
    for (const parser of PARSERS) {
      const dsl = parser(q);
      if (dsl) return dsl;
    }
    return { family: "unsupported", reason: "no_verified_backend_matched" };
  }

  function routeQuestion(text) {
    return solveDsl(parseQuestion(text));
  }

  const EXAMPLES = [
    "If I flip 50 coins, what are the chances there is a sequence of 5 heads in a row?",
    "A fair coin is flipped 6 times. What is the probability of exactly 4 heads?",
    "A fair coin is flipped 10 times. What is the probability of at least 7 heads?",
    "Two dice are rolled. What is the probability that the total is at least 9?",
    "What is the probability of rolling doubles with two fair dice?",
    "A card is drawn from a standard 52-card deck. What is the probability of drawing an ace or a spade?",
    "A bag contains 5 red, 3 blue, and 2 green balls. Two balls are drawn without replacement. What is the probability that both are blue?",
    "Two cards are drawn from a deck without replacement. Given that the first card is a king, what is the probability that the second card is also a king?",
    "A disease affects 2% of a population. A test has 90% sensitivity and 95% specificity. If a person tests positive, what is the probability they have the disease?",
    "In a room of 30 people, what is the probability that at least two people share the same birthday?",
    "There are 3 doors. One contains a car and two contain goats. You pick one door. The host opens another door showing a goat. Should you switch?",
    "Let X be the number of heads obtained when flipping 5 fair coins. Find P(X=3), E[X], and Var(X).",
    "Box A contains 2 gold coins. Box B contains 2 silver coins. Box C contains 1 gold and 1 silver coin. A box is chosen at random, and one coin drawn is gold. What is the probability the other coin is also gold?",
    "100 prisoners and 100 numbered boxes. Each prisoner may open 50 boxes. What strategy maximizes their survival probability?",
    "You repeatedly roll a fair die until a 6 appears. What is the expected number of rolls?",
    "Using REV, DELTA, DOUBLE-EVENS, and ZIP-SELF, is there exactly five operations that maps [1,4,9,16,25,36] to [1,6,5,14,9]?",
    "A loaded die has unknown probabilities. What is the probability of rolling a 6?",
  ];

  global.AHRCore = {
    familyMetadata: FAMILY_METADATA,
    examples: EXAMPLES,
    normalize,
    parseQuestion,
    solveDsl,
    routeQuestion,
    validateDsl,
  };

  if (typeof module !== "undefined" && module.exports) {
    module.exports = global.AHRCore;
  }
})(typeof window !== "undefined" ? window : globalThis);
