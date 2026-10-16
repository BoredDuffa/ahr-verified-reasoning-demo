"use strict";

const assert = require("node:assert/strict");
const core = require("../docs/ahr_core.js");

function check(question, expectedFamily, expectedFraction) {
  const result = core.routeQuestion(question);
  assert.equal(result.abstain, false, question);
  assert.equal(result.dsl.family, expectedFamily, question);
  if (expectedFraction) {
    assert.equal(result.verification.exact_fraction, expectedFraction, question);
  }
  assert.ok(result.trace.length >= 2, question);
  return result;
}

check(
  "If I flip 50 coins, what are the chances there is a sequence of 5 heads in a row?",
  "coin_run",
  "155339093675555/281474976710656",
);
check(
  "A fair coin is tossed 10 times. What is the probability of getting at least 7 heads?",
  "binomial_tail",
  "11/64",
);
check(
  "Two fair six-sided dice are rolled. What is the probability that the total is at least 9?",
  "dice_event",
  "5/18",
);
check(
  "A card is drawn from a standard deck. What is the probability of drawing an ace or a spade?",
  "card_event",
  "4/13",
);
check(
  "A card is drawn from a standard deck. What is the probability of drawing the ace of spades?",
  "card_event",
  "1/52",
);
check(
  "A bag contains 5 red, 3 blue, and 2 green balls. Two balls are drawn without replacement. What is the probability that both are blue?",
  "urn_draw",
  "1/15",
);
check(
  "Two cards are drawn without replacement. Given that the first card is a king, what is the probability that the second card is also a king?",
  "card_event",
  "1/17",
);
check(
  "In a room of 30 people, what is the probability that at least two share a birthday?",
  "birthday_event",
);
check(
  "A disease affects 2% of people. A test has 90% sensitivity and 95% specificity. If positive, what is the probability they have the disease?",
  "bayes_event",
  "18/67",
);

const rv = check(
  "Let X be the number of heads obtained when flipping 5 fair coins. Find P(X=3), E[X], Var(X).",
  "binomial_tail",
  "5/16",
);
assert.match(rv.text, /E\[X\] = 5\/2/);
assert.match(rv.text, /Var\(X\) = 5\/4/);

const unsupported = core.routeQuestion("A loaded die has unknown probabilities. What is the probability of rolling a 6?");
assert.equal(unsupported.abstain, true);
assert.equal(unsupported.dsl.family, "unsupported");

console.log("ahr_demo_core_routes: ok");
