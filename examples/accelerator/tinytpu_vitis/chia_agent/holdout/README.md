# Held-out rediscovery

One experiment with a known correct answer, for calibrating how much guidance an
agent needs before open-ended abstraction work is worth paying for.

An agent is given a symptom and asked to find an abstraction that addresses it.
The abstraction it should arrive at already exists in this repository, was
written by a person, and is covered by tests — so grading needs no judgement
call. The agent is not told that, is not told the primitive's name, and works
from a tree in which it genuinely does not exist.

## Why a base commit rather than a redaction

Deleting a primitive from the current tree leaves its traces: tests that
reference it, a design that calls it, documentation that names it, a commit in
`git log`. Every one of those is a leak, and missing one invalidates the run
without anyone noticing.

So the held-out tree is an **earlier commit**, from before the primitive was
written:

    git worktree add <path> a4151ca0

`a4151ca0` is the parent of the commit that introduced the answer. At that
commit the primitive has never existed, the design does not use it, no test
mentions it, and nothing in the documentation describes it. The absence is real
rather than simulated.

The grader's reference is the child commit, `bbea2af0`, together with the tests
it added. Read those **after** the run, not before.

## What the agent is given

`symptom.md` — the measurement and nothing else. It states what was observed,
what was tried in hardware, and what that cost. It does not name the mechanism,
the pragma, the primitive, or the vendor feature, and it does not say that a
one-line fix exists. Read it before running the experiment and check it still
leaks nothing; a symptom statement drifts toward its answer every time someone
edits it.

The agent must also be given the ordinary context a compiler engineer would
have: the repository at the held-out commit, the ability to build and measure,
and the tool documentation. Withholding those would not make the experiment
harder in an interesting way — it would only make it a guessing game.

### One deliberate near-leak, recorded rather than removed

`symptom.md` uses the word *dependence*, twice, to describe a loop-carried
dependence through the accumulator array. That is the vendor's own term and it
is what the scheduling report says, so removing it would make the symptom
artificially obscure rather than making the experiment harder — an engineer
reading a real report would see that word. It is also the first word of the
answer's name, and of the pragma's.

This is a judgement call and it is written down so that a later reader does not
mistake it for an oversight. What is withheld: the pragma, the vendor feature,
the primitive's name and signature, the existence of a one-line fix, and any
suggestion that the fix belongs in the compiler rather than in the design. If a
run succeeds, note in the record whether the agent got there from that word or
from the symptom.

## Grading

Four outcomes, in decreasing strength. Record which one, plus the transcript.

1. **Arrives at the same abstraction**, at the same level — a schedule
   primitive that carries the claim to the emitter. Whether its argument list
   matches does not matter; whether it puts the claim in the same place does.
2. **Arrives at an abstraction of equal power at a different level** — for
   instance a type or an IR attribute that expresses the same claim. This is
   not a failure and may be a better answer than the one in the tree; judge it
   on whether the claim is checkable and on what it costs the user to state.
3. **Solves the instance without an abstraction** — patches the emitted C++,
   hard-codes the pragma, or engineers the recurrence away in hardware. This is
   what the project did before the primitive existed, and it is the outcome
   that most needs to be distinguished from success, because it *works*: the
   hardware redesign was bit-exact and it was still the wrong answer.
4. **Does not reach a solution.** Record where it stopped and what it was
   missing, because that is the evidence about what guidance the open-ended
   attempts need.

A further question worth recording for any outcome above 3: **does the agent
notice that the claim it is making is unchecked?** The primitive in the tree
does not verify its own claim — a false claim is invisible to every simulator
and only appears in RTL. An agent that proposes the abstraction *and* says what
would have to hold for it to be sound has done something better than the
original commit did.

## Cost and discipline

This is the cheapest experiment in the plan and it must stay that way: one
symptom, a bounded attempt, a recorded outcome. Run it before spending on
open-ended extension attempts. If an agent cannot reach an answer that exists
in the tree with its tests, the open-ended runs will not succeed either, and
knowing that costs a few dollars here rather than a few hundred there.

Do not iterate on the symptom statement to get a better result. If the symptom
needs three rewrites before an agent succeeds, that is the finding — the
guidance required *is* the measurement.
