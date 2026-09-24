# How the fork's doc pages should be written

Feedback received 2026-09-24: *"the docs are poorly written."* This is the
diagnosis and the template. It applies to every fork-only page, and to the new
ones now needed for the PD flow, the ACT flow, the SystemC emitter and the CHIA
loops.

## The diagnosis, from the contrast

Upstream's `backends/vitis.rst` opens:

> The Vitis HLS FPGA backend leverages the Vivado/Vitis HLS toolchain to
> generate hardware accelerators for FPGA devices. **This document demonstrates
> how to** define a general matrix multiplication kernel using the Allo ADL and
> generate HLS code for FPGA synthesis.

Then, immediately: **Kernel Definition**, with runnable code.

One of our pages opens:

> An LLM-agent loop that co-designs a TPU's instruction set and
> microarchitecture together, where a proposal counts only if a real tool
> measures it. […] **The principle** — An agent can make an informed co-design
> decision only after it has measured how a software function performs […]

Five faults, in the order they hurt a reader:

1. **It opens with a thesis instead of with the thing.** A reader arrives
   wanting to know what this is and how to run it. We give them an argument.
2. **There is no runnable code.** Upstream pages are mostly code with prose
   between it. Ours are mostly prose with measurement tables between it.
3. **Findings and retractions are interleaved with reference material**, so
   neither is usable: you cannot skim for the parameter you need, and you
   cannot read the findings without wading through parameters.
4. **The register is argumentative.** *"which is why"*, *"the honest version
   is"*, *"stated at its true strength"*. That is right for a paper and wrong
   for a page whose job is to let someone use a tool.
5. **They are too long.** Several run past a thousand lines. Length is not
   thoroughness; it is a refusal to decide what matters.

None of this is about accuracy. The content is measured and the corrections are
real. It is organised as a notebook rather than as documentation.

## The template

Every fork page gets these sections, in this order. Omit a section only when it
genuinely does not apply — not because there is a lot to say elsewhere.

```
#####################
<Thing> (<what it targets>)
#####################

One paragraph: what it is, what it produces, and what it needs. No claims, no
principles, no history. A reader should be able to decide from this paragraph
whether the page is relevant to them.

Quick start
-----------
The command. Copy-pasteable, with the environment it assumes stated in a
comment. What it prints when it works. How long it takes.

How it works
------------
The mechanism, with code or a file listing. What is generated and what is
hand-written. Where the pieces live.

Reference
---------
Parameters, environment variables, file layout, exit codes. Tables. This is the
part a returning reader skims, so it must be skimmable.

Limits and known failures
-------------------------
What it cannot do, what breaks, and the gates that would catch it. Short, and
links to the limitations register rather than restating it.

Results and history
-------------------
Measurements, findings, corrections, retractions — last, or on their own page
if there is a lot. Never above the Reference section.
```

## Rules

- **Quick start comes before explanation.** If a reader cannot run the thing in
  the first screenful, the page has failed.
- **Show code, not only numbers.** A code block a reader can paste is worth
  more than a table of what happened when we pasted it.
- **One claim per figure, and cite the file it comes from** rather than
  retyping it — `allo/backend/asic/tools/check_numbers.py --reports <design>/asic_synthesis/reports`
  enforces this for area.
- **Move history down or out.** Dated measurements belong in `dev/records/`;
  the page states the current state, with a pointer.
- **Cut the hedges.** Write the measured fact. Caveats belong in one place next
  to the number, not distributed through the prose as qualifiers.
- **A page over ~400 lines needs splitting** into a reference page and a
  results page.

## Pages that need this

| page | what is wrong | priority |
| --- | --- | --- |
| `extensions/chia.rst` | opens with a principle; findings above reference; very long | high |
| `designs/tinytpu_isa.rst` | reference and history interleaved; ~2000 lines | high |
| `designs/gemmini_comparison.rst` | argumentative throughout; no quick start | high |
| `extensions/act.rst`, `act_specs.rst` | no quick start; two pages that overlap | medium |
| `designs/benchmarks.rst` | a results page, which is correct — but it is where other pages' history should go | keep |

## Pages that do not exist yet and should

- **The PD / ASIC flow** — preflight, the documented command sequence, the
  settings snapshot, what is committed and what stays on scratch.
- **The SystemC emitter** — currently only reachable through the Catapult page.
- **The CHIA loop as a tool** — how to run it, what it costs, what the guards
  are. The existing page is about what it *found*.
- **The workload suite** — how to add a model, what maps and what does not.
