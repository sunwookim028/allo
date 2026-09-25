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
- **Associate a figure with its run by declaring it, not by inferring it.**
  `check_numbers.py` matches area-shaped figures against the set of committed
  totals, which is decidable. Associating a figure with the *cycle counts near
  it in prose* is not: a tool that guesses at that would produce false
  failures, and a checker that cries wolf stops being run. If that association
  is ever enforced, the honest form is a directive in the page naming the run
  it quotes -- declared by the author, checkable by a tool. That is a docs
  change, not a checker change, and it is not done.
- **Record the derivation beside the figure.** *A derived number inherits the
  truth of its derivation and carries none of the derivation with it.* Once
  "36.4 % of peak" is written down, nothing in the number objects when the
  mechanism it was derived from turns out not to exist -- and a second page
  quoting it inherits the error with none of the provenance. This is not
  hypothetical: `designs/gemmini_results.rst` carried 36.4 % as a bound set by
  "one `mxu_matrix_ctrl` FSM" that does not exist, priced our emitter at "52 %
  of the bound" on top of it, and built a ceiling-versus-overhead argument on
  top of that. All three fell together in September 2026, and the only thing
  that would have caught it earlier was the derivation sitting next to the
  figure where a reader could check it.

  `check_numbers.py` already enforces this for area: every area-shaped figure
  in the docs must match a committed report or carry a named exemption.
  Nothing enforced it for an efficiency figure. Until something does, write
  the derivation out -- the workload, the instrument, the date -- next to any
  number that is not read directly off a report, and say plainly whether it is
  a measurement or a bound. The two are different objects and neither survives
  being mistaken for the other.

- **Move history down or out.** Dated measurements belong in `dev/records/`;
  the page states the current state, with a pointer.
- **Cut the hedges.** Write the measured fact. Caveats belong in one place next
  to the number, not distributed through the prose as qualifiers.
- **A page over ~400 lines needs splitting** into a reference page and a
  results page.

## Pages that need this

**Applied 2026-09-24** in `95bfd77a`, merged as `5d7223e9` ("the fork's pages
open with the thing"). All four pages now carry the template's sections in the
template's order, and each one over ~400 lines was split into a reference page
and a results page. No number was changed; every correction and retraction
moved with its page's history rather than being summarised.

| page | what was wrong | what was done | state |
| --- | --- | --- | --- |
| `extensions/chia.rst` | opens with a principle; findings above reference; very long | opens with what the loop is and what it produces; Quick start second; 1232 → 454 lines, findings to `chia_results.rst`, the co-design argument to `chia_codesign.rst`. Cap and spend arithmetic kept on the page, under Reference → Billing | done |
| `designs/tinytpu_isa.rst` | reference and history interleaved; ~2000 lines | 2266 → 952; the instruction/encoding reference to `tinytpu_isa_spec.rst`, measurements and retractions to `tinytpu_isa_results.rst`, older history already on `tinytpu_history.rst` | done |
| `designs/gemmini_comparison.rst` | argumentative throughout; no quick start | 2462 → 962; Quick start is `reproduce.sh` plus the parity configs; every measurement and withdrawn claim to `gemmini_results.rst`, including the 36.4 % restatement and the withdrawn "52 % of the bound" attribution | done |
| `extensions/act.rst`, `act_specs.rst` | no quick start; two pages that overlap | both open by saying what they are for and pointing at the other: `act.rst` is the mapper, `act_specs.rst` is the corpus and the judge, `act_results.rst` is the measurements. 949 → 557 and 748 → 543 | done |
| `designs/benchmarks.rst` | a results page, which is correct — but it is where other pages' history should go | not restyled, as intended; grew an index of where each other page's results now live | keep |

Left open, with the reason:

- **`extensions/chia.rst` has regrown to 929 lines** (454 at the restyle) from
  a day of real feature work on 2026-09-24/25 — the runbook, the two derived
  guards, the candidate-proposed configuration. It is still in template order
  with Quick start in the first screenful, so this is new content rather than
  undone restyling, but it is over the ~400-line threshold again and its
  Reference section is now the largest part of it. The next split is
  `chia_runbook.rst`, and it is not done. `act.rst` (557 → 690, the TOSA front
  end) and `tinytpu_isa.rst` (952 → 999) regrew the same way, less far.
- The threshold itself wants restating. A page splits when its *Reference*
  section stops being skimmable, not at a line count — `gemmini_comparison.rst`
  is 962 lines and skims fine because its reference is tables.

## Pages that do not exist yet and should

Two of these four now exist: **the SystemC emitter** is `backends/systemc.rst`
(`4a089373`, 2026-09-24) and **the PD / ASIC flow** is partly
`backends/asic_manifest.rst` (`0f6bdcd3`), which covers the manifest but not
preflight or the settings snapshot. **The workload suite** is
`designs/workload_suite.rst` (`46e73b39`). **The CHIA loop as a tool** remains
the open one: `extensions/chia.rst` now has a Quick start and a runbook, but
the page is still organised around what the loop found.

- **The PD / ASIC flow** — preflight, the documented command sequence, the
  settings snapshot, what is committed and what stays on scratch.
- **The SystemC emitter** — currently only reachable through the Catapult page.
- **The CHIA loop as a tool** — how to run it, what it costs, what the guards
  are. The existing page is about what it *found*.
- **The workload suite** — how to add a model, what maps and what does not.
