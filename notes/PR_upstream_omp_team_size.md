# [Simulator][Dataflow] Size the OpenMP team to the section count, not the core count

Branch: `upstream-omp-team-size` (2 commits' worth of change in one commit, on top of `upstream/main` @ 8bafb0dc)

## The bug

`_inject_omp_parallel_sections` in `allo/backend/simulator.py` wraps the PE calls
in an `omp.parallel` > `omp.sections`, and leaves the team size to default — which
is the core count.

A PE blocked on a stream spins inside its section. It does not yield. So if the
OpenMP team is smaller than the section count, the runtime never starts the
sections that would unblock it, and a region with more `df.kernel` instances than
threads deadlocks.

Three things make this much worse than an ordinary deadlock:

- **It is silent and unbounded.** No message, no timeout, no indication of which
  process is blocked on which channel. The simulator simply never returns.
- **Deep FIFOs mask it.** With a stream deep enough that producers run to
  completion before anyone has to block, the same region works. So the bug
  appears and disappears with FIFO depth and with the host's core count, which
  makes it present as a *design* bug in the user's dataflow graph rather than a
  simulator one.
- **The threshold is invisible.** It is the kernel-instance count, which the user
  never writes down — it is the product of the `mapping=[...]` dimensions summed
  over the kernels in the region.

## Two independent discoveries

This was found twice, from opposite directions, by unrelated projects:

- **@chhzh123**, on the SPMW branch (`a03edb85`, 2026-09-05): 56 PEs for an 8x8
  FEATHER on a 48-core host.
- **This author**: a 22-PE instruction-programmable TPU at `OMP_NUM_THREADS=8`.
  It cost multiple sessions, because the symptom is indistinguishable from a
  genuine dataflow deadlock in one's own design.

Two projects hitting the same wall and each carrying a local patch is the
argument for fixing it here rather than in each fork.

## The fix

The team is now `len(pe_call_define_ops)` — as many threads as there are
sections — passed as `num_threads` on the `omp.parallel`.

```python
num_threads = arith_d.ConstantOp(
    IntegerType.get_signless(32), len(pe_call_define_ops), ip=omp_ip
)
omp_parallel_op = openmp_d.ParallelOp(
    [], [], [], [], num_threads=num_threads.result, ip=omp_ip
)
```

The diff is @chhzh123's; this PR adds the docstring and the test.

Oversubscribing the machine is correct here: these are not compute threads
competing for cores, they are coroutines that spend their time blocked on
channels. The simulator's concurrency requirement is a *correctness*
requirement, not a performance one.

## The test

`tests/dataflow/test_omp_team_size.py` — a 16-stage relay chain over **depth-1**
streams, run under `OMP_NUM_THREADS=2`.

Depth 1 is the point: no stage can run ahead, so every stage must block and the
deadlock is immediate rather than timing-dependent. 16 sections against a team of
2 is far enough over the line that the result does not depend on the host.

The region runs in a **subprocess with a 120s timeout**, because the pre-fix
failure mode is an unbounded hang and a test that hangs is worse than a test that
fails. On timeout the test fails with a message naming the function and the
mechanism — the diagnosis the original bug never printed.

## Verification

Both directions, on this branch, against `upstream/main`'s own simulator:

| | result |
|---|---|
| new test, with the fix | `1 passed in 3.56s` |
| new test, fix reverted | `1 failed in 121.12s` (subprocess killed at the timeout) |
| `tests/dataflow/` with the fix | `29 passed, 3 skipped in 49.54s` |

Run bare rather than under the harness, the unfixed case was still spinning after
60s with no output.

## Still open, and not addressed here

The *diagnosis* is still absent. What cost the time was not the deadlock but the
silence: a region that hangs forever with no indication of which process is
blocked on which channel. A deadlock report — every blocked PE, the channel it is
waiting on, and that channel's occupancy — is roughly 30 lines of bookkeeping over
the stream structs the simulator already builds, and would turn this whole class
of bug from a multi-session hunt into one run. Worth a follow-up issue.

🤖 Generated with [Claude Code](https://claude.com/claude-code)

https://claude.ai/code/session_01BmrdaXYkbAqVL8kc9ikwRk
