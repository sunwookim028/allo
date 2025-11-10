# Experimental Guide: Understanding Static → RTL Mapping

This guide walks you through experiments to understand how `static` affects hardware generation.

## Experiment 1: Basic Static Behavior

**Goal**: Understand the fundamental difference between static and non-static variables.

**Steps**:
1. Navigate to `ex01_basic_static`
2. Run `make csim`
3. Observe how results change between calls

**Questions to Answer**:
- Why does the result increase in Call 2?
- What hardware represents `static_var`?
- What hardware represents `non_static`?

**Expected Observations**:
- Non-static resets each call → wire/logic
- Static accumulates → register (FF)

**Hardware Mapping**:
```
non_static: Combinational logic (wire)
static_var: Register (Flip-flop) with initialization
```

## Experiment 2: Static in Loops

**Goal**: See how static variables affect loop behavior.

**Steps**:
1. Navigate to `ex02_static_variables`
2. Run `make test_a test_b test_c`
3. Compare results across versions

**Questions to Answer**:
- Why does Version B accumulate across calls?
- How does Version C handle reset?
- Which version is better for pipelines?

**Expected Observations**:
- Version A: Clean behavior, resets each call
- Version B: Accumulates, may break pipelines
- Version C: Controlled reset, best of both

**Hardware Mapping**:
```
Version A: acc → register (within loop only)
Version B: acc → register (persists across calls)
Version C: acc → register + reset control logic
```

## Experiment 3: Static Arrays and Memory

**Goal**: Understand how static arrays map to FPGA memory resources.

**Steps**:
1. Navigate to `ex03_static_arrays`
2. Run `make test_a test_b test_c`
3. Run `make synth view` to see resource usage

**Questions to Answer**:
- Why does Version B use BRAM?
- How does initialization affect static arrays?
- What's the difference between static and non-static arrays?

**Expected Observations**:
- Small arrays → registers (if partitioned)
- Large arrays → BRAM/URAM
- Static arrays persist across calls

**Hardware Mapping**:
```
Version A: local_array → registers/wires
Version B: static_array → BRAM
Version C: static_init → registers/BRAM (with initialization)
```

**Check Resources**:
```bash
make synth
cat ex03.prj/solution1/syn/report/top_csynth.rpt | grep -E "BRAM|URAM|FF"
```

## Experiment 4: Static in Pipelines

**Goal**: Understand pipeline implications of static variables.

**Steps**:
1. Navigate to `ex04_pipelines`
2. Run `make test_a test_b test_c test_d`
3. Run `make synth` to check II values

**Questions to Answer**:
- Why can't Version B achieve II=1 easily?
- How does Version C (shift register) work?
- When is static appropriate in pipelines?

**Expected Observations**:
- Version A: Clean pipeline, II=1 achievable
- Version B: Static accumulator prevents II=1
- Version C: Shift register pattern works well
- Version D: State machine uses static appropriately

**Hardware Mapping**:
```
Version A: acc → register (pipeline-friendly)
Version B: acc → register (persists, breaks pipeline)
Version C: shift_reg → registers (shift pattern)
Version D: state → register (state machine)
```

**Check Pipeline Performance**:
```bash
make synth
cat ex04.prj/solution1/syn/report/top_csynth.rpt | grep "Initiation Interval"
```

## Interactive Experiments

### Experiment A: Modify Array Size

**Task**: Change array size and observe resource mapping

1. Edit `ex03_static_arrays/kernel.cpp`
2. Change `static_array[100]` to `static_array[10]`
3. Run `make test_b synth view`
4. Compare BRAM usage

**Hypothesis**: Smaller arrays should use fewer resources
**Verify**: Check synthesis report for BRAM usage

### Experiment B: Remove Static Keyword

**Task**: See what happens without static

1. Edit `ex01_basic_static/kernel.cpp`
2. Remove `static` from `static_var`
3. Run `make csim`
4. Compare results

**Hypothesis**: Variable should reset each call
**Verify**: Results should be identical across calls

### Experiment C: Add Pipeline Pragma

**Task**: Add pipeline to static accumulator

1. Edit `ex02_static_variables/kernel.cpp` Version B
2. Add `#pragma HLS PIPELINE II=1`
3. Run `make test_b synth`
4. Check II value

**Hypothesis**: II should be > 1 due to static dependency
**Verify**: Check synthesis report for actual II

### Experiment D: Compare Resource Usage

**Task**: Compare static vs non-static resource usage

1. Run `make synth` for both versions
2. Compare synthesis reports:
   ```bash
   grep -E "BRAM|FF|LUT" ex02.prj/solution1/syn/report/top_csynth.rpt
   ```

**Hypothesis**: Static version uses more registers
**Verify**: Count FF usage in both versions

## Understanding Synthesis Reports

### Key Sections to Check

1. **Timing Summary**
   ```
   Timing (ns):
   - Clock period
   - Clock uncertainty
   - Total latency
   ```

2. **Resource Usage**
   ```
   BRAM: Block RAM usage
   DSP:  DSP slice usage
   FF:   Flip-flop usage
   LUT:  Look-up table usage
   ```

3. **Pipeline Information**
   ```
   Initiation Interval: Cycles between new inputs
   Latency: Total cycles for execution
   ```

4. **Loop Information**
   ```
   Loop pipeline status
   Unroll status
   Dependencies
   ```

### How to Read Reports

```bash
# View full report
cat <project>/solution1/syn/report/top_csynth.rpt

# Extract specific sections
grep -A 10 "Resource" <project>/solution1/syn/report/top_csynth.rpt
grep -A 5 "Pipeline" <project>/solution1/syn/report/top_csynth.rpt
```

## Common Observations

### Observation 1: Static Variables = More Registers

**Check**: Compare FF usage between static and non-static versions
**Expected**: Static version uses more FFs
**Reason**: Static variables become registers

### Observation 2: Static Arrays = BRAM Usage

**Check**: Compare BRAM usage for large arrays
**Expected**: Static arrays use BRAM
**Reason**: Large arrays mapped to Block RAM

### Observation 3: Static Can Break Pipelines

**Check**: Compare II values with/without static
**Expected**: Static accumulator may have II > 1
**Reason**: Dependency across function calls

### Observation 4: Shift Registers Work Well

**Check**: Pipeline performance of shift register pattern
**Expected**: II=1 achievable
**Reason**: Regular, predictable pattern

## Troubleshooting Experiments

### Problem: Simulation doesn't match expectations

**Check**:
- Are static variables initialized?
- Is reset logic correct?
- Are testbench expectations correct?

**Fix**:
- Initialize static variables
- Add reset control if needed
- Review testbench logic

### Problem: Synthesis fails

**Check**:
- Are pragmas correct?
- Are array sizes reasonable?
- Are there syntax errors?

**Fix**:
- Review pragma syntax
- Reduce array sizes if needed
- Check compiler errors

### Problem: II > 1 when expecting II=1

**Check**:
- Are there static dependencies?
- Are there data dependencies?
- Are pragmas correct?

**Fix**:
- Remove static if not needed
- Check for loop-carried dependencies
- Verify pragma placement

## Next Steps

1. **Run all experiments**: Understand each pattern
2. **Modify code**: Experiment with changes
3. **Check reports**: Understand hardware mapping
4. **Compare versions**: See trade-offs
5. **Build intuition**: Connect C++ to hardware

## Questions for Reflection

After each experiment, ask:

1. **What hardware does this generate?**
   - Registers? BRAM? Logic?

2. **Why does this behavior occur?**
   - Static persistence? Reset logic?

3. **What are the trade-offs?**
   - Area vs performance? Pipelining vs state?

4. **When would I use this pattern?**
   - Shift registers? State machines? Accumulators?

5. **How can I optimize this?**
   - Different pragmas? Array partitioning? Pipeline optimization?

Happy experimenting! 🚀

