# Framework Traps: Checks That Silently Pass

This page covers one kind of fact: **the verification method itself fails without reporting an error**. These cases are more dangerous than ordinary bugs: tests are green and the IR looks correct, but the conclusion is false.

Each entry includes a discriminating experiment, because these problems cannot be found by reading code alone.

---

## 1. `pytest.raises(..., match=...)` Can Match the Assertion's Own Line

Write a test that asserts an error message and it may pass **regardless of what the message says**.

### Mechanism

`match` was **not skipped**; it ran and **matched successfully** because the regular expression found itself in the message.

`PyptoProError.__str__` (`pypto_pro/_errors.py:560`) returns a rendered form rather than plain diagnostic text: the first line + `--> file:line:col` + **source excerpts spanning two lines before and after the error** (rendered at `_errors.py:606`; the window is defined by `_PRIOR_CONTEXT_LINES` / `_FOLLOWING_CONTEXT_LINES` at `:370-371`). The source comes from `source_lines`, and `runtime/kernel.py:127-129` passes in the **entire file** (`f.read().split("\n")`), not just the lines containing the kernel.

When the test's `@pl.jit` kernel is written in the test file, the span's filename is that test file. The ±2-line window can extend beyond the end of the kernel and copy the following line into the message:

```python
    with pytest.raises(InvalidArgument, match=r"ZZZZ_IMPOSSIBLE"):
```

pytest applies `re.search` to `str(exc)` (`_pytest/raises.py:478` → `_code/code.py:467`), so the pattern matches its own source text.

### Discriminating Experiment: One Blank Line Makes the Difference

```python
        pl.gemm(t, t, t, l0a=t, l0b=t, nonsense_kw=1)      # line N
                                                            # N+1
    with pytest.raises(InvalidArgument, match=r"ZZZZ_IMPOSSIBLE"):   # N+2, inside the window
```
→ **PASSED** (incorrect)

With the same code, move the assertion down one line to N+3 (outside the window) → **correctly FAILED**.

**So “raising directly fails, but going through the parser passes” is an illusion**—it depends on the code's **layout**, not on the exception. Put the kernel in a helper function and the `with` line is far from the span, so the test behaves normally again. This is also why the first guess (a custom `__str__` prevents pytest from reading the message) was completely off track.

### Scope (Corrected After Remeasurement)

The repository has **547** uses of `pytest.raises(..., match=...)` across 84 files; **458** are under `pypto_pro` (`ut/` 440, `st/` 18). Non-Pro tests are unaffected.

Dynamic measurement (replace `_pytest.raises.stringify_exception`, run all of `ut/pypto_pro`, and count 644 matches against `PyptoProError`):

| Category | Count | Meaning |
| --- | ---: | --- |
| **Currently vacuous** | **1** | The pattern matches neither the diagnostic nor the hint; it matches only the quoted source. |
| **Potential** | **50** | The current assertion is correct, but the pattern also appears in the quoted source—**a diagnostic change could make it silently keep passing**. |
| Matches a hint | 6 | `.message` does not contain the hint, so it is included in the initial scan, but the assertion is **valid**. |

**The previously recorded total of 54 combined all three categories.** The accurate summary is “1 currently vacuous and 50 potential.”

The only currently vacuous case is the `[set]` argument in `test_buffer_manager.py:1000`: the parser fails earlier with `Unsupported closure variable type: set`, before reaching the `mutex_ids` container check. Both errors are `InvalidType`, so the exception type cannot distinguish them either. Fixed on 2026-09-25: closure evaluation accepts only `int / float / bool / list / tuple / IR expressions`, so neither set nor str **can reach that check**; using them as examples of “invalid container types” is incorrect.

### Guard Added

`python/tests/ut/pypto_pro/conftest.py` replaces `_pytest.raises.stringify_exception` and **strips the quoted source block** before matching, while preserving the message, the `--> location` line, and the hint. This makes `match=` mean “match the diagnostic,” which is what each test author intended.

- All 50 potential cases become safe, **with no need to edit them individually**.
- The one vacuous case immediately becomes a loud failure (measured: this exact test fails after adding the guard).
- New tests are safe by default.

Five self-tests in `test_match_guard.py` pin down the guard itself—**an unchecked guard could silently stop working, bringing back all 50 cases without anyone noticing**.

**There are another 18 uses of `match=` under `st/pypto_pro` outside the guard's coverage** (they use a different `conftest.py` tree). They have the same pattern and remain to be addressed.

### This Is a pypto Issue, Not a pytest Issue

pytest has always matched against `str(exception)`; this documented behavior did not change in pytest 9. pypto makes `str()` unsuitable for regex matching by folding excerpts from arbitrary user files into the exception's string form. **The non-Pro side of pypto does not have this issue**: `pypto/frontend/parser/diagnostics.py:163-211` prints source context to stdout instead of appending it to the message.

### Recommended Form

In order of reliability:

1. **`excinfo.value.message`** — `PyptoProError.__init__` stores plain diagnostic text, without a first-line marker, a location, or a source excerpt. It is exact and requires no regex.
2. **`check=lambda e: "..." in e.message`** — use this when you want the check inside the `with` statement.
3. **`match=r"^.*<text>"`** — without `re.MULTILINE`, `re.search` anchors `^` only at position 0 and `.` does not cross `\n`, so matching is limited to the first line. **This fails if someone passes a precompiled pattern with `re.M` / `re.S`.**

The `conftest` fixture cannot fix this: `pytest.raises` itself chooses which string to match, and the test's own source really is in that string.

## 2. ST Tests Without a `soc` Marker Are Silently Dropped on A5

If a new test under `python/tests/st/` lacks `@pytest.mark.soc("950")`, the root `conftest.py:296-310` treats it as a 910 test:

```python
soc_marker = item.get_closest_marker("soc")
if soc_marker is None:
    supported_socs = ["910"]        # default: 910
```

It does not match an Ascend950 card, so the test is dropped, **with no skip and no deselected count**. The only output is:

```
collected 5 items
============================= 14 warnings in 1.96s =============================
```

`--collect-only` sees 5 tests, but a normal run executes none. It is easy to misread this as “an environment problem” and investigate elsewhere.

**Recommended form:** module-level `pytestmark = pytest.mark.soc("950")`.

---

## 3. IR Comparison Cannot Verify Transposes or Stripes

`pl.reinterpret` is a compile-time view declaration and **produces no instructions**. A kernel with `trans_a` / `trans_b` therefore has exactly the same op sequence as one without them. Comparing op sequences for equivalence will report “pass” without checking whether the transpose is correct.

Likewise, an incorrect column-stripe address in split-N does not raise an error; it simply writes somewhere else.

**Recommended form:** directly assert the shape and layout of the tile read by `move` when checking a transpose; validate numerical correctness on the board against a reference. ascriptor recorded the on-board behavior of an incorrect transpose (D-122): the result was **completely unrelated** to the reference, with no error.

---

## 4. Narrowing and `compact` Must Be Used Together or Elements Shift Silently

`set_validshape` narrows a tile to its valid region, but **the data remains arranged in the full physical layout**. Data is contiguous within the valid region only when `compact=1`. If these disagree, matmul follows `valid_shape` while reading data laid out at the full shape—every element lands at the wrong position.

Measured on Ascend950PR with a 48×40 block in a 64×64 tile, compared against `torch.matmul`:

| Configuration | max\|diff\| |
| --- | ---: |
| Neither compact | 56.0 |
| Only acc `compact=1` | 47.4 |
| Only l0a/l0b `compact=1` | 62.5 |
| **Both L0 and Acc use `compact=1`** | **1.7e-05** ✓ |

The most deceptive part is that **the region outside the valid area is zero under all four configurations**—the boundaries are always correct, and only values inside the region are wrong. A seemingly reasonable check for out-of-bounds writes will not catch this.

### But “Everything Needs `compact`” Is the Wrong Conclusion

The table above measured l0a and l0b as a pair and led to the conclusion that “all L0 tiles need compact.” The actual rule is in the official documentation (the `matmul_m_tail` comment in `TileType.md`):

> L0A Buffer: matmul always reads using a compact layout; `compact=1` makes `move` write with the same stride.
> L0C Buffer: matmul always writes using a compact layout; `compact=1` makes `store` read with the same stride.

Thus, **only tiles that carry a narrowed axis need compact**, not every tile:

| Dimension | Narrowed and needs compact | Leave unchanged |
| --- | --- | --- |
| `m` | l0a `[m,K]`, acc `[m,N]` | **l0b** |
| `n` | l0b `[K,n]`, acc `[M,n]` | **l0a** |
| `k` | l0a `[m,k]`, l0b `[k,n]` | **acc** (K is reduced and does not affect the output shape) |

The official `matmul_m_tail` sets `compact=1` only for `a_l0a` and `c_l0c`, not for `b_l0b` or either L1 tile; it also calls `set_validshape` only twice. This was verified on board: with only `m=` specified and `l0b` declared exactly as in the aligned kernel (neither compact nor narrowed), the correct result was 1.9e-05.

**A check requiring every participating tile to be compact would incorrectly reject this correct kernel.**

### Do Not Narrow on the L1 Side

- Documentation: whether compact is configured on an L1 Buffer “does not affect the result”; without compact, “`set_validshape` has no effect on the L1→L0 transfer path.”
- In official examples, L1 narrowing always happens **before** `pl.load`; it decides how much to transfer from GM, which is the load's job. Narrowing L1 after matmul is a no-op and leaves a descriptor that has not been restored.

### `compact` Has Opposite Requirements Depending on Whether Narrowing Happens Before or After matmul

Two examples in the same document use opposite configurations, and both are correct:

| Example | When narrowing happens | L0C compact |
| --- | --- | --- |
| `matmul_m_tail` | **Before** matmul (matmul writes to the narrowed range) | **Must be `1`** |
| `matmul_wide_alias` | **After** matmul, only to limit how much `store` transfers | **Must be omitted** |

The criterion is: **has narrowing taken effect when matmul runs?** If so, compact is needed to keep both sides consistent. If not, data remains in the full layout and `store` must read using the full stride. Do not mix these two forms.

### Two More Rules

- With `compact=1` **combined with `phase` and a tail block, L0C must also use `compact=1` or it may hang** (`TileType.md:47`).
- `compact` **does not change buffer allocation size; it changes only the layout inside it**. Therefore, splitting one buffer into address-based stripes (as in split-N) is incompatible with compact: addresses are calculated from the declared stride and no longer point to those stripes after compaction.

### Two Traps When Reading `compact`

- It is stored at `tile_type.hardware_info.compact`, **not on the tile type itself**.
- The value is returned as an `ir.CompactMode` enum and **does not compare equal to the `1` written in the kernel**. Compare with `ir.CompactMode.normal` (`null`=0, `normal`=1, `row_plus_one`=2).

Also, the parser rejects an explicit `None` in `pl.TileType(..., compact=None)` (`InvalidType`). Therefore, an inline helper cannot use `compact=None` as its default: either omit this keyword argument or define two versions.

`valid_shape=[-1, -1]` is prominent in official examples, but it is the default (`valid_shape=None` is equivalent to dynamic mode `[-1,-1]`); `set_validshape` works even when it is omitted.

## 5. `memref_id` Is a Spurious Difference When Comparing IR Across Kernels

`memref_id` **increases globally within a process**. A second kernel parsed in the same process will always receive IDs different from the first, flooding the diff with false differences and hiding the real ones.

**Recommended form:** remove `memref_id` and SSA variable names before comparing; retain only op names and the `pipe` / `mutex_ids` values of `mutex_*` operations.

## 6. The MX Scale Plane Address Is Derived; Scalar `addrs` Can Silently Produce Wrong Results

The tile in `ScaleLeft` / `ScaleRight` (called a **plane** by ascriptor) cannot use an arbitrary address: it must equal the **paired L0 data tile address `>> 4`**. Hardware `mad_mx` does not take scale as an explicit operand; it finds it implicitly through the SFractal layout, which means through the address.

### Why the Framework Did Not Catch It: the Check Saw Only Slot 0

The failure described below is a slot-coverage defect in the recorded framework version. Distinguish it from a missing check or an early return for nonconstant addresses by inspecting the operands actually received by the validator.

`_check_mx_scale_tile` in `pypto_pro/ir/op/block_ops.py` checks four things: the dtype is `FP8E8M0`, the memory space is ScaleLeft/ScaleRight, the shape is derived from `_MX_GROUP_SIZE`, and the **address**:

```python
expected_scale_addr = data_addr.value >> _MX_SCALE_ADDR_SHIFT   # = 4
if scale_addr.value != expected_scale_addr:
    raise InvalidVal(f"{op_name}: {scale_name} address must equal {data_name} address >> 4, ...")
```

There really is a gate before it (the comment beginning at `block_ops.py:4383`):

> Skip auto-allocated tiles or **non-constant addresses**, which cannot be proven here.

**But this gate never fires in this case.** Measurement on 2026-09-26 counted all 2,458 cases in `ut/pypto_pro`: the branch that skips because “the address is not a `ConstInt`” was hit **0 times**.

The actual mechanism is different: **for the operand returned by `.next()`, `.type.memref.addr` is the slot-0 `ConstInt`**, not a runtime selection. Attach a probe to `_check_mx_scale_tile` and print what it receives:

```
--- 2 slot(s): 4 calls to _check_mx_scale_tile
    is_left=True   data addr ConstInt=0x0   scale addr ConstInt=0x0
    ...
```

The check therefore **ran normally and passed**: it compared `0x0 >> 4 == 0x0`, meaning it **checked only slot 0**. Slot 1 and later were never inspected, and slot 0 happened to satisfy the condition unconditionally.

Distinguish two ways to make a mistake; their outcomes are completely different. This was measured with the new parser-side check disabled to isolate `block_ops` behavior:

| Declaration | With `block_ops` alone |
| --- | --- |
| 2 slots, correct addresses | Pass ✓ |
| 2 slots, incorrect **base address** (aligned but off by `0x100`) | **Raises `InvalidVal`**—slot 0 already fails |
| 2 slots, incorrect **slot spacing** (scalar form `[0x0, 0x80]`) | **Silently passes** ← blind spot |
| 1 slot, incorrect base address | **Raises `InvalidVal`** |

**The precise blind spot is “correct base address, incorrect slot spacing.”** The semantics of `addrs=<scalar>` guarantee a correct base address and derive slot spacing from the tile's own size—**so this form hits exactly that blind spot**, not by coincidence.

Two pieces of evidence previously cited incorrectly are clarified here: `_errors.py:412` (*"ScaleLeft and ScaleRight never reach the check"*) refers to the **capacity** check, not this one; the branch described by the comment at `block_ops.py:4383` exists but is not the cause here.

### Where the Trap Is

`make_tile_group(addrs=<scalar>)` means “start here and place tiles contiguously using **their own** size.” The plane's required slot spacing is the **data** tile's size `>> 4`, which differs:

| Data dtype | Data/block | Plane/block | Slot 1 from `addrs=<scalar>` | Required by hardware | Result |
| --- | --- | --- | --- | --- | --- |
| FP8 | 4096 | 128 | `0x80` | `0x100` | **Silently wrong** |
| FP4 | 2048 | 128 | `0x80` | `0x80` | Correct by coincidence |

**The same form is silently wrong for FP8 and correct for FP4.** “I copied the FP4 example” is therefore not evidence.

### Discriminating Experiment (2026-09-26, Ascend950PR, MXFP8 128³)

Same operands on the same card; only the plane's `addrs` changed:

| Form | Result |
| --- | --- |
| `addrs=[0x0, 0x100]` (= data `>> 4`) | **Bitwise identical** to the reference |
| `addrs=0x0` (scalar → `[0x0, 0x80]`) | `max\|diff\|` 138.6, output magnitude capped at 252.25, 11,143 of 16,384 values wrong, **no error or warning** (correct base, wrong spacing; exactly the blind spot) |
| Only one plane slot provided (data still has two slots) | Incorrect output **bitwise identical** to the previous row |

The last row shows the error is deterministic; it depends only on what hardware reads at that derived address.

**The symptom is hard to recognize.** The hypothesis “half the scales become zero and contribute nothing” fails its control: the wrong output matches neither the reference nor “computing only the first K tile,” and it is even farther from the latter (202.5 > 138.6). **It is a plausible-looking wrong result**, not an obvious “half the values are missing” pattern.

### How to Prevent It

**The framework fix is in** (`pypto` branch `fix/mx-plane-slot-addresses`, commit `a5c48ce63`): for every `matmul_mx` / `matmul_mx_acc`, the parser compares the slot addresses of both tile groups **slot by slot**, instead of inspecting only the slot-0 address carried by the operand. The “incorrect slot spacing” case in the table above now raises `InvalidVal`. Users of an older environment can still get silent wrong results, so the guidance below remains.

- **Use an explicit address list** for the plane; do not pass a scalar.
- In tests, assert `plane.addr == data.addr >> 4` for every slot, reading addresses from parsed IR (`ir.MemRef(ir.MemorySpace.ScaleLeft, <addr>, <size>)`). See `test_the_plane_addresses_pair_with_their_data_tiles` in [`examples/matmul/fundamental_usage/matmul_mx/test.py`](../../examples/matmul/fundamental_usage/matmul_mx/test.py).
- **Two slots are required to expose the bug**: with one slot, the plane address is always 0 and the rule reduces to `0 >> 4 == 0`, so the issue is invisible.

**The root cause is in the framework, not in user code.** The check and derivation were correct, but they ran on the op, where a multi-slot operand exposes only the slot-0 address. The fix is to check where all slots are visible: trace the operand back to its tile group on the parser side (`_gemm_require_compact` is an existing precedent). See “Maintenance and regression” in [api/gemm-mx.md](../api/gemm-mx.md).

The related-check scope is in “A multi-slot check cannot use slot 0 as a proxy for the address” in [diagnostics/check-coverage.md](check-coverage.md). In that source audit, address was the only per-slot property and only four op-time sites read it; do not infer that every property check has the same defect.

## 7. `auto_mutex=True` Inserts No Synchronization for a Bare `pl.make_tile`

`@pl.jit(auto_mutex=True)` sounds like “leave synchronization to the framework.” It applies only to the `mutex_ids` of `pl.make_tile_group`. When a tile comes from bare `pl.make_tile`, there is no mutex ID to infer, so **the framework inserts nothing and raises no error**.

### Discriminating Experiment: Parsed IR, No Device Required

Keep the same load → fillpad → store sequence and change only the synchronization form; after parsing to `ir.SectionKind.Vector`, inspect `@system.*` / `@block.*`:

| Form | Synchronization ops |
| --- | --- |
| `auto_mutex=True` + `make_tile_group(mutex_ids=[0, 1])` | **6**: three pairs of `mutex_lock_dyn` / `mutex_unlock_dyn`, around load, fillpad, and store |
| `auto_mutex=True` + bare `make_tile` | **0** |
| `@pl.jit()` + manual `sync_src` / `sync_dst` | **4**: two pairs of `sync_src_dyn` / `sync_dst_dyn` |

```python
program, _ = kernel.to_kernel_def().parse_target_program(ir.SectionKind.Vector)
text = pypto_pro.ir.python_print(program.get_function(kernel.__name__))
ops = re.findall(r"@(system\.[\w.]+|block\.[\w.]+)", text)
```

### On-Board Symptom: Stable, Regular, and Wrong

The MTE2 load, V fill, and MTE3 store race; the read sees **bytes left by a previous run at the same UB address**. On 2026-09-28, `dma_pad` produced `-2147483648` for `TilePad.zero`, `0` for `max`, and `2147483647` for `min`—three values neatly shifted by one, consistent across four same-state measurements. The pattern alone does not establish an enum-mapping defect.

**“Consistent across four runs” was misleading**: the process layout and call order were the same each time, so it was effectively one measurement repeated four times. Changing the order from `zero,max,min` to `min,max,zero` changed the answer; running only one pad in a process produced the correct result. See the full account in [`verification/evidence.md`](../verification/evidence.md) #16.

### Two Correct Forms

```python
@pl.jit(auto_mutex=True)                      # Framework-managed: tiles must come from a group
def a(x, out):
    g = pl.make_tile_group(type=t, addrs=[0x0, 0x100], mutex_ids=[0, 1])

@pl.jit()                                     # Manual: switch between pipes with a matched pair each time
def b(x, out):
    src = pl.make_tile(t, addr=0x0)
    ...
    pl.system.sync_src(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.V, event_id=0)
    pl.system.sync_dst(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.V, event_id=0)
```

**`sync_src` / `sync_dst` must be paired and use matching parameters**: the set half goes in the former and the wait half in the latter. Writing only one does not raise an error and prevents nothing. See [`examples/basic_api/datamove/dma_pad`](../../examples/basic_api/datamove/dma_pad). The primary source is `pypto/python/tests/st/pypto_pro/frontend/datacopy/test_pro_fillpad.py`.

**A tool catches this case:** `auto-mutex-needs-tile-group` in `python scripts/check.py rules <path>`.

### 7.1 A Tile Group May Still Lack Protection: Grouping Uses `mutex_ids`, Not Addresses

**First define the mutex domain; the entire argument in §7.1 depends on it:** **the mutex key is the ID itself, across pipes**. `Lock<pipe>(id)` blocks **the instruction queue for the current pipe** while waiting for token `id` to be released; `Unlock<pipe>(id)` releases the token after earlier instructions on that pipe drain. **The pipe identifies which queue is blocked; it is not part of the token** (the Semantics comment in `ir/op/sync_ops/sync.cpp`; ID range 0–31; the original wording is “per Ascend C Mutex ISASI spec”).

Four sources of evidence were reconciled on 2026-09-28. This had been an open question (“the sanitizer pairs by `(region, id, pipe)`, suggesting that pipe is part of the key”), but that interpretation is now discarded:

| Evidence | What it says |
| --- | --- |
| ISA semantics comment (`sync.cpp`) | As above—the token is keyed by ID, and the pipe determines which queue is blocked. |
| Official documentation `tile_creation_and_operations.md` | “A mutex ID identifies an on-chip buffer mutex synchronization resource, **used to coordinate access to Tiles by different Pipes**.” |
| Backend behavior | `ShouldSkipVPipeMutex` skips only when an ID is used **exclusively** for `PIPE_V`; **if the same ID is also used by another pipe, it still emits the V lock**. This condition would make no sense if mutexes were pipe-scoped. |
| On-board measurement | `get_buf(PIPE_MTE2, 0, 0)` from `make_tile_group` blocked the **MTE3** `TSTORE` from `pl.init_output` (both use ID 0): a bare tile failed 6 of 12 times; the group failed 0. **The same ID does provide mutual exclusion across different pipes** (§11). |

**The sanitizer's triple is a pairing key, not a mutual-exclusion key.** The `_check_mutex_pairing` docstring says: "every lock must be matched by an unlock on the same `(region, mutex_id, pipe)` key". This checks whether every lock is released; it says nothing about which operations exclude one another. **The two concepts share an ID field, which makes them easy to confuse.**

`auto_mutex` orders uses of a tile slot through mutex acquire/release operations. The frontend records per-tile runtime IDs and candidate groups; codegen deduplicates cross-tile IDs at runtime while retaining each tile's ownership.

`mutex_ids="auto"` plans overlapping automatic groups together and validates any manual pins in their overlap component. Manual-only declarations retain their explicit IDs. `pipeline` separately checks regions within its analysis scope through `_check_reuse_mutex_ids`; the source rule `overlapping-groups-need-shared-mutex` is a limited manual-layout precheck.

Mutex IDs are per core. Matching IDs do not establish mutual exclusion between Cube and Vector; shared cross-core buffers require event synchronization. See the [allocation contract](../api/memory-allocation.md) for automatic alias rules and explicit placement responsibilities.

**Phase exemptions are a separate case:** `matmul` / `store` / `move` / `insert` with `phase=Partial/Final` **drop the lock on the Acc tile** and let the hardware `unit_flag` take over. This is intentional, and the comments record the real cost: removing the L0A/L0B mutex causes an MTE1→cube RAW (device error 507015, measured 2026-05-29). When using `phase=`, verify that `unit_flag` covers the dependency.

## 8. `bar_*` Is an **Intra-Pipeline** Barrier; Using It to Order Across Pipelines Silently Produces Wrong Results

`pl.system.bar_mte2()` sounds like “wait for MTE2 to finish.” It drains **that pipeline itself** and does not establish ordering between MTE2 and V. **The only way to order across pipelines is with a flag:** paired `sync_src` / `sync_dst`.

A per-pipe barrier drains its own pipeline; it does not establish a dependency on another pipe. Use it when operations within that pipe must not overlap, and use the appropriate cross-pipe synchronization for a producer and consumer on different pipes. `bar_all` has the broader drain scope described below.

### Discriminating Experiment: Same Chain, Only the Synchronization in the Middle Changes

`pl.load` (MTE2) → `pl.add` (V) → `pl.store` (MTE3), with `out == x * 2` as the criterion. **Fix the initial state by immediately preceding each run with a kernel that writes known nonzero data to both UB regions**; run each form three times in separate processes:

| Synchronization in the middle | After the preceding UB-dirtying kernel | Alone in a fresh process |
| --- | --- | --- |
| Two pairs of `sync_src` / `sync_dst` | **3/3 correct** | Correct |
| Two `bar_all()` calls | 3/3 correct | Correct |
| `bar_mte2()` + `bar_mte3()` | **3/3 wrong** | Correct |
| No synchronization | **3/3 wrong** | Correct |

Read these two observations together:

1. **`bar_mte2()` + `bar_mte3()` produces the same result as no synchronization.** On failure, it reads the data left in `dst` by the previous kernel; in other words, the store moves the data before V has finished computing it.
2. **The last column is all “correct.”** Even no synchronization works in a fresh process. A single run in a fresh process would never expose the missing synchronization, which is why this repository's criterion does not rely on numerical input values.

`bar_all()` works because it drains **all** pipelines. It is a “global barrier,” not a barrier for one pipeline, and has a different cost.

### Actual Uses of `bar_*` (`easyasc`, the Previous Generation of the Same Hardware)

- `kernels/a2/vec_only/mxfp8e5m2.py:185` — `bar_mte3()` is at the **end** of the loop body, ordering MTE3 after itself: let this iteration's output land before the next iteration reuses the same UB scratch.
- `kernels/a2/vec_only/dflash_anchor_mask/dflash_anchor_mask_kernel.py:180` — the comment is explicit: automatic synchronization builds dependencies from DBuff slots, cannot see across two loops, so `bar_all()` forces a drain. “**The barrier drains it.**”

Both examples “drain one pipeline (or all pipelines)”; neither makes “A wait for B.”

Using a Cube-only barrier in Vector is rejected during parsing (`3322ea999`); see [diagnostic entry points](messages.md#parse-error-localization-and-regressions) for valid sides and the location regression.

**Example policy:** `examples/basic_api/datamove/move_sync` **deliberately excludes `bar_*`** (user decision on 2026-09-28). An example that “uses `bar_*` for synchronization” would teach the wrong lesson; correct usage is clear only in a case where automatic synchronization analysis cannot see the dependency. The rationale is in §6 of that example's `test.py`.

**That case was later found**, see #9 below: two `pl.store` operations on MTE3 write overlapping GM ranges. A flag cannot be used (`set_pipe != wait_pipe` blocks MTE3→MTE3), leaving `bar_mte3()` as the only option. The barrier in `examples/basic_api/datamove/strided_view` is the only use of `bar_*` in this repository.

## 9. Two `pl.store` Operations Writing Overlapping GM Regions Are **Not Guaranteed to Follow Program Order**, Even on the Same Pipeline

The framework cannot see that two GM views point to the same memory—**there is no alias check, error, or warning**. The intuition that operations on “the same pipeline follow program order” is also wrong.

### Discriminating Experiment

Use a `16 × 16` GM output. View A covers rows 0–7, and view B covers 8 rows starting at row 4 (overlap is rows 4–7). Call `pl.store` through view A and then view B; both operations use MTE3. Run 20 times (5 runs in each of 4 processes):

| Between the two stores | Which store wins the overlapping region |
| --- | --- |
| Nothing | **B (later write) 14 times, A (earlier write) 6 times**; the outcome can change between consecutive runs in the same process. |
| One `pl.system.bar_mte3()` | **B all 20 times**, consistent in cold and warm processes, with or without warmup. |

The two non-overlapping regions and the trailing sentinel are stable in both cases—**only the overlapping region is nondeterministic**. The kernel is not “completely broken”; the completion order of the two stores is undefined.

### Why a Barrier Is the Only Option

`sync_src` / `sync_dst` require `set_pipe != wait_pipe` (`system_ops.py::validate_sync_path`), so an MTE3→MTE3 flag is rejected during parsing. **This is a valid use of `bar_*`**: the goal is “prevent parallel execution within a pipeline,” not “make A wait for B.” This does not conflict with #8, which describes using `bar_*` for cross-pipeline ordering; this case serializes work within one pipeline.

### How to Avoid It

- **Avoid overlap when possible.** Only the author knows whether the overlap is intentional; the framework cannot help.
- If the overlap is intentional, add `pl.system.bar_mte3()` (or the barrier for the relevant pipeline) between the two writes.
- **Do not include values from the overlapping region in the criterion** unless the writes have been ordered. The unbarriered “later write wins” assertion failed 6 of 20 recorded runs. Retain exclusive regions and sentinels as controls; require the explicit barrier before asserting which overlapping write wins.

See verification cases 5 / 6 / 7 in [`examples/basic_api/datamove/strided_view`](../../examples/basic_api/datamove/strided_view).

---

## 10. Five Silent Failures Measured in the Support Chapter (Data in ../api/support.md)

All five are cases where “the verification method looks right, but the conclusion is false.” **Maintain only one full table and set of criteria**, in [`api/support.md`](../api/support.md), where every row is more complete (including the entry-value column, the six-row CTRL[60] truth table, and the 16-block column). This page keeps only “what can bite you and where”:

| Fact | Where it fails silently | Data |
| --- | --- | --- |
| `reset_ctrl_spr` restores the **hardware default**, not the kernel entry value | “Restore the original state” changes bit 48 from 1 to 0, switching floating-point conversion from non-saturating to saturating | §2 |
| The CTRL bit set by `set_saturation_flag` is not the flag value (FLOAT / FLOAT8 / CAST have inverted polarity) | `get_ctrl_spr` reads 0 and appears ineffective, leading to suspicion of an interface that works correctly | §3 |
| CTRL[60] is 1 at entry, so `vf.astype(saturate=)` is ineffective by default | Passing `saturate=` has no effect; the first guess is “the argument was passed incorrectly” | §4 |
| `get_subblock_num()` depends on whether the kernel has a cube section | In a pure vector kernel, `subblock_idx` is always 0; “split work between subcores” sends every block to the same half | §5 |
| Multiple blocks writing their own fields in GM can lose writes when the stride is under 64 bytes; `dcci` cannot fix it | Only 2–3 of 8 blocks remain, and the result varies | §7 |

## 11. `pl.init_output` Overwrites 8 KB of UB Starting at `0x0` and Emits No Barrier After `TSTORE`

The docstring says only “Must be called before Tile buffer allocation (`make_tile_group` etc.).” **The behavior has a specific address and width, but the documentation gives neither number.**

Read the codegen in `backend_cce_block_out_ops.cpp:548`; it generates CCE like this:

```cpp
Tile<Vec, float, 1, 2048, ...> tmp(1, 2048);
TASSIGN(tmp, 0x0);                    // ← temporary tile fixed at UB 0x0
get_buf(PIPE_V, 0, 0);
TEXPANDS(tmp, -1.0f);                 // ← fills the full 8 KB before SetValidShape
rls_buf(PIPE_V, 0, 0);
get_buf(PIPE_MTE3, 0, 0);
for (...) { tmp.SetValidShape(1, chunk); ...; TSTORE(gm, tmp); }
rls_buf(PIPE_MTE3, 0, 0);
// no barrier
```

### Overwrite Width

`chunk_elems = (256 * 32) / sizeof(T)` (`MAX_REPEAT_TIMES × ONE_BLK_SIZE`):

| dtype width | Chunk elements | Bytes overwritten |
| --- | ---: | ---: |
| b8 / b16 / b32 | 8192 / 4096 / 2048 | **8192** |
| b64 | 64 (clamped) | 512 |

**`TEXPANDS` runs before `SetValidShape`**, so even `size=128` overwrites 8 KB—**it is independent of `size=`**.

The boundary was measured on board (verification 11 in `examples/basic_api/datamove/init_and_insert`): tiles at `0x0` and `0x1000` were filled with the fill value, while **`0x2000` remained intact**.

### Second Half: No MTE3 → MTE2 Synchronization After `TSTORE`

The docstring guarantees **V→MTE3** and separately warns that cross-core visibility requires the caller to use `sync_all`. **It says nothing about later writes to UB at `0x0`.** Thus, while MTE3 is still transferring data from UB `0x0` to GM, a `TLOAD` (MTE2) that writes to the same UB region may happen concurrently—**a WAR race**.

**Users cannot fix this themselves**: the temporary tile is internal to the framework, so users neither know that it exists nor where it is.

### Reproduction: Widen the Window to Expose It

`size=65536` → 32 rounds of `TSTORE`, 8 KB per round, immediately followed by a `pl.load` into the same UB region. Each group ran 12 times:

| Kernel | Failures in 12 runs | Wrong values |
| --- | ---: | --- |
| `@pl.jit()` + bare `make_tile(0x0)` | **6** | 128 or 384 values, **all elements from input `x`** |
| Same + `pl.system.bar_all()` | 0 | — |
| Same + `pl.system.bar_mte3()` | **6** | Same |
| `auto_mutex=True` + bare `make_tile(0x0)` | **6** | Same |
| `@pl.jit()` + `make_tile_group(addrs=0x0, mutex_ids=[0])` | 0 | — |
| `auto_mutex=True` + the same group | 0 | — |
| `@pl.jit()` + bare `make_tile(0x2000)` | 0 | — |

**The decisive factors are the address and how the tile is declared, not call order**—every row above follows the legal order required by the docstring (`init_output` before tile declaration), yet still has 6/12 failures. The ineffectiveness of `bar_mte3()` confirms §8.

### Why `make_tile_group` Fixes It but `auto_mutex` Does Not

Dump `build/<kernel>__a5__d0/tk_none/kernel.cpp` and inspect around the user's `TLOAD`:

| Kernel | Around `TLOAD` |
| --- | --- |
| Bare `make_tile` + `@pl.jit()` | `rls_buf(PIPE_MTE3,0,0) TLOAD` — **no `get_buf` at all** |
| Bare `make_tile` + `auto_mutex=True` | **Exactly the same as the previous row** |
| `make_tile_group` + `@pl.jit()` | `rls_buf(PIPE_MTE3,0,0) get_buf(PIPE_MTE2,0,0) TLOAD rls_buf(PIPE_MTE2,0,0)` |
| `make_tile_group` + `auto_mutex=True` | **Exactly the same as the previous row** |

`get_buf`/`rls_buf` are cross-pipeline buffer-ownership tokens. The `TSTORE` from `init_output` uses buffer ID `(0, 0)`, and the tile declared by `make_tile_group` also uses `(0, 0)`, so the two are ordered.

**The identical pairs of rows locate the mechanism: `make_tile_group` emits `get_buf`; toggling `auto_mutex` has no effect in this probe.** Keep those two variables separate when attributing the result; see [evidence assessment](../verification/evidence.md) #19.

### Two Fixes and Their Costs

| Fix | Cost |
| --- | --- |
| `pl.system.bar_all()` | Drains all pipelines once, removing all overlap in the chain after `init_output`. |
| Use `make_tile_group` | Orders only accesses to the same buffer ID, but requires managing `mutex_ids` and placing the tile in a group. |

Static gate: `init-output-needs-a-drain` in `scripts/_checks/init_output.py`; it checks for “`init_output` + bare `make_tile` + no `bar_all`.” **It does not check addresses**—kernels with sufficiently distant addresses use the `反例：` waiver.

### Resolution: Documentation Only (User Decision, 2026-09-28)

There were three candidate fixes, each with a different cost:

| | Fix | Why it was rejected or selected |
| --- | --- | --- |
| 1 | Unconditionally emit a barrier after `TSTORE` in `init_output` | Correct, but every call pays for a full-pipeline drain; the common `make_tile_group` path is already safe, so that drain would be wasted. |
| 2 | Include user bare tiles in the buffer-ID protocol | Changes behavior and affects every use of `make_tile`. |
| 3 | **Update only the docstring** | **Selected.** No behavior change and zero cost. |

Implemented in `pypto` `dev/pro-agentkit-build` at `428f3e8d9`: the `_api.py` `init_output` docstring gained a `.. warning::` describing the fixed address 0, the width `256*32/sizeof(dtype)`, the fact that “filling the full width is independent of `size=`,” and the two fixes (including that “`auto_mutex` is not what makes this work”). There is **no page for `init_output` under the official `docs/`**; this docstring is its only documentation.

**Option 3 does not prevent silent wrong results; users must take care when writing kernels.** The practical defense is therefore two documentation references and one gate:

| Defense | Location | When it applies |
| --- | --- | --- |
| Docstring `.. warning::` | `pypto/python/pypto_pro/language/_api.py` | When looking up how to use this API |
| Usage guidance | `references/api/constraints.md`, “UB tile adjacent to `pl.init_output`” | Before writing the kernel |
| Static rule | `scripts/_checks/init_output.py` | After writing, before board execution |

**The gate is the only one of the three that does not depend on someone remembering to read.** Always run `python scripts/check.py rules <path>` before running a newly written kernel that uses `init_output` on the board.

This also explains the mechanism behind the open issue in §5.8.1 of `references/verification/board-runs.md` (one `init_and_insert` criterion failed once in the full suite): that kernel's `src` was at `0x0`, exactly overlapping the temporary tile. **It still could not be reproduced at the original size** (100 clean consecutive runs in isolation)—the window contained only one 512-byte `TSTORE`; widening it to 32 rounds produced a stable 6/12 failure rate.

## Empty consumer loops need their own reuse check

When a tile load precedes a runtime row/segment loop, include executions where this worker owns no rows. They exercise a different dependency path from nonempty iterations. Check the generated loads, tile aliases, mutex IDs and `get_buf/rls_buf` on both paths; `auto_mutex=True` alone is not proof that the intended physical lifetime was described.

If guarding an unused load removes a failure, that alone does not isolate an MTE2 write/write reordering mechanism. Do not infer that all consecutive same-pipe loads are unordered, or attribute the failure to a compiler omission without checking its emitted dependencies.

A useful guard is the actual per-worker trip count, rather than only the group's global nonempty condition:

```python
first_owned = row_start + worker
if first_owned < row_end:
    ws_tile = ws.next()
    pl.load(ws_tile, weight_scale, [group, 0])
    for row in pl.range(first_owned, row_end, workers):
        ...
```

Apply this only when the skipped path owes no work, initialization, cursor advancement or cross-core token to another stage. Test consecutive empty groups, nonempty groups smaller than the worker count, active work following skipped work and repeated launches. Removing an unnecessary load can eliminate a trigger; it does not by itself prove the original root cause. Rotating two slots is not a general synchronization fix, and a barrier must correspond to a demonstrated dependency and its actual pipe.

The [grouped GEMM regression](../../examples/algorithms/grouped_gemm_swiglu_quant/test.py) changes runtime group boundaries at fixed tensor addresses, reuses poisoned outputs, and exercises both empty groups and nonempty groups with idle workers. The correct control guards actual local work; do not put a known unsafe no-consumer preload into the default on-board test suite merely to demonstrate the failure.
