# Opaque error / error-code playbook (DEBUG_GUIDEBOOK.md §1–§7)

**Audience:** Agents working on complex PyPTO kernels under this bundle.

**Do not skip:** This file is part of the mandatory path when stuck — see **skill `pypto-orchestration-manual`'s `references/rules.md` → Zero tolerance**.

**Policy:** **`Errcode: FFFFF!`**, **`UNKNOWN`**, **`0x3FFFF`**, or other opaque / scary-looking error codes are **not** a reason to end the session, report failure, or "give up" until the **Stop Conditions** in skill `pypto-orchestration-manual`'s `references/rules.md` actually apply. **Token and turn budget are not a constraint** for this workflow: keep iterating, gathering evidence, and trying the next structured strategy until the problem is resolved or a true blocker is proven.

**What "give up" looks like (forbidden here):** stopping after only printing the error code, declaring the task impossible without consulting docs, skipping logs, or refusing to try another angle.

---

## 1. First actions (always)

1. **Capture the full message** — stderr, Python traceback, and any **`pypto-log*.log`** / device log lines. Search for `Errcode`, `ErrCode`, `F` + digits, `aicore`.
2. **Route the error code** — read `https://raw.gitcode.com/cann/pypto/raw/master/docs/zh/trouble_shooting/README.md` and open the component doc for that prefix (e.g. FUNCTION / `https://raw.gitcode.com/cann/pypto/raw/master/docs/zh/trouble_shooting/function.md` for many `F2xxxx`-style codes). For **`F21004`** / **`REGISTER_COPY`** / invalid vector tile, see **§4** below first.
3. **Append to `custom/<operator_name>/MEMORY.md` → Development & debug log** — what failed, command, hypothesis, next step. No empty "stopped" endings.

---

## 2. UNKNOWN / FFFFF / insufficient detail / silent failure

This section covers **three related failure modes**:
- An **opaque error code** (`FFFFF`, `UNKNOWN`, `0x3FFFF`) with no narrative.
- A run that **looks successful in Python** but produces wrong / zero outputs and emits **no Python-level error message** at all.
- A run that dies early on the NPU with only a terse line like `Inner Error, please contact Huawei Engineer`.

In all three cases, the **Python stderr is insufficient**. The real error is in the device slog. You must pull it into stdout, capture to a file, and search it efficiently.

### 2a. Force the device log into stdout and capture it

Set the two ASCEND env vars and re-run with stdout redirected to a log file:

```bash
export ASCEND_GLOBAL_LOG_LEVEL=0       # 0 = DEBUG, 1 = INFO, 2 = WARN, 3 = ERROR, 4 = NULL
export ASCEND_SLOG_PRINT_TO_STDOUT=1    # route slog to stdout instead of on-disk slog files
python <failing_file>.py >> result.log 2>&1
```

Notes:
- **Use `>>` (append) rather than `>`** so repeated runs accumulate evidence you can diff later.
- `2>&1` is required — some ASCEND layers emit to stderr even with `PRINT_TO_STDOUT=1`.
- `ASCEND_GLOBAL_LOG_LEVEL=0` is verbose; expect `result.log` to commonly exceed **10 MB** on any non-trivial kernel (single op, L1 shape, full NT loop).
- Run from the NPU server over `Run <file> on npu:<N>`; the env vars apply to that process.

### 2b. Search `result.log` efficiently with ast-grep

Straight `grep -n ERROR result.log` gets you the list of ERROR-lines, but **loses the structural context** around each one (the preceding `[TRACE]` breadcrumbs, the CCE file path, the stack frame). And scrolling 10 MB+ of slog by hand is a dead end.

The **ast-grep agent-skill** (`https://github.com/ast-grep/agent-skill/tree/main/ast-grep/skills/ast-grep`) gives you **syntax-aware** and **context-preserving** pattern search over large structured logs at high throughput. Install it per that repo's README before starting. It is the preferred search tool when `result.log` exceeds a few MB or when the failing pattern spans multiple lines (e.g. an `ERROR:` line followed by 10 lines of `[INFO]` device context that is the actual signal).

**Recommended first pass** (run in this order — each narrows the noise further):

1. **All ERROR-class lines with 5 lines of trailing context** (the trailing lines usually carry the device stack frame the ERROR line references):

   ```bash
   ast-grep run --pattern 'ERROR' --context-after 5 result.log
   ```

   Fall back to `grep -nA 5 ERROR result.log` only if `ast-grep` is not installed.

2. **Filter to ERROR lines that carry a code** (`F2xxxx`, `E1xxxx`, `0xXXXXX`, `Errcode`, `ErrCode`):

   ```bash
   ast-grep run --pattern 'ERROR $CODE' --regex 'F[0-9]{5}|E[0-9]{5}|0x[0-9A-Fa-f]+|Errcode|ErrCode' result.log
   ```

3. **Pull matching CCE file paths** (when the error references generated device code):

   ```bash
   ast-grep run --pattern '$PATH.cce' --regex '/.*\.cce' result.log
   ```

4. **Narrow to the last ERROR block only** (if the run died early, the last error before the Python traceback is usually the real cause):

   ```bash
   tac result.log | ast-grep run --pattern 'ERROR' --context-before 5 --context-after 20 --max-count 1
   ```

If the op uses `pypto.loop` and you suspect an iteration-dependent failure, also search for the loop-index breadcrumbs the framework emits: `loop iter=`, `chunk=`, `NT step`. These are in the same log but often **below** the ERROR line, so `--context-after 20` on step 1 captures them.

### 2c. Classify what you found

Once the relevant ERROR block is isolated (usually 20–50 lines, not 10 MB):

- If the code starts with `F2xxxx`: route via `https://raw.gitcode.com/cann/pypto/raw/master/docs/zh/trouble_shooting/function.md` → see **§4** if it is `F21004`.
- If the code starts with `F4` / `F5`: **pass / compile error** — see §3a.
- If the run **completed cleanly** but outputs are wrong and **no ERROR line appears even at `LOG_LEVEL=0`**: this is a **true silent precision / correctness failure**. Proceed to §3 (narrow blast radius) and §9 lookup.

### 2d. Computation-graph / program dump

- If the graph is suspect: follow the **computation-graph / program-dump** guidance linked from troubleshooting docs (upstream docs may label this "view computation graph").
- **Do not** treat "unknown" as terminal; treat it as **need more signal** (logs, smaller repro, earlier checkpoint).

### 2e. Append findings to the memory

Every ASCEND-log search that produced a useful signal (ERROR code, CCE file, loop-index breadcrumb) is logged to `custom/<op>/MEMORY.md` → Development & debug log with:
- the exact `ast-grep` command used,
- the matched block (not the full 10 MB log — include only the narrowed result),
- the classification it led to.

This preserves the search trail for future bisection and for the verification handoff.

---

## 3. Kernel-specific: narrow the blast radius

1. Confirm which **staged file** is current (`<op>_module12.py`, etc. — see **skill `pypto-orchestration-manual`'s `references/rules.md`** rule 14). Run **`extract_pypto_calls.py`** on **that** file (or the final kernel).
2. Run **`python3 skills/pypto-op-review/scripts/extract_pypto_calls.py <kernel.py>`** and follow the **op-by-op check protocol** in skill `pypto-general-debug` (SKILL.md auto-loads).
3. Stay in **one active module** at a time; stub downstream with golden-fed tensors if needed.
4. Re-run **module boundary** checks with **`detailed_tensor_compare`** and log results in **Per-module verification log**.
5. After fixes, re-run the **current staged file** and/or **`test_<operator_name>.py`** and confirm **every** kernel output (see **skill `pypto-orchestration-manual`'s `references/rules.md`** rule 10 / **skill `pypto-orchestration-manual`'s `references/rules.md`** rule 10) — not only the first tensor.

### 3a. Layout CI and pass regressions (quick pointers)

- **Layout / loops:** re-check the layout and structure requirements in **skill `pypto-op-review`'s `references/CI.md`** — memory / `test_<op>.py` / staged naming, **and** no Python loop driving kernel work. Express that iteration with **`pypto.loop`** in **`_your_op_kernel_impl`** / the JIT kernel: a host wrapper must not drive the kernel with `for ... in range(...)`, and `while` or non-range Python `for` is not allowed inside the JIT graph — both per **skill `pypto-orchestration-manual`'s `references/rules.md`** Prohibition B / rule 18.
- **Pass / compile failure right after a graph edit:** If logs show **PASS**-range codes (**`F4` / `F5`**, see **`https://raw.gitcode.com/cann/pypto/raw/master/docs/zh/trouble_shooting/README.md`** → **`pass.md`**), **bisect** the PyPTO graph (e.g. last known-good **staged** file vs current), and re-check API constraints by reading the op's API doc `pypto-<op>.md` (use `pypto-docs-search` to find it, or read the raw URL `https://raw.gitcode.com/cann/pypto/raw/master/docs/zh/api/operation/pypto-<op>.md`) before large rewrites. Re-run **`extract_pypto_calls.py`** on the failing file to see whether a new op ordering triggered the pass.

---

## 4. F21004 (`INVALID_VAL` / `TileShape::Current().GetVecTile()` invalid) — `REGISTER_COPY` and vector tile shapes

**Cause (framework):** **`F21004`** is raised in **`Operation`'s constructor** when **`TileShape::Current().GetVecTile()`** is invalid (e.g. `framework/src/interface/operation/operation.cpp` ~191–195).

**Why `REGISTER_COPY` shows up:** **`REGISTER_COPY`** is an **AIV** op. It needs a **valid vector tile** even when the **compiler inserts** that op (e.g. memory-conflict passes). There is **no** separate "REGISTER_COPY tiling" registration — the same **vec tile** rules apply.

**When is `VecTile` valid?** The stored list must be **non-empty** and **every value must be &gt; 0** (`tile_shape.cpp`).

### Fix — what to do in PyPTO Python

1. **Before any vector/tensor work in that scope** (including code that eventually leads to **`REGISTER_COPY`**), set **vector** tile sizes:

   ```python
   import pypto
   pypto.set_vec_tile_shapes(1, 1, 128, 128)   # positive ints per doc
   ```

2. Call it at the **start** of your **`@jit` / kernel function body**, and **again after any scope change** if your API uses **nested scopes**.

3. **Do not rely only on `set_cube_tile_shapes`** — cube tile does **not** replace vector tile for AIV ops like **`REGISTER_COPY`**.

4. If you use **both** cube and vector ops:

   ```python
   pypto.set_cube_tile_shapes([128, 128], [128, 128], [128, 128])
   pypto.set_vec_tile_shapes(1, 1, 128, 128)
   ```

### If it still fails

- **Invalid args or zeros** — ensure all args are positive integers. Refer to **`https://raw.gitcode.com/cann/pypto/raw/master/docs/zh/api/config/pypto-set_vec_tile_shapes.md`** for your PyPTO version's requirements (see **skill `pypto-orchestration-manual`'s `references/rules.md`** rule 16 / **skill `pypto-op-develop`'s `references/pypto-kernel-design-format.md`** §11c).
- **Wrong order** — anything that **adds ops** (reshape, view, passes inserting copy) must run **after** **`set_vec_tile_shapes`** on that execution path.
- **Symbolic / dynamic shapes** — ensure tile args resolve to **concrete positive integers** (see **`set_vec_tile_shapes`** + **`SymbolicScalar`** in `python/pypto/_controller.py` in your tree).

**Bottom line:** **`F21004` on `REGISTER_COPY`** is almost always **"no valid `vec_tile_shapes` in the current scope when this op was created."** Fix it with **`pypto.set_vec_tile_shapes(...)`** with **all positive sizes** before those ops run.

---

## 6. Error-code quick reference (repo)

- skill `pypto-op-develop`'s `references/error-code-troubleshooting.md` — flow for `Errcode: Fxxxxx!`.
- `https://raw.gitcode.com/cann/pypto/raw/master/docs/zh/trouble_shooting/function.md` — e.g. **INVALID_VAL (0x21004)**, **UNKNOWN (0x3FFFF)**.
- **F21004 / vec tile / `REGISTER_COPY`:** see **§4** above.

---

## 7. When stopping is allowed

Only align with the **Stop Conditions** in skill `pypto-orchestration-manual`'s `references/rules.md` (missing reference, impossible golden, fundamental framework block, **missing** user-provided logs when required, or **proven** blind speculation after exhaustive structured attempts). **A single cryptic error line is never enough.**

---

## Final reminder

**Prefer ten documented failed attempts with log citations over one early exit.** Unknown error codes mean **escalate evidence and method**, not **stop**.
