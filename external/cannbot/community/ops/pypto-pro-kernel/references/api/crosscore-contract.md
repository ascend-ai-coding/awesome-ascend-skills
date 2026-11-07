# Cross-core Handover: Implementation Maintenance Contract

Keep usage routes and examples only in [api/crosscore-sync.md](crosscore-sync.md). This page preserves the boundaries that implementations must uphold. GM binding, explicit tile groups, event conflicts, and multi-slot checks are implemented; the historical baseline is pypto `f89df4f72`, Ascend950PR. Verify the matching PyPTO checkout before applying version-specific behavior.

## Declarations and Lowering

| Entry point | Contract |
| --- | --- |
| `pl.bind_crosscore_sync(ws, ids=[...])` | Bind a GM tensor outside a section; one ID per slot, with list length expressing depth; fixed to INTRA_BLOCK |
| Hand-written `make_tile_group` | `fwd_ids/bwd_ids` express the cross-core protocol; mutexes still govern only within a core |

GM depth is a declaration promise; the framework cannot prove that the actual workspace partition matches it. Declaring too much can leave synchronization insufficient and silently produce wrong values; declaring too little can over-synchronize or trigger a slot shortage. Passing a check does not verify the physical layout.

`bind_crosscore_sync` emits no IR in the parser; the pipeline transform reads and removes it. Rejecting its use without pipeline must happen at the JIT layer because the probe phase parses once first. The GM workspace must be present in both the helper argument mapping and `slot_to_buffer`; otherwise access outside a stage may escape validation.

## ID and Pipe Invariants

- The same buffer may use the same ID in both directions; the hardware counter key includes direction.
- Different buffers handed over in the same pipeline loop cannot share an ID in the same direction; compare the actual producer side, not only the fwd/bwd labels.
- Different pipeline loops may reuse IDs; each loop must balance prefire, steady state, and drain independently.
- Publish on the pipe of the last write on the producer side, and wait on the pipe of the first read on the consumer side; the two ends need not match. When adding a buffer source, confirm the scanner recognizes it; syntax acceptance alone is insufficient.

**The default is one ID per slot.** A hand-written group may share one ID only in supported forms where the consumer has a single gating pipe. When multiple pipes release, a shared counter mixes tokens from different tasks and lets the producer overwrite a slot before it has been fully consumed. The criterion is the number of pipes in `_gating_readers`, not whether updates are in place.

## Regression Entry Points

Framework tests: `python/tests/st/pypto_pro/frontend/auto_pipeline/`:

| File | Property that must be preserved |
| --- | --- |
| `test_gm_handover.py` | Real GM synchronization, rejection of duplicate bindings, multiple workspaces, and preload |
| `test_pipelined_crosscore_handover.py` | Explicit event IDs, local slot diagnostics, and numerical results at different preload values |
| `test_cross_core_id_collisions.py` | Reject same-side, same-direction ID collisions; allow reverse-direction and cross-loop reuse |
| `test_shared_event_id.py` | Reject sharing an ID across multiple pipes; keep the single-pipe positive case |

Implementation entry points: `_check_event_id_collisions` and slot-count checks in `runtime/pipeline/_validate.py`, plus role and pipe derivation in `_cross_core_scanner.py`. Transform tests verify generated synchronization; numerical behavior and reuse races require board runs. `preload=0` is not a substitute for pipeline validation.

## Extension Conditions

Hand-written cross-core channel wrappers, detailed event-lifetime analysis, and a new automatic route for one-shot handovers are not currently scheduled. Reopen the design only when the existing three-layer mechanisms cannot express a concrete kernel or a profile shows conservative synchronization is a bottleneck. Do not introduce `CvChannel` for API symmetry or integrate an external compiler; simulation and schedule analysis need separately defined scope and acceptance criteria.
