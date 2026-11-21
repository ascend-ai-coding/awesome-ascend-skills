# Tunable constants — extending INIT normalisation past the tile call

Status: design agreed, not implemented.

## What exists

`bayesian_optimization.normalize` runs at INIT and already carries the whole pattern: the model
rewrites, the harness checks the rewrite is admissible, and the BO gains sites.
Its one claim is

> normalisation exposes parameters and does not change values

checked by resolving the symbolic arguments against the constant bindings and
comparing them to the literals the model wrote. It does not claim semantics are
preserved — constant folding cannot know that, and the golden answers it.

It covers the two tile calls and nothing else. Whether the deterministic core
owns a tile therefore depends on how a generated kernel happened to spell one
call — a kernel written with literals gets a BO lever, the same kernel written
with names gets none.

## The gap

The same asymmetry one level out. A named constant that drives a loop trip count
or a view extent is invisible to `bayesian_optimization.apply`, so the search cannot move it and the
model has to sweep it by hand as a structural delta — paying a full evaluation
per point for what a single BO dimension would cover.

## Ownership

| step | owner |
|---|---|
| enumerate integer constants and their uses | DET |
| select the admitted ones | DET, by the rule below |
| rewrite the kernel so each is a single literal site | LLM, at INIT |
| verify no value moved | DET |
| verify each incumbent value is inside its own domain | DET |
| record the declaration | DET |

The model's remaining job is the same one it already has in normalisation: decide
which uses of a name are the parameter and which are the extent, and spell the
rewrite. Selection is computable and is not asked.

The declaration lives in `search_state.json`, never in the kernel source: PANKO
runs on kernels that never opted in, and the delivered kernel carries no marking
that means nothing outside PANKO. It also survives the snapshot/restore ratchet
there.

## Domain

Deterministic, anchored on the incumbent — the ladder around the value the kernel
is running, the way the L1 multiplier is already anchored so a warm start is
always reachable. No author knowledge, no model guess. The domain does not have
to be correct; it has to CONTAIN the good values, and the search prunes the rest.

## Which constants are admitted

Two stages, and the order matters.

1. A constant read DIRECTLY inside a `@jit` type annotation or a `torch.*` call
   is part of the operator's declared interface. Excluded.
2. Of the rest, admit only those that reach a `pypto.loop` trip count, directly
   or through a derivation. Reaching a loop bound means the constant sets an
   iteration count, which is what a tiling or chunking parameter does and what a
   declared extent does not.
3. Everything else is left alone.

The rule is default-deny. Stage 2 admits a narrow, positively identified class
rather than admitting everything and gating the dangerous cases out.

Derivation matters in both directions and asymmetrically. A constant typically
reaches its loop bound one derivation removed, so stage 2 must follow the chain.
Stage 1 must NOT: a constant whose DERIVED name reaches an annotation is not
itself part of the interface, and treating it as one loses real levers.

## Legality is not declared

There is no legality check to build. Almost every illegal value announces itself
when measured — a compile error, a capacity refusal, a device fault, a precision
failure — and costs one evaluation, which learned ceilings and carried front-end
refusals then amortise across the rest of the ladder.

The one case measurement cannot catch is a read past a bound that lands on memory
the fixture happens to have allocated: the golden passes, a paired re-measurement
agrees with itself, and a wrong kernel is banked. The admission rule is what puts
that case out of reach — a constant admitted for setting a loop trip count is not
an index, so moving it cannot make a read run past a parameter's extent, and
nothing outside that class is tuned.
