# Fused operator integration workflow

1. **Find the production path.** Find the target checkout and kernel
   source or installed binding. Trace caller → public wrapper → dispatch or
   registration → forward and backward. Record the caller and backend that
   are in scope. Use a benchmark adapter as a source of fixtures and contract
   evidence. Check production reachability through the normal entry point.
2. **Compare the contracts.** Compare the existing path with the kernel's
   supported forward and backward contracts. Record input/output shape,
   dtype, device, layout, casts, argument defaults, saved tensors and gradient
   requirements where they affect the caller. Include sharding or recompute
   when they affect what reaches the kernel. Decide which cases the new
   backend can serve and where the existing path must handle the request.
3. **Wire the smallest compatible change.** Reuse the semantic wrapper and
   the target repo's dispatch and configuration conventions. Keep kernel-specific
   imports and input adaptation in the backend. Bridge explicit kernel backward
   through the project's autograd pattern when needed. Preserve parameter
   identity, state-dict keys, default behavior and explicit eager overrides unless
   requested otherwise. Reuse an existing backend selector when available.
   If opting in needs a new setting, follow the repo's config validation and
   document how to disable it. Keep fallback scoped to known unsupported
   inputs or unavailable optional dependencies as allowed by the selector's
   contract. Surface unexpected execution errors.
4. **Validate from the caller.** Start with a supported case through the
   actual fused adapter. Check dispatch and keep forward/backward evidence.
   Compare outputs and relevant gradients with the existing implementation
   and, when needed, an independent reference. Exercise the most important
   compatibility risk and verify original behavior for disabled and applicable
   fallback paths. Where practical use a small module/caller smoke test to
   verify the model reaches the wrapper. Use
   [the test checklist](test-checklist.md) for execution and reporting.
5. **Deliver the integration.** Point to the changed entry point and backend,
   explain how to select or disable it if configurable, and record executed
   checks, unverified behavior and any blockers. For plan-only requests,
   give the same concrete file/symbol map and verification plan.
