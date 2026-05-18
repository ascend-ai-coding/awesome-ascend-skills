# README Outline

Use this outline when the user provides a target operator plus a reference template README.

## Recommended sections

1. `Operator Name`
2. `Overview`
3. `Inputs And Outputs`
4. `Runtime Parameters`
5. `Tiling Design`
6. `Kernel Design`
7. `File Layout Mapping`
8. `Build Entry`
9. `Limitations`
10. `Summary`

## Section guidance

### Overview

- State what the operator computes.
- Name the most important inputs.
- Describe the high-level pipeline in one short paragraph or a compact code block.

### Inputs And Outputs

- Pull names, dtypes, and formats from the host `OpDef`.
- Separate confirmed shape facts from inferred shape semantics.
- Call out auxiliary outputs such as flags or reduced extrema.

### Runtime Parameters

- Document any packed runtime arrays like `actualSize`.
- Use index-based explanations such as `actualSize[0] -> actualNum`.
- Only document indices that the kernel actually reads.

### Tiling Design

- Summarize the tiling struct fields.
- Explain how host code derives dimensions, block sizes, and core counts.
- Include fixed constants if the host hardcodes them.
- Include workspace formulas when present.

### Kernel Design

- Describe init, compute loop, and writeback in order.
- Highlight reuse of query tensors, block-based base processing, and overlap between stages when present.
- Explain mask and reduction behavior carefully.

### File Layout Mapping

- Add this section when the template repo and target repo organize operator files differently.
- Show where the host, tiling, kernel, and documentation files live.

### Build Entry

- Point to real build entry files such as `CMakeLists.txt`.
- Do not invent commands that are not supported by the checked-in project context.

### Limitations

- Include platform gates, dtype restrictions, fixed tile sizes, missing infer-shape coverage, or other code-visible limits.

## Writing style

- Prefer short technical paragraphs.
- Use lists only for inherently list-shaped facts.
- Mark uncertain conclusions as inference.
- Do not copy irrelevant details from the template README.
