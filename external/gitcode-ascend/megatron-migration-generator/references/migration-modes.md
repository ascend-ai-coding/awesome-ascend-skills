# Migration Modes

Choose the output mode based on confidence and review needs.

## report

Use when:

- alignment is unresolved
- impact paths are speculative
- the user wants a quick architectural summary

Output:

- migration report only

## patch

Use when:

- alignment is confirmed
- candidate files are known
- the user wants reviewable code guidance

Output:

- migration report
- one unified patch or a small set of patch snippets

This is the default mode for the first version of the skill.

## apply

Use when:

- alignment is confirmed
- impact confidence is high
- the user wants the local workspace updated

Output:

- updated files in workspace
- short explanation of what changed

Do not create a commit automatically.

## commit

Use only on explicit request after review. Prefer a clean patch and explanation first.
