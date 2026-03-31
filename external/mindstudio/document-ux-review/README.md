# document-ux-review

`document-ux-review` is a skill for auditing repository documentation from a real first-time user perspective.

It is designed for code agents such as Claude Code / OpenCode. Given a repository URL or local repo, it follows the `README` and directly linked install / quick start docs step by step, tries to execute the documented flow in a realistic environment, and produces a structured report about whether the docs are actually usable for newcomers.

## What This Skill Does

- Treats itself like a first-time user instead of a project insider
- Follows `README` and directly linked install / quick start docs in order
- Prefers real execution over paper review
- Uses isolated environments when possible, such as Docker, `uv`, `venv`, or `conda`
- Avoids silently fixing missing steps in the docs
- Treats “had to read source code / scripts / CI / Dockerfile to infer the next step” as a documentation completeness problem, not as proof that the docs are sufficient
- Produces a report covering:
  - usability
  - correctness
  - readability
  - completeness
  - environment safety and best practices
  - newcomer friendliness

## Typical Use Cases

Use this skill when you want an agent to do things like:

- verify whether a repo's `README` can actually guide a new user to success
- audit onboarding quality for an open source project
- test whether install / quick start docs are runnable as written
- identify missing prerequisites, broken commands, unclear placeholders, or platform gaps
- generate a documentation UX report with concrete problems and suggested fixes

## Output Style

The skill is optimized to generate a structured review report, typically including:

- an overall score
- a high-level conclusion that clearly states the reviewed branch and commit id
- a step-by-step experience flow
- a problem summary table
- detailed issues with exact doc locations, observed behavior, impact, and recommended fixes
- notes on newcomer experience and open source documentation best practices

When needed, the bundled script can also render a more polished HTML report set from the generated Markdown output. Put those generated HTML files in a dedicated report folder, because a single review may include an overview page plus one or more scenario detail pages.

## Repository Structure

```text
document-ux-review/
├── SKILL.md
├── README.md
├── evals/
│   └── evals.json
└── scripts/
    └── render_report_html.py
```

## Files

### `SKILL.md`

The main skill definition.

It contains:

- the trigger description
- execution principles
- environment safety guidance
- reporting format requirements
- review dimensions and workflow

### `evals/evals.json`

Example evaluation prompts used while developing the skill.

These are useful if you want to:

- regression test future edits
- compare prompt behavior across iterations
- build a larger benchmark set later

### `scripts/render_report_html.py`

A helper script that converts generated Markdown review outputs into a more readable HTML report layout.

This is useful when you want to share results with humans in a cleaner format than raw Markdown. It writes an overview HTML page plus sibling scenario detail HTML pages into the same output folder.

## How the Skill Works

At a high level, the skill asks the agent to:

1. Identify the real documentation entry path from the repository `README`
2. Confirm environment constraints and already-installed prerequisites
3. Choose the least disruptive execution strategy
4. Follow the documented steps literally
5. Record blockers, ambiguities, unsafe assumptions, missing guidance, and any places where code-reading was required to infer undocumented steps
6. Summarize findings in a structured review report

Important behavior:

- If the user says some prerequisites are already installed, the skill should verify and continue instead of reinstalling them
- If the host OS does not match the target doc path and Docker or another isolated option is available, the skill should prefer a closer target environment and explicitly record that as an execution deviation
- If continuing requires reading source code, scripts, CI, Dockerfile, Makefile, or tests to infer how to install, start, configure, or verify the project, the skill should report that as a documentation completeness issue instead of treating it as “docs passed”
- If the docs require secrets, paid services, internal networks, or risky global changes, the skill should stop and report the blocker clearly

## Example Prompts

Here are example user prompts that should trigger this skill:

```text
Please audit this repo's README experience: https://github.com/example/project .
Act like a first-time user, follow the README and directly linked install docs,
try to actually run the setup, and give me a Markdown report on where newcomers
would get stuck.
```

```text
帮我检查这个仓库文档到底能不能带新用户走通：<repo-url>
当前机器 Python 已有，Docker 可用，不允许全局安装。请严格按 README
和直接关联文档执行，并输出完整体验报告。
```

```text
这个项目没有 macOS 安装包，只给了源码编译路径。请你像小白用户一样
按文档跑一遍，看看能不能走通，并指出文档问题和改进建议。
```

## Development Notes

If you continue evolving this skill, recommended next steps are:

- expand the eval set with more repos and platform combinations
- separate generic repo onboarding cases from specialized AI / CUDA / CANN cases
- keep the trigger description focused on user intent, not internal implementation detail
- validate changes against both should-trigger and should-not-trigger queries

## Packaging

If you want to distribute this skill as a `.skill` bundle, package the folder rather than copying generated test outputs into the release artifact.

The source tree in this repository is intended to be the editable, version-controlled version of the skill.
