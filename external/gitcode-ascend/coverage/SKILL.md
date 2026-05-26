---
name: external-gitcode-ascend-coverage
description: Use when working with coverage
doc_version: null
original-name: coverage
synced-from: https://gitcode.com/Ascend/agent-skills
synced-date: '2026-05-26'
synced-commit: 1f7666e7768a0ceb21bb1d40ce4b5179fcb6f1d6
license: UNKNOWN
---

# Coverage Skill

Use when working with coverage, generated from official documentation.

## When to Use This Skill

This skill should be triggered when:
- Working with coverage
- Asking about coverage features or APIs
- Implementing coverage solutions
- Debugging coverage code
- Learning coverage best practices

## Quick Reference

### Common Patterns

**Pattern 1:** XML reporting: coverage xml The xml command writes coverage data to a “coverage

```
coverage xml
```

**Pattern 2:** Combining data files: coverage combine Often test suites are run under different conditions, for example, with different versions of Python, or de...

```
coverage combine
```

**Pattern 3:** HTML reporting: coverage html Coverage

```
coverage html
```

**Pattern 4:** Diagnostics: coverage debug The debug command shows internal information to help diagnose problems

```
coverage debug
```

**Pattern 5:** Installation You can install coverage

```
$ python3 -m pip install coverage
```

**Pattern 6:** If you are installing on Linux, you may need to install the python-dev and gcc support files before installing coverage via pip

```
$ sudo apt-get install python-dev gcc
$ sudo yum install python-devel gcc

$ sudo apt-get install python3-dev gcc
$ sudo yum install python3-devel gcc
```

**Pattern 7:** Text annotation: coverage annotate Note The annotate command has been obsoleted by more modern reporting tools, including the html command

```
coverage annotate
```


## Reference Files

This skill includes comprehensive documentation in `references/`:

- **7.13.4.md** - 7.13.4 documentation

Use `view` to read specific reference files when detailed information is needed.

## Working with This Skill

### For Beginners
Start with the getting_started or tutorials reference files for foundational concepts.

### For Specific Features
Use the appropriate category reference file (api, guides, etc.) for detailed information.

### For Code Examples
The quick reference section above contains common patterns extracted from the official docs.

## Resources

### references/
Organized documentation extracted from official sources. These files contain:
- Detailed explanations
- Code examples with language annotations
- Links to original documentation
- Table of contents for quick navigation

### scripts/
Add helper scripts here for common automation tasks.

### assets/
Add templates, boilerplate, or example projects here.

## Notes

- This skill was automatically generated from official documentation
- Reference files preserve the structure and examples from source docs
- Code examples include language detection for better syntax highlighting
- Quick reference patterns are extracted from common usage examples in the docs

## Updating

To refresh this skill with updated documentation:
1. Re-run the scraper with the same configuration
2. The skill will be rebuilt with the latest information
