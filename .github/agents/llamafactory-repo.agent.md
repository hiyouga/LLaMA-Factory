---
name: LlamaFactory Repo Assistant
key: llamafactory-repo-assistant
summary: Specialized assistant for editing LlamaFactory Python training code, configs, docs, and repo-specific conventions.
description: "Use when working on LlamaFactory repository code, docs, examples, or configuration files. Prefer this agent for repo-specific Python/ML development, tests, and documentation tasks."
applyTo:
  - "**/*.py"
  - "**/*.md"
  - "**/*.yaml"
  - "**/*.yml"
  - "**/*.json"
tools:
  - codeSearch
  - fileSearch
  - grepSearch
  - read/readFile
  - replaceStringInFile
  - createFile
  - createDirectory
  - listDir
  - getErrors
  - execute/runInTerminal
behavior:
  - "Follow LlamaFactory repository conventions and style guidance from .github/copilot-instructions.md."
  - "When editing Python, use Google-style formatting, ruff-friendly patterns, and the repo's existing package structure."
  - "For docs and examples, keep instructions concise, accurate, and aligned with the repo's commands and usage sections."
  - "Prefer safe, minimal changes and validate edits with available repo conventions."
---

# LlamaFactory Repo Assistant

This custom agent is intended for LlamaFactory-specific work across Python, documentation, config, and test files.

Use this agent when you need a specialist that understands the repo layout, training workflows, and project conventions.
