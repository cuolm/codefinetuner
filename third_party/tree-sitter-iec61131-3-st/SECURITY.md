# Security Policy

## Reporting a Vulnerability

If you discover a security vulnerability in tree-sitter-iec61131-3-st, please report it responsibly.

**Preferred channel:** open a private security advisory through GitHub: [Report a vulnerability](https://github.com/HeytalePazguato/tree-sitter-iec61131-3-st/security/advisories/new).

**Do NOT** open a public GitHub issue for security vulnerabilities.

You can expect an initial response within 7 days. Confirmed issues will be fixed in the next release; the advisory will be published with credit (unless you prefer to remain anonymous).

## Scope

This is a tree-sitter parser. Its purpose is to produce a syntax tree from arbitrary IEC 61131-3 ST source text. In-scope concerns:

- **Parser crashes / hangs** on malformed input — denial-of-service via pathological inputs that consume excessive CPU or memory.
- **Stack overflows** from deeply-nested expressions or statements.
- **Incorrect tree output** that leads downstream tools (linters, code search, security scanners) to miss code they should have flagged.

Out of scope:

- Vulnerabilities in third-party tree-sitter language bindings — report those upstream.
- Vulnerabilities in vendor dialect grammars — report to the dialect's own repo.
- Vulnerabilities in tools that consume the grammar (editors, language servers, formatters) — report to those projects.
- Functional bugs (wrong tree shape, missing token coverage) — file a regular issue.

## Supported Versions

Only the latest released minor version is supported with security updates.

| Version | Supported          |
| ------- | ------------------ |
| latest  | :white_check_mark: |
| older   | :x:                |
