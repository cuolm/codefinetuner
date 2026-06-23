---
layout: default
title: "tree-sitter-iec61131-3-st"
description: "tree-sitter grammar for IEC 61131-3 Structured Text — standard, dialect-extensible"
---

# tree-sitter-iec61131-3-st

A [tree-sitter](https://tree-sitter.github.io/tree-sitter/) grammar for the IEC 61131-3 (3rd edition, 2013) **Structured Text** language. Standard-compliant first; vendor dialects (Beckhoff TwinCAT, Codesys, B&R, Siemens, Rockwell) live in separate dialect grammars that extend this base.

[View on GitHub](https://github.com/HeytalePazguato/tree-sitter-iec61131-3-st){: .btn } [Latest release](https://github.com/HeytalePazguato/tree-sitter-iec61131-3-st/releases/latest){: .btn }

## Highlights

- Full IEC 61131-3 ST coverage: POUs, all VAR blocks, every type, every operator with correct precedence, all statements, OOP, namespaces.
- Case-insensitive keywords; identifiers preserve case.
- Error-tolerant parser — useful trees on partial / broken input.
- Six editor query files (highlights / locals / tags / folds / indents / injections) using the standard tree-sitter capture vocabulary.
- Bindings for Node, Rust, Python, and Go.
- Designed for vendor-dialect extension via `grammar(base, {…})` — see [EXTENDING](https://github.com/HeytalePazguato/tree-sitter-iec61131-3-st/blob/main/EXTENDING.md).

## Install

### Node
```sh
npm install tree-sitter tree-sitter-iec61131-3-st
```

### Rust
```toml
tree-sitter-iec61131-3-st = "0.0"
```

### Python
```sh
pip install tree-sitter tree-sitter-iec61131-3-st
```

### Go
```go
import iec61131_3_st "github.com/HeytalePazguato/tree-sitter-iec61131-3-st/bindings/go"
```

## Links

- [README](https://github.com/HeytalePazguato/tree-sitter-iec61131-3-st/blob/main/README.md)
- [Spec compliance map](https://github.com/HeytalePazguato/tree-sitter-iec61131-3-st/blob/main/SPEC-COMPLIANCE.md)
- [Extension architecture](https://github.com/HeytalePazguato/tree-sitter-iec61131-3-st/blob/main/EXTENDING.md)
- [Issues](https://github.com/HeytalePazguato/tree-sitter-iec61131-3-st/issues)
- [Discussions](https://github.com/HeytalePazguato/tree-sitter-iec61131-3-st/discussions)
- [Releases](https://github.com/HeytalePazguato/tree-sitter-iec61131-3-st/releases)
- [Security policy](https://github.com/HeytalePazguato/tree-sitter-iec61131-3-st/blob/main/SECURITY.md)
