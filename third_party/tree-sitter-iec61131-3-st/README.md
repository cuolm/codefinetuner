# tree-sitter-iec61131-3-st

[![ci](https://github.com/HeytalePazguato/tree-sitter-iec61131-3-st/actions/workflows/ci.yml/badge.svg)](https://github.com/HeytalePazguato/tree-sitter-iec61131-3-st/actions/workflows/ci.yml)
[![release](https://img.shields.io/github/v/release/HeytalePazguato/tree-sitter-iec61131-3-st?display_name=tag&sort=semver)](https://github.com/HeytalePazguato/tree-sitter-iec61131-3-st/releases)
[![npm](https://img.shields.io/npm/v/tree-sitter-iec61131-3-st?label=npm&logo=npm)](https://www.npmjs.com/package/tree-sitter-iec61131-3-st)
[![crates.io](https://img.shields.io/crates/v/tree-sitter-iec61131-3-st?logo=rust)](https://crates.io/crates/tree-sitter-iec61131-3-st)
[![PyPI](https://img.shields.io/pypi/v/tree-sitter-iec61131-3-st?logo=pypi&logoColor=white)](https://pypi.org/project/tree-sitter-iec61131-3-st/)
[![license](https://img.shields.io/github/license/HeytalePazguato/tree-sitter-iec61131-3-st)](LICENSE)
[![tree-sitter](https://img.shields.io/badge/tree--sitter-0.26%2B-7c3aed)](https://tree-sitter.github.io/tree-sitter/)
[![IEC 61131-3](https://img.shields.io/badge/IEC%2061131--3-Structured%20Text-005f87)](https://en.wikipedia.org/wiki/IEC_61131-3)

A [tree-sitter] grammar for [IEC 61131-3][iec61131] **Structured Text** (ST) — the standard programming language for industrial PLCs. Standard-compliant first; vendor dialects (Beckhoff TwinCAT, Codesys, B&R Automation Studio, Siemens TIA, Rockwell) are deferred to separate dialect grammars that extend this base.

> **About the name** — IEC 61131 is the umbrella PLC-programming standard. **Part 3** (`IEC 61131-3`) defines the programming languages: ST (Structured Text), LD (Ladder Diagram), FBD (Function Block Diagram), IL (Instruction List, deprecated), and SFC (Sequential Function Chart). This repo covers ST only — the `-3-st` suffix encodes both: Part 3 of the standard, ST language specifically.

![tree-sitter-iec61131-3-st demo — playground showing the parse tree for a state machine](assets/demo.gif)

## Features

- IEC 61131-3 (3rd edition, 2013) Structured Text — POUs, all VAR blocks, every elementary / derived / generic type, every operator with correct precedence, every statement, full OOP (METHOD / PROPERTY / EXTENDS / IMPLEMENTS / INTERFACE), namespaces, configuration / resource / task.
- **Case-insensitive keywords** (`IF`, `if`, `If` all parse as the same keyword) implemented at the lexer level.
- **Error-tolerant**: produces a useful tree even on partial or broken input — usable in editors during typing.
- **Dialect-extensible**: the base grammar exposes named hidden rules (`_declaration`, `_statement`, `_expression`, `_type_specifier`, `_var_block`) so dialect grammars can add vendor-specific constructs via `grammar(base, {…})` without forking. See [EXTENDING.md](EXTENDING.md).
- **Editor-ready** queries: `highlights.scm`, `locals.scm`, `tags.scm`, `folds.scm`, `indents.scm`, `injections.scm`. Standard tree-sitter capture vocabulary.
- Bindings for **Node, Rust, Python, Go**.

## Quick demo

```st
FUNCTION_BLOCK PID
VAR_INPUT
    setpoint, process_var : REAL;
    Kp, Ki, Kd            : REAL;
END_VAR
VAR_OUTPUT
    output : REAL;
END_VAR
VAR
    error, prev_error, integral : REAL;
END_VAR

error := setpoint - process_var;
integral := integral + error;
output := Kp * error + Ki * integral + Kd * (error - prev_error);
prev_error := error;
END_FUNCTION_BLOCK
```

Parsing this with `tree-sitter parse` produces a clean tree with `function_block_declaration` → `var_input` / `var_output` / `var_block` → assignments with `binary_expression` operands at the right precedence.

## Install

### Node

```sh
npm install tree-sitter tree-sitter-iec61131-3-st
```

### Rust

```toml
# Cargo.toml
[dependencies]
tree-sitter = "0.25"
tree-sitter-iec61131-3-st = "0.0"
```

### Python

```sh
pip install tree-sitter tree-sitter-iec61131-3-st
```

```python
import tree_sitter, tree_sitter_iec61131_3_st
language = tree_sitter.Language(tree_sitter_iec61131_3_st.language())
parser = tree_sitter.Parser(language)
tree = parser.parse(b"PROGRAM Hello END_PROGRAM")
```

### Go

```go
import (
    sitter "github.com/tree-sitter/go-tree-sitter"
    iec61131_3_st "github.com/HeytalePazguato/tree-sitter-iec61131-3-st/bindings/go"
)
```

## Editor setup

### Neovim with `nvim-treesitter`

```lua
require('nvim-treesitter.configs').setup {
  ensure_installed = { 'iec61131_3_st' },   -- once published; pre-publish, install from local path
  highlight = { enable = true },
  indent    = { enable = true },
  fold      = { enable = true },
}
```

For a local development install before the parser is on the npm/CDN registry, add to your `init.lua`:

```lua
local parser_config = require'nvim-treesitter.parsers'.get_parser_configs()
parser_config.iec61131_3_st = {
  install_info = {
    url = 'https://github.com/HeytalePazguato/tree-sitter-iec61131-3-st',
    files = { 'src/parser.c' },
    branch = 'main',
  },
  filetype = 'st',
}
```

### Helix

`languages.toml`:

```toml
[[language]]
name = "iec61131-3-st"
scope = "source.iec61131-3.st"
file-types = ["st", "iecst"]
roots = []
comment-token = "//"
indent = { tab-width = 4, unit = "    " }

[[grammar]]
name = "iec61131_3_st"
source = { git = "https://github.com/HeytalePazguato/tree-sitter-iec61131-3-st", rev = "main" }
```

### Zed

Zed picks up tree-sitter grammars from extensions; see Zed's docs for the current recommended packaging path.

### VSCode

VSCode does not natively use tree-sitter for grammar parsing — its highlighting comes from TextMate grammars and its semantic tokens come from language servers. A future companion repo will provide a VSCode extension that loads this grammar via the [`vscode-tree-sitter`][vsts] integration.

## What's covered, what's not

Implemented in v0.0.1:
- POU declarations: `PROGRAM`, `FUNCTION` (with return type), `FUNCTION_BLOCK`, `INTERFACE`, `TYPE`, `NAMESPACE`, `CONFIGURATION`, `RESOURCE`.
- All `VAR_*` block kinds with `CONSTANT` / `RETAIN` / `NON_RETAIN` qualifiers, `AT %{I,Q,M}{X,B,W,D,L}` direct addresses, initial values.
- All elementary types, generic `ANY_*` types, `STRING(N)` / `WSTRING(N)`, `ARRAY [a..b, …] OF`, structures, enumerations, subranges, `POINTER TO`, `REF_TO`.
- All literals: integers (plain, `2#…`, `8#…`, `16#…`, with `_` separators), reals with exponent, strings with `$` escapes, `T#…`, `D#…`, `TOD#…`, `DT#…`, typed-prefixed (`INT#42`, `REAL#3.14`, etc).
- All statements: assignment (`:=`), reference assignment (`REF=`), function/method calls with positional and named (`:=` / `=>`) arguments, `IF`/`ELSIF`/`ELSE`/`END_IF`, `CASE` with single, list, range values + `ELSE`, `FOR … TO … BY … DO … END_FOR`, `WHILE`, `REPEAT`, `EXIT`, `CONTINUE`, `RETURN`.
- All operators with IEC 61131-3 §6.6.5 precedence: parentheses, calls, indexing `[…,…]`, member access, dereference `^`, `ADR()`, unary `-` / `+` / `NOT`, right-associative `**`, `*` / `/` / `MOD`, `+` / `-`, comparisons, equality, `AND` / `&`, `XOR`, `OR`.
- OOP (3rd edition): `METHOD` / `END_METHOD`, `PROPERTY` with `GET` / `SET` accessor bodies, `EXTENDS`, `IMPLEMENTS`, `INTERFACE`, `ABSTRACT` / `FINAL` / `OVERRIDE`, `PUBLIC` / `PRIVATE` / `PROTECTED` / `INTERNAL`, `THIS` / `SUPER`.
- Comments (`(* … *)`, `//`) and pragmas (`{ … }`, opaque body).

Out of scope for v0.0.1:
- Vendor dialect extensions (TwinCAT `__VERSION`, Codesys structured pragmas, B&R `ACTION`, etc.) — those will live in dialect repos that extend this grammar.
- Other IEC 61131-3 languages — Ladder Diagram, Function Block Diagram, Instruction List, Sequential Function Chart.
- Type checking, symbol resolution, code generation, formatting — this is a parser, not a compiler.

## Performance

The CI benchmark parses a synthetic ~10 000-line ST file (200× the combined PID + conveyor + state-machine examples) and fails the build if the parse exceeds **200 ms** on a `ubuntu-latest` runner. Typical run is well under that.

## Development

```sh
# Install tree-sitter-cli and a C compiler.
npm install -g tree-sitter-cli

# Generate the parser (writes src/parser.c).
tree-sitter generate

# Run the corpus.
tree-sitter test

# Parse a single file.
tree-sitter parse examples/blink.st
```

See the project's [BLUEPRINT.md](BLUEPRINT.md) for branch / release conventions: `develop → release/<version> → main`, semver from a single `VERSION` file.

## Roadmap

Future repos that will extend this base grammar:

- `tree-sitter-iec61131-3-st-twincat` — Beckhoff TwinCAT 3 (TwinCAT-specific pragmas, `S=` / `R=` set/reset, `OR_ELSE` / `AND_THEN` short-circuit operators, conditional compilation, `ACTION` blocks).
- `tree-sitter-iec61131-3-st-codesys` — Codesys 3 (attribute pragmas with structured contents, action / transition blocks).
- `tree-sitter-iec61131-3-st-br` — B&R Automation Studio (`ACTION`, task-specific extensions).
- `tree-sitter-iec61131-3-st-siemens` — Siemens TIA Portal SCL.
- `tree-sitter-iec61131-3-st-rockwell` — Rockwell Studio 5000 ST.

Pull requests welcome — see [CONTRIBUTING.md](CONTRIBUTING.md) and [EXTENDING.md](EXTENDING.md).

## Acknowledgments

Patterns, organization, and corpus references from prior work in the ecosystem (all MIT or otherwise permissively licensed):

- [`tmatijevich/tree-sitter-structured-text`][prior1] — B&R-leaning partial grammar; this project's literal regex shape and basic precedence layout drew from it.
- [`teunreyniers/tree-sitter-structured-text`][prior2] — generic ST grammar; informed the named-precedence-table approach and VAR-block factoring.
- [`klauer/blark`][blark] — Lark-based TwinCAT parser; the most comprehensive open-source IEC 61131-3 grammar. Used as a reference for the rule organization, the OOP/extension surface, and which features are truly TwinCAT-only vs. standard.

## License

MIT — see [LICENSE](LICENSE).

[tree-sitter]: https://tree-sitter.github.io/tree-sitter/
[iec61131]: https://en.wikipedia.org/wiki/IEC_61131-3
[prior1]: https://github.com/tmatijevich/tree-sitter-structured-text
[prior2]: https://github.com/teunreyniers/tree-sitter-structured-text
[blark]: https://github.com/klauer/blark
[vsts]: https://marketplace.visualstudio.com/search?term=tree-sitter
