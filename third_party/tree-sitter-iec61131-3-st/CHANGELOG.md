# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased] — next: 0.1.3

## [0.1.2] - 2026-06-15

### Added

- External scanner (`src/scanner.c`) that recognizes the 19 block and POU terminator keywords (`END_IF`, `END_CASE`, `END_FOR`, `END_WHILE`, `END_REPEAT`, `END_VAR`, `END_STRUCT`, `END_TYPE`, `END_PROGRAM`, `END_FUNCTION`, `END_FUNCTION_BLOCK`, `END_INTERFACE`, `END_METHOD`, `END_PROPERTY`, `END_GET`, `END_SET`, `END_NAMESPACE`, `END_CONFIGURATION`, `END_RESOURCE`) as reserved tokens. They were previously case-insensitive regex tokens built by the `kw()` helper, which tree-sitter cannot reserve. All language bindings already compile `src/scanner.c` when present, so no binding changes were required.

### Fixed

- Error recovery for a missing block terminator. A forgotten `END_IF` (or any `END_*`) used to collapse the entire enclosing POU into a single `ERROR` node, with the real terminator mis-lexed as an `identifier`, so a consuming tool could only report "something is wrong in this POU", not which block was left open. The terminators are now reserved keywords lexed in every state (including during recovery), so the parser keeps the enclosing `function_block_declaration` / `program_declaration` / etc. and localizes the fault. Statement blocks (`IF`, `CASE`, `FOR`, `WHILE`, `REPEAT`) now produce a precise `MISSING "END_*"` node at the unterminated block instead of a POU-wide error.

## [0.1.1] - 2026-06-08

### Added

- Prebuilt **WebAssembly grammar** (`tree-sitter-iec61131_3_st.wasm`) is now built in CI and shipped in the npm package (matched by the `*.wasm` files glob). Consumers can load the parser through [`web-tree-sitter`](https://www.npmjs.com/package/web-tree-sitter) with no native toolchain, sidestepping the `tree-sitter` (node-tree-sitter) native build that fails on Node 24 / Electron when no prebuild matches. A CI `wasm` job builds the module and parses every `examples/*.st` file through the web-tree-sitter runtime as a smoke test.

### Changed

- npm releases now publish via **OIDC Trusted Publishing** instead of a long-lived `NPM_TOKEN` (npm bumped to latest in CI for OIDC; the npm preflight check was dropped). Requires a one-time trusted-publisher setup on npmjs.com for this repo + `release.yml`; the `NPM_TOKEN` secret is no longer used and can be removed.
- Bumped pinned GitHub Actions to current majors: `actions/checkout` v4/v5 → v6, `actions/setup-node` v5 → v6, `actions/setup-go` v5 → v6, `actions/deploy-pages` v4 → v5, `actions/upload-pages-artifact` v3 → v5.

## [0.1.0] - 2026-05-15

### Added

- C# bindings via `TreeSitter.DotNet`: `Language.Create()` returns a typed `TreeSitter.Language` object; `Language.HighlightsQuery`, `InjectionsQuery`, `LocalsQuery`, and `TagsQuery` expose the bundled `.scm` files as embedded-resource strings. Targets `net8.0`. NuGet package id `TreeSitterIec61131_3St`. Initial scaffolding contributed by @beslst in #11; finished and brought up to current `develop` here.
- CI matrix job for C# (Linux / macOS / Windows) that builds the native parser library and runs `dotnet test`. The Windows leg loads the MSVC environment via `ilammy/msvc-dev-cmd` before invoking `cl`.
- Native library build artifacts (`*.so` / `*.a` / `*.dylib` / `*.dll` / `*.dll.a` / `*.lib` / `*.pdb` / `src/*.o` / `*.pc`) and C# `bin/` / `obj/` directories added to `.gitignore`.

## [0.0.2] - 2026-05-11

### Changed

- Contributions now require a `Signed-off-by` trailer on every commit (`git commit -s`) per the [Developer Certificate of Origin](https://developercertificate.org). See [CONTRIBUTING.md](CONTRIBUTING.md#sign-your-work-dco) for details. No CLA — DCO is purely an attestation of contribution rights.

### Fixed

- PyPI package now renders a non-empty `Author` field. Worked around a PEP 621 + setuptools quirk where `authors = [{ name, email }]` writes only `Author-email` to core metadata, leaving `Author` null on PyPI, by splitting the entry into `authors = [{ name }]` and a separate `maintainers = [{ name, email }]` in `pyproject.toml`.

## [0.0.1] - 2026-05-09

### Changed

- Renamed the project to `tree-sitter-iec61131-3-st` so the name encodes both the IEC 61131 part number (Part 3 — programming languages) **and** the specific language (ST), distinguishing it from future grammars for the other Part-3 languages (FBD, LD, IL, SFC). The `-3-st-` carries through to all future vendor-dialect repos (e.g. `tree-sitter-iec61131-3-st-twincat`). Internal grammar name is `iec61131_3_st`; C function symbol is `tree_sitter_iec61131_3_st`; TextMate scope is `source.iec61131-3.st`. Distribution names align across npm, crates.io, and PyPI. No published artifacts existed under any prior name, so this is purely a pre-release rename.

### Added

- Initial tree-sitter grammar for IEC 61131-3 (3rd edition, 2013) Structured Text. Standard-compliant; vendor dialects deferred to separate dialect repos that extend this base.
- All POU declarations: `PROGRAM`, `FUNCTION` (with return type), `FUNCTION_BLOCK`, `INTERFACE`, `TYPE`, `NAMESPACE`, `CONFIGURATION`, `RESOURCE`, `TASK`.
- All `VAR_*` block kinds with `CONSTANT` / `RETAIN` / `NON_RETAIN` qualifiers, `AT %{I,Q,M}{X,B,W,D,L}<addr>` direct addresses, and initial values.
- All elementary types, generic `ANY_*` types, sized `STRING(N)` / `WSTRING(N)`, multi-dimensional `ARRAY [a..b, …] OF`, structures, enumerations, subranges, `POINTER TO`, `REF_TO`.
- All literals: integers (plain, `2#…`, `8#…`, `16#…`, with `_` separators), reals with exponent, single- and double-quoted strings with `$`-escape sequences, time/date/TOD/DT prefixed forms, typed-prefix (`INT#42`, `REAL#3.14`, `BOOL#TRUE`, `WORD#16#FF`).
- All operators with IEC 61131-3 §6.6.5 precedence, right-associative `**`.
- All statements: `:=`, `REF=`, function/method calls with positional and named (`:=` / `=>`) arguments, `IF`/`ELSIF`/`ELSE`, `CASE` with single, list, range values, `FOR … TO … BY … DO`, `WHILE`, `REPEAT`, `EXIT`, `CONTINUE`, `RETURN`.
- OOP (3rd edition): `METHOD` / `END_METHOD`, `PROPERTY` with `GET` / `SET` accessor bodies, `EXTENDS`, `IMPLEMENTS`, `INTERFACE`, `ABSTRACT` / `FINAL` / `OVERRIDE`, `PUBLIC` / `PRIVATE` / `PROTECTED` / `INTERNAL`, `THIS` / `SUPER`.
- Comments (`(* … *)`, `//`) and pragmas (`{ … }`, opaque body).
- Case-insensitive keyword matching at the lexer level via per-letter regex character classes; identifier maximal-munch resolves keyword-vs-identifier conflicts.
- Editor query files: `highlights.scm`, `locals.scm`, `tags.scm`, `folds.scm`, `indents.scm`, `injections.scm` — all using the standard tree-sitter highlight capture vocabulary.
- Bindings scaffolded for Node, Rust, Python, Go.
- Test corpus split by feature area (literals, expressions, statements, pou-declarations, var-blocks, data-types, oop, comments-and-pragmas).
- 13 real-world ST examples in `examples/` covering blink, traffic light, PID, debouncer, ramp generator, state machine, OOP shapes, conveyor with jam detection, temperature monitor with hysteresis, moving average, edge detectors, lookup table with linear interpolation, and a top-level `CONFIGURATION`.
- CI workflows (`ci.yml`, `prerelease.yml`, `release.yml`) per `BLUEPRINT.md` conventions, with grammar tests, examples parse-check, 10 000-line benchmark gate (200 ms budget), and per-binding builds on Ubuntu / macOS / Windows.
- `EXTENDING.md` with worked TwinCAT dialect-extension example.
- `SPEC-COMPLIANCE.md` mapping every IEC 61131-3 §6.x and §7 section to its grammar rule(s) and listing deliberate deviations.

<!--
  Add entries here while working on `develop` or `release/*`. The heading
  carries the next target version (`next: X.Y.Z`) so it's obvious what
  these entries will ship as. On release, rename to `[X.Y.Z] - YYYY-MM-DD`
  and start a new `[Unreleased] — next: X.Y.Z+1` block.

  Use these subsections, omitting any that don't apply:
    ### Added       — new features
    ### Changed     — changes in existing functionality
    ### Deprecated  — soon-to-be removed features
    ### Removed     — removed features
    ### Fixed       — bug fixes
    ### Security    — vulnerability fixes
-->
