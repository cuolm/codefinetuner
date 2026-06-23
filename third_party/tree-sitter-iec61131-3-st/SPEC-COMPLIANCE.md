# IEC 61131-3 specification compliance

This document maps the IEC 61131-3 (3rd edition, 2013) Structured Text language sections to the grammar rules in `grammar.js`. It also notes deliberate deviations and out-of-scope sections.

The clause numbers below follow IEC 61131-3:2013 (the 3rd edition). Where the standard is paywalled, the OSCAT, OpenPLC, and blark projects' public source corpora were used to validate behavior on real-world ST.

---

## §6.1  Lexical elements

| §        | Topic                                | Rule(s)                                |
|----------|--------------------------------------|----------------------------------------|
| §6.1.1   | Whitespace and case-insensitivity    | `extras: [/\s/, $.comment]`; `kw()` helper |
| §6.1.2   | Identifiers                          | `identifier`, `qualified_identifier`   |
| §6.1.3   | Comments                             | `comment` (line `//`, block `(* *)`)   |
| §6.1.4   | Pragmas (3rd edition addition)       | `pragma` (opaque body, `{ … }`)        |

### Deviations
- **Block comment nesting:** the standard does NOT allow nested `(* … *)`. This grammar follows the standard. Vendor dialects (notably TwinCAT) that allow nesting must override the `comment` rule and introduce an external scanner if they want true nesting support.
- **`{ … }` braces:** the 2nd edition (2003) historically permitted `{ … }` as an alternative comment form. The 3rd edition (2013) repurposes them for **pragmas**. This grammar follows the 3rd edition: `{ … }` is always a `pragma` node, never a comment.

## §6.2  Numeric literals

| §        | Topic                                  | Rule(s)               |
|----------|----------------------------------------|-----------------------|
| §6.2.1   | Integer literals (plain, base-prefixed)| `integer_literal`     |
| §6.2.2   | Real literals with optional exponent   | `real_literal`        |
| §6.2.3   | Underscore separators                   | both rules accept `_` |
| §6.2.4   | Type-prefixed literals (`INT#42`, …)    | `typed_literal`       |

## §6.3  String literals

| §        | Topic                                | Rule(s)             |
|----------|--------------------------------------|---------------------|
| §6.3.1   | Single-byte (`'…'`)                  | `string_literal`    |
| §6.3.2   | Double-byte (`"…"`)                  | `string_literal`    |
| §6.3.3   | `$` escape sequences                 | `string_literal`    |

Supported escapes: `$$`, `$'`, `$"`, `$L`, `$N`, `$P`, `$R`, `$T`, `$<2-hex>` for 8-bit strings, `$<4-hex>` for 16-bit (WSTRING) strings.

## §6.4  Time, date, time-of-day, date-and-time

| §        | Topic                       | Rule(s)                  |
|----------|-----------------------------|--------------------------|
| §6.4.1   | `T#…` / `TIME#…` / `LTIME#`  | `time_literal`           |
| §6.4.2   | `D#…` / `DATE#…` / `LDATE#…` | `date_literal`           |
| §6.4.3   | `TOD#…` / `TIME_OF_DAY#…` / `LTOD#…` | `time_of_day_literal` |
| §6.4.4   | `DT#…` / `DATE_AND_TIME#…` / `LDT#…` | `date_and_time_literal` |

## §6.5  Data types

| §        | Topic                              | Rule(s)                      |
|----------|------------------------------------|------------------------------|
| §6.5.1   | Elementary types                   | `elementary_type`            |
| §6.5.2   | Generic `ANY_*` types              | `generic_type`               |
| §6.5.3.1 | Subrange types                     | `subrange_type`              |
| §6.5.3.2 | Enumerated types                   | `enumerated_type_inline`     |
| §6.5.3.3 | Array types                        | `array_type`, `subrange`     |
| §6.5.3.4 | Structure types                    | `structure_type_inline`      |
| §6.5.3.5 | String types with length           | `string_type`                |
| §6.5.4   | `POINTER TO`                       | `pointer_type`               |
| §6.5.5   | `REF_TO`                           | `reference_type`             |
| §6.5.6   | TYPE … END_TYPE                    | `type_declaration`, `type_definition` |

## §6.6  Variables and direct addressing

| §        | Topic                                | Rule(s)                  |
|----------|--------------------------------------|--------------------------|
| §6.6.1   | Variable declaration                 | `variable_declaration`   |
| §6.6.2   | Initial value                        | `_initializer`, `structure_initializer` |
| §6.6.3   | `AT %{I,Q,M}{X,B,W,D,L}<addr>`       | `direct_address`         |
| §6.6.4   | Array initialization with repetition `[1, 2(3), 4]` | `_array_initializer`, `array_repetition` |
| §6.6.5   | Operator precedence (Table 55)       | `binary_expression`, `unary_expression`, `PREC` table |

### Operator precedence ladder
Higher number = tighter binding. Right-associative only on `**`.

| Level | Operators                  |
|-------|----------------------------|
| 14    | literals, identifiers, `()` (primary) |
| 13    | `^` deref, `[i]`, `.field`, `(args)` (postfix) |
| 12    | `-` `+` `NOT` (unary)       |
| 11    | `**` (right-associative)    |
| 10    | `*` `/` `MOD`              |
|  9    | `+` `-`                    |
|  8    | `<` `>` `<=` `>=`          |
|  7    | `=` `<>`                   |
|  6    | `AND` `&`                  |
|  5    | `XOR`                      |
|  4    | `OR`                       |

## §6.7  Statements

| §        | Topic                                | Rule(s)                   |
|----------|--------------------------------------|---------------------------|
| §6.7.1.1 | Assignment `:=`                      | `assignment_statement`    |
| §6.7.1.2 | Reference assignment `REF=`          | `reference_assignment_statement` |
| §6.7.2   | Function/method invocation as stmt   | `invocation_statement`    |
| §6.7.3   | `RETURN` (with optional value)       | `return_statement`        |
| §6.7.4.1 | `IF / ELSIF / ELSE / END_IF`         | `if_statement`, `elsif_clause`, `else_clause` |
| §6.7.4.2 | `CASE / OF / ELSE / END_CASE`        | `case_statement`, `case_clause`, `case_value`  |
| §6.7.4.3 | `FOR / TO / BY / DO / END_FOR`       | `for_statement`           |
| §6.7.4.4 | `WHILE / DO / END_WHILE`             | `while_statement`         |
| §6.7.4.5 | `REPEAT / UNTIL / END_REPEAT`        | `repeat_statement`        |
| §6.7.4.6 | `EXIT`                               | `exit_statement`          |
| §6.7.4.7 | `CONTINUE`                           | `continue_statement`      |
| §6.7.5   | Empty statement (`;`)                | `empty_statement`         |

## §6.8  Program Organization Units (POUs)

| §        | Topic                                | Rule(s)                       |
|----------|--------------------------------------|-------------------------------|
| §6.8.1   | `FUNCTION … END_FUNCTION`            | `function_declaration`        |
| §6.8.2   | `FUNCTION_BLOCK … END_FUNCTION_BLOCK` | `function_block_declaration`  |
| §6.8.3   | `PROGRAM … END_PROGRAM`              | `program_declaration`         |
| §6.8.4   | `INTERFACE … END_INTERFACE`          | `interface_declaration`       |
| §6.8.5   | `METHOD … END_METHOD`                | `method_declaration`, `method_signature` |
| §6.8.6   | `PROPERTY … END_PROPERTY` with GET/SET | `property_declaration`, `property_accessor`, `property_signature` |
| §6.8.7   | `EXTENDS` / `IMPLEMENTS`             | inline in `function_block_declaration`, `interface_declaration` |
| §6.8.8   | Access modifiers                     | `_access_modifier` (`PUBLIC` / `PRIVATE` / `PROTECTED` / `INTERNAL`) |
| §6.8.9   | `ABSTRACT` / `FINAL` / `OVERRIDE`    | `_fb_modifier`, `_method_modifier` |
| §6.8.10  | `THIS` / `SUPER`                     | `this_expression`, `super_expression` |

## §6.9  Namespaces (3rd edition addition)

| §        | Topic                                | Rule(s)                  |
|----------|--------------------------------------|--------------------------|
| §6.9.1   | `NAMESPACE … END_NAMESPACE`          | `namespace_declaration`  |
| §6.9.2   | `USING` directive                    | `using_directive`        |
| §6.9.3   | Qualified names (`A.B.C`)            | `qualified_identifier`   |

## §7  Configuration and resources

| §        | Topic                                | Rule(s)                       |
|----------|--------------------------------------|-------------------------------|
| §7.1     | `CONFIGURATION … END_CONFIGURATION`  | `configuration_declaration`   |
| §7.2     | `RESOURCE … ON cpu_type … END_RESOURCE` | `resource_declaration`     |
| §7.3     | `TASK` declaration                   | `task_declaration`, `task_parameter` |
| §7.4     | `PROGRAM ... WITH ... :` assignment  | `program_assignment`          |

## §B  Generic / `ANY_*` types

`ANY`, `ANY_DERIVED`, `ANY_ELEMENTARY`, `ANY_MAGNITUDE`, `ANY_NUM`, `ANY_REAL`, `ANY_INT`, `ANY_BIT`, `ANY_STRING`, `ANY_DATE`, `ANY_CHAR`, `ANY_CHARS` are all matched by `generic_type`.

---

## Out of scope for this grammar

The following are deliberately NOT covered. Each is either a different IEC 61131-3 language (out-of-scope per the project's stated scope) or a concern that belongs in a downstream tool.

| What                       | Why out of scope                                         | Where it should live                |
|----------------------------|----------------------------------------------------------|-------------------------------------|
| Ladder Diagram (LD)        | Different language family (2-D graphical).              | A separate grammar (likely not tree-sitter-friendly). |
| Function Block Diagram (FBD) | Different language family (2-D graphical).            | Out of scope. |
| Instruction List (IL)      | Different language family. Deprecated in 3rd ed annex.   | A separate grammar if anyone needs it. |
| Sequential Function Chart (SFC) | Different language family (2-D graphical with embedded ST/IL transitions). | A separate grammar; could embed this one for transition expressions. |
| Type checking              | A semantic concern.                                      | A separate analyzer / language server. |
| Symbol resolution          | A semantic concern.                                      | A separate analyzer / language server. |
| Code generation            | Out of scope.                                            | A separate tool. |
| Cross-file analysis        | Per-file parsing is the contract here.                   | A separate analyzer. |
| Vendor-specific extensions | Documented in EXTENDING.md.                              | `tree-sitter-iec61131-3-st-<vendor>` repos. |

---

## Known deliberate deviations from a strict reading of the standard

1. **Lvalue is any `_expression`.** The standard restricts the left side of `:=` to assignable expressions (variables, indexing, member access, dereference). This grammar accepts any expression and defers the semantic check downstream. Rationale: keeps the grammar smaller and avoids LR(1) conflicts; the same approach is used by `klauer/blark`.

2. **Trailing `;` after `END_IF` / `END_CASE` / etc.** is NOT consumed by the compound-statement rule. If a `;` appears after a compound statement's closing keyword, it parses as an `empty_statement`. Rationale: consuming it optionally inside the compound rule creates a parse ambiguity that tree-sitter cannot resolve cleanly.

3. **Inline default in `array_type` and `enumerated_type_inline`.** The standard syntax allows `(A, B, C) := A` and `ARRAY[1..3] OF INT := [1, 2, 3]` as part of the type itself. This grammar parses the `:= initial_value` portion at the surrounding context (variable declaration / type definition), not inside the type rule. Rationale: avoids a parse conflict with the surrounding declaration's own initializer.

4. **`AND_THEN` / `OR_ELSE` short-circuit operators** are NOT in the base grammar. They are TwinCAT/Codesys extensions; standard ST uses non-short-circuiting `AND` / `OR`. Dialect grammars may add them.

5. **`S=` / `R=` set/reset assignments** are NOT in the base grammar. Same reason as above; they are TwinCAT-style extensions.

6. **`PERSISTENT` / `NON_RETAIN` qualifiers** — `NON_RETAIN` IS recognized (it appears in the IEC 61131-3 amendment 1); `PERSISTENT` is a TwinCAT-only extension and lives in the dialect grammar.
