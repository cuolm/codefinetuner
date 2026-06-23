/**
 * @file Worked dialect-extension example for tree-sitter-iec61131-3-st.
 * @license MIT
 *
 * This is a *minimal* TwinCAT-flavored dialect that adds two constructs
 * not present in the IEC 61131-3 standard:
 *
 *   1. `__SYSTEM.X(args)` — TwinCAT runtime intrinsic call. Parses as a
 *      `twincat_system_call` node distinct from a regular call.
 *
 *   2. `<expr> AND_THEN <expr>` and `<expr> OR_ELSE <expr>` — short-circuit
 *      logical operators. Inserted as new alternatives in the binary
 *      expression precedence ladder, just below the standard `AND` and
 *      `OR`.
 *
 * The point of this example is to demonstrate the EXTENSION MECHANISM, not
 * to be a complete TwinCAT grammar. A real `tree-sitter-iec61131-3-st-twincat`
 * repo would live in its own GitHub repository and add ~30+ more
 * constructs (S=, R=, REF=, attribute pragmas with structured contents,
 * conditional compilation, ACTION blocks, …).
 *
 * To run:
 *
 *   cd examples/dialect-extension
 *   npm link tree-sitter-iec61131-3-st          # or `npm install tree-sitter-iec61131-3-st` once published
 *   tree-sitter generate
 *   tree-sitter parse sample.st
 */

// In a real dialect repo this would be:
//   import base, { kw } from 'tree-sitter-iec61131-3-st/grammar';
// Inside this examples/ folder, we resolve to the parent directory:
import base, { kw } from '../../grammar.js';

export default grammar(base, {
  name: 'iec61131_3_st_twincat',

  // Extra conflicts created by the new alternatives in `_expression`.
  conflicts: ($, original) => [
    ...(original || []),
  ],

  rules: {
    // Add `twincat_system_call` as a new alternative to `_expression`. The
    // `original` parameter resolves to the base grammar's `_expression`
    // rule, so all standard expression forms still work.
    _expression: ($, original) =>
      choice(
        original,
        $.twincat_system_call,
        $.short_circuit_expression,
      ),

    twincat_system_call: $ =>
      seq(
        '__SYSTEM',
        '.',
        field('intrinsic', $.identifier),
        field('arguments', $.argument_list),
      ),

    // Short-circuit operators bind looser than `AND` / `OR` (per common
    // TwinCAT/Codesys convention) so they group correctly relative to
    // standard logical operators.
    short_circuit_expression: $ =>
      choice(
        prec.left(
          3,
          seq(
            field('left', $._expression),
            field('operator', kw('AND_THEN')),
            field('right', $._expression),
          ),
        ),
        prec.left(
          2,
          seq(
            field('left', $._expression),
            field('operator', kw('OR_ELSE')),
            field('right', $._expression),
          ),
        ),
      ),
  },
});
