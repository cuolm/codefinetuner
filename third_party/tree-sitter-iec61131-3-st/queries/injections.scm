; Language-injection queries — IEC 61131-3 Structured Text
;
; Inject the standard `comment` mini-grammar into our `comment` nodes so
; editors with TODO/FIXME/NOTE highlighting (Helix, nvim-treesitter) can
; find them inside both line and block comments.
;
; Pragmas are not injected — their content is dialect-specific and is
; left for vendor dialect grammars to interpret structurally.

((comment) @injection.content
 (#set! injection.language "comment")
 (#set! injection.include-children))
