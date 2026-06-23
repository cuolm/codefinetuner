; Syntax highlighting queries — IEC 61131-3 Structured Text
;
; Capture names follow the standard tree-sitter highlight capture vocabulary
; so the grammar Just Works in any editor that uses tree-sitter highlighting
; (Helix, Neovim's nvim-treesitter, Zed, …).

; ---------------------------------------------------------------------------
; Comments and pragmas
; ---------------------------------------------------------------------------
(comment) @comment
(pragma)  @attribute

; ---------------------------------------------------------------------------
; Literals
; ---------------------------------------------------------------------------
(boolean_literal)        @boolean
(integer_literal)        @number
(real_literal)           @number.float
(string_literal)         @string
(time_literal)           @string.special
(date_literal)           @string.special
(time_of_day_literal)    @string.special
(date_and_time_literal)  @string.special

(typed_literal
  type: (identifier) @type)

; ---------------------------------------------------------------------------
; Types
; ---------------------------------------------------------------------------
(elementary_type) @type.builtin
(generic_type)    @type.builtin

(string_type
  (_) @type.builtin)

; User-defined type names in declarations
(type_definition
  name: (identifier) @type)

(structure_type_inline) @type
(enumerated_type_inline) @type

; Type references in variable declarations
(variable_declaration
  type: (identifier) @type)

(variable_declaration
  type: (qualified_identifier) @type)

(structure_field
  type: (identifier) @type)

; ---------------------------------------------------------------------------
; Identifiers — names of program/function/FB/method/property declarations.
; ---------------------------------------------------------------------------
(program_declaration         name: (identifier) @function)
(function_declaration        name: (identifier) @function)
(function_block_declaration  name: (identifier) @type)
(interface_declaration       name: (identifier) @type)
(method_declaration          name: (identifier) @function.method)
(method_signature            name: (identifier) @function.method)
(property_declaration        name: (identifier) @property)
(property_signature          name: (identifier) @property)
(namespace_declaration       name: (_) @namespace)

; Variable names in declarations
(variable_declaration
  names: (identifier) @variable)

(structure_field
  name: (identifier) @field)

(enumerator
  name: (identifier) @constant)

; FOR loop control variable
(for_statement
  control: (identifier) @variable)

; Named arguments in function calls — `param := value`
(named_argument
  name: (identifier) @parameter)

; Member access — the right-hand side names a field/method
(member_access_expression
  member: (identifier) @field)

; Function/method calls — highlight the call target
(call_expression
  function: (identifier) @function.call)
(call_expression
  function: (member_access_expression
    member: (identifier) @function.method))

; ---------------------------------------------------------------------------
; Special identifiers
; ---------------------------------------------------------------------------
(this_expression)  @variable.builtin
(super_expression) @variable.builtin

; Direct addresses are physical I/O references
(direct_address) @constant.builtin

; ---------------------------------------------------------------------------
; Keywords
; ---------------------------------------------------------------------------

; Top-level / POU
[
  "PROGRAM" "END_PROGRAM"
  "FUNCTION" "END_FUNCTION"
  "FUNCTION_BLOCK" "END_FUNCTION_BLOCK"
  "INTERFACE" "END_INTERFACE"
  "METHOD" "END_METHOD"
  "PROPERTY" "END_PROPERTY"
  "GET" "END_GET"
  "SET" "END_SET"
  "TYPE" "END_TYPE"
  "STRUCT" "END_STRUCT"
  "NAMESPACE" "END_NAMESPACE"
  "USING"
  "CONFIGURATION" "END_CONFIGURATION"
  "RESOURCE" "END_RESOURCE"
  "TASK"
] @keyword

; Variable blocks
[
  "VAR" "END_VAR"
  "VAR_INPUT" "VAR_OUTPUT" "VAR_IN_OUT" "VAR_TEMP"
  "VAR_GLOBAL" "VAR_EXTERNAL" "VAR_ACCESS" "VAR_CONFIG"
] @keyword

; Variable qualifiers
[
  "CONSTANT" "RETAIN" "NON_RETAIN"
] @keyword.modifier

; Class/method modifiers
[
  "ABSTRACT" "FINAL" "OVERRIDE"
  "PUBLIC" "PRIVATE" "PROTECTED" "INTERNAL"
] @keyword.modifier

; Inheritance / interface
[
  "EXTENDS" "IMPLEMENTS"
] @keyword

; Control flow
[
  "IF" "THEN" "ELSIF" "ELSE" "END_IF"
  "CASE" "OF" "END_CASE"
  "FOR" "TO" "BY" "DO" "END_FOR"
  "WHILE" "END_WHILE"
  "REPEAT" "UNTIL" "END_REPEAT"
  "EXIT" "CONTINUE" "RETURN"
] @keyword.control

; Type-construction keywords
[
  "ARRAY" "AT" "POINTER" "REF_TO"
  "READ_WRITE" "READ_ONLY"
  "ON" "WITH"
] @keyword

; Operator keywords
[
  "AND" "OR" "XOR" "NOT" "MOD" "ADR"
] @keyword.operator

; THIS / SUPER are highlighted as variable.builtin above; the keyword
; tokens themselves are styled here when they appear standalone.
"THIS"  @variable.builtin
"SUPER" @variable.builtin

; ---------------------------------------------------------------------------
; Operators (symbol tokens)
; ---------------------------------------------------------------------------
[
  ":=" "=>"
  "="  "<>"
  "<"  ">"  "<=" ">="
  "+"  "-"  "*"  "/"  "**"
  "&"  "^"
  ".."
] @operator

(ref_assign) @operator

; ---------------------------------------------------------------------------
; Punctuation
; ---------------------------------------------------------------------------
[ "(" ")" "[" "]" ] @punctuation.bracket
[ "," ":" ";" "."  ] @punctuation.delimiter
"#" @punctuation.special
