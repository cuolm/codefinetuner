; Code-navigation tags — IEC 61131-3 Structured Text
;
; Used by editors and code-search tools (ctags-style) to extract a
; per-file symbol index. Capture names follow tree-sitter-tags conventions:
;   @definition.<kind>  for declared symbols
;   @name               for the symbol's identifier (sub-capture of definition)

; --- POUs ---------------------------------------------------------------

(program_declaration
  name: (identifier) @name) @definition.module

(function_declaration
  name: (identifier) @name) @definition.function

(function_block_declaration
  name: (identifier) @name) @definition.class

(interface_declaration
  name: (identifier) @name) @definition.interface

; --- Members ------------------------------------------------------------

(method_declaration
  name: (identifier) @name) @definition.method

(method_signature
  name: (identifier) @name) @definition.method

(property_declaration
  name: (identifier) @name) @definition.property

(property_signature
  name: (identifier) @name) @definition.property

; --- Types --------------------------------------------------------------

(type_definition
  name: (identifier) @name) @definition.type

(structure_field
  name: (identifier) @name) @definition.field

(enumerator
  name: (identifier) @name) @definition.constant

; --- Globals ------------------------------------------------------------

(var_global
  (variable_declaration
    names: (identifier) @name)) @definition.variable

; --- Namespaces ---------------------------------------------------------

(namespace_declaration
  name: (_) @name) @definition.namespace

; --- References ---------------------------------------------------------

(call_expression
  function: (identifier) @name) @reference.call

(call_expression
  function: (member_access_expression
    member: (identifier) @name)) @reference.call
