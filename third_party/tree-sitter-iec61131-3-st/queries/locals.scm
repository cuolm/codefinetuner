; Local-scope queries — IEC 61131-3 Structured Text
;
; Marks scopes (POU bodies, methods, variable blocks) and the definitions /
; references that drive go-to-definition and rename refactorings.

; ---------------------------------------------------------------------------
; Scopes
; ---------------------------------------------------------------------------
(program_declaration)         @local.scope
(function_declaration)        @local.scope
(function_block_declaration)  @local.scope
(interface_declaration)       @local.scope
(method_declaration)          @local.scope
(method_signature)            @local.scope
(property_declaration)        @local.scope
(property_signature)          @local.scope
(property_accessor)           @local.scope
(namespace_declaration)       @local.scope
(type_declaration)            @local.scope
(configuration_declaration)   @local.scope
(resource_declaration)        @local.scope

; Loop body introduces a scope (for the FOR control variable, mostly).
(for_statement) @local.scope

; ---------------------------------------------------------------------------
; Definitions
; ---------------------------------------------------------------------------
(program_declaration         name: (identifier) @local.definition.function)
(function_declaration        name: (identifier) @local.definition.function)
(function_block_declaration  name: (identifier) @local.definition.type)
(interface_declaration       name: (identifier) @local.definition.type)
(method_declaration          name: (identifier) @local.definition.method)
(method_signature            name: (identifier) @local.definition.method)
(property_declaration        name: (identifier) @local.definition.property)
(property_signature          name: (identifier) @local.definition.property)

(type_definition             name: (identifier) @local.definition.type)
(structure_field             name: (identifier) @local.definition.field)
(enumerator                  name: (identifier) @local.definition.constant)

(variable_declaration
  names: (identifier) @local.definition.var)

(named_argument
  name: (identifier) @local.definition.parameter)

(for_statement
  control: (identifier) @local.definition.var)

; ---------------------------------------------------------------------------
; References
; ---------------------------------------------------------------------------
(identifier) @local.reference
