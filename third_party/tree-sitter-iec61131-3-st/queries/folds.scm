; Code-folding regions — IEC 61131-3 Structured Text

; POU bodies
(program_declaration)        @fold
(function_declaration)       @fold
(function_block_declaration) @fold
(interface_declaration)      @fold
(method_declaration)         @fold
(method_signature)           @fold
(property_declaration)       @fold
(property_signature)         @fold
(property_accessor)          @fold
(namespace_declaration)      @fold

; Type/configuration containers
(type_declaration)          @fold
(configuration_declaration) @fold
(resource_declaration)      @fold

; Variable blocks
(var_block)    @fold
(var_input)    @fold
(var_output)   @fold
(var_in_out)   @fold
(var_temp)     @fold
(var_global)   @fold
(var_external) @fold
(var_access)   @fold
(var_config)   @fold

; Compound statements
(if_statement)     @fold
(case_statement)   @fold
(for_statement)    @fold
(while_statement)  @fold
(repeat_statement) @fold

; Inline structure / enum
(structure_type_inline)  @fold
(enumerated_type_inline) @fold

; Comments long enough to be worth folding
(comment) @fold
