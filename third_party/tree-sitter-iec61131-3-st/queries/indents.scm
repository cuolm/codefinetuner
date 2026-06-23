; Indentation queries — IEC 61131-3 Structured Text
;
; Capture vocabulary follows nvim-treesitter / Helix conventions:
;   @indent.begin    — node's children should be indented one level deeper
;   @indent.end      — node ends an indented region
;   @indent.dedent   — current line should outdent
;   @indent.branch   — line aligns with the parent (e.g. ELSE, ELSIF)
;   @indent.zero     — current line is forced to column zero

; --- Increase indent inside POU bodies ---------------------------------

(program_declaration)        @indent.begin
(function_declaration)       @indent.begin
(function_block_declaration) @indent.begin
(interface_declaration)      @indent.begin
(method_declaration)         @indent.begin
(method_signature)           @indent.begin
(property_declaration)       @indent.begin
(property_signature)         @indent.begin
(property_accessor)          @indent.begin
(namespace_declaration)      @indent.begin
(type_declaration)           @indent.begin
(configuration_declaration)  @indent.begin
(resource_declaration)       @indent.begin

; --- Variable blocks ---------------------------------------------------

(var_block)    @indent.begin
(var_input)    @indent.begin
(var_output)   @indent.begin
(var_in_out)   @indent.begin
(var_temp)     @indent.begin
(var_global)   @indent.begin
(var_external) @indent.begin
(var_access)   @indent.begin
(var_config)   @indent.begin

; --- Compound statements -----------------------------------------------

(if_statement)     @indent.begin
(elsif_clause)     @indent.begin
(else_clause)      @indent.begin
(case_statement)   @indent.begin
(case_clause)      @indent.begin
(for_statement)    @indent.begin
(while_statement)  @indent.begin
(repeat_statement) @indent.begin

; --- Aligned-with-parent keywords --------------------------------------

"ELSE"  @indent.branch
"ELSIF" @indent.branch
"UNTIL" @indent.branch

; --- Closing keywords end the region -----------------------------------

[
  "END_PROGRAM" "END_FUNCTION" "END_FUNCTION_BLOCK"
  "END_INTERFACE" "END_METHOD" "END_PROPERTY"
  "END_GET" "END_SET" "END_TYPE" "END_STRUCT"
  "END_NAMESPACE" "END_CONFIGURATION" "END_RESOURCE"
  "END_VAR"
  "END_IF" "END_CASE" "END_FOR" "END_WHILE" "END_REPEAT"
] @indent.end
