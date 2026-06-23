(*
 * Rising / falling edge detector — minimal R_TRIG / F_TRIG style FBs
 * implemented in ST.
 *)

FUNCTION_BLOCK RisingEdge
VAR_INPUT  CLK : BOOL; END_VAR
VAR_OUTPUT Q   : BOOL; END_VAR
VAR        prev : BOOL; END_VAR

Q := CLK AND NOT prev;
prev := CLK;
END_FUNCTION_BLOCK

FUNCTION_BLOCK FallingEdge
VAR_INPUT  CLK : BOOL; END_VAR
VAR_OUTPUT Q   : BOOL; END_VAR
VAR        prev : BOOL := FALSE; END_VAR

Q := NOT CLK AND prev;
prev := CLK;
END_FUNCTION_BLOCK

FUNCTION_BLOCK BothEdges
VAR_INPUT  CLK : BOOL; END_VAR
VAR_OUTPUT Q   : BOOL; END_VAR
VAR        prev : BOOL; END_VAR

Q := CLK <> prev;
prev := CLK;
END_FUNCTION_BLOCK
