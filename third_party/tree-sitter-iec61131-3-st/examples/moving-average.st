(*
 * N-sample circular-buffer moving average. Demonstrates ARRAY indexing,
 * MOD arithmetic, and instance-level state preservation.
 *)

FUNCTION_BLOCK MovingAverage
VAR_INPUT
    sample : REAL;
    reset  : BOOL;
END_VAR

VAR_OUTPUT
    average : REAL;
    full    : BOOL;
END_VAR

VAR CONSTANT
    WINDOW : INT := 16;
END_VAR

VAR
    buffer : ARRAY [0..15] OF REAL;
    head   : INT := 0;
    count  : INT := 0;
    sum    : REAL := 0.0;
    i      : INT;
END_VAR

IF reset THEN
    FOR i := 0 TO WINDOW - 1 DO
        buffer[i] := 0.0;
    END_FOR
    head := 0;
    count := 0;
    sum := 0.0;
    average := 0.0;
    full := FALSE;
    RETURN;
END_IF

(* Replace oldest sample, then advance the head pointer. *)
sum := sum - buffer[head] + sample;
buffer[head] := sample;
head := (head + 1) MOD WINDOW;

IF count < WINDOW THEN
    count := count + 1;
END_IF

average := sum / INT_TO_REAL(count);
full := count = WINDOW;
END_FUNCTION_BLOCK
