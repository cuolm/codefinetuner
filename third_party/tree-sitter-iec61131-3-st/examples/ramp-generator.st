(*
 * Ramp generator — drives `output` from its current value toward
 * `target` with a fixed slope (units per second).
 *)

FUNCTION_BLOCK Ramp
VAR_INPUT
    target     : REAL;
    slope      : REAL := 10.0;       (* units per second *)
    cycle_time : TIME := T#10ms;
    enable     : BOOL := TRUE;
END_VAR

VAR_OUTPUT
    output : REAL;
    done   : BOOL;
END_VAR

VAR
    delta_per_cycle : REAL;
    diff            : REAL;
END_VAR

IF NOT enable THEN
    done := FALSE;
    RETURN;
END_IF

delta_per_cycle := slope * TIME_TO_REAL(cycle_time) / 1000.0;
diff := target - output;

IF ABS(diff) <= delta_per_cycle THEN
    output := target;
    done := TRUE;
ELSIF diff > 0.0 THEN
    output := output + delta_per_cycle;
    done := FALSE;
ELSE
    output := output - delta_per_cycle;
    done := FALSE;
END_IF
END_FUNCTION_BLOCK
