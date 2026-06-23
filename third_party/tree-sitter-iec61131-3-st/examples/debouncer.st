(*
 * Debouncer FB — filters a noisy boolean input by requiring it to stay
 * stable for `debounce_time` before reporting the new state.
 *)

FUNCTION_BLOCK Debounce
VAR_INPUT
    input         : BOOL;
    debounce_time : TIME := T#20ms;
END_VAR

VAR_OUTPUT
    output : BOOL;
END_VAR

VAR
    timer        : TON;
    pending      : BOOL;
END_VAR

(* Detect change, start the timer, latch the new value when stable. *)
IF input <> output THEN
    pending := TRUE;
    timer(IN := TRUE, PT := debounce_time);
    IF timer.Q THEN
        output := input;
        pending := FALSE;
        timer(IN := FALSE);
    END_IF
ELSE
    pending := FALSE;
    timer(IN := FALSE);
END_IF
END_FUNCTION_BLOCK
