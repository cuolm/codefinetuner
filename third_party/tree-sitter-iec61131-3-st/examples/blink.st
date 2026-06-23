(*
 * Blink — toggle an output at a fixed period using TON timers.
 *)

PROGRAM Blink
VAR
    on_timer  : TON;
    off_timer : TON;
    output    : BOOL;
    period_on  : TIME := T#500ms;
    period_off : TIME := T#500ms;
END_VAR

on_timer(IN := NOT output, PT := period_on);
off_timer(IN := output, PT := period_off);

IF on_timer.Q THEN
    output := TRUE;
END_IF

IF off_timer.Q THEN
    output := FALSE;
END_IF
END_PROGRAM
