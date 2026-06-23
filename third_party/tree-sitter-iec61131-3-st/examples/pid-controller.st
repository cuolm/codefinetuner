(*
 * PID controller as a FUNCTION_BLOCK.
 *
 * Output = Kp*e + Ki*∫e dt + Kd*(de/dt), with anti-windup clamp on the
 * integral term.
 *)

FUNCTION_BLOCK PID
VAR_INPUT
    setpoint     : REAL;
    process_var  : REAL;
    Kp, Ki, Kd   : REAL;
    cycle_time   : TIME;
    out_min      : REAL := -100.0;
    out_max      : REAL :=  100.0;
    enable       : BOOL := TRUE;
END_VAR

VAR_OUTPUT
    output       : REAL;
END_VAR

VAR
    error        : REAL;
    prev_error   : REAL;
    integral     : REAL;
    derivative   : REAL;
    dt           : REAL;
END_VAR

IF NOT enable THEN
    integral := 0.0;
    output := 0.0;
    RETURN;
END_IF

dt := TIME_TO_REAL(cycle_time) / 1000.0;
error := setpoint - process_var;

(* Trapezoidal integration with anti-windup. *)
integral := integral + 0.5 * (error + prev_error) * dt;
IF integral > out_max THEN
    integral := out_max;
ELSIF integral < out_min THEN
    integral := out_min;
END_IF

IF dt > 0.0 THEN
    derivative := (error - prev_error) / dt;
ELSE
    derivative := 0.0;
END_IF

output := Kp * error + Ki * integral + Kd * derivative;
IF output > out_max THEN
    output := out_max;
ELSIF output < out_min THEN
    output := out_min;
END_IF

prev_error := error;
END_FUNCTION_BLOCK
