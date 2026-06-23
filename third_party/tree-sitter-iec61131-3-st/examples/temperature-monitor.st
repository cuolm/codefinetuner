(*
 * Temperature monitor — reads a sensor, scales raw counts to engineering
 * units, applies hysteresis, and triggers high/low alarms.
 *)

FUNCTION_BLOCK TempMonitor
VAR_INPUT
    raw_counts        : INT;
    raw_min, raw_max  : INT  := 0;
    eu_min,  eu_max   : REAL := 0.0;
    high_setpoint     : REAL;
    low_setpoint      : REAL;
    hysteresis        : REAL := 1.0;
END_VAR

VAR_OUTPUT
    temperature : REAL;
    high_alarm  : BOOL;
    low_alarm   : BOOL;
    in_range    : BOOL;
END_VAR

VAR
    span_raw : REAL;
    span_eu  : REAL;
END_VAR

(* Linear scaling. *)
span_raw := REAL_TO_LREAL(raw_max - raw_min);
span_eu  := eu_max - eu_min;
IF span_raw > 0.0 THEN
    temperature := eu_min + (REAL_TO_LREAL(raw_counts - raw_min) / span_raw) * span_eu;
ELSE
    temperature := eu_min;
END_IF

(* High alarm with hysteresis: trips at high_setpoint, clears at high_setpoint - hysteresis. *)
IF high_alarm THEN
    IF temperature < (high_setpoint - hysteresis) THEN
        high_alarm := FALSE;
    END_IF
ELSE
    IF temperature >= high_setpoint THEN
        high_alarm := TRUE;
    END_IF
END_IF

(* Low alarm with hysteresis. *)
IF low_alarm THEN
    IF temperature > (low_setpoint + hysteresis) THEN
        low_alarm := FALSE;
    END_IF
ELSE
    IF temperature <= low_setpoint THEN
        low_alarm := TRUE;
    END_IF
END_IF

in_range := NOT high_alarm AND NOT low_alarm;
END_FUNCTION_BLOCK
