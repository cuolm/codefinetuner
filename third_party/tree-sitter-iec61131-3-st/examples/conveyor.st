(*
 * Conveyor with infeed/outfeed sensors, jam detection, and maintenance
 * counter.
 *)

PROGRAM Conveyor
VAR_INPUT
    start_button   : BOOL;
    stop_button    : BOOL;
    estop_button   : BOOL;
    infeed_sensor  : BOOL;
    outfeed_sensor : BOOL;
    motor_overload : BOOL;
END_VAR

VAR_OUTPUT
    motor_run      : BOOL;
    fault_lamp     : BOOL;
    parts_counted  : DINT;
END_VAR

VAR
    jam_timer       : TON;
    edge_outfeed    : R_TRIG;
    last_seen_part  : TIME;
    runtime_hours   : DINT;
    runtime_seconds : TON;
END_VAR

VAR CONSTANT
    JAM_TIMEOUT : TIME := T#10s;
END_VAR

(* E-stop and overload force shutdown immediately. *)
IF estop_button OR motor_overload THEN
    motor_run := FALSE;
    fault_lamp := TRUE;
    RETURN;
END_IF

(* Latch start/stop. *)
IF start_button AND NOT stop_button THEN
    motor_run := TRUE;
    fault_lamp := FALSE;
END_IF
IF stop_button THEN
    motor_run := FALSE;
END_IF

(* Jam detection: if the infeed sees a part but the outfeed never does. *)
jam_timer(IN := infeed_sensor AND NOT outfeed_sensor, PT := JAM_TIMEOUT);
IF jam_timer.Q THEN
    motor_run := FALSE;
    fault_lamp := TRUE;
END_IF

(* Count rising edges on the outfeed sensor. *)
edge_outfeed(CLK := outfeed_sensor);
IF edge_outfeed.Q THEN
    parts_counted := parts_counted + 1;
END_IF

(* Run-time counter — increments every hour the motor is on. *)
runtime_seconds(IN := motor_run, PT := T#1h);
IF runtime_seconds.Q THEN
    runtime_hours := runtime_hours + 1;
    runtime_seconds(IN := FALSE);
END_IF
END_PROGRAM
