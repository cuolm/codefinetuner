(*
 * Linear interpolation across a lookup table. Useful for sensor
 * linearization curves and gain schedulers.
 *)

TYPE
    LookupRow : STRUCT
        x : REAL;
        y : REAL;
    END_STRUCT;
END_TYPE

FUNCTION_BLOCK LookupTable
VAR_INPUT
    x          : REAL;
    table      : ARRAY [0..15] OF LookupRow;
    table_size : INT := 16;
    extrapolate : BOOL := FALSE;
END_VAR

VAR_OUTPUT
    y          : REAL;
    out_of_range : BOOL;
END_VAR

VAR
    i          : INT;
    span       : REAL;
    fraction   : REAL;
END_VAR

out_of_range := FALSE;

(* Below the first point. *)
IF x <= table[0].x THEN
    IF extrapolate AND table_size >= 2 THEN
        span := table[1].x - table[0].x;
        IF span > 0.0 THEN
            fraction := (x - table[0].x) / span;
            y := table[0].y + fraction * (table[1].y - table[0].y);
        ELSE
            y := table[0].y;
        END_IF
    ELSE
        y := table[0].y;
        out_of_range := x < table[0].x;
    END_IF
    RETURN;
END_IF

(* Above the last point. *)
IF x >= table[table_size - 1].x THEN
    IF extrapolate AND table_size >= 2 THEN
        span := table[table_size - 1].x - table[table_size - 2].x;
        IF span > 0.0 THEN
            fraction := (x - table[table_size - 2].x) / span;
            y := table[table_size - 2].y + fraction * (table[table_size - 1].y - table[table_size - 2].y);
        ELSE
            y := table[table_size - 1].y;
        END_IF
    ELSE
        y := table[table_size - 1].y;
        out_of_range := x > table[table_size - 1].x;
    END_IF
    RETURN;
END_IF

(* Find bracketing pair and linearly interpolate. *)
FOR i := 0 TO table_size - 2 DO
    IF x >= table[i].x AND x <= table[i + 1].x THEN
        span := table[i + 1].x - table[i].x;
        IF span > 0.0 THEN
            fraction := (x - table[i].x) / span;
            y := table[i].y + fraction * (table[i + 1].y - table[i].y);
        ELSE
            y := table[i].y;
        END_IF
        RETURN;
    END_IF
END_FOR
END_FUNCTION_BLOCK
