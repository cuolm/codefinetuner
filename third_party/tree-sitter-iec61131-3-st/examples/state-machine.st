(*
 * Sequential machine controller for a simple pick-and-place: HOME → DOWN →
 * GRIP → UP → MOVE → DOWN → RELEASE → UP → HOME.
 *)

TYPE
    Step : (
        HOME, MOVE_DOWN, GRIP, MOVE_UP, MOVE_OVER,
        DROP_DOWN, RELEASE, DROP_UP, RETURN_TO_HOME
    );
END_TYPE

PROGRAM PickAndPlace
VAR
    step : Step := HOME;
    cycle : DINT;
    z_axis_at_top, z_axis_at_bottom : BOOL;
    x_axis_at_pickup, x_axis_at_drop : BOOL;
    gripper_closed : BOOL;
    cmd_z_down, cmd_z_up : BOOL;
    cmd_x_to_drop, cmd_x_to_pickup : BOOL;
    cmd_grip_close, cmd_grip_open : BOOL;

    settle : TON;
END_VAR

CASE step OF
    HOME:
        IF z_axis_at_top AND x_axis_at_pickup THEN
            step := MOVE_DOWN;
        END_IF

    MOVE_DOWN:
        cmd_z_down := TRUE; cmd_z_up := FALSE;
        IF z_axis_at_bottom THEN
            cmd_z_down := FALSE;
            step := GRIP;
        END_IF

    GRIP:
        cmd_grip_close := TRUE;
        settle(IN := TRUE, PT := T#200ms);
        IF settle.Q THEN
            settle(IN := FALSE);
            gripper_closed := TRUE;
            step := MOVE_UP;
        END_IF

    MOVE_UP:
        cmd_z_up := TRUE;
        IF z_axis_at_top THEN
            cmd_z_up := FALSE;
            step := MOVE_OVER;
        END_IF

    MOVE_OVER:
        cmd_x_to_drop := TRUE;
        IF x_axis_at_drop THEN
            cmd_x_to_drop := FALSE;
            step := DROP_DOWN;
        END_IF

    DROP_DOWN:
        cmd_z_down := TRUE;
        IF z_axis_at_bottom THEN
            cmd_z_down := FALSE;
            step := RELEASE;
        END_IF

    RELEASE:
        cmd_grip_close := FALSE; cmd_grip_open := TRUE;
        settle(IN := TRUE, PT := T#200ms);
        IF settle.Q THEN
            settle(IN := FALSE);
            cmd_grip_open := FALSE;
            gripper_closed := FALSE;
            step := DROP_UP;
        END_IF

    DROP_UP:
        cmd_z_up := TRUE;
        IF z_axis_at_top THEN
            cmd_z_up := FALSE;
            step := RETURN_TO_HOME;
        END_IF

    RETURN_TO_HOME:
        cmd_x_to_pickup := TRUE;
        IF x_axis_at_pickup THEN
            cmd_x_to_pickup := FALSE;
            cycle := cycle + 1;
            step := HOME;
        END_IF
END_CASE
END_PROGRAM
