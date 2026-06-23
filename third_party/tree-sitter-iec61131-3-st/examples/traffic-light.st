(*
 * Traffic Light controller — single-intersection with NS / EW phases.
 *)

TYPE
    Phase : (NS_GREEN, NS_YELLOW, ALL_RED1, EW_GREEN, EW_YELLOW, ALL_RED2);
END_TYPE

PROGRAM TrafficLight
VAR
    state          : Phase := NS_GREEN;
    phase_timer    : TON;
    green_time     : TIME := T#15s;
    yellow_time    : TIME := T#3s;
    all_red_time   : TIME := T#1s;

    ns_red, ns_yellow, ns_green : BOOL;
    ew_red, ew_yellow, ew_green : BOOL;
END_VAR

CASE state OF
    NS_GREEN:
        ns_green := TRUE; ns_yellow := FALSE; ns_red := FALSE;
        ew_green := FALSE; ew_yellow := FALSE; ew_red := TRUE;
        phase_timer(IN := TRUE, PT := green_time);
        IF phase_timer.Q THEN
            phase_timer(IN := FALSE);
            state := NS_YELLOW;
        END_IF

    NS_YELLOW:
        ns_green := FALSE; ns_yellow := TRUE;
        phase_timer(IN := TRUE, PT := yellow_time);
        IF phase_timer.Q THEN
            phase_timer(IN := FALSE);
            state := ALL_RED1;
        END_IF

    ALL_RED1:
        ns_yellow := FALSE; ns_red := TRUE;
        phase_timer(IN := TRUE, PT := all_red_time);
        IF phase_timer.Q THEN
            phase_timer(IN := FALSE);
            state := EW_GREEN;
        END_IF

    EW_GREEN:
        ew_red := FALSE; ew_green := TRUE;
        phase_timer(IN := TRUE, PT := green_time);
        IF phase_timer.Q THEN
            phase_timer(IN := FALSE);
            state := EW_YELLOW;
        END_IF

    EW_YELLOW:
        ew_green := FALSE; ew_yellow := TRUE;
        phase_timer(IN := TRUE, PT := yellow_time);
        IF phase_timer.Q THEN
            phase_timer(IN := FALSE);
            state := ALL_RED2;
        END_IF

    ALL_RED2:
        ew_yellow := FALSE; ew_red := TRUE;
        phase_timer(IN := TRUE, PT := all_red_time);
        IF phase_timer.Q THEN
            phase_timer(IN := FALSE);
            state := NS_GREEN;
        END_IF
END_CASE
END_PROGRAM
