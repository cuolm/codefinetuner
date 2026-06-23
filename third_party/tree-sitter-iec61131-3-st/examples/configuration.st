(*
 * Top-level CONFIGURATION / RESOURCE / TASK / PROGRAM mapping.
 * Demonstrates the IEC 61131-3 system architecture layer.
 *)

CONFIGURATION PlantConfig

VAR_GLOBAL
    g_emergency_stop : BOOL;
    g_total_uptime   : DINT;
END_VAR

RESOURCE Cpu1 ON CPU_001
    VAR_GLOBAL
        line1_state : INT;
    END_VAR

    TASK FastScan(INTERVAL := T#10ms, PRIORITY := 1);
    TASK SlowScan(INTERVAL := T#100ms, PRIORITY := 5);

    PROGRAM IO_Update     WITH FastScan : Conveyor;
    PROGRAM Coordinator   WITH SlowScan : PickAndPlace;
END_RESOURCE

RESOURCE Cpu2 ON CPU_002
    VAR_GLOBAL
        line2_state : INT;
    END_VAR

    TASK Periodic(INTERVAL := T#50ms, PRIORITY := 3);
    PROGRAM Hmi WITH Periodic : TempMonitor;
END_RESOURCE

END_CONFIGURATION
