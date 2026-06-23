(*
 * Sample TwinCAT-flavored ST that exercises the dialect's extensions.
 *)

PROGRAM SystemReader
VAR
    task_idx : DINT;
    safe     : BOOL;
    p        : POINTER TO INT;
    val      : INT;
END_VAR

task_idx := __SYSTEM.GetCurrentTaskIndex();

(* Short-circuit AND_THEN: only dereference p if it's non-NULL. *)
safe := (p <> 0) AND_THEN (p^ > 0);
IF safe THEN
    val := p^;
END_IF
END_PROGRAM
