"""Stop Windows sleeping while a job runs.

Two modes:
  keep_awake.py growth_slot.py --rebalance   WRAPPER: holds the flag for exactly as long as the
                                             child runs, returns its exit code
  keep_awake.py 200                          SIDECAR: hold for N minutes
  keep_awake.py --selftest                   check both paths

Why the wrapper exists: Task Scheduler's WakeToRun can wake the host for a scheduled job, but
Windows can return to sleep seconds after a timer wake unless a power request is held, leaving the
job suspended mid-run. ES_SYSTEM_REQUIRED is exactly the flag that prevents that.

What it CANNOT do: override an explicit sleep or shutdown (a SetSuspendState call, or a user
choosing Sleep), and nothing helps if the host is powered off. It only stops the automatic
return-to-sleep while the job is running.
"""
import ctypes
import subprocess
import sys
import time

ES_CONTINUOUS = 0x80000000
ES_SYSTEM_REQUIRED = 0x00000001


def _set(flags):
    """-> previous state (0 means the call FAILED; Windows-only, absent elsewhere)."""
    k32 = getattr(ctypes, "windll", None)
    if k32 is None:
        return None                      # not Windows: nothing to hold, never block the job
    return k32.kernel32.SetThreadExecutionState(flags)


def hold():
    ok = _set(ES_CONTINUOUS | ES_SYSTEM_REQUIRED)
    if ok == 0:
        print("keep-awake: SetThreadExecutionState FAILED - the host may still sleep mid-run")
    return ok


def release():
    _set(ES_CONTINUOUS)


def run(cmd, _run=subprocess.run):
    """Run cmd under a power request; always release, even if it raises. -> child's exit code."""
    hold()
    try:
        return _run(cmd).returncode
    finally:
        release()


def main(argv):
    if not argv:
        argv = ["200"]
    head = argv[0]
    try:
        mins = float(head)
    except ValueError:
        pass                              # not a number -> it is a command to wrap
    else:
        hold()
        print(f"keep-awake armed for {mins:.0f} min", flush=True)
        try:
            time.sleep(mins * 60)
        finally:
            release()
            print("keep-awake released")
        return 0
    return run([sys.executable, *argv])


def selftest():
    rc = run([sys.executable, "-c", "import sys; sys.exit(7)"])
    assert rc == 7, f"wrapper must return the child's exit code, got {rc}"
    calls = []
    assert run(["x"], _run=lambda c: calls.append(c) or type("R", (), {"returncode": 0})()) == 0
    assert calls == [["x"]], calls
    try:                                  # the flag must be released even when the child blows up
        run(["y"], _run=lambda c: (_ for _ in ()).throw(OSError("boom")))
    except OSError:
        pass
    else:
        raise AssertionError("exception should propagate")
    assert isinstance(hold(), int) and hold() != 0, "SetThreadExecutionState should succeed here"
    release()
    print("selftest OK")
    return 0


if __name__ == "__main__":
    sys.exit(selftest() if "--selftest" in sys.argv[1:] else main(sys.argv[1:]))
