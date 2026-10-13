"""Wait (briefly) for the broker's API host before a scheduled job starts.

Run it just before a scheduled job (e.g. growth_slot.py --rebalance --live). After a wake or reboot,
Task Scheduler can start the job the instant Windows is back, before DNS is up; the job then cannot
reach the broker and does nothing that day. The host is ALPACA_BASE_URL_GROWTH's, defaulting to
the Alpaca paper endpoint.

Always exits 0. If the network never comes up, the job still runs and fails cleanly on its own;
this only removes the "woke up one second too early" failure. Waits at most MAX_WAIT_S.
    python wait_for_network.py --selftest
"""
import datetime as dt
import os
import socket
import sys
import time
from urllib.parse import urlparse

DEFAULT_HOST = "paper-api.alpaca.markets"
HOST = urlparse(os.environ.get("ALPACA_BASE_URL_GROWTH", "")).hostname or DEFAULT_HOST
MAX_WAIT_S = 180
POLL_S = 5


def up(host=HOST):
    """True when `host` resolves AND accepts a TCP connection on 443 (DNS alone can come up first)."""
    try:
        socket.create_connection((host, 443), timeout=5).close()
        return True
    except OSError:
        return False


def wait(up=up, sleep=time.sleep, clock=time.monotonic):
    """-> (network_is_up, seconds_waited)."""
    t0 = clock()
    while not up():
        if clock() - t0 >= MAX_WAIT_S:
            return False, clock() - t0
        sleep(POLL_S)
    return True, clock() - t0


def main():
    ok, waited = wait()
    if waited >= 1 or not ok:          # silent in the normal case - no log noise on ordinary days
        stamp = dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f"{stamp}  NETWORK WAIT  {HOST}: " + (f"up after {waited:.0f}s" if ok else
              f"STILL DOWN after {waited:.0f}s - starting the job anyway (it will fail cleanly)"))
    return 0


def selftest():
    def fake(up_after):                # a clock that only moves when we sleep
        t, calls = [0.0], [0]
        def _up():
            calls[0] += 1
            return up_after is not None and t[0] >= up_after
        def _sleep(s):
            t[0] += s
        return wait(up=_up, sleep=_sleep, clock=lambda: t[0]), calls[0]
    assert fake(0) == ((True, 0.0), 1)                       # already up: no waiting at all
    assert fake(12) == ((True, 15.0), 4)                     # up on the 4th probe
    (ok, waited), _ = fake(None)                             # never up: gives up at the cap
    assert ok is False and waited == MAX_WAIT_S
    assert urlparse("https://paper-api.alpaca.markets/v2").hostname == DEFAULT_HOST
    print("selftest OK")
    return 0


if __name__ == "__main__":
    sys.exit(selftest() if "--selftest" in sys.argv else main())
