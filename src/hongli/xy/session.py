"""AmazingData session — logout iron rule (spec §3.4).

Every code path that logs in MUST leave via AmazingSession context manager;
__exit__ guarantees ad.logout(username) in a finally block.
"""
from __future__ import annotations

import re
import sys
import time
from contextlib import contextmanager
from pathlib import Path

import pandas as pd


def read_account(account_file: str) -> tuple[str, str, str, int]:
    """Parse account.txt -> (username, password, host, port).

    Handles lines like:
      密码（AD_PASSWORD）：your_password_here
      set AD_USERNAME=550100093586        (CMD export)
      $env:AD_HOST="101.230.159.234"      (PowerShell export)
      101.230.159.234                     (bare host)
    """
    text = Path(account_file).read_text(encoding="utf-8", errors="ignore")
    user = pw = host = ""
    port = 8600
    bare_hosts = []
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        # env-var style: KEY=VALUE possibly wrapped in quotes / $env: prefix
        m = re.search(r"AD_USERNAME\s*=\s*\"?([A-Za-z0-9_.-]+)\"?", line)
        if m:
            user = m.group(1)
            continue
        m = re.search(r"AD_PASSWORD\s*=\s*\"?([^\s\"']+)\"?", line)
        if m:
            pw = m.group(1)
            continue
        m = re.search(r"AD_HOST\s*=\s*\"?([\d.]+)\"?", line)
        if m:
            host = m.group(1)
            continue
        m = re.search(r"AD_PORT\s*=\s*\"?(\d{4,5})\"?", line)
        if m:
            port = int(m.group(1))
            continue
        # labeled CN style: 密码（AD_PASSWORD）：your_password_here
        if "AD_USERNAME" in line:
            user = _last_token(line)
        elif "AD_PASSWORD" in line:
            pw = _last_token(line)
        elif "AD_PORT" in line:
            m = re.search(r"(\d{4,5})", line)
            if m:
                port = int(m.group(1))
        elif "AD_HOST" in line:
            m = re.search(r"(\d+\.\d+\.\d+\.\d+)", line)
            if m:
                host = m.group(1)
        elif re.fullmatch(r"\d+\.\d+\.\d+\.\d+", line):
            bare_hosts.append(line)
    if not host and bare_hosts:
        host = bare_hosts[0]
    if not user or not pw:
        raise ValueError(f"cannot parse credentials from {account_file}")
    return user, pw, host, port


def _last_token(line: str) -> str:
    token = re.split(r"[:：=]", line)[-1].strip().strip('"').strip("'")
    token = token.split()[-1] if token.split() else token
    return token


class AmazingSession:
    """Context manager. Login on __enter__, mandatory logout on __exit__."""

    def __init__(self, account_file: str, host_primary: str, host_backup: str, port: int):
        self.account_file = account_file
        self.host_primary = host_primary
        self.host_backup = host_backup
        self.port = port
        self.ad = None
        self.base_data = None
        self.market_data = None
        self.info_data = None
        self.username = ""
        self.logged_in = False

    def __enter__(self) -> "AmazingSession":
        import AmazingData as ad  # local import: heavy dependency
        self.ad = ad
        user, pw, host_file, port_file = read_account(self.account_file)
        self.username = user
        host = self.host_primary or host_file
        port = self.port or port_file
        last_err = None
        for h in [host, self.host_backup]:
            if not h:
                continue
            try:
                t0 = time.time()
                ok = ad.login(user, pw, h, port)
                dt = time.time() - t0
                if ok in (True, 0, None) or ok is not None:
                    self.logged_in = True
                    print(f"[session] login ok via {h}:{port} ({dt:.1f}s)", file=sys.stderr)
                    break
            except Exception as e:  # noqa: BLE001 — try backup host
                last_err = e
                print(f"[session] login failed via {h}: {e}", file=sys.stderr)
        if not self.logged_in:
            raise ConnectionError(f"AmazingData login failed on both hosts: {last_err}")
        self.base_data = ad.BaseData()
        # get_calendar occasionally returns None on transient push-channel failure
        # (seen after prior logout). Retry with backoff before giving up.
        cal = None
        for attempt in range(4):
            try:
                cal = self.base_data.get_calendar()
            except Exception as e:  # noqa: BLE001
                last_err = e
            if cal is not None and len(cal) > 0:
                break
            time.sleep(2 ** attempt)
        if cal is None:
            raise ConnectionError(f"get_calendar returned None after retries: {last_err}")
        self.market_data = ad.MarketData(cal)
        self.info_data = ad.InfoData()
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        # IRON RULE: always logout, even on exception.
        try:
            if self.ad is not None and self.username:
                self.ad.logout(self.username)
                print("[session] logout ok", file=sys.stderr)
        except Exception as e:  # noqa: BLE001 — never mask original exception
            print(f"[session] logout error (ignored): {e}", file=sys.stderr)
        return False  # do not suppress exceptions
