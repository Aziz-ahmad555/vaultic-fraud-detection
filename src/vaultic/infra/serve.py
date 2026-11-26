"""Start the platform API safely (Sprint 3 step 3.2a, D238).

    python -m vaultic.infra.serve [--port 8000]                    from the repository root
    VAULTIC_AUTH=jwt python -m vaultic.infra.serve                 login on

- Host: 127.0.0.1 only.
- Proxy headers are never trusted (A3). uvicorn otherwise takes X-Forwarded-For from loopback clients, and
  every client of this server is on loopback, so the per-IP login limit could be dodged.
- No root path (B7).
- The active mode is printed at start-up.
"""

from __future__ import annotations

import argparse
import sys

from vaultic.infra.db import REPO

HOST = "127.0.0.1"


def config(app, port: int = 8000, host: str = HOST):
    """The uvicorn settings of every documented start (D238): no proxy headers, no root path."""
    import uvicorn

    return uvicorn.Config(app, host=host, port=port, proxy_headers=False, forwarded_allow_ips="",
                          root_path="", log_level="info")  # fmt: skip


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--port", type=int, default=8000)
    a = p.parse_args(argv)
    import uvicorn

    sys.path.insert(0, str(REPO))
    from services.api.platform_server import app  # loads the bundle; prints the login mode

    if app is None:
        raise SystemExit("no demo bundle: set VAULTIC_DEMO_BUNDLE or VAULTIC_DATA_DIR")
    uvicorn.Server(config(app, a.port)).run()


if __name__ == "__main__":
    main()
