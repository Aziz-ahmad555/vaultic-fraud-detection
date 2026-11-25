"""docker-compose.yml follows D231: its own project name and 'vaultic-' volumes, every port bound to
127.0.0.1 and set by a VAULTIC_*_PORT key whose default avoids the usual ports and the other local
project's (PostgreSQL 5433, Redis 6379), and every service has a memory limit and a health check;
images are official and pinned to an exact version."""

from __future__ import annotations

import re
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
COMPOSE = yaml.safe_load((ROOT / "docker-compose.yml").read_text("utf-8"))
TAKEN = {5432, 5433, 6379, 9092}  # native PostgreSQL (D14), the other project, the usual Kafka port
PORT = re.compile(
    r"^127\.0\.0\.1:\$\{(VAULTIC_[A-Z]+_PORT):-(?:\$\{POSTGRES_PORT:-)?(\d+)\}?\}:(\d+)$"
)


def test_the_project_is_named_vaultic_and_its_volumes_are_prefixed():
    assert COMPOSE["name"] == "vaultic"
    vols = COMPOSE.get("volumes", {})
    assert vols and all(k.startswith("vaultic-") and v["name"] == k for k, v in vols.items())
    for svc in COMPOSE["services"].values():
        for v in svc.get("volumes", []):
            src = v.split(":")[0]
            assert src in vols, f"{v}: only named vaultic- volumes, no bind mounts"


def test_every_port_is_local_configurable_and_clear_of_other_projects():
    seen = set()
    for name, svc in COMPOSE["services"].items():
        for p in svc.get("ports", []):
            m = PORT.match(p)
            assert m, f"{name}: {p} must be 127.0.0.1:${{VAULTIC_..._PORT:-default}}:container"
            default = int(m.group(2))
            assert default not in TAKEN and default not in seen
            seen.add(default)
    assert 5442 in seen  # PostgreSQL (D231)


def test_every_service_has_a_memory_limit_a_health_check_and_a_pinned_official_image():
    for name, svc in COMPOSE["services"].items():
        assert svc.get("mem_limit"), name
        assert svc.get("healthcheck", {}).get("test"), name
        image = svc["image"]
        repo, _, tag = image.partition("@")[0].partition(":")
        assert "/" not in repo or repo.startswith(
            "apache/"
        ), f"{name}: official images only ({image})"
        assert re.match(r"^\d+(\.\d+)+", tag), f"{name}: an exact version tag, not '{tag}'"
        assert re.search(r"@sha256:[0-9a-f]{64}$", image), f"{name}: pinned to its digest ({image})"


def test_only_postgres_runs_without_a_profile():
    no_profile = [n for n, s in COMPOSE["services"].items() if not s.get("profiles")]
    assert no_profile == ["postgres"]
