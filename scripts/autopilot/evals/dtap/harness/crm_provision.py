"""Explicit isolated service provisioning for the CRM pair.

Caller must own an admitted CPU claim, pre-pull these digest-pinned images and
supply the pinned upstream seed bytes. Never uses host networking or host volumes.
"""
from __future__ import annotations

import hashlib
import json
import subprocess
import time
import urllib.request
import uuid
from pathlib import Path
from typing import Any

CRM_DB_IMAGE = "decodingtrustagent/salesforce-crm@sha256:4a47fd1b2d39345299e107a46abb91aaee2227226adeb7729cf72c26be739025"
CRM_APP_IMAGE = "decodingtrustagent/salesforce-crm@sha256:ec1ee52b0af706908c55178bb49831211ceeb6c8914e8efef82fee1d260619bb"
# The publisher's legacy API image resolved on 2026-11-02; this is a separate
# image lineage from the pinned 2026-05 upstream MCP source, not a source build.
SLACK_IMAGE = "decodingtrustagent/slack@sha256:7e864c1706d77a2932ce488ea76f4ab9057687a7ed379b707289f2c2442a0f4d"
SEED_SHA256 = "f8da9246199906e77c95e6b9afd4857604967a393ff829a9268d5f515cd2916f"
SEED_ACCOUNT_ID = "3f000000-0000-4000-8000-000000000040"
SEED_DATETIME = "2026-01-01 00:00:00"


def prepare_seed(path: Path) -> tuple[str, str]:
    data = path.read_bytes()
    if hashlib.sha256(data).hexdigest() != SEED_SHA256:
        raise ValueError("upstream seed digest mismatch")
    sql = data.decode("utf-8")
    if sql.count("UUID()") != 1 or sql.count("NOW()") != 2:
        raise ValueError("unexpected upstream seed expressions")
    sql = sql.replace("UUID()", "'" + SEED_ACCOUNT_ID + "'").replace("NOW()", "'" + SEED_DATETIME + "'")
    return sql, hashlib.sha256(sql.encode()).hexdigest()


class PrivateServices:
    """Own exact Docker IDs only; the caller records returned evidence/receipts."""
    def __init__(self, *, cpus: str, seed_path: Path, receipt: Any = None):
        if not cpus or any(c not in "0123456789,-" for c in cpus):
            raise ValueError("explicit cpuset is required")
        self.cpus = cpus
        self.seed_sql, self.seed_sha256 = prepare_seed(seed_path)
        self.receipt = receipt
        self.tag = "dtap-crm-" + uuid.uuid4().hex
        self.network = None
        self.containers: list[str] = []
        self.crm_url = None
        self.slack_url = None

    def _emit(self, kind: str, payload: Any) -> None:
        if self.receipt is not None:
            self.receipt.append(kind, payload)

    def _docker(self, *args: str) -> str:
        p = subprocess.run(["docker", *args], check=True, text=True, stdout=subprocess.PIPE,
                           stderr=subprocess.PIPE, timeout=90)
        return p.stdout.strip()

    def _create(self, suffix: str, image: str, extra: list[str]) -> str:
        name = self.tag + "-" + suffix
        # The exact unique name provides recovery if create succeeds but its response is lost.
        try:
            ident = self._docker("create", "--name", name, "--label", f"dtap.owner={self.tag}",
                                 "--network", self.network, "--cpuset-cpus", self.cpus,
                                 "--pids-limit", "150", *extra, image)
        except Exception:
            try:
                ident = self._docker("inspect", "--format", "{{.Id}}", name)
                label = self._docker("inspect", "--format", '{{index .Config.Labels "dtap.owner"}}', ident)
                if label == self.tag:
                    self.containers.append(ident)
            except Exception:
                pass
            raise
        self.containers.append(ident)
        self._emit("container_created", {"id": ident, "image": image})
        self._docker("start", ident)
        return ident

    def _ip(self, ident: str) -> str:
        value = self._docker("inspect", "--format", "{{range .NetworkSettings.Networks}}{{.IPAddress}}{{end}}", ident)
        # This only accepts Docker's private IPv4 bridge addresses, not arbitrary endpoints.
        import ipaddress
        ip = ipaddress.ip_address(value)
        if ip.version != 4 or not ip.is_private or ip.is_loopback:
            raise ValueError("container does not have a private bridge address")
        return str(ip)

    @staticmethod
    def _http(url: str, *, body: Any = None) -> Any:
        data = None if body is None else json.dumps(body).encode()
        request = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        with opener.open(request, timeout=5) as response:
            raw = response.read(4 * 1024 * 1024 + 1)
            if len(raw) > 4 * 1024 * 1024:
                raise ValueError("service response too large")
            if not 200 <= response.status < 300:
                raise ValueError("service returned non-success status")
            if not raw:
                return None
            try:
                return json.loads(raw)
            except json.JSONDecodeError:
                return raw.decode("utf-8", errors="replace")

    def _health(self, url: str, container: str) -> None:
        deadline = time.monotonic() + 90
        while time.monotonic() < deadline:
            try:
                self._http(url)
                return
            except Exception:
                status = self._docker("inspect", "--format", "{{.State.Status}}", container)
                if status != "running":
                    raise RuntimeError(f"private service exited before health: {container}")
                time.sleep(1)
        raise TimeoutError(f"private service health failed: {url}")

    def start(self) -> None:
        try:
            # Images must already be present: no implicit registry pull during a claim.
            for image in (CRM_DB_IMAGE, CRM_APP_IMAGE, SLACK_IMAGE):
                self._docker("image", "inspect", image)
            try:
                self.network = self._docker("network", "create", "--internal", "--label",
                                            f"dtap.owner={self.tag}", self.tag)
            except Exception:
                try:
                    candidate = self._docker("network", "inspect", "--format", "{{.Id}}", self.tag)
                    label = self._docker("network", "inspect", "--format", '{{index .Labels "dtap.owner"}}', candidate)
                    if label == self.tag:
                        self.network = candidate
                except Exception:
                    pass
                raise
            self._emit("network_created", {"id": self.network})
            db = self._create("db", CRM_DB_IMAGE, ["--network-alias", "mariadb", "--memory", "768m",
                "--tmpfs", "/var/lib/mysql:rw,size=512m", "-e", "MYSQL_DATABASE=salesforce_crm",
                "-e", "MYSQL_USER=sf_user", "-e", "MYSQL_PASSWORD=bitnami123", "-e", "MYSQL_RANDOM_ROOT_PASSWORD=yes"])
            deadline = time.monotonic() + 90
            while True:
                try:
                    self._docker("exec", db, "mysqladmin", "ping", "-h", "localhost", "-u", "sf_user", "-pbitnami123")
                    break
                except subprocess.CalledProcessError:
                    if time.monotonic() >= deadline:
                        raise TimeoutError("private CRM database did not become ready")
                    time.sleep(1)
            crm = self._create("crm", CRM_APP_IMAGE, ["--memory", "1536m", "-e", "SALESFORCE_DATABASE_HOST=mariadb",
                "-e", "SALESFORCE_DATABASE_PORT_NUMBER=3306", "-e", "SALESFORCE_DATABASE_USER=sf_user",
                "-e", "SALESFORCE_DATABASE_NAME=salesforce_crm", "-e", "SALESFORCE_DATABASE_PASSWORD=bitnami123"])
            slack = self._create("slack", SLACK_IMAGE, ["--memory", "768m", "--tmpfs", "/app/data:rw,size=256m",
                "-e", "SLACK_JWT_SECRET=dev-secret", "-e", "SLACK_DB_PATH=/app/data/slack.db",
                "-e", "SLACK_FROM_EMAIL=noreply@slack.local", "-e", "SLACK_API_PORT=8034"])
            self.crm_url = f"http://{self._ip(crm)}:8080"
            self.slack_url = f"http://{self._ip(slack)}:8034"
            self._health(self.crm_url + "/", crm)
            self._health(self.slack_url + "/health", slack)
            self._emit("services_started", {"crm_url": self.crm_url, "slack_url": self.slack_url,
                "images": [CRM_DB_IMAGE, CRM_APP_IMAGE, SLACK_IMAGE], "cpus": self.cpus,
                "original_seed_sha256": SEED_SHA256, "deterministic_seed_sha256": self.seed_sha256})
        except BaseException:
            self.close()
            raise

    def reset(self) -> None:
        if not self.crm_url or not self.slack_url:
            raise RuntimeError("private services not started")
        actions = [(self.crm_url + "/Api/V8/admin/reset-database", {}),
                   (self.slack_url + "/api/v1/reset", {}),
                   (self.crm_url + "/Api/V8/admin/import-data", {"sql": self.seed_sql}),
                   (self.slack_url + "/api/v1/admin.init", {"workspace_id": "W01",
                      "users": [{"email": "sales.ops@company.com", "name": "Sales Operations", "access_token": "slack_token_sales_ops"}],
                      "channels": [{"name": "sales", "is_private": False, "members": ["sales.ops@company.com"]}]})]
        for url, payload in actions:
            result = self._http(url, body=payload)
            if isinstance(result, dict) and (result.get("error") or result.get("ok") is False):
                raise RuntimeError("private reset/seed operation failed")
            self._emit("reset_step", {"url": url.split("/", 3)[-1], "result": result})

    def close(self) -> None:
        errors = []
        for ident in reversed(self.containers):
            try:
                try:
                    self._docker("rm", "-f", ident)
                except subprocess.CalledProcessError as exc:
                    if "no such container" not in (exc.stderr or "").lower():
                        raise
                try:
                    self._docker("inspect", ident)
                except subprocess.CalledProcessError as exc:
                    if "no such object" in (exc.stderr or "").lower() or "no such container" in (exc.stderr or "").lower():
                        continue
                    raise
                errors.append(f"container still exists: {ident}")
            except Exception as exc:
                errors.append(f"container {ident}: {exc}")
        if not errors:
            self.containers.clear()
            if self.network:
                try:
                    network = self.network
                    self._docker("network", "rm", network)
                    try:
                        self._docker("network", "inspect", network)
                    except subprocess.CalledProcessError as exc:
                        message = (exc.stderr or "").lower()
                        if "no such network" not in message and f"network {network} not found" not in message:
                            raise
                    else:
                        raise RuntimeError("network still exists after removal")
                    self.network = None
                except Exception as exc:
                    errors.append(f"network removal: {exc}")
        if errors:
            raise RuntimeError("; ".join(errors))
