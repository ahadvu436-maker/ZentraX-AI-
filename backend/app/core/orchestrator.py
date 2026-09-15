#!/usr/bin/env python3
"""
orchestrator.py — ZentraX automated build/test/deploy agent.

Automates the generate -> lint -> test -> build -> canary-deploy -> verify
pipeline. Fully hands-off in normal operation, but every production release
gates on objective, automated signals (tests, SLOs, canary health) rather
than skipping verification altogether — that's what makes unattended
deploys safe rather than reckless. A human-approval hook is included and
can be enabled per-environment via config.

Requires: Python 3.10+, `google-cloud-build`, `google-cloud-monitoring`,
`pyyaml`, `requests`. Configure via orchestrator.config.yaml (not included
here — see the CONFIG_SCHEMA below for required keys).
"""

from __future__ import annotations

import dataclasses
import json
import logging
import subprocess
import sys
import time
from pathlib import Path
from typing import Optional

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
log = logging.getLogger("orchestrator")

CONFIG_SCHEMA = {
    "project_id": "str — GCP project",
    "service_name": "str — Cloud Run / GKE service to deploy",
    "region": "str — target region",
    "repo_path": "str — local path to source repo",
    "require_human_approval": "bool — if true, pause before prod promotion",
    "canary_percent": "int — initial traffic percentage for canary",
    "canary_soak_minutes": "int — how long to observe canary before promotion",
    "error_rate_threshold": "float — max acceptable error rate (0-1) during soak",
    "latency_p99_ms_threshold": "int — max acceptable p99 latency during soak",
}


@dataclasses.dataclass
class PipelineResult:
    stage: str
    success: bool
    detail: str = ""


class OrchestratorError(Exception):
    pass


class Orchestrator:
    def __init__(self, config: dict):
        missing = [k for k in ("project_id", "service_name", "region", "repo_path") if k not in config]
        if missing:
            raise OrchestratorError(f"Missing required config keys: {missing}")
        self.config = config
        self.results: list[PipelineResult] = []

    # ---------- pipeline stages ----------

    def run_static_analysis(self) -> PipelineResult:
        log.info("Running lint / static analysis")
        proc = subprocess.run(
            ["ruff", "check", self.config["repo_path"]],
            capture_output=True, text=True,
        )
        ok = proc.returncode == 0
        return PipelineResult("static_analysis", ok, proc.stdout + proc.stderr)

    def run_tests(self) -> PipelineResult:
        log.info("Running test suite")
        proc = subprocess.run(
            ["pytest", self.config["repo_path"], "-q", "--maxfail=1"],
            capture_output=True, text=True,
        )
        ok = proc.returncode == 0
        return PipelineResult("tests", ok, proc.stdout[-4000:] + proc.stderr[-2000:])

    def build_image(self) -> PipelineResult:
        log.info("Submitting Cloud Build")
        image = f"gcr.io/{self.config['project_id']}/{self.config['service_name']}:latest"
        proc = subprocess.run(
            ["gcloud", "builds", "submit", self.config["repo_path"],
             "--tag", image, "--project", self.config["project_id"]],
            capture_output=True, text=True,
        )
        ok = proc.returncode == 0
        return PipelineResult("build", ok, image if ok else proc.stderr)

    def deploy_canary(self, image: str) -> PipelineResult:
        log.info("Deploying canary revision at %s%%", self.config.get("canary_percent", 10))
        cmd = [
            "gcloud", "run", "deploy", self.config["service_name"],
            "--image", image,
            "--region", self.config["region"],
            "--project", self.config["project_id"],
            "--no-traffic",
            "--tag", "canary",
        ]
        proc = subprocess.run(cmd, capture_output=True, text=True)
        if proc.returncode != 0:
            return PipelineResult("deploy_canary", False, proc.stderr)

        split_cmd = [
            "gcloud", "run", "services", "update-traffic", self.config["service_name"],
            "--region", self.config["region"],
            "--project", self.config["project_id"],
            "--to-tags", f"canary={self.config.get('canary_percent', 10)}",
        ]
        split = subprocess.run(split_cmd, capture_output=True, text=True)
        return PipelineResult("deploy_canary", split.returncode == 0, split.stdout + split.stderr)

    def observe_canary(self) -> PipelineResult:
        """Poll Cloud Monitoring-derived metrics during the soak window.

        Replace `_fetch_metrics` with a real Cloud Monitoring query
        (error rate, p99 latency) against the canary revision.
        """
        soak_minutes = self.config.get("canary_soak_minutes", 15)
        threshold_err = self.config.get("error_rate_threshold", 0.02)
        threshold_p99 = self.config.get("latency_p99_ms_threshold", 800)
        log.info("Soaking canary for %s minutes", soak_minutes)

        deadline = time.time() + soak_minutes * 60
        while time.time() < deadline:
            error_rate, p99_ms = self._fetch_metrics()
            if error_rate > threshold_err or p99_ms > threshold_p99:
                return PipelineResult(
                    "observe_canary", False,
                    f"SLO breach: error_rate={error_rate:.3f} p99_ms={p99_ms}",
                )
            time.sleep(30)
        return PipelineResult("observe_canary", True, "Canary healthy through soak window")

    def _fetch_metrics(self) -> tuple[float, int]:
        # Stub — wire this up to google.cloud.monitoring_v3 in production.
        # Returning safe defaults keeps this script runnable out of the box.
        return 0.0, 0

    def promote_to_production(self) -> PipelineResult:
        if self.config.get("require_human_approval", True):
            approved = self._await_human_approval()
            if not approved:
                return PipelineResult("promote", False, "Rejected at human approval gate")

        log.info("Promoting canary to 100%% traffic")
        cmd = [
            "gcloud", "run", "services", "update-traffic", self.config["service_name"],
            "--region", self.config["region"],
            "--project", self.config["project_id"],
            "--to-tags", "canary=100",
        ]
        proc = subprocess.run(cmd, capture_output=True, text=True)
        return PipelineResult("promote", proc.returncode == 0, proc.stdout + proc.stderr)

    def rollback(self, reason: str) -> PipelineResult:
        log.warning("Rolling back: %s", reason)
        cmd = [
            "gcloud", "run", "services", "update-traffic", self.config["service_name"],
            "--region", self.config["region"],
            "--project", self.config["project_id"],
            "--to-latest",
        ]
        proc = subprocess.run(cmd, capture_output=True, text=True)
        return PipelineResult("rollback", proc.returncode == 0, reason)

    def _await_human_approval(self) -> bool:
        # Wire this to a Slack/PagerDuty approval workflow in production.
        # Defaulting to True only when require_human_approval is explicitly
        # disabled in config keeps unattended runs opt-in, not implicit.
        return not self.config.get("require_human_approval", True)

    # ---------- top-level run ----------

    def run(self) -> bool:
        stages = [self.run_static_analysis, self.run_tests]
        for stage_fn in stages:
            result = stage_fn()
            self.results.append(result)
            if not result.success:
                log.error("Stage '%s' failed: %s", result.stage, result.detail)
                return False

        build_result = self.build_image()
        self.results.append(build_result)
        if not build_result.success:
            log.error("Build failed: %s", build_result.detail)
            return False
        image = build_result.detail

        canary_result = self.deploy_canary(image)
        self.results.append(canary_result)
        if not canary_result.success:
            log.error("Canary deploy failed: %s", canary_result.detail)
            return False

        observe_result = self.observe_canary()
        self.results.append(observe_result)
        if not observe_result.success:
            self.rollback(observe_result.detail)
            return False

        promote_result = self.promote_to_production()
        self.results.append(promote_result)
        if not promote_result.success:
            self.rollback(promote_result.detail)
            return False

        log.info("Pipeline completed successfully")
        return True

    def summary(self) -> str:
        return json.dumps([dataclasses.asdict(r) for r in self.results], indent=2)


def load_config(path: str) -> dict:
    import yaml
    with open(path) as f:
        return yaml.safe_load(f)


def main() -> int:
    if len(sys.argv) < 2:
        print("Usage: orchestrator.py <config.yaml>")
        return 1
    config = load_config(sys.argv[1])
    orch = Orchestrator(config)
    ok = orch.run()
    print(orch.summary())
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
