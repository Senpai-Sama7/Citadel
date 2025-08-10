#!/usr/bin/env python3
"""
freeze_latest_requirements.py

Given a hardcoded list of PyPI package names (from the user's list), resolve the
LATEST STABLE (non pre-release/dev) versions from PyPI's JSON API and write a
pinned requirements.txt. Handles common Google Cloud naming quirks like
"-v1", "-v1beta1" suffixes by falling back to the base client package when the
suffix isn't a separate distribution on PyPI.

Usage:
  python freeze_latest_requirements.py [-o requirements.txt] [--strict]
                                      [--max-workers N] [--self-test]

Notes:
  * Requires internet access to query PyPI.
  * No third-party deps required at start; the script will auto-install
    `packaging` if missing.
  * If a package name doesn't exist on PyPI (common for v1/v1beta1 module
    namespaces), the script will try a base-name fallback and report the mapping.
  * Duplicates are deduped by the final resolved PyPI project name.
  * Concurrency is conservative to avoid environments with strict thread limits.
    Override with --max-workers or env FREEZE_MAX_WORKERS.
"""
from __future__ import annotations

import json
import os
import re
import sys
import time
import subprocess
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple

# Attempt to import packaging, auto-install if not present to keep this script
# fully self-contained for environments that start bare.
try:
    from packaging.version import Version
except Exception:  # pragma: no cover
    print("[setup] Installing 'packaging'...", file=sys.stderr)
    subprocess.check_call([sys.executable, "-m", "pip", "install", "--disable-pip-version-check", "packaging"])
    from packaging.version import Version

PYPI_JSON_URL = "https://pypi.org/pypi/{project}/json"
UA = "freeze-latest-requirements/1.1 (+https://pypi.org/)"

# ----------------------------------------------------------------------------
# Input packages from the user's provided list (lines 102-200 in their prompt)
# ----------------------------------------------------------------------------
INPUT_PACKAGES: List[str] = [
    "PyQt6",
    "accelerate",
    "aiofiles",
    "aiohttp",
    "annotated-types",
    "anyio",
    "async-timeout",
    "attrs",
    "backoff",
    "beautifulsoup4",
    "bcrypt",
    "bitsandbytes",
    "build",
    "cachetools",
    "certifi",
    "charset-normalizer",
    "chromadb",
    "click",
    "coloredlogs",
    "distro",
    "dowhy",
    "duckduckgo-search",
    "durationpy",
    "econml",
    "experta",
    "faiss-cpu",
    "fastapi",
    "filelock",
    "flatbuffers",
    "fsspec",
    "gitpython",
    "google-api-python-client",
    "google-auth",
    "google-auth-httplib2",
    "google-auth-oauthlib",
    "google-cloud-aiplatform",
    "google-cloud-asset",
    "google-cloud-automl",
    "google-cloud-automl-v1beta1",
    "google-cloud-bigquery",
    "google-cloud-bigquery-datatransfer",
    "google-cloud-bigquery-datatransfer-v2",
    "google-cloud-bigquery-storage",
    "google-cloud-bigquery-storage-v1",
    "google-cloud-bigtable",
    "google-cloud-billing",
    "google-cloud-build",
    "google-cloud-container",
    "google-cloud-containeranalysis",
    "google-cloud-containeranalysis-v1beta1",
    "google-cloud-dataflow",
    "google-cloud-datacatalog",
    "google-cloud-datacatalog-v1beta1",
    "google-cloud-datafusion",
    "google-cloud-datafusion-v1beta1",
    "google-cloud-dataproc",
    "google-cloud-dataproc-v1beta2",
    "google-cloud-dialogflow-cx",
    "google-cloud-dlp",
    "google-cloud-dlp-v2beta1",
    "google-cloud-documentai",
    "google-cloud-documentai-v1beta2",
    "google-cloud-dns",
    "google-cloud-error-reporting",
    "google-cloud-eventarc",
    "google-cloud-eventarc-publishing",
    "google-cloud-eventarc-publishing-v1",
    "google-cloud-functions",
    "google-cloud-functions-v1beta2",
    "google-cloud-game-servers",
    "google-cloud-game-servers-v1beta",
    "google-cloud-gke-hub",
    "google-cloud-gke-hub-v1beta1",
    "google-cloud-iam",
    "google-cloud-iam-admin",
    "google-cloud-iam-admin-v1",
    "google-cloud-iot",
    "google-cloud-iot-v1",
    "google-cloud-kms",
    "google-cloud-language",
    "google-cloud-language-v1",
    "google-cloud-logging",
    "google-cloud-logging-handlers",
    "google-cloud-logging-v2",
    "google-cloud-memcache",
    "google-cloud-memcache-v1beta1",
    "google-cloud-monitoring",
    "google-cloud-monitoring-dashboards",
    "google-cloud-monitoring-v3",
    "google-cloud-network-management",
    "google-cloud-network-management-v1",
    "google-cloud-orchestration-airflow",
    "google-cloud-orchestration-airflow-v1beta1",
    "google-cloud-phishing-protection",
    "google-cloud-phishing-protection-v1beta1",
    "google-cloud-private-catalog",
    "google-cloud-private-catalog-v1beta1",
    "google-cloud-pubsub",
    "google-cloud-recaptcha-enterprise",
    "google-cloud-recaptcha-enterprise-v1beta1",
    "google-cloud-redis",
    "google-cloud-redis-v1beta1",
    "google-cloud-resource-manager",
    "google-cloud-retail",
    "google-cloud-retail-v2",
    "google-cloud-run",
    "google-cloud-run-v1",
    "google-cloud-scheduler",
    "google-cloud-scheduler-v1beta1",
    "google-cloud-secret-manager",
    "google-cloud-security-center",
    "google-cloud-security-center-v1",
    "google-cloud-securitycenter",
    "google-cloud-securitycenter-v1beta1",
    "google-cloud-spanner",
    "google-cloud-spanner-v1",
    "google-cloud-speech",
    "google-cloud-speech-v1p1beta1",
    "google-cloud-sql",
    "google-cloud-sql-v1beta4",
    "google-cloud-storage",
    "google-cloud-storage-transfer",
    "google-cloud-storage-transfer-v1",
    "google-cloud-talent",
    "google-cloud-talent-v4beta1",
    "google-cloud-tasks",
    "google-cloud-texttospeech",
    "google-cloud-tpu",
    "google-cloud-tpu-v1",
    "google-cloud-trace",
    "google-cloud-translate",
    "google-cloud-translate-v3beta1",
    "google-cloud-video-intelligence",
    "google-cloud-video-intelligence-v1",
    "google-cloud-vision",
    "google-cloud-vision-v1p3beta1",
    "google-cloud-web-security-scanner",
    "google-cloud-web-security-scanner-v1beta",
    "google-cloud-webrisk",
    "google-cloud-workflows",
    "google-cloud-workflows-v1beta",
    "googleapis-common-protos",
    "grpcio",
    "h11",
    "hf-xet",
    "hiclass",
    "html5lib",
    "httpcore",
    "httptools",
    "httpx",
    "huggingface-hub",
    "humanfriendly",
    "idna",
    "importlib-metadata",
    "importlib-resources",
    "joblib",
    "jsonschema",
    "jsonschema-specifications",
    "kubernetes",
    "llama-cpp-python",
    "lxml",
    "markdown-it-py",
    "mdurl",
    "mmh3",
    "moviepy",
    "mpmath",
    "neo4j",
    "numpy",
    "oauthlib",
    "onnxruntime",
    "opencv-python",
    "opentelemetry-api",
    "opentelemetry-exporter-otlp-proto-common",
    "opentelemetry-exporter-otlp-proto-grpc",
    "opentelemetry-proto",
    "opentelemetry-sdk",
    "opentelemetry-semantic-conventions",
    "orjson",
    "overrides",
    "packaging",
    "pandas",
    "passlib",
    "pillow",
    "posthog",
    "prophet",
    "protobuf",
    "psycopg2-binary",
    "psutil",
    "pydantic",
    "pydantic-core",
    "pydub",
    "pygments",
    "pyinstaller",
    "pypika",
    "pyproject-hooks",
    "pyqt5",
    "pyqt5-tools",
    "pytest",
    "python-dateutil",
    "python-dotenv",
    "python-jose",
    "python-magic",
    "python-multipart",
    "python-ulid",
    "pyyaml",
    "qfluentwidgets",
    "redis",
    "referencing",
    "requests",
    "requests-oauthlib",
    "rich",
    "rpds-py",
    "rsa",
    "scikit-learn",
    "sentence-transformers",
    "shellingham",
    "six",
    "sniffio",
    "starlette",
    "sympy",
    "tenacity",
    "tokenizers",
    "torch",
    "tqdm",
    "transformers",
    "typer",
    "typing-extensions",
    "typing-inspection",
    "urllib3",
    "uvicorn",
    "uvloop",
    "watchdog",
    "watchfiles",
    "websocket-client",
    "websockets",
    "zipp",
]

# ----------------------------------------------------------------------------
# Core
# ----------------------------------------------------------------------------
@dataclass(frozen=True)
class ResolveResult:
    input_name: str
    resolved_project: Optional[str]  # actual PyPI project we pinned
    version: Optional[str]           # pinned version
    note: Optional[str] = None       # mapping/explanation


def _http_get(url: str, timeout: int = 20, retry: int = 3) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    last_err: Optional[Exception] = None
    for attempt in range(1, retry + 1):
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return resp.read()
        except Exception as e:  # pragma: no cover
            last_err = e
            # Polite backoff
            time.sleep(min(2 ** attempt, 5))
    assert last_err is not None
    raise last_err


def fetch_project_json(project: str) -> Optional[dict]:
    url = PYPI_JSON_URL.format(project=project)
    try:
        data = _http_get(url)
    except urllib.error.HTTPError as e:
        if e.code == 404:
            return None
        raise
    return json.loads(data.decode("utf-8"))


SEMVER_SUFFIX_RE = re.compile(r"^(?P<base>.+?)-v\d+(?:[a-zA-Z]\d+)?$")


def maybe_base_package(name: str) -> Optional[str]:
    """If name looks like google-cloud-foo-v1beta1, suggest google-cloud-foo.
    Returns None if no transformation applies.
    """
    m = SEMVER_SUFFIX_RE.match(name)
    if not m:
        return None
    return m.group("base")


def latest_stable_version_from_json(doc: dict) -> Optional[str]:
    # Prefer explicit releases over info.version to avoid pre/dev and yanked.
    releases = doc.get("releases", {})
    candidates: List[Tuple[Version, str]] = []
    for ver_str, files in releases.items():
        try:
            v = Version(ver_str)
        except Exception:
            continue
        # Skip pre/dev releases, allow post releases
        if v.is_prerelease or v.is_devrelease:
            continue
        # Skip yanked or empty release entries
        if not files or any(f.get("yanked", False) for f in files):
            continue
        candidates.append((v, ver_str))
    if not candidates:
        # Fallback to info.version if present
        info_ver = (doc.get("info") or {}).get("version")
        return info_ver or None
    candidates.sort(key=lambda t: t[0])
    return candidates[-1][1]


def resolve_one(name: str) -> ResolveResult:
    # Handle qfluentwidgets special case
    if name == "qfluentwidgets":
        return ResolveResult(input_name=name, resolved_project="PyQt6-Fluent-Widgets", version="1.8.6", note="mapped qfluentwidgets to PyQt6-Fluent-Widgets (PyQt6 stack)")

    # First try the name as-is
    doc = fetch_project_json(name)
    if doc is not None:
        ver = latest_stable_version_from_json(doc)
        return ResolveResult(input_name=name, resolved_project=name, version=ver, note=None)

    # Try base-package fallback for google-cloud-foo-v1 / -v1beta1 etc.
    base = maybe_base_package(name)
    if base:
        doc2 = fetch_project_json(base)
        if doc2 is not None:
            ver2 = latest_stable_version_from_json(doc2)
            return ResolveResult(
                input_name=name,
                resolved_project=base,
                version=ver2,
                note=f"mapped '{name}' -> '{base}' (versioned namespace on PyPI)",
            )

    # Not found
    return ResolveResult(input_name=name, resolved_project=None, version=None, note="not on PyPI")


def dedupe_results(results: Iterable[ResolveResult]) -> List[ResolveResult]:
    by_project: Dict[str, ResolveResult] = {}
    unresolved: List[ResolveResult] = []
    for r in results:
        if r.resolved_project is None:
            unresolved.append(r)
            continue
        # Keep the first occurrence (all versions should be same since we query once per project)
        if r.resolved_project not in by_project:
            by_project[r.resolved_project] = r
    # Return deterministic order: sorted by project name, followed by unresolved inputs
    resolved_sorted = [by_project[k] for k in sorted(by_project)]
    return resolved_sorted + unresolved


def write_requirements(path: str, results: Iterable[ResolveResult]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        f.write(f"# Pinned by freeze_latest_requirements.py on {time.strftime('%Y-%m-%d %H:%M:%S %Z')}\n")
        for r in results:
            if r.resolved_project and r.version:
                f.write(f"{r.resolved_project}=={r.version}\n")
        # Also annotate unresolved cases at the bottom for visibility
        unresolved = [r for r in results if not r.resolved_project or not r.version]
        if unresolved:
            f.write("\n# Unresolved inputs (not separate packages on PyPI or missing):\n")
            for r in unresolved:
                note = f"  # {r.note}" if r.note else ""
                f.write(f"# {r.input_name}{note}\n")


# ----------------------------------------------------------------------------
# Concurrency utilities
# ----------------------------------------------------------------------------

def _env_max_workers() -> Optional[int]:
    val = os.environ.get("FREEZE_MAX_WORKERS")
    if not val:
        return None
    try:
        n = int(val)
        if n >= 1:
            return n
    except ValueError:
        pass
    return None


def _default_max_workers() -> int:
    # Conservative default to avoid RuntimeError: can't start new thread
    # Hint suggested: min(16, os.cpu_count() or 8)
    cpu = os.cpu_count() or 8
    return min(16, cpu)


def resolve_all(names: List[str], max_workers: int) -> List[ResolveResult]:
    results: List[ResolveResult] = []
    if max_workers <= 1:
        for name in names:
            try:
                results.append(resolve_one(name))
            except Exception as e:  # pragma: no cover
                results.append(ResolveResult(input_name=name, resolved_project=None, version=None, note=str(e)))
        return results

    try:
        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            futs = {pool.submit(resolve_one, name): name for name in names}
            for fut in as_completed(futs):
                name = futs[fut]
                try:
                    r = fut.result()
                except Exception as e:  # pragma: no cover
                    print(f"error resolving {name}: {e}", file=sys.stderr)
                    results.append(ResolveResult(input_name=name, resolved_project=None, version=None, note=str(e)))
                    continue
                # Log helpful mapping info
                if r.note:
                    print(f"[map] {r.note}", file=sys.stderr)
                elif r.resolved_project and r.resolved_project != r.input_name:
                    print(f"[map] '{r.input_name}' -> '{r.resolved_project}'", file=sys.stderr)
                elif r.resolved_project is None:
                    print(f"[miss] '{r.input_name}' not found on PyPI", file=sys.stderr)
                results.append(r)
        return results
    except RuntimeError as e:
        # Typical in sandboxes or very constrained systems: can't start new thread
        msg = str(e).lower()
        if "can't start new thread" in msg or "cannot start new thread" in msg:
            print("[warn] Thread creation failed; falling back to sequential resolution.", file=sys.stderr)
            return resolve_all(names, max_workers=1)
        raise


# ----------------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------------

def main(argv: List[str]) -> int:
    out_path = "requirements.txt"
    strict = False
    max_workers: Optional[int] = _env_max_workers()
    run_self_test = False

    # Minimal CLI parsing
    it = iter(argv)
    for token in it:
        if token in ("-o", "--output"):
            try:
                out_path = next(it)
            except StopIteration:
                print("error: -o/--output requires a path", file=sys.stderr)
                return 2
        elif token == "--strict":
            strict = True
        elif token == "--self-test":
            run_self_test = True
        elif token == "--max-workers":
            try:
                max_workers = int(next(it))
            except Exception:
                print("error: --max-workers requires an integer >= 1", file=sys.stderr)
                return 2
            if max_workers < 1:
                print("error: --max-workers must be >= 1", file=sys.stderr)
                return 2
        else:
            print(f"warning: ignoring unknown argument: {token}", file=sys.stderr)

    if run_self_test:
        ok = _self_test()
        return 0 if ok else 1

    if max_workers is None:
        max_workers = _default_max_workers()

    # Resolve
    results = resolve_all(INPUT_PACKAGES, max_workers=max_workers)
    final_results = dedupe_results(results)
    write_requirements(out_path, final_results)

    unresolved = [r for r in final_results if not r.resolved_project or not r.version]
    if unresolved:
        print("\n[summary] Some inputs did not resolve as standalone PyPI projects:", file=sys.stderr)
        for r in unresolved:
            print(f"  - {r.input_name}: {r.note}", file=sys.stderr)
        if strict:
            return 1

    print(f"[ok] wrote pinned requirements to: {out_path}")
    return 0


# ----------------------------------------------------------------------------
# Minimal self-tests (no external framework). Run with --self-test
# ----------------------------------------------------------------------------

def _assert(cond: bool, msg: str) -> None:
    if not cond:
        raise AssertionError(msg)


def _fake_doc(releases: Dict[str, List[Dict[str, object]]], info_version: Optional[str] = None) -> dict:
    return {
        "info": {"version": info_version or "0.0.0"},
        "releases": releases,
    }


def _self_test() -> bool:
    print("[test] running self-tests…", file=sys.stderr)

    # maybe_base_package
    _assert(maybe_base_package("google-cloud-redis-v1beta1") == "google-cloud-redis", "base mapping failed")
    _assert(maybe_base_package("google-cloud-run-v1") == "google-cloud-run", "base mapping v1 failed")
    _assert(maybe_base_package("requests") is None, "should not map plain name")

    # latest_stable_version_from_json: prefer highest stable, ignore pre/dev, ignore yanked
    doc = _fake_doc(
        releases={
            "1.0.0": [{"yanked": False}],
            "1.1.0a1": [{"yanked": False}],  # pre-release ignored
            "1.0.1": [{"yanked": False}],
            "1.0.2": [{"yanked": True}],     # yanked ignored
            "1.0.1.post1": [{"yanked": False}],
        },
        info_version="9.9.9",
    )
    _assert(latest_stable_version_from_json(doc) == "1.0.1.post1", "stable selection incorrect")

    # resolve_one on a real, tiny package: 'pip' should exist and have a version
    r = resolve_one("pip")
    _assert(r.resolved_project == "pip" and isinstance(r.version, str) and len(r.version) > 0, "pip resolve failed")

    # resolve_one on a non-existent package
    r2 = resolve_one("definitely-not-a-real-pypi-distribution-xyz")
    _assert(r2.resolved_project is None, "nonexistent should not resolve")

    print("[test] ok", file=sys.stderr)
    return True


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))

