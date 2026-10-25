"""Build the project's CPython 3.12.13 with the pinned SQLite library.

Run with ``.venv/bin/python scripts/setup_sqlite_runtime.py`` on macOS/Linux.
Sources, build logs, and the private interpreter stay under .fibmind/runtime.
Only the existing venv's interpreter link and configuration are switched, after
the new runtime passes its checks. Installed packages and memory files stay put.
"""

from __future__ import annotations

import hashlib
import json
import os
import shlex
import shutil
import ssl
import subprocess
import sys
import tarfile
import urllib.request
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SQLITE_VERSION = (ROOT / ".sqlite-version").read_text(encoding="utf-8").strip()
PYTHON_VERSION = "3.12.13"
PYTHON_SHA256 = "c08bc65a81971c1dd5783182826503369466c7e67374d1646519adf05207b684"
OPENSSL_VERSION = "3.5.5"
# Same OpenSSL release as the project's original uv-managed interpreter.
OPENSSL_SHA256 = "b28c91532a8b65a1f983b4c28b7488174e4a01008e29ce8e69bd789f28bc2a89"
# https://www.sqlite.org/releaselog/3_51_3.html
SQLITE_SOURCE_SHA3 = "32d5424f97e0a7fc5ed2f6335afbb58be4e0298bd7117a34e39d345ff13d859e"
WORK = ROOT / ".fibmind" / "runtime"
PREFIX = WORK / f"cpython-{PYTHON_VERSION}-sqlite-{SQLITE_VERSION}"
PYTHON = PREFIX / "bin" / "python3.12"


def run(args: list[str], *, cwd: Path, log_name: str, env: dict[str, str] | None = None) -> None:
    log = WORK / f"{log_name}.log"
    print(f"{log_name}: {shlex.join(args)} (log: {log.relative_to(ROOT)})", flush=True)
    with log.open("w", encoding="utf-8") as output:
        subprocess.run(args, cwd=cwd, env=env, stdout=output, stderr=subprocess.STDOUT, check=True)


def download(url: str, target: Path) -> None:
    if target.exists():
        return
    print(f"Downloading {url}", flush=True)
    partial = target.with_suffix(target.suffix + ".part")
    with urllib.request.urlopen(url, timeout=60) as response, partial.open("wb") as output:
        shutil.copyfileobj(response, output)
    partial.replace(target)


def check_runtime(python: Path, *, project_import: bool = False) -> dict[str, str]:
    code = """
import bz2, ctypes, json, lzma, sqlite3, ssl, sys
assert sys.version_info[:3] == (3, 12, 13), sys.version
assert sqlite3.sqlite_version == '3.51.3', sqlite3.sqlite_version
assert ssl.OPENSSL_VERSION.split()[1] == '3.5.5', ssl.OPENSSL_VERSION
assert ssl.create_default_context().cert_store_stats()['x509_ca'] > 0
with sqlite3.connect(':memory:') as conn:
    assert conn.execute('select sqlite_version()').fetchone()[0] == '3.51.3'
    conn.execute('CREATE VIRTUAL TABLE probe USING fts5(content)')
    conn.execute("INSERT INTO probe VALUES ('SQLite runtime check')")
    assert conn.execute("SELECT count(*) FROM probe WHERE probe MATCH 'runtime'").fetchone()[0] == 1
print(json.dumps({'python': sys.version.split()[0], 'sqlite': sqlite3.sqlite_version, 'openssl': ssl.OPENSSL_VERSION, 'executable': sys.executable}))
"""
    if project_import:
        code = "import fibmind, mcp\n" + code
    return json.loads(subprocess.check_output([str(python), "-c", code], cwd=ROOT, text=True))


def build_openssl(build: Path, env: dict[str, str]) -> Path:
    prefix = WORK / f"openssl-{OPENSSL_VERSION}"
    if all((prefix / name).is_file() for name in ("lib/libssl.a", "lib/libcrypto.a", "ssl/cert.pem")):
        return prefix
    archive = build / f"openssl-{OPENSSL_VERSION}.tar.gz"
    download(f"https://github.com/openssl/openssl/releases/download/openssl-{OPENSSL_VERSION}/{archive.name}", archive)
    if hashlib.sha256(archive.read_bytes()).hexdigest() != OPENSSL_SHA256:
        raise RuntimeError("OpenSSL source does not match its published SHA-256")
    source = build / f"openssl-{OPENSSL_VERSION}"
    if not source.exists():
        with tarfile.open(archive) as sources:
            sources.extractall(build, filter="data")
    run(
        [
            "./Configure",
            f"--prefix={prefix}",
            f"--openssldir={prefix / 'ssl'}",
            "--libdir=lib",
            "no-shared",
            "no-tests",
            "-fPIC",
        ],
        cwd=source,
        log_name="configure-openssl",
        env=env,
    )
    jobs = str(min(os.cpu_count() or 2, 8))
    run(["make", f"-j{jobs}"], cwd=source, log_name="build-openssl", env=env)
    run(["make", "install_sw"], cwd=source, log_name="install-openssl", env=env)
    original_cafile = ssl.get_default_verify_paths().cafile
    if not original_cafile:
        raise RuntimeError("Cannot locate the original interpreter's trusted CA bundle")
    cert = prefix / "ssl" / "cert.pem"
    cert.parent.mkdir(exist_ok=True)
    if not cert.exists():
        cert.symlink_to(original_cafile)
    return prefix


def build_runtime() -> None:
    if PYTHON.is_file():
        try:
            check_runtime(PYTHON)
            return
        except subprocess.CalledProcessError:
            print("Existing private runtime failed its checks; completing the build", flush=True)
    build = WORK / "build"
    build.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ)
    if sys.platform == "darwin" and Path("/Library/Developer/CommandLineTools").is_dir():
        env.setdefault("DEVELOPER_DIR", "/Library/Developer/CommandLineTools")
    openssl = build_openssl(build, env)
    sqlite = build / "sqlite-3.51.3"
    sqlite.mkdir(exist_ok=True)
    archive = build / "sqlite-amalgamation-3510300.zip"
    download("https://www.sqlite.org/2026/sqlite-amalgamation-3510300.zip", archive)
    with zipfile.ZipFile(archive) as sources:
        for name in ("sqlite3.c", "sqlite3.h", "sqlite3ext.h"):
            (sqlite / name).write_bytes(sources.read(f"sqlite-amalgamation-3510300/{name}"))
    if hashlib.sha3_256((sqlite / "sqlite3.c").read_bytes()).hexdigest() != SQLITE_SOURCE_SHA3:
        raise RuntimeError("SQLite source does not match the official 3.51.3 SHA3-256")

    cc = shlex.split(os.environ.get("CC", "cc"))
    run(
        [
            *cc,
            "-O2",
            "-fPIC",
            "-DSQLITE_THREADSAFE=1",
            "-DSQLITE_ENABLE_FTS5",
            "-DSQLITE_ENABLE_RTREE",
            "-DSQLITE_ENABLE_COLUMN_METADATA",
            "-c",
            "sqlite3.c",
            "-o",
            "sqlite3.o",
        ],
        cwd=sqlite,
        log_name="build-sqlite",
        env=env,
    )
    run(["ar", "rcs", "libsqlite3.a", "sqlite3.o"], cwd=sqlite, log_name="archive-sqlite", env=env)

    archive = build / f"Python-{PYTHON_VERSION}.tar.xz"
    download(f"https://www.python.org/ftp/python/{PYTHON_VERSION}/{archive.name}", archive)
    if hashlib.sha256(archive.read_bytes()).hexdigest() != PYTHON_SHA256:
        raise RuntimeError("CPython source does not match its published SHA-256")
    source = build / f"Python-{PYTHON_VERSION}"
    if not source.exists():
        with tarfile.open(archive) as sources:
            sources.extractall(build, filter="data")
    env["LIBSQLITE3_CFLAGS"] = f"-I{shlex.quote(str(sqlite))}"
    env["LIBSQLITE3_LIBS"] = f"{shlex.quote(str(sqlite / 'libsqlite3.a'))} -lm"
    args = [
        str(source / "configure"),
        f"--prefix={PREFIX}",
        "--with-ensurepip=no",
        "--disable-test-modules",
        "--enable-loadable-sqlite-extensions",
        f"--with-openssl={openssl}",
    ]
    if sys.platform == "darwin":
        xz = next((p for p in (Path("/opt/homebrew/opt/xz"), Path("/usr/local/opt/xz")) if p.is_dir()), None)
        if xz is None:
            raise RuntimeError("macOS builds require the Homebrew xz development library")
        env["CPPFLAGS"] = f"-I{shlex.quote(str(xz))}/include " + env.get("CPPFLAGS", "")
        env["LDFLAGS"] = f"-L{shlex.quote(str(xz))}/lib " + env.get("LDFLAGS", "")
    run(args, cwd=source, log_name="configure-python", env=env)
    jobs = str(min(os.cpu_count() or 2, 8))
    run(["make", f"-j{jobs}"], cwd=source, log_name="build-python", env=env)
    run(["make", f"-j{jobs}", "altinstall"], cwd=source, log_name="install-python", env=env)
    check_runtime(PYTHON)


def activate() -> None:
    venv = ROOT / ".venv"
    link = venv / "bin" / "python"
    config = venv / "pyvenv.cfg"
    if not link.is_symlink() or not config.is_file():
        raise RuntimeError("Expected an existing uv venv with .venv/bin/python as a symlink")
    old_link = os.readlink(link)
    old_config = config.read_text(encoding="utf-8")
    backup = WORK / "venv-before-sqlite-3.51.3.json"
    if not backup.exists():
        backup.write_text(json.dumps({"python_link": old_link, "pyvenv_cfg": old_config}, indent=2), encoding="utf-8")
    lines = old_config.splitlines()
    updated = []
    for line in lines:
        key = line.partition("=")[0].strip()
        if key == "home":
            line = f"home = {PREFIX / 'bin'}"
        elif key == "version_info":
            line = f"version_info = {PYTHON_VERSION}"
        elif key in {"executable", "command"}:
            continue
        updated.append(line)
    try:
        replacement = link.with_name("python-sqlite-new")
        replacement.symlink_to(PYTHON)
        replacement.replace(link)
        config.write_text("\n".join(updated) + "\n", encoding="utf-8")
        print(json.dumps(check_runtime(link, project_import=True), indent=2), flush=True)
    except BaseException:
        replacement = link.with_name("python-sqlite-rollback")
        replacement.symlink_to(old_link)
        replacement.replace(link)
        config.write_text(old_config, encoding="utf-8")
        raise


def main() -> None:
    if SQLITE_VERSION != "3.51.3":
        raise SystemExit("This installer only supports the reviewed SQLite 3.51.3 source and checksum")
    if sys.platform not in {"darwin", "linux"} or sys.version_info[:3] != (3, 12, 13):
        raise SystemExit("Run this installer with the project's CPython 3.12.13 venv on macOS/Linux")
    if sys.prefix == sys.base_prefix or Path(sys.prefix).resolve() != ROOT / ".venv":
        raise SystemExit("Run .venv/bin/python scripts/setup_sqlite_runtime.py from this checkout")
    WORK.mkdir(parents=True, exist_ok=True)
    build_runtime()
    activate()


if __name__ == "__main__":
    main()
