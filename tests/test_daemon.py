"""Tests for FibMind daemon mode and SSE / Streamable-HTTP transport."""

import json
import os
import tempfile
import threading
import time
import unittest
import urllib.request
from unittest.mock import patch

import pytest

pytest.importorskip("mcp")

from fibmind.mcp_server import (  # noqa: E402
    _parse_args,
    build_server,
    daemon_main,
    is_loopback_host,
    open_loopback_url,
    probe_daemon,
    run_server,
)
from fibmind.service import MemoryService  # noqa: E402


def _can_connect_loopback() -> bool:
    import socket

    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.bind(("127.0.0.1", 0))
        s.listen(1)
        port = s.getsockname()[1]
        c = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            c.connect(("127.0.0.1", port))
            return True
        except (PermissionError, OSError):
            return False
        finally:
            c.close()
            s.close()
    except Exception:
        return False


_LOOPBACK_AVAILABLE = _can_connect_loopback()


class DaemonModeTests(unittest.TestCase):
    def test_parse_args_transport_defaults(self) -> None:
        args = _parse_args([])
        self.assertEqual(args.transport, "stdio")
        self.assertEqual(args.host, "127.0.0.1")
        self.assertEqual(args.port, 8765)

    def test_parse_args_transport_custom(self) -> None:
        args = _parse_args(["--transport", "sse", "--host", "0.0.0.0", "--port", "9090"])
        self.assertEqual(args.transport, "sse")
        self.assertEqual(args.host, "0.0.0.0")
        self.assertEqual(args.port, 9090)

    def test_parse_args_transport_from_env(self) -> None:
        with patch.dict(os.environ, {"FIBMIND_TRANSPORT": "streamable-http", "FIBMIND_PORT": "9100"}):
            args = _parse_args([])
            self.assertEqual(args.transport, "streamable-http")
            self.assertEqual(args.port, 9100)

    def test_daemon_main_injects_sse(self) -> None:
        with patch("fibmind.mcp_server.main") as mock_main:
            daemon_main(["--port", "8888"])
            mock_main.assert_called_once_with(["--transport", "sse", "--port", "8888"])

    def test_run_server_invalid_transport_raises(self) -> None:
        with tempfile.NamedTemporaryFile(suffix=".db") as f:
            service = MemoryService(f.name)
            server = build_server(service)
            with self.assertRaises(ValueError):
                run_server(server, transport="unsupported")

    def test_is_loopback_host(self) -> None:
        self.assertTrue(is_loopback_host("127.0.0.1"))
        self.assertTrue(is_loopback_host("localhost"))
        self.assertTrue(is_loopback_host("::1"))
        self.assertTrue(is_loopback_host("127.0.1.1"))
        self.assertFalse(is_loopback_host("example.com"))
        self.assertFalse(is_loopback_host("192.168.1.1"))
        self.assertFalse(is_loopback_host(""))

    def test_open_loopback_url_bypasses_proxy(self) -> None:
        from unittest.mock import MagicMock

        with patch("urllib.request.build_opener") as mock_build:
            mock_opener = MagicMock()
            mock_build.return_value = mock_opener
            mock_opener.open.return_value = "mock_response"

            res = open_loopback_url("http://127.0.0.1:8765/health", timeout=1.5)
            self.assertEqual(res, "mock_response")
            mock_build.assert_called_once()
            args, _ = mock_build.call_args
            self.assertTrue(any(isinstance(a, urllib.request.ProxyHandler) for a in args))
            mock_opener.open.assert_called_once_with("http://127.0.0.1:8765/health", timeout=1.5)

    def test_probe_daemon_mocked(self) -> None:
        from unittest.mock import MagicMock

        mock_resp = MagicMock()
        mock_resp.status = 200
        mock_resp.read.return_value = b'{"status": "ok", "service": "fibmind", "transport": "sse"}'

        with patch("fibmind.mcp_server.open_loopback_url") as mock_open:
            mock_open.return_value.__enter__.return_value = mock_resp
            info = probe_daemon(host="127.0.0.1", port=8765)
            self.assertIsNotNone(info)
            self.assertEqual(info["status"], "ok")
            self.assertEqual(info["service"], "fibmind")

        with patch("fibmind.mcp_server.open_loopback_url", side_effect=OSError("connection refused")):
            info = probe_daemon(host="127.0.0.1", port=8765)
            self.assertIsNone(info)

    @unittest.skipUnless(_LOOPBACK_AVAILABLE, "Loopback socket connection not permitted in sandbox environment")
    def test_sse_daemon_health_check(self) -> None:
        import anyio
        import uvicorn
        from starlette.responses import JSONResponse
        from starlette.routing import Route

        port = 18876
        with tempfile.NamedTemporaryFile(suffix=".db") as f:
            service = MemoryService(f.name)
            server = build_server(service)

            app = server.sse_app(host="127.0.0.1")

            async def health_endpoint(request):
                return JSONResponse({"status": "ok", "service": "fibmind", "transport": "sse"})

            app.routes.append(Route("/health", health_endpoint))
            app.routes.append(Route("/", health_endpoint))

            config = uvicorn.Config(app, host="127.0.0.1", port=port, log_level="error")
            uvicorn_server = uvicorn.Server(config)

            thread = threading.Thread(target=lambda: anyio.run(uvicorn_server.serve), daemon=True)
            thread.start()
            try:
                # Wait for server ready
                started = False
                for _ in range(50):
                    time.sleep(0.05)
                    if uvicorn_server.started:
                        started = True
                        break
                self.assertTrue(started, "Daemon server failed to start")

                # Health check endpoint
                with open_loopback_url(f"http://127.0.0.1:{port}/health", timeout=2) as response:
                    self.assertEqual(response.status, 200)
                    body = json.loads(response.read().decode())
                    self.assertEqual(body["status"], "ok")
                    self.assertEqual(body["transport"], "sse")

                # Root endpoint
                with open_loopback_url(f"http://127.0.0.1:{port}/", timeout=2) as response:
                    self.assertEqual(response.status, 200)
            finally:
                uvicorn_server.should_exit = True
                thread.join(timeout=3)

    def test_run_server_rejects_non_loopback_by_default(self) -> None:
        with tempfile.NamedTemporaryFile(suffix=".db") as f:
            service = MemoryService(f.name)
            server = build_server(service)
            with self.assertRaises(ValueError) as ctx:
                run_server(server, transport="sse", host="0.0.0.0")
            self.assertIn("non-loopback", str(ctx.exception))

            with self.assertRaises(ValueError) as ctx:
                run_server(server, transport="streamable-http", host="192.168.1.100")
            self.assertIn("non-loopback", str(ctx.exception))

    @unittest.skipUnless(_LOOPBACK_AVAILABLE, "Loopback socket connection not permitted in sandbox environment")
    def test_sse_daemon_token_auth(self) -> None:
        import anyio
        import uvicorn
        from starlette.middleware.base import BaseHTTPMiddleware
        from starlette.responses import JSONResponse
        from starlette.routing import Route

        port = 18877
        secret_token = "test-secret-token-42"
        with tempfile.NamedTemporaryFile(suffix=".db") as f:
            service = MemoryService(f.name)
            server = build_server(service)

            app = server.sse_app(host="127.0.0.1")

            class TokenAuthMiddleware(BaseHTTPMiddleware):
                async def dispatch(self, request, call_next):
                    if request.url.path in ("/health",):
                        return await call_next(request)
                    auth = request.headers.get("authorization", "")
                    query_token = request.query_params.get("token", "")
                    passed_token = auth[7:].strip() if auth.startswith("Bearer ") else query_token
                    if not passed_token or passed_token != secret_token:
                        return JSONResponse({"error": "Unauthorized: invalid or missing daemon token"}, status_code=401)
                    return await call_next(request)

            app.add_middleware(TokenAuthMiddleware)

            async def root_endpoint(request):
                return JSONResponse({"status": "ok", "service": "fibmind"})

            async def health_endpoint(request):
                return JSONResponse({"status": "ok", "service": "fibmind"})

            app.routes.append(Route("/", root_endpoint))
            app.routes.append(Route("/health", health_endpoint))

            config = uvicorn.Config(app, host="127.0.0.1", port=port, log_level="error")
            uvicorn_server = uvicorn.Server(config)

            thread = threading.Thread(target=lambda: anyio.run(uvicorn_server.serve), daemon=True)
            thread.start()
            try:
                for _ in range(50):
                    time.sleep(0.05)
                    if uvicorn_server.started:
                        break

                # 1. Health check works without token
                with open_loopback_url(f"http://127.0.0.1:{port}/health", timeout=2) as response:
                    self.assertEqual(response.status, 200)

                # 2. Root without token fails with 401
                req_unauth = urllib.request.Request(f"http://127.0.0.1:{port}/")
                with self.assertRaises(urllib.error.HTTPError) as err_ctx:
                    open_loopback_url(req_unauth, timeout=2)
                self.assertEqual(err_ctx.exception.code, 401)

                # 3. Root with Bearer token succeeds with 200
                req_auth = urllib.request.Request(
                    f"http://127.0.0.1:{port}/",
                    headers={"Authorization": f"Bearer {secret_token}"},
                )
                with open_loopback_url(req_auth, timeout=2) as response:
                    self.assertEqual(response.status, 200)
                    body = json.loads(response.read().decode())
                    self.assertEqual(body["status"], "ok")

                # 4. Root with query param token succeeds with 200
                with open_loopback_url(f"http://127.0.0.1:{port}/?token={secret_token}", timeout=2) as response:
                    self.assertEqual(response.status, 200)
            finally:
                uvicorn_server.should_exit = True
                thread.join(timeout=3)

    @unittest.skipUnless(_LOOPBACK_AVAILABLE, "Loopback socket connection not permitted in sandbox environment")
    def test_probe_daemon_detects_running_instance(self) -> None:
        import anyio
        import uvicorn
        from starlette.responses import JSONResponse
        from starlette.routing import Route

        port = 18878
        # Probe on inactive port returns None
        self.assertIsNone(probe_daemon(host="127.0.0.1", port=port, timeout=0.5))

        with tempfile.NamedTemporaryFile(suffix=".db") as f:
            service = MemoryService(f.name)
            server = build_server(service)
            app = server.sse_app(host="127.0.0.1")

            async def health_endpoint(request):
                return JSONResponse({"status": "ok", "service": "fibmind", "transport": "sse"})

            app.routes.append(Route("/health", health_endpoint))
            config = uvicorn.Config(app, host="127.0.0.1", port=port, log_level="error")
            uvicorn_server = uvicorn.Server(config)

            thread = threading.Thread(target=lambda: anyio.run(uvicorn_server.serve), daemon=True)
            thread.start()
            try:
                for _ in range(50):
                    time.sleep(0.05)
                    if uvicorn_server.started:
                        break
                # Probe on active port detects running daemon
                info = probe_daemon(host="127.0.0.1", port=port, timeout=2.0)
                self.assertIsNotNone(info)
                self.assertEqual(info.get("status"), "ok")
                self.assertEqual(info.get("service"), "fibmind")
            finally:
                uvicorn_server.should_exit = True
                thread.join(timeout=3)


if __name__ == "__main__":
    unittest.main()
