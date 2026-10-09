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

from fibmind.mcp_server import _parse_args, build_server, daemon_main, run_server  # noqa: E402
from fibmind.service import MemoryService  # noqa: E402


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
                with urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=2) as response:
                    self.assertEqual(response.status, 200)
                    body = json.loads(response.read().decode())
                    self.assertEqual(body["status"], "ok")
                    self.assertEqual(body["transport"], "sse")

                # Root endpoint
                with urllib.request.urlopen(f"http://127.0.0.1:{port}/", timeout=2) as response:
                    self.assertEqual(response.status, 200)
            finally:
                uvicorn_server.should_exit = True
                thread.join(timeout=3)


if __name__ == "__main__":
    unittest.main()
