import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.testclient import TestClient

import web_api


class UiEntryCacheTests(unittest.TestCase):
    def test_html_entries_are_uncached_but_hashed_assets_keep_static_cache_policy(self):
        with tempfile.TemporaryDirectory(prefix="vt-ui-dist-") as directory:
            dist = Path(directory)
            assets = dist / "assets"
            assets.mkdir()
            (dist / "index.html").write_text("<html>fresh UI</html>", encoding="utf-8")
            (assets / "index-AbCd1234.js").write_text("window.ui = 'fresh';", encoding="utf-8")
            app = FastAPI()
            app.mount("/assets", StaticFiles(directory=assets), name="test-assets")
            app.add_api_route("/", web_api.serve_ui)
            app.add_api_route("/{asset_path:path}", web_api.serve_ui_asset)
            with patch.object(web_api, "UI_DIST", dist):
                client = TestClient(app)
                for path in ("/", "/history"):
                    response = client.get(path)
                    self.assertEqual(response.status_code, 200)
                    self.assertEqual(response.text, "<html>fresh UI</html>")
                    self.assertEqual(response.headers.get("cache-control"), "no-store")
                    self.assertEqual(response.headers.get("pragma"), "no-cache")

                asset = client.get("/assets/index-AbCd1234.js")
                self.assertEqual(asset.status_code, 200)
                self.assertEqual(asset.text, "window.ui = 'fresh';")
                self.assertNotEqual(asset.headers.get("cache-control"), "no-store")


if __name__ == "__main__":
    unittest.main()
