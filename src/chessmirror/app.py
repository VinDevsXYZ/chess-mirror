"""Desktop app entry point (the executable inside "Chess Mirror.app").

    Chess Mirror            -> native window with the dashboard
    Chess Mirror --uci      -> UCI engine (what En Croissant runs, via Contents/MacOS/mirror-engine)
    Chess Mirror --cli ...  -> the `mirror` command line (e.g. `--cli status`)
"""

from __future__ import annotations

import multiprocessing
import socket
import subprocess
import sys
import threading


def _free_port(preferred: int = 8765) -> int:
    for port in (preferred, 0):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            try:
                s.bind(("127.0.0.1", port))
                return s.getsockname()[1]
            except OSError:
                continue
    raise RuntimeError("no free port")


class JsApi:
    """Native helpers exposed to the page as window.pywebview.api.*"""

    def __init__(self, cfg):
        self.cfg = cfg
        self.window = None

    def pick_pgn_files(self):
        import webview
        paths = self.window.create_file_dialog(webview.FileDialog.OPEN, allow_multiple=True,
                                               file_types=("PGN files (*.pgn)", "All files (*.*)"))
        return list(paths or [])

    def reveal_data_dir(self):
        subprocess.run(["open", str(self.cfg.data_dir)])
        return True

    def open_external(self, url: str):
        if url.startswith(("http://", "https://")):
            subprocess.run(["open", url])
        return True


def run_window() -> None:
    import webview

    from .config import ensure_config
    from .dashboard.server import make_server

    cfg = ensure_config()
    cfg.ensure_dirs()
    port = _free_port()
    httpd = make_server(cfg, port)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    api = JsApi(cfg)
    window = webview.create_window("Chess Mirror", f"http://127.0.0.1:{port}/", width=1320, height=900,
                                   min_size=(900, 620), js_api=api, text_select=True)
    api.window = window
    webview.start()
    httpd.shutdown()


def main() -> None:
    multiprocessing.freeze_support()  # required for the analysis worker pool in the frozen app
    args = sys.argv[1:]
    if args and args[0] == "--uci":
        from .uci.engine import main as uci_main
        uci_main()
    elif args and args[0] == "--cli":
        from .cli import main as cli_main
        cli_main(args[1:])
    else:
        run_window()


if __name__ == "__main__":
    main()
