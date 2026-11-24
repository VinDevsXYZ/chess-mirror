# PyInstaller spec for "Chess Mirror.app" (build with packaging/build_app.sh)
import os
import shutil

from PyInstaller.utils.hooks import collect_submodules

ROOT = os.path.abspath(os.path.join(SPECPATH, ".."))
VERSION = "0.1.0"
stockfish = os.path.realpath(os.environ.get("MIRROR_STOCKFISH") or shutil.which("stockfish") or "/opt/homebrew/bin/stockfish")

a = Analysis(
    [os.path.join(ROOT, "packaging", "app_main.py")],
    pathex=[os.path.join(ROOT, "src")],
    binaries=[(stockfish, ".")],
    datas=[(os.path.join(ROOT, "src", "chessmirror", "dashboard", "static"), "chessmirror/dashboard/static")],
    hiddenimports=collect_submodules("chessmirror") + collect_submodules("webview"),
    excludes=["tkinter", "matplotlib", "PIL", "pytest", "IPython", "pandas"],
    noarchive=False,
)
pyz = PYZ(a.pure)
exe = EXE(pyz, a.scripts, [], exclude_binaries=True, name="Chess Mirror", console=False,
          target_arch="arm64", codesign_identity=None)
coll = COLLECT(exe, a.binaries, a.datas, name="Chess Mirror")
app = BUNDLE(
    coll,
    name="Chess Mirror.app",
    icon=os.path.join(ROOT, "packaging", "ChessMirror.icns"),
    bundle_identifier="local.chessmirror.app",
    version=VERSION,
    info_plist={
        "CFBundleName": "Chess Mirror",
        "CFBundleDisplayName": "Chess Mirror",
        "CFBundleShortVersionString": VERSION,
        "CFBundleVersion": VERSION,
        "LSMinimumSystemVersion": "12.0",
        "NSHighResolutionCapable": True,
        "LSApplicationCategoryType": "public.app-category.board-games",
        # the dashboard is served from 127.0.0.1 inside the app
        "NSAppTransportSecurity": {"NSAllowsLocalNetworking": True},
    },
)
