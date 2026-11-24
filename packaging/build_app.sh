#!/bin/bash
# Build "Chess Mirror.app" and a DMG in dist/.   Usage: packaging/build_app.sh
set -euo pipefail
cd "$(dirname "$0")/.."
VERSION=0.1.0
APP="dist/Chess Mirror.app"

echo "==> icon"
uv run python packaging/make_icon.py packaging/ChessMirror.icns

echo "==> PyInstaller"
uv run pyinstaller --noconfirm --clean --log-level WARN --distpath dist --workpath build/pyinstaller \
    packaging/chess_mirror.spec

echo "==> UCI launcher for En Croissant"
cat > "$APP/Contents/MacOS/mirror-engine" <<'EOS'
#!/bin/sh
# En Croissant runs this file as a UCI engine (no shell, minimal PATH).
exec "$(dirname "$0")/Chess Mirror" --uci "$@"
EOS
chmod +x "$APP/Contents/MacOS/mirror-engine"

echo "==> ad-hoc code signature (local use)"
codesign --force --deep --sign - "$APP"
codesign --verify --deep --strict "$APP" && echo "signature ok"

echo "==> smoke test (UCI)"
printf 'uci\nquit\n' | "$APP/Contents/MacOS/mirror-engine" | grep -q uciok && echo "uci ok"

echo "==> DMG"
rm -rf build/dmg && mkdir -p build/dmg
ditto "$APP" "build/dmg/Chess Mirror.app"
ln -s /Applications build/dmg/Applications
rm -f "dist/Chess-Mirror-$VERSION.dmg"
hdiutil create -volname "Chess Mirror" -srcfolder build/dmg -ov -format UDZO "dist/Chess-Mirror-$VERSION.dmg" >/dev/null
rm -rf build/dmg  # a leftover copy here gets launched by Spotlight/Launch Services instead of /Applications
du -sh "$APP" "dist/Chess-Mirror-$VERSION.dmg"
echo "Built dist/Chess-Mirror-$VERSION.dmg"
