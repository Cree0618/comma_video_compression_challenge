#!/usr/bin/env bash
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PD="$(cd "${HERE}/../.." && pwd)"

IN_DIR="${PD}/videos"
VIDEO_NAMES_FILE="${PD}/public_test_video_names.txt"
ARCHIVE_DIR="${HERE}/archive"
JOBS="1"
CRF="28"
PRESET="medium"
SCALE="0.45"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --in-dir|--in_dir)
      IN_DIR="${2%/}"; shift 2 ;;
    --jobs)
      JOBS="$2"; shift 2 ;;
    --video-names-file|--video_names_file)
      VIDEO_NAMES_FILE="$2"; shift 2 ;;
    --crf)
      CRF="$2"; shift 2 ;;
    --preset)
      PRESET="$2"; shift 2 ;;
    --scale)
      SCALE="$2"; shift 2 ;;
    *)
      echo "Unknown arg: $1" >&2
      exit 2 ;;
  esac
done

rm -rf "$ARCHIVE_DIR"
mkdir -p "$ARCHIVE_DIR"

export IN_DIR ARCHIVE_DIR CRF PRESET SCALE

head -n "$(wc -l < "$VIDEO_NAMES_FILE")" "$VIDEO_NAMES_FILE" | xargs -P"$JOBS" -I{} bash -lc '
  rel="$1"
  [[ -z "$rel" ]] && exit 0

  IN="${IN_DIR}/${rel}"
  BASE="${rel%.*}"
  OUT="${ARCHIVE_DIR}/${BASE}.mkv"

  echo "CRF=${CRF}, preset=${PRESET}, scale=${SCALE}"

  ffmpeg -nostdin -y -hide_banner -loglevel warning \
    -r 20 -fflags +genpts -i "$IN" \
    -vf "scale=trunc(iw*${SCALE}/2)*2:trunc(ih*${SCALE}/2)*2:flags=lanczos" \
    -c:v libx265 -preset "${PRESET}" -crf "${CRF}" \
    -x265-params "keyint=60:min-keyint=1:bframes=4:frame-threads=4:log-level=warning" \
    -r 20 "$OUT"
' _ {}

# zip archive using python (zip may not be installed)
python3 -c "
import zipfile, os, sys
arc_dir = '${ARCHIVE_DIR}'
out_zip = '${HERE}/archive.zip'
with zipfile.ZipFile(out_zip, 'w', zipfile.ZIP_STORED) as zf:
    for root, dirs, files in os.walk(arc_dir):
        for f in files:
            full = os.path.join(root, f)
            arcname = os.path.relpath(full, arc_dir)
            zf.write(full, arcname)
print(f'Compressed to {out_zip}')
"
