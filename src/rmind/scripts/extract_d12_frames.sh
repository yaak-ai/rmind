#!/usr/bin/env bash
# Extract every camera of every given D12 job to per-frame JPEGs.
#
# Job and camera discovery, idempotency and count checking live here; the ffmpeg
# invocation itself is `ffmpeg_extract_frames.sh` in this directory, which this
# calls once per video. Run that directly for a single video.
#
# Layout follows yaak's `{drive}/frames/.../{W}x{H}/` in keeping the resolution
# in the path, so several resolutions coexist rather than overwrite. Frames land
# NEXT TO the video they came from, inside the job directory:
#
#   {job-dir}/{camera}/frames/{W}x{H}/%06d.jpg
#
# which means this WRITES INTO the job directories. Pass --out-root to mirror the
# same tree somewhere else instead, leaving the source read-only.
#
# Why JPEGs at all, rather than decoding mp4 in the dataloader: training reads
# frames shuffled, and random access into h264 costs a keyframe seek plus forward
# decode -- measured 1958ms per 6 frames from mp4 against 1.6ms from pre-resized
# JPEGs, so the GPU would sit idle. Resizing on the way out also makes the JPEGs
# smaller than the source video.
#
#   src/rmind/scripts/extract_d12_frames.sh /nasa/data/d12/yaak/<job-id>
#   src/rmind/scripts/extract_d12_frames.sh --size 256x144 --gpu 1 /nasa/.../d12/*/
#
# Cameras are discovered per job as the subdirectories holding a video.mp4, since
# jobs do not all carry the same set; --cameras overrides with an explicit list.
#
# Re-running is a no-op where the frame count already matches; --force re-encodes.
set -euo pipefail

SIZE="256x144"
OUT_ROOT=""  # empty = write next to the videos, inside the job directory
CAMERAS=""   # empty = discover from each job's videos/ directory
QUALITY=16
GPU=1
USE_GPU=1
FORCE=0
PATTERN="%06d.jpg"

usage() {
  sed -n '2,31p' "$0" | sed 's/^# \{0,1\}//'
  exit "${1:-0}"
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --size)     SIZE="$2"; shift 2 ;;
    --out-root) OUT_ROOT="$2"; shift 2 ;;
    --cameras)  CAMERAS="$2"; shift 2 ;;
    --quality)  QUALITY="$2"; shift 2 ;;
    --gpu)      GPU="$2"; shift 2 ;;
    --no-gpu)   USE_GPU=0; shift ;;
    --pattern)  PATTERN="$2"; shift 2 ;;
    --force)    FORCE=1; shift ;;
    -h|--help)  usage 0 ;;
    --*)        echo "unknown option: $1" >&2; usage 1 ;;
    *)          break ;;
  esac
done

[[ $# -gt 0 ]] || { echo "no job directories given" >&2; usage 1; }
command -v ffmpeg >/dev/null || { echo "ffmpeg not on PATH" >&2; exit 1; }
command -v ffprobe >/dev/null || { echo "ffprobe not on PATH" >&2; exit 1; }

WIDTH="${SIZE%x*}"
HEIGHT="${SIZE#*x}"

# "w,h,codec" header then one timestamp per encoded frame, and no trailing
# newline -- so awk's NR, minus the header, is the encoded frame count
encoded_frames() {
  awk 'END{print NR-1}' "$1"
}

total_written=0
for job_dir in "$@"; do
  job_dir="${job_dir%/}"
  job_id="$(basename "$job_dir")"

  cameras="$CAMERAS"
  if [[ -z "$cameras" ]]; then
    cameras="$(find "$job_dir" -mindepth 2 -maxdepth 2 -name video.mp4 -printf '%h\n' \
      | xargs -r -n1 basename | sort | tr '\n' ' ')"
  fi
  if [[ -z "$cameras" ]]; then
    echo "  skip $job_id: no <camera>/video.mp4 found" >&2
    continue
  fi

  for camera in $cameras; do
    video="$job_dir/$camera/video.mp4"
    info="$job_dir/$camera/frame_info_$camera.txt"

    if [[ ! -s "$video" ]]; then
      echo "  skip $job_id/$camera: video.mp4 missing or empty" >&2
      continue
    fi

    expected=""
    [[ -f "$info" ]] && expected="$(encoded_frames "$info")"

    if [[ -n "$OUT_ROOT" ]]; then
      out_dir="$OUT_ROOT/$job_id/$camera/frames/${WIDTH}x${HEIGHT}"
    else
      out_dir="$job_dir/$camera/frames/${WIDTH}x${HEIGHT}"
    fi
    have=0
    [[ -d "$out_dir" ]] && have="$(find "$out_dir" -name '*.jpg' | wc -l)"

    if [[ "$FORCE" -eq 0 && -n "$expected" && "$have" -eq "$expected" ]]; then
      echo "  have $job_id/$camera: $have frames, up to date"
      continue
    fi

    gpu_args=(--gpu "$GPU")
    [[ "$USE_GPU" -eq 0 ]] && gpu_args=(--no-gpu)

    # via bash, not the exec bit, so a fresh checkout works without chmod
    written="$(bash "$(dirname "$0")/ffmpeg_extract_frames.sh" "$video" "$out_dir" \
      --size "$SIZE" --quality "$QUALITY" --pattern "$PATTERN" "${gpu_args[@]}")"
    total_written=$((total_written + written))

    if [[ -n "$expected" && "$written" -ne "$expected" ]]; then
      echo "  WARN $job_id/$camera: wrote $written but frame_info says $expected;" \
           "frame indices will not line up with rmind.data.d12" >&2
    else
      echo "  done $job_id/$camera: $written frames at ${WIDTH}x${HEIGHT} (q:v $QUALITY)"
    fi
  done
done

echo "extracted $total_written frames into ${OUT_ROOT:-the job directories}"
