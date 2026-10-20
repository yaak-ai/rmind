#!/usr/bin/env bash
# Extract one video to per-frame JPEGs. NVDEC decode, GPU scale, one frame out per
# frame in. This is the ffmpeg invocation on its own; `extract_d12_frames.sh` is
# the wrapper that discovers jobs and cameras and calls this per video.
#
#   src/rmind/scripts/ffmpeg_extract_frames.sh <video.mp4> <out-dir> [options]
#
#   --size WxH      output resolution   (default 256x144)
#   --quality N     ffmpeg -q:v         (default 16; 2 is near-lossless, ~3x larger)
#   --gpu N         -hwaccel_device     (default 1)
#   --no-gpu        CPU decode and scale
#   --pattern PAT   filename pattern    (default %06d.jpg)
#   --start N       -start_number       (default 0)
#
# The command it runs, with defaults, is exactly:
#
#   ffmpeg -y -vsync 0 -threads 0 -hide_banner -loglevel error \
#     -hwaccel cuda -hwaccel_output_format cuda -c:v h264_cuvid -hwaccel_device 1 \
#     -i <video.mp4> \
#     -filter_complex 'scale_cuda=256:144,hwdownload,format=nv12' \
#     -f image2 -q:v 16 -start_number 0 <out-dir>/%06d.jpg
#
# Three things it does around that command:
#
#   * probes the codec, so `-c:v` is ${codec}_cuvid rather than a hardcoded
#     hevc_cuvid -- the D12 videos are h264, which hevc_cuvid refuses;
#   * stages the NVIDIA driver libraries. The nix ffmpeg has no driver on its
#     library path, and prepending the system libdir breaks it outright (system
#     glibc is older than nix's), so it symlinks just the four libraries NVDEC and
#     scale_cuda need into a temp dir and points LD_LIBRARY_PATH there;
#   * falls back to CPU (`-vf scale=WxH`) when there is no driver, or when GPU
#     decode fails for this particular file.
#
# `scale_cuda` rather than `scale_npp` because scale_npp needs an ffmpeg built
# with --enable-libnpp (non-free); it does the same job on the same hardware.
#
# `-start_number 0` makes the filename the 0-based encoded frame index, which is
# what `rmind.data.d12` records as `frame_idx`.
set -euo pipefail

SIZE="256x144"
QUALITY=16
GPU=1
USE_GPU=1
PATTERN="%06d.jpg"
START=0

usage() {
  sed -n '2,15p' "$0" | sed 's/^# \{0,1\}//'
  exit "${1:-0}"
}

VIDEO=""
OUT_DIR=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --size)    SIZE="$2"; shift 2 ;;
    --quality) QUALITY="$2"; shift 2 ;;
    --gpu)     GPU="$2"; shift 2 ;;
    --no-gpu)  USE_GPU=0; shift ;;
    --pattern) PATTERN="$2"; shift 2 ;;
    --start)   START="$2"; shift 2 ;;
    -h|--help) usage 0 ;;
    --*)       echo "unknown option: $1" >&2; usage 1 ;;
    *)
      if [[ -z "$VIDEO" ]]; then VIDEO="$1"
      elif [[ -z "$OUT_DIR" ]]; then OUT_DIR="$1"
      else echo "unexpected argument: $1" >&2; usage 1
      fi
      shift ;;
  esac
done

[[ -n "$VIDEO" && -n "$OUT_DIR" ]] || { echo "need <video.mp4> and <out-dir>" >&2; usage 1; }
[[ -s "$VIDEO" ]] || { echo "$VIDEO: missing or empty" >&2; exit 1; }
command -v ffmpeg >/dev/null || { echo "ffmpeg not on PATH" >&2; exit 1; }
command -v ffprobe >/dev/null || { echo "ffprobe not on PATH" >&2; exit 1; }

WIDTH="${SIZE%x*}"
HEIGHT="${SIZE#*x}"
CODEC="$(ffprobe -v error -select_streams v:0 -show_entries stream=codec_name -of csv=p=0 "$VIDEO")"

# libcuda: CUDA driver API. libnvcuvid: NVDEC. ptxjitcompiler: scale_cuda JITs a
# PTX kernel at runtime. nvidia-ml: device enumeration.
DRIVER_LIBS=""
stage_driver_libs() {
  local dir lib found=0
  DRIVER_LIBS="$(mktemp -d)"
  trap 'rm -rf "$DRIVER_LIBS"' EXIT
  for dir in /run/opengl-driver/lib /usr/lib/x86_64-linux-gnu; do
    [[ -d "$dir" ]] || continue
    for lib in libcuda.so.1 libnvcuvid.so.1 libnvidia-ptxjitcompiler.so.1 libnvidia-ml.so.1; do
      [[ -e "$dir/$lib" ]] && ln -sf "$dir/$lib" "$DRIVER_LIBS/$lib" && found=1
    done
    [[ "$found" -eq 1 ]] && break
  done
  [[ "$found" -eq 1 ]]
}

extract_gpu() {
  LD_LIBRARY_PATH="$DRIVER_LIBS" ffmpeg -y -vsync 0 -threads 0 -hide_banner -loglevel error \
    -hwaccel cuda -hwaccel_output_format cuda -c:v "${CODEC}_cuvid" -hwaccel_device "$GPU" -i "$VIDEO" \
    -filter_complex "scale_cuda=${WIDTH}:${HEIGHT},hwdownload,format=nv12" \
    -f image2 -q:v "$QUALITY" -start_number "$START" "$OUT_DIR/$PATTERN"
}

extract_cpu() {
  ffmpeg -y -vsync 0 -threads 0 -hide_banner -loglevel error -i "$VIDEO" \
    -vf "scale=${WIDTH}:${HEIGHT}" \
    -f image2 -q:v "$QUALITY" -start_number "$START" "$OUT_DIR/$PATTERN"
}

mkdir -p "$OUT_DIR"

if [[ "$USE_GPU" -eq 1 ]] && ! stage_driver_libs; then
  echo "  no NVIDIA driver libraries found, using CPU" >&2
  USE_GPU=0
fi

if [[ "$USE_GPU" -eq 1 ]]; then
  if ! extract_gpu 2>/dev/null; then
    echo "  GPU decode failed (codec $CODEC), using CPU" >&2
    extract_cpu
  fi
else
  extract_cpu
fi

find "$OUT_DIR" -name '*.jpg' | wc -l
