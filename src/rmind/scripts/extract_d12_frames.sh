#!/usr/bin/env bash
# Extract D12 job videos to per-frame JPEGs, NVDEC-decoded and GPU-scaled.
#
# Follows the yaak frame-extraction command, with three forced deviations, all
# checked on this host:
#
#   * `-c:v hevc_cuvid` -> `-c:v ${codec}_cuvid`, probed per file. The D12 job
#     videos are h264, not hevc, and hevc_cuvid refuses them.
#   * `scale_npp` -> `scale_cuda`. scale_npp needs an ffmpeg built with
#     --enable-libnpp (non-free); this build has only scale_cuda, which does the
#     same job on the same hardware.
#   * `-start_number 0` added, so the filename IS the 0-based encoded frame index
#     that `rmind.scripts.prepare_d12` writes as `frame_idx`. The yaak command
#     omits it and starts at 1, where a PathScanner recovers the mapping.
#
# Layout mirrors yaak's `{drive}/frames/{camera}.pii.mp4/{W}x{H}/`, so several
# resolutions coexist rather than overwrite:
#
#   {out-root}/{job-id}/frames/{camera}/{W}x{H}/%09d.jpg
#
# Why JPEGs at all, rather than decoding mp4 in the dataloader: training reads
# frames shuffled, and random access into h264 costs a keyframe seek plus forward
# decode -- measured 1958ms per 6 frames from mp4 against 1.6ms from pre-resized
# JPEGs, so the GPU would sit idle. Resizing on the way out also makes the JPEGs
# smaller than the source video.
#
#   src/rmind/scripts/extract_d12_frames.sh /nasa/team-space/nikita/data/d12/<job-id>
#   src/rmind/scripts/extract_d12_frames.sh --size 256x144 --gpu 1 /nasa/.../d12/*/
#
# Re-running is a no-op where the frame count already matches; --force re-encodes.
set -euo pipefail

SIZE="256x144"
OUT_ROOT="data/palletjack/d12_frames"
CAMERAS="cam_fork cam_left_forward cam_right_forward"
QUALITY=16
GPU=1
USE_GPU=1
FORCE=0
PATTERN="%09d.jpg"

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

# The nix-provided ffmpeg has no NVIDIA driver on its library path, and putting
# the whole system libdir there breaks it (the system glibc is older than nix's).
# So expose exactly the four driver libraries NVDEC + scale_cuda need, by symlink.
DRIVER_LIBS=""
setup_driver_libs() {
  local dir found=0
  DRIVER_LIBS="$(mktemp -d)"
  trap 'rm -rf "$DRIVER_LIBS"' EXIT
  for dir in /run/opengl-driver/lib /usr/lib/x86_64-linux-gnu; do
    [[ -d "$dir" ]] || continue
    # libcuda: CUDA driver API. libnvcuvid: NVDEC. ptxjitcompiler: scale_cuda
    # JITs a PTX kernel at runtime. nvidia-ml: device enumeration.
    for lib in libcuda.so.1 libnvcuvid.so.1 libnvidia-ptxjitcompiler.so.1 libnvidia-ml.so.1; do
      [[ -e "$dir/$lib" ]] && ln -sf "$dir/$lib" "$DRIVER_LIBS/$lib" && found=1
    done
    [[ "$found" -eq 1 ]] && break
  done
  [[ "$found" -eq 1 ]]
}

if [[ "$USE_GPU" -eq 1 ]] && ! setup_driver_libs; then
  echo "  no NVIDIA driver libraries found, falling back to CPU" >&2
  USE_GPU=0
fi

# "w,h,codec" header then one timestamp per encoded frame, and no trailing
# newline -- so awk's NR, minus the header, is the encoded frame count
encoded_frames() {
  awk 'END{print NR-1}' "$1"
}

extract_gpu() {
  local video="$1" codec="$2" out="$3"
  LD_LIBRARY_PATH="$DRIVER_LIBS" ffmpeg -y -vsync 0 -threads 0 -hide_banner -loglevel error \
    -hwaccel cuda -hwaccel_output_format cuda -c:v "${codec}_cuvid" -hwaccel_device "$GPU" -i "$video" \
    -filter_complex "scale_cuda=${WIDTH}:${HEIGHT},hwdownload,format=nv12" \
    -f image2 -q:v "$QUALITY" -start_number 0 "$out/$PATTERN"
}

extract_cpu() {
  local video="$1" out="$2"
  ffmpeg -y -vsync 0 -threads 0 -hide_banner -loglevel error -i "$video" \
    -vf "scale=${WIDTH}:${HEIGHT}" \
    -f image2 -q:v "$QUALITY" -start_number 0 "$out/$PATTERN"
}

total_written=0
for job_dir in "$@"; do
  job_dir="${job_dir%/}"
  job_id="$(basename "$job_dir")"

  if [[ ! -d "$job_dir/videos" ]]; then
    echo "  skip $job_id: no videos/ directory" >&2
    continue
  fi

  for camera in $CAMERAS; do
    video="$job_dir/videos/$camera/video.mp4"
    info="$job_dir/videos/$camera/frame_info_$camera.txt"

    if [[ ! -s "$video" ]]; then
      echo "  skip $job_id/$camera: video.mp4 missing or empty" >&2
      continue
    fi

    expected=""
    [[ -f "$info" ]] && expected="$(encoded_frames "$info")"

    out_dir="$OUT_ROOT/$job_id/frames/$camera/${WIDTH}x${HEIGHT}"
    have=0
    [[ -d "$out_dir" ]] && have="$(find "$out_dir" -name '*.jpg' | wc -l)"

    if [[ "$FORCE" -eq 0 && -n "$expected" && "$have" -eq "$expected" ]]; then
      echo "  have $job_id/$camera: $have frames, up to date"
      continue
    fi

    mkdir -p "$out_dir"
    codec="$(ffprobe -v error -select_streams v:0 -show_entries stream=codec_name -of csv=p=0 "$video")"

    if [[ "$USE_GPU" -eq 1 ]]; then
      if ! extract_gpu "$video" "$codec" "$out_dir" 2>/dev/null; then
        echo "  note $job_id/$camera: GPU decode failed (codec $codec), using CPU" >&2
        extract_cpu "$video" "$out_dir"
      fi
    else
      extract_cpu "$video" "$out_dir"
    fi

    written="$(find "$out_dir" -name '*.jpg' | wc -l)"
    total_written=$((total_written + written))

    if [[ -n "$expected" && "$written" -ne "$expected" ]]; then
      echo "  WARN $job_id/$camera: wrote $written but frame_info says $expected;" \
           "frame indices will not line up with prepare_d12" >&2
    else
      echo "  done $job_id/$camera: $written frames at ${WIDTH}x${HEIGHT} (${codec}, q:v $QUALITY)"
    fi
  done
done

echo "extracted $total_written frames into $OUT_ROOT"
