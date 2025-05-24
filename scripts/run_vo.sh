#!/usr/bin/env bash
# usage: run_vo.sh --video myvideo.mp4 [--scale 1.0] [--calib camera_calib.npz]

ARGS=$(getopt -o '' -l video:,scale:,calib:,output:,no_save,no_plot -- "$@")
eval set -- "$ARGS"

VIDEO="" SCALE="1.0" CALIB="camera_calib.npz" OUTPUT="vo_output_tracked.mp4"
NO_SAVE="" NO_PLOT=""
while true; do
  case "$1" in
    --video)   VIDEO="$2"; shift 2;;
    --scale)   SCALE="$2"; shift 2;;
    --calib)   CALIB="$2"; shift 2;;
    --output)  OUTPUT="$2"; shift 2;;
    --no_save) NO_SAVE="--no_save"; shift ;;
    --no_plot) NO_PLOT="--no_plot"; shift ;;
    --) shift; break;;
  esac
done

python src/vo_pipeline.py \
  --video "$VIDEO" \
  --scale "$SCALE" \
  --calib "$CALIB" \
  --output "$OUTPUT" \
  $NO_SAVE $NO_PLOT