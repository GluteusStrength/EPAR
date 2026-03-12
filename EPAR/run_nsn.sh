#!/bin/bash
# run_nsn.sh — Stage 2 (NSN / Memory Bank) runner
#
# Usage:
#   ./run_nsn.sh mvtec   [class_name]   [mode]
#   ./run_nsn.sh mvtec   all            combined
#   ./run_nsn.sh eyecandies [class_name]
#
# mode: combined | only_jepa | only_memory  (default: combined)
#
# Examples:
#   ./run_nsn.sh mvtec cookie combined
#   ./run_nsn.sh eyecandies all

DATASET=${1:-mvtec}
CLASS=${2:-all}
MODE=${3:-combined}

MVTEC_CLASSES="bagel cable_gland carrot cookie dowel foam peach potato rope tire"
EYECANDIES_CLASSES="CandyCane ChocolateCookie ChocolatePraline Confetto GummyBear HazelnutTruffle LicoriceSandwich Lollipop Marshmallow PeppermintCandy"

if [ "$DATASET" = "mvtec" ]; then
    CONFIG="configs/train_nsn_mvtec.yaml"
    if [ "$CLASS" = "all" ]; then
        CLASSES=$MVTEC_CLASSES
    else
        CLASSES=$CLASS
    fi
elif [ "$DATASET" = "eyecandies" ]; then
    CONFIG="configs/train_nsn_eyecandies.yaml"
    if [ "$CLASS" = "all" ]; then
        CLASSES=$EYECANDIES_CLASSES
    else
        CLASSES=$CLASS
    fi
else
    echo "Unknown dataset: $DATASET  (choose: mvtec | eyecandies)"
    exit 1
fi

for cls in $CLASSES; do
    echo "========================================"
    echo "  Stage-2 NSN | dataset=$DATASET | class=$cls | mode=$MODE"
    echo "========================================"
    python train_nsn.py --config "$CONFIG" --class_name "$cls" --mode "$MODE"
    if [ $? -ne 0 ]; then
        echo "[ERROR] train_nsn failed for class: $cls"
        exit 1
    fi
done

echo "All Stage-2 jobs finished."
