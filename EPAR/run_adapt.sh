#!/bin/bash
# run_adapt.sh — Stage 1 (JEPA Adaptation) runner
#
# Usage:
#   ./run_adapt.sh mvtec   [class_name]   # single class
#   ./run_adapt.sh mvtec   all            # all classes
#   ./run_adapt.sh eyecandies [class_name]
#   ./run_adapt.sh eyecandies all
#
# Examples:
#   ./run_adapt.sh mvtec cookie
#   ./run_adapt.sh eyecandies all

DATASET=${1:-mvtec}
CLASS=${2:-all}

MVTEC_CLASSES="bagel cable_gland carrot cookie dowel foam peach potato rope tire"
EYECANDIES_CLASSES="CandyCane ChocolateCookie ChocolatePraline Confetto GummyBear HazelnutTruffle LicoriceSandwich Lollipop Marshmallow PeppermintCandy"

if [ "$DATASET" = "mvtec" ]; then
    CONFIG="configs/train_adapt_mvtec.yaml"
    if [ "$CLASS" = "all" ]; then
        CLASSES=$MVTEC_CLASSES
    else
        CLASSES=$CLASS
    fi
elif [ "$DATASET" = "eyecandies" ]; then
    CONFIG="configs/train_adapt_eyecandies.yaml"
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
    echo "  Stage-1 Adapt | dataset=$DATASET | class=$cls"
    echo "========================================"
    python train_adapt.py --config "$CONFIG" --class_name "$cls"
    if [ $? -ne 0 ]; then
        echo "[ERROR] train_adapt failed for class: $cls"
        exit 1
    fi
done

echo "All Stage-1 jobs finished."
