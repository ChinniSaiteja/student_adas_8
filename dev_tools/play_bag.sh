#!/usr/bin/env bash
# =============================================================================
# HSHL ADAS Student Lab — Bag Replay Entrypoint
#
# Finds and plays an instructor-provided ROS 2 bag on a loop.
#
# Usage
# -----
#   Automatic (plays the first bag found in bags/):
#     docker compose --profile bag up --build
#
#   Specific bag (set BAG_NAME in a .env file next to docker-compose.yaml):
#     echo "BAG_NAME=session_2025-03-01_14-30" > .env
#     docker compose --profile bag up --build
# =============================================================================
set -e

source /opt/ros/humble/setup.bash

# ── Find the bag to play ─────────────────────────────────────────────────────
if [ -n "$BAG_NAME" ]; then
    TARGET="/bags/$BAG_NAME"
    if [ ! -d "$TARGET" ]; then
        echo ""
        echo "ERROR: Bag '$BAG_NAME' not found inside the bags/ folder."
        echo "  Available bags:"
        find /bags -name "metadata.yaml" -maxdepth 2 2>/dev/null | xargs -r dirname | xargs -r -I{} basename {} | sed 's/^/    - /'
        echo ""
        exit 1
    fi
else
    # Auto-detect: find the first directory that contains a metadata.yaml
    TARGET=$(find /bags -name "metadata.yaml" -maxdepth 2 2>/dev/null | head -1 | xargs -r dirname)

    if [ -z "$TARGET" ]; then
        echo ""
        echo "ERROR: No bag found in bags/."
        echo ""
        echo "  1. Ask your instructor for a bag file (.zip or .tar.gz)."
        echo "  2. Unzip it into:  student_adas/bags/"
        echo "     The folder must contain a file called  metadata.yaml"
        echo "  3. Re-run:  docker compose --profile bag up --build"
        echo ""
        echo "  Optional — to choose a specific bag when you have multiple:"
        echo "    echo \"BAG_NAME=<folder-name>\" > .env"
        echo ""
        exit 1
    fi
fi

# ── Play ─────────────────────────────────────────────────────────────────────
echo ""
echo "========================================"
echo " HSHL ADAS Lab — Bag Replay"
echo "========================================"
echo " Bag    : $(basename "$TARGET")"
echo " Mode   : loop (Ctrl+C to stop)"
echo "========================================"
echo ""

exec ros2 bag play "$TARGET" --loop
