#!/usr/bin/env bash
# =============================================================================
# start_ros_node.sh — wrapper that injects a correct Zenoh session config
#
# The default rmw_zenoh_cpp session config shipped with ROS Humble hardcodes
#   connect.endpoints = ["tcp/localhost:7447"]
# which means every node tries to find the Zenoh router on its own loopback.
# That works on a single machine, but breaks in Docker where the router runs
# in a separate container.
#
# This script generates a minimal session config file that:
#   • connects to the router given by $ZENOH_CONNECT_ENDPOINTS
#   • disables UDP multicast scouting (unsupported on Docker bridge networks)
# and then runs the command supplied as arguments.
#
# Usage
# -----
#   bash start_ros_node.sh <command> [args...]
#
# Environment
# -----------
#   ZENOH_CONNECT_ENDPOINTS   Router endpoint (default: tcp/localhost:7447)
# =============================================================================
set -e

source /opt/ros/humble/setup.bash

ENDPOINT="${ZENOH_CONNECT_ENDPOINTS:-tcp/localhost:7447}"
CONFIG_FILE="/tmp/zenoh_student_session_$$.json5"

cat > "$CONFIG_FILE" << ZENOH_EOF
{
  mode: "client",
  connect: {
    timeout_ms: { router: -1, peer: -1, client: 0 },
    endpoints: ["${ENDPOINT}"],
    exit_on_failure: { router: false, peer: false, client: false },
    retry: {
      period_init_ms: 500,
      period_max_ms:  4000,
      period_increase_factor: 2,
    },
  },
  listen: {
    timeout_ms: 0,
    endpoints: [],
    exit_on_failure: false,
    retry: {
      period_init_ms: 1000,
      period_max_ms:  4000,
      period_increase_factor: 2,
    },
  },
  scouting: {
    multicast: { enabled: false },
    gossip:    { enabled: false },
  },
  transport: {
    shared_memory: {
      enabled: false,
    },
  },
}
ZENOH_EOF

export ZENOH_SESSION_CONFIG_URI="$CONFIG_FILE"

echo "[Zenoh] connect endpoint : $ENDPOINT"
echo "[Zenoh] session config   : $CONFIG_FILE"

exec "$@"
