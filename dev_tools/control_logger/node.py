#!/usr/bin/env python3
"""
HSHL ADAS Control Logger — dev tool for home testing.

Subscribes to your ADAS node's output topics and prints a live, formatted
log so you can verify that your send_control() and HUD calls are doing what
you expect — without needing the CARLA window.

Run alongside the bag profile:
    docker compose --profile bag up --build

Then in a second terminal:
    docker compose --profile bag logs -f ctrl-logger
"""
import json
import os

import rclpy          # type: ignore
from rclpy.node import Node                  # type: ignore
from geometry_msgs.msg import Twist          # type: ignore
from std_msgs.msg import Float32, String     # type: ignore

ROLE   = os.getenv("ROLE_NAME", "hero")
PREFIX = f"/carla/{ROLE}"

# ANSI colours (disabled if the terminal does not support them)
_R = "\033[91m"   # red
_Y = "\033[93m"   # yellow
_G = "\033[92m"   # green
_W = "\033[97m"   # white
_D = "\033[0m"    # reset


class ControlLogger(Node):

    def __init__(self):
        super().__init__("ctrl_logger")

        self._speed: float = 0.0

        self.create_subscription(Float32, f"{PREFIX}/speed",        self._on_speed, 10)
        self.create_subscription(Twist,   f"{PREFIX}/cmd_vel_ext",  self._on_ctrl,  10)
        self.create_subscription(String,  f"{PREFIX}/hud_event",    self._on_hud,   10)

        print(
            f"\n{_W}=== HSHL ADAS Control Logger ==={_D}\n"
            f"  Listening for control commands on  {PREFIX}/cmd_vel_ext\n"
            f"  Listening for HUD events on        {PREFIX}/hud_event\n"
            f"  Waiting for data...\n",
            flush=True,
        )

    # ── Subscribers ──────────────────────────────────────────────────────────

    def _on_speed(self, msg: Float32):
        self._speed = float(msg.data)

    def _on_ctrl(self, msg: Twist):
        thr = msg.linear.x
        brk = msg.linear.y
        ste = msg.angular.z

        warnings = []
        if not (0.0 <= thr <= 1.0):
            warnings.append(f"{_R}[!] throttle {thr:.3f} out of range [0,1]{_D}")
        if not (0.0 <= brk <= 1.0):
            warnings.append(f"{_R}[!] brake {brk:.3f} out of range [0,1]{_D}")
        if not (-1.0 <= ste <= 1.0):
            warnings.append(f"{_R}[!] steer {ste:.3f} out of range [-1,1]{_D}")

        warn_str = "  " + "  ".join(warnings) if warnings else ""

        print(
            f"{_G}[CTRL]{_D}"
            f"  throttle={thr:+.3f}"
            f"  brake={brk:.3f}"
            f"  steer={ste:+.3f}"
            f"  {_W}|{_D} speed={self._speed:.1f} km/h"
            f"{warn_str}",
            flush=True,
        )

    def _on_hud(self, msg: String):
        try:
            data     = json.loads(msg.data)
            level    = data.get("level", "info").upper()
            text     = data.get("text", "")
            duration = float(data.get("duration", 0))

            colour = {
                "INFO":    _W,
                "WARNING": _Y,
                "ALERT":   _R,
            }.get(level, _W)

            print(
                f"{colour}[HUD {level:<7}]{_D}  {text}  ({duration:.1f}s)",
                flush=True,
            )
        except Exception:
            pass


def main():
    rclpy.init()
    node = ControlLogger()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
