"""
CarlaADASInterface — base class for student ADAS nodes.

Students subclass this and override the callback methods (or register their own
via the on_* helpers).  The class handles all ROS 2 plumbing.

Quick-start example
-------------------
    class MyADAS(CarlaADASInterface):
        def __init__(self):
            super().__init__("my_adas")
            self.on_camera_image(self.process_image)
            self.on_speed_update(self.on_speed)
            self.on_imu_update(self.on_imu)           # optional
            self.on_gnss_update(self.on_gnss)         # optional
            self.on_collision(self.on_collision_event) # optional
            self.on_lane_invasion(self.on_lane_event)  # optional

Always-available sensors
------------------------
    Camera  : on_camera_image(cb)  →  cb(image: np.ndarray)  BGR, shape (H, W, 3)
    Speed   : on_speed_update(cb)  →  cb(speed_kmh: float)
              self.current_speed   →  float  (km/h, updated automatically)

Optional sensors
----------------
    IMU           : on_imu_update(cb)    →  cb(msg: sensor_msgs.msg.Imu)
                      msg.linear_acceleration.{x, y, z}  – m/s²
                      msg.angular_velocity.{x, y, z}     – rad/s
    GNSS / GPS    : on_gnss_update(cb)   →  cb(msg: sensor_msgs.msg.NavSatFix)
                      msg.latitude, msg.longitude         – degrees
                      msg.altitude                        – metres
    Collision     : on_collision(cb)     →  cb(info: dict)
                      info["actor"]    – description of what was hit
                      info["impulse"]  – [x, y, z] impulse vector (N·s)
    Lane invasion : on_lane_invasion(cb) →  cb(lane_types: list[str])
                      lane_types – e.g. ["Solid", "SolidSolid"]

Sending control
---------------
    self.send_control(throttle=0.4, steer=-0.1)

HUD events (shown on the simulation screen)
-------------------------------------------
    self.show_notification("Lane-keep active")
    self.show_warning("Vehicle ahead!")
    self.show_alert("COLLISION IMMINENT")

Periodic tasks
--------------
    self.create_periodic_task(0.1, self.my_controller)   # called every 100 ms
"""
import json

import cv2          # type: ignore
import numpy as np  # type: ignore
import rclpy        # type: ignore
from rclpy.node import Node                      # type: ignore
try:
    from rclpy.qos import qos_profile_sensor_data  # type: ignore
except Exception:
    qos_profile_sensor_data = 10
from geometry_msgs.msg import Twist              # type: ignore
from sensor_msgs.msg import CompressedImage, Imu, NavSatFix  # type: ignore
from std_msgs.msg import Float32, String         # type: ignore

from . import topics


class CarlaADASInterface(Node):
    """
    Base ROS 2 node for student ADAS systems.

    Wraps all ROS topics so students can focus on perception and control logic
    rather than middleware boilerplate.
    """

    def __init__(self, node_name: str = "adas_student"):
        super().__init__(node_name)

        # ── Publishers ────────────────────────────────────────────────────────
        self._pub_ctrl = self.create_publisher(Twist, topics.CMD_VEL_EXT, 10)
        self._pub_hud  = self.create_publisher(String, topics.HUD_EVENT, 10)

        # ── Core subscriptions (always active) ────────────────────────────────
        self.create_subscription(
            CompressedImage, topics.CAMERA_IMAGE, self._on_camera_msg, qos_profile_sensor_data
        )
        self.create_subscription(
            Float32, topics.VEHICLE_SPEED, self._on_speed_msg, 10
        )
        self.create_subscription(
            String, topics.HUD_EVENT, self._on_hud_event_msg, 10
        )
        self.create_subscription(
            String, topics.UI_STATE, self._on_ui_state_msg, 10
        )

        # ── State ─────────────────────────────────────────────────────────────
        self.current_speed: float = 0.0  # km/h, updated automatically

        # Optional FrameViewer for live telemetry (set via register_viewer)
        self._viewer = None

        # User-registered callbacks
        self._camera_cb       = None
        self._speed_cb        = None
        self._imu_cb          = None
        self._gnss_cb         = None
        self._collision_cb    = None
        self._lane_cb         = None

        # Optional subscriptions (created on first use)
        self._sub_imu      = None
        self._sub_gnss     = None
        self._sub_collision = None
        self._sub_lane     = None

        self.get_logger().info(
            f"[CarlaADASInterface] Node '{node_name}' ready.\n"
            f"  Subscribing : {topics.CAMERA_IMAGE}\n"
            f"               {topics.VEHICLE_SPEED}\n"
            f"  Publishing  : {topics.CMD_VEL_EXT}\n"
            f"               {topics.HUD_EVENT}"
        )

    # ── Callback registration — core sensors ─────────────────────────────────

    def on_camera_image(self, callback):
        """
        Register a callback that is called for every camera frame.

        callback(image: np.ndarray)
            image — BGR frame, shape (H, W, 3), same convention as OpenCV.
        """
        self._camera_cb = callback

    def on_speed_update(self, callback):
        """
        Register a callback that is called whenever the vehicle speed changes.

        callback(speed_kmh: float)
        """
        self._speed_cb = callback

    def register_viewer(self, viewer) -> None:
        """
        Connect a FrameViewer so sensor values appear automatically in the
        browser dashboard at http://localhost:8080.

        Call this once in your ``__init__`` after creating the viewer::

            self._viewer = FrameViewer(port=8080)
            self._viewer.start()
            self.register_viewer(self._viewer)   # ← add this line

        Speed is forwarded automatically.  IMU, GNSS, collision and lane data
        appear in the sidebar as soon as you register the corresponding
        callbacks with ``on_imu_update``, ``on_gnss_update``, etc.
        """
        self._viewer = viewer

    # ── Callback registration — optional sensors ──────────────────────────────

    def on_imu_update(self, callback):
        """
        Register a callback for IMU data (accelerometer + gyroscope).

        callback(msg: sensor_msgs.msg.Imu)
            msg.linear_acceleration.x / .y / .z  — m/s²
            msg.angular_velocity.x    / .y / .z  — rad/s

        Example::
            def handle_imu(self, msg):
                ax = msg.linear_acceleration.x
                gz = msg.angular_velocity.z
        """
        self._imu_cb = callback
        if self._sub_imu is None:
            self._sub_imu = self.create_subscription(
                Imu, topics.IMU, self._on_imu_msg, 10
            )
            self.get_logger().info(f"[CarlaADASInterface] IMU subscription active: {topics.IMU}")

    def on_gnss_update(self, callback):
        """
        Register a callback for GNSS / GPS data.

        callback(msg: sensor_msgs.msg.NavSatFix)
            msg.latitude   — degrees
            msg.longitude  — degrees
            msg.altitude   — metres above sea level

        Example::
            def handle_gnss(self, msg):
                lat, lon = msg.latitude, msg.longitude
        """
        self._gnss_cb = callback
        if self._sub_gnss is None:
            self._sub_gnss = self.create_subscription(
                NavSatFix, topics.GNSS, self._on_gnss_msg, 10
            )
            self.get_logger().info(f"[CarlaADASInterface] GNSS subscription active: {topics.GNSS}")

    def on_collision(self, callback):
        """
        Register a callback that fires on every collision event.

        callback(info: dict)
            info["actor"]   — string description of the object that was hit
            info["impulse"] — [x, y, z] impulse in N·s

        Example::
            def handle_collision(self, info):
                self.show_alert(f"Collision with {info['actor']}!")
        """
        self._collision_cb = callback
        if self._sub_collision is None:
            self._sub_collision = self.create_subscription(
                String, topics.COLLISION, self._on_collision_msg, 10
            )
            self.get_logger().info(
                f"[CarlaADASInterface] Collision subscription active: {topics.COLLISION}"
            )

    def on_lane_invasion(self, callback):
        """
        Register a callback that fires whenever the vehicle crosses a lane marking.

        callback(lane_types: list[str])
            lane_types — list of crossed marking type strings,
                         e.g. ["Solid"], ["Broken", "Solid"]

        Example::
            def handle_lane(self, lane_types):
                if "Solid" in lane_types:
                    self.show_warning("Solid line crossed!")
        """
        self._lane_cb = callback
        if self._sub_lane is None:
            self._sub_lane = self.create_subscription(
                String, topics.LANE_INVASION, self._on_lane_msg, 10
            )
            self.get_logger().info(
                f"[CarlaADASInterface] Lane-invasion subscription active: {topics.LANE_INVASION}"
            )

    # ── Periodic task helper ──────────────────────────────────────────────────

    def create_periodic_task(self, period_sec: float, callback):
        """
        Create a ROS 2 timer that calls *callback* at a fixed rate.

        Parameters
        ----------
        period_sec : float
            Interval in seconds (e.g. 0.1 for 10 Hz).
        callback : callable
            Function with no arguments, called every *period_sec* seconds.

        Returns
        -------
        rclpy.timer.Timer
            The timer object (store it if you need to cancel it later).

        Example::
            self.create_periodic_task(0.05, self.run_controller)  # 20 Hz
        """
        return self.create_timer(period_sec, callback)

    # ── Control ───────────────────────────────────────────────────────────────

    def send_control(self, throttle: float = 0.0, brake: float = 0.0, steer: float = 0.0):
        """
        Send a driving command to the simulation.

        Parameters
        ----------
        throttle : float  [0.0 – 1.0]   Gas pedal pressure.
        brake    : float  [0.0 – 1.0]   Brake pedal pressure.
        steer    : float  [-1.0 – 1.0]  Steering angle (negative = left).
        """
        msg = Twist()
        msg.linear.x  = float(max(0.0, min(1.0, throttle)))
        msg.linear.y  = float(max(0.0, min(1.0, brake)))
        msg.angular.z = float(max(-1.0, min(1.0, steer)))
        self._pub_ctrl.publish(msg)

    # ── HUD helpers ───────────────────────────────────────────────────────────

    def show_notification(self, text: str, duration: float = 3.0):
        """Display a white info message on the driver HUD."""
        self._send_hud_event("info", text, duration)

    def show_warning(self, text: str, duration: float = 5.0):
        """Display a yellow warning banner on the driver HUD."""
        self._send_hud_event("warning", text, duration)

    def show_alert(self, text: str, duration: float = 5.0):
        """Display a red alert banner on the driver HUD."""
        self._send_hud_event("alert", text, duration)

    # ── Internal ──────────────────────────────────────────────────────────────

    def _send_hud_event(self, level: str, text: str, duration: float):
        payload = json.dumps({"level": level, "text": text, "duration": float(duration)})
        self._pub_hud.publish(String(data=payload))

    def _on_camera_msg(self, msg: CompressedImage):
        if self._camera_cb is None:
            return
        try:
            buf   = np.frombuffer(msg.data, dtype=np.uint8)
            frame = cv2.imdecode(buf, cv2.IMREAD_COLOR)
            if frame is not None:
                self._camera_cb(frame)
        except Exception as exc:
            self.get_logger().warn(f"Camera decode error: {exc}")

    def _on_speed_msg(self, msg: Float32):
        self.current_speed = float(msg.data)
        if self._viewer is not None:
            self._viewer.push_telemetry(speed=self.current_speed)
        if self._speed_cb:
            self._speed_cb(self.current_speed)

    def _on_hud_event_msg(self, msg: String):
        if self._viewer is None:
            return
        try:
            self._viewer.push_hud_event(json.loads(msg.data))
        except Exception as exc:
            self.get_logger().warn(f"HUD event decode error: {exc}")

    def _on_ui_state_msg(self, msg: String):
        if self._viewer is None:
            return
        try:
            self._viewer.push_ui_state(json.loads(msg.data))
        except Exception as exc:
            self.get_logger().warn(f"UI state decode error: {exc}")

    def _on_imu_msg(self, msg: Imu):
        if self._viewer is not None:
            self._viewer.push_telemetry(imu={
                "ax": msg.linear_acceleration.x,
                "ay": msg.linear_acceleration.y,
                "az": msg.linear_acceleration.z,
                "gx": msg.angular_velocity.x,
                "gy": msg.angular_velocity.y,
                "gz": msg.angular_velocity.z,
            })
        if self._imu_cb:
            self._imu_cb(msg)

    def _on_gnss_msg(self, msg: NavSatFix):
        if self._viewer is not None:
            self._viewer.push_telemetry(gnss={
                "lat": msg.latitude,
                "lon": msg.longitude,
                "alt": msg.altitude,
            })
        if self._gnss_cb:
            self._gnss_cb(msg)

    def _on_collision_msg(self, msg: String):
        try:
            info = json.loads(msg.data)
        except Exception as exc:
            self.get_logger().warn(f"Collision message parse error: {exc}")
            return
        if self._viewer is not None:
            imp = info.get("impulse", [0, 0, 0])
            self._viewer.push_telemetry(collision={
                "actor":      str(info.get("actor", "unknown")),
                "impulse_mag": round(sum(v ** 2 for v in imp) ** 0.5, 2),
            })
        if self._collision_cb is not None:
            self._collision_cb(info)

    def _on_lane_msg(self, msg: String):
        try:
            data = json.loads(msg.data)
        except Exception as exc:
            self.get_logger().warn(f"Lane invasion message parse error: {exc}")
            return
        lane_types = data.get("lane_types", [])
        if self._viewer is not None:
            self._viewer.push_telemetry(lane={"types": lane_types})
        if self._lane_cb is not None:
            self._lane_cb(lane_types)
