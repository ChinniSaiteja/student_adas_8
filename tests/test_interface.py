"""
Pure-Python unit tests for adas/interface.py
============================================
No ROS installation required — all ROS and display dependencies are mocked.

Run from inside student_adas/:
    python -m pytest tests/test_interface.py -v

Or directly with unittest:
    python tests/test_interface.py
"""

import json
import sys
import types
import unittest
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np

# ─── Inject ROS mock modules BEFORE importing student code ───────────────────
#
# We replace every ROS package with lightweight stubs so the tests run without
# a ROS installation.  The fake Node class is a real Python class (not a Mock)
# so that CarlaADASInterface can subclass it normally.

class _Twist:
    def __init__(self):
        self.linear  = types.SimpleNamespace(x=0.0, y=0.0, z=0.0)
        self.angular = types.SimpleNamespace(x=0.0, y=0.0, z=0.0)

class _CompressedImage:
    def __init__(self):
        self.data = b""

class _Imu:
    pass

class _NavSatFix:
    pass

class _Float32:
    def __init__(self, data=0.0):
        self.data = data

class _String:
    def __init__(self, data=""):
        self.data = data


class _FakeNode:
    """Minimal stand-in for rclpy.node.Node used by CarlaADASInterface."""

    def __init__(self, name: str):
        self._node_name  = name
        self._pubs: dict  = {}
        self._subs: dict  = {}
        self._timers: list = []

    def create_publisher(self, msg_type, topic, qos):
        pub = MagicMock(name=f"pub:{topic}")
        self._pubs[topic] = pub
        return pub

    def create_subscription(self, msg_type, topic, callback, qos):
        sub = MagicMock(name=f"sub:{topic}")
        self._subs[topic] = (callback, sub)
        return sub

    def create_timer(self, period, callback):
        timer = MagicMock()
        self._timers.append((period, callback, timer))
        return timer

    def get_logger(self):
        return MagicMock()


def _mock(name, **attrs):
    m = MagicMock(name=name)
    for k, v in attrs.items():
        setattr(m, k, v)
    return m


_geom_msg   = _mock("geometry_msgs.msg", Twist=_Twist)
_sens_msg   = _mock("sensor_msgs.msg",
                    CompressedImage=_CompressedImage, Imu=_Imu, NavSatFix=_NavSatFix)
_std_msg    = _mock("std_msgs.msg", Float32=_Float32, String=_String)
_rclpy_node = _mock("rclpy.node", Node=_FakeNode)

sys.modules.update({
    "rclpy":             MagicMock(name="rclpy"),
    "rclpy.node":        _rclpy_node,
    "geometry_msgs":     _mock("geometry_msgs",   msg=_geom_msg),
    "geometry_msgs.msg": _geom_msg,
    "sensor_msgs":       _mock("sensor_msgs",     msg=_sens_msg),
    "sensor_msgs.msg":   _sens_msg,
    "std_msgs":          _mock("std_msgs",        msg=_std_msg),
    "std_msgs.msg":      _std_msg,
})

# ─── Now import the student module ───────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).parent.parent))
from adas.interface import CarlaADASInterface  # noqa: E402
from adas import topics as T                   # noqa: E402


# ─── Helper: build a minimal valid CompressedImage message ───────────────────

def _make_jpeg_msg() -> _CompressedImage:
    """Return a _CompressedImage whose data bytes contain a tiny JPEG."""
    try:
        import cv2
        img = np.zeros((4, 4, 3), dtype=np.uint8)
        _, buf = cv2.imencode(".jpg", img)
        jpeg = buf.tobytes()
    except Exception:
        # Minimal JFIF JPEG that cv2 will still decode to a tiny black frame
        jpeg = (
            b"\xff\xd8\xff\xe0\x00\x10JFIF\x00\x01\x01\x00\x00\x01"
            b"\x00\x01\x00\x00\xff\xd9"
        )
    msg = _CompressedImage()
    msg.data = np.frombuffer(jpeg, dtype=np.uint8)
    return msg


# ═══════════════════════════════════════════════════════════════════════════════
# Tests: send_control
# ═══════════════════════════════════════════════════════════════════════════════

class TestSendControl(unittest.TestCase):

    def setUp(self):
        self.node = CarlaADASInterface("test_ctrl")
        self.pub  = self.node._pubs[T.CMD_VEL_EXT]

    def _sent(self):
        args, _ = self.pub.publish.call_args
        return args[0]  # the Twist object

    def test_normal_values_forwarded(self):
        self.node.send_control(throttle=0.5, brake=0.2, steer=-0.3)
        m = self._sent()
        self.assertAlmostEqual(m.linear.x,   0.5)
        self.assertAlmostEqual(m.linear.y,   0.2)
        self.assertAlmostEqual(m.angular.z, -0.3)

    def test_defaults_are_zero(self):
        self.node.send_control()
        m = self._sent()
        self.assertAlmostEqual(m.linear.x,  0.0)
        self.assertAlmostEqual(m.linear.y,  0.0)
        self.assertAlmostEqual(m.angular.z, 0.0)

    def test_throttle_clamped_above_1(self):
        self.node.send_control(throttle=5.0)
        self.assertAlmostEqual(self._sent().linear.x, 1.0)

    def test_throttle_clamped_below_0(self):
        self.node.send_control(throttle=-0.5)
        self.assertAlmostEqual(self._sent().linear.x, 0.0)

    def test_brake_clamped_above_1(self):
        self.node.send_control(brake=99.0)
        self.assertAlmostEqual(self._sent().linear.y, 1.0)

    def test_brake_clamped_below_0(self):
        self.node.send_control(brake=-1.0)
        self.assertAlmostEqual(self._sent().linear.y, 0.0)

    def test_steer_clamped_above_1(self):
        self.node.send_control(steer=5.0)
        self.assertAlmostEqual(self._sent().angular.z, 1.0)

    def test_steer_clamped_below_minus1(self):
        self.node.send_control(steer=-5.0)
        self.assertAlmostEqual(self._sent().angular.z, -1.0)

    def test_boundary_values_pass_through(self):
        self.node.send_control(throttle=1.0, brake=1.0, steer=-1.0)
        m = self._sent()
        self.assertAlmostEqual(m.linear.x,   1.0)
        self.assertAlmostEqual(m.linear.y,   1.0)
        self.assertAlmostEqual(m.angular.z, -1.0)

    def test_each_call_publishes_once(self):
        self.node.send_control(throttle=0.3)
        self.node.send_control(throttle=0.4)
        self.assertEqual(self.pub.publish.call_count, 2)


# ═══════════════════════════════════════════════════════════════════════════════
# Tests: HUD events
# ═══════════════════════════════════════════════════════════════════════════════

class TestHUDEvents(unittest.TestCase):

    def setUp(self):
        self.node = CarlaADASInterface("test_hud")
        self.pub  = self.node._pubs[T.HUD_EVENT]

    def _payload(self) -> dict:
        args, _ = self.pub.publish.call_args
        return json.loads(args[0].data)

    def test_notification_level_is_info(self):
        self.node.show_notification("Hello")
        self.assertEqual(self._payload()["level"], "info")

    def test_notification_text_correct(self):
        self.node.show_notification("Hello world")
        self.assertEqual(self._payload()["text"], "Hello world")

    def test_notification_default_duration_3s(self):
        self.node.show_notification("x")
        self.assertAlmostEqual(self._payload()["duration"], 3.0)

    def test_warning_level_is_warning(self):
        self.node.show_warning("Watch out")
        self.assertEqual(self._payload()["level"], "warning")

    def test_warning_custom_duration(self):
        self.node.show_warning("!", duration=7.0)
        self.assertAlmostEqual(self._payload()["duration"], 7.0)

    def test_alert_level_is_alert(self):
        self.node.show_alert("DANGER")
        self.assertEqual(self._payload()["level"], "alert")

    def test_alert_default_duration_5s(self):
        self.node.show_alert("x")
        self.assertAlmostEqual(self._payload()["duration"], 5.0)

    def test_duration_serialised_as_float(self):
        self.node.show_notification("x", duration=2)
        self.assertIsInstance(self._payload()["duration"], float)

    def test_hud_payload_is_valid_json(self):
        self.node.show_warning("test")
        args, _ = self.pub.publish.call_args
        json.loads(args[0].data)  # must not raise


# ═══════════════════════════════════════════════════════════════════════════════
# Tests: speed callbacks
# ═══════════════════════════════════════════════════════════════════════════════

class TestSpeedCallbacks(unittest.TestCase):

    def setUp(self):
        self.node = CarlaADASInterface("test_spd")

    def _deliver(self, value: float):
        cb, _ = self.node._subs[T.VEHICLE_SPEED]
        cb(_Float32(data=value))

    def test_current_speed_starts_at_zero(self):
        self.assertAlmostEqual(self.node.current_speed, 0.0)

    def test_current_speed_updated(self):
        self._deliver(42.0)
        self.assertAlmostEqual(self.node.current_speed, 42.0)

    def test_user_callback_receives_value(self):
        received = []
        self.node.on_speed_update(received.append)
        self._deliver(60.0)
        self.assertEqual(received, [60.0])

    def test_no_callback_does_not_crash(self):
        self._deliver(10.0)  # no on_speed_update registered — must not raise

    def test_speed_zero_handled(self):
        self._deliver(0.0)
        self.assertAlmostEqual(self.node.current_speed, 0.0)

    def test_successive_updates_keep_latest(self):
        self._deliver(10.0)
        self._deliver(20.0)
        self._deliver(30.0)
        self.assertAlmostEqual(self.node.current_speed, 30.0)


# ═══════════════════════════════════════════════════════════════════════════════
# Tests: camera callback
# ═══════════════════════════════════════════════════════════════════════════════

class TestCameraCallback(unittest.TestCase):

    def setUp(self):
        self.node = CarlaADASInterface("test_cam")

    def _deliver(self, msg):
        cb, _ = self.node._subs[T.CAMERA_IMAGE]
        cb(msg)

    def test_callback_receives_numpy_array(self):
        frames = []
        self.node.on_camera_image(frames.append)
        self._deliver(_make_jpeg_msg())
        self.assertEqual(len(frames), 1)
        self.assertIsInstance(frames[0], np.ndarray)
        self.assertEqual(frames[0].ndim, 3)      # shape (H, W, 3) BGR

    def test_no_callback_does_not_crash(self):
        # No on_camera_image registered — must not raise
        self._deliver(_make_jpeg_msg())

    def test_corrupt_image_bytes_do_not_crash(self):
        self.node.on_camera_image(lambda img: None)
        bad = _CompressedImage()
        bad.data = np.array([0xDE, 0xAD, 0xBE, 0xEF], dtype=np.uint8)
        self._deliver(bad)  # must not raise


# ═══════════════════════════════════════════════════════════════════════════════
# Tests: optional sensor subscriptions
# ═══════════════════════════════════════════════════════════════════════════════

class TestOptionalSensors(unittest.TestCase):

    def setUp(self):
        self.node = CarlaADASInterface("test_opt")

    # ── subscription creation ────────────────────────────────────────────────

    def test_imu_subscription_created_on_first_register(self):
        self.assertNotIn(T.IMU, self.node._subs)
        self.node.on_imu_update(lambda msg: None)
        self.assertIn(T.IMU, self.node._subs)

    def test_imu_subscription_object_not_replaced_on_re_register(self):
        self.node.on_imu_update(lambda msg: None)
        sub_first = self.node._sub_imu
        self.node.on_imu_update(lambda msg: None)
        self.assertIs(self.node._sub_imu, sub_first)

    def test_gnss_subscription_created_on_register(self):
        self.assertNotIn(T.GNSS, self.node._subs)
        self.node.on_gnss_update(lambda msg: None)
        self.assertIn(T.GNSS, self.node._subs)

    def test_collision_subscription_created_on_register(self):
        self.assertNotIn(T.COLLISION, self.node._subs)
        self.node.on_collision(lambda info: None)
        self.assertIn(T.COLLISION, self.node._subs)

    def test_lane_subscription_created_on_register(self):
        self.assertNotIn(T.LANE_INVASION, self.node._subs)
        self.node.on_lane_invasion(lambda types: None)
        self.assertIn(T.LANE_INVASION, self.node._subs)

    # ── callback delivery ────────────────────────────────────────────────────

    def test_imu_callback_receives_message(self):
        received = []
        self.node.on_imu_update(received.append)
        cb, _ = self.node._subs[T.IMU]
        fake_imu = _Imu()
        cb(fake_imu)
        self.assertEqual(received, [fake_imu])

    def test_gnss_callback_receives_message(self):
        received = []
        self.node.on_gnss_update(received.append)
        cb, _ = self.node._subs[T.GNSS]
        fake_gnss = _NavSatFix()
        cb(fake_gnss)
        self.assertEqual(received, [fake_gnss])


class _ViewerProbe:
    def __init__(self):
        self.ui_states = []
        self.hud_events = []

    def push_ui_state(self, state):
        self.ui_states.append(state)

    def push_hud_event(self, event):
        self.hud_events.append(event)


class TestRemoteDisplayForwarding(unittest.TestCase):

    def setUp(self):
        self.node = CarlaADASInterface("test_remote_display")
        self.viewer = _ViewerProbe()
        self.node.register_viewer(self.viewer)

    def test_ui_state_forwarded_to_viewer(self):
        cb, _ = self.node._subs[T.UI_STATE]
        cb(_String(data=json.dumps({"mode_text": "REMOTE", "recording": False})))
        self.assertEqual(self.viewer.ui_states[-1]["mode_text"], "REMOTE")

    def test_hud_event_forwarded_to_viewer(self):
        cb, _ = self.node._subs[T.HUD_EVENT]
        cb(_String(data=json.dumps({"level": "warning", "text": "Vehicle ahead"})))
        self.assertEqual(self.viewer.hud_events[-1]["text"], "Vehicle ahead")


# ═══════════════════════════════════════════════════════════════════════════════
# Tests: collision message parsing
# ═══════════════════════════════════════════════════════════════════════════════

class TestCollisionParsing(unittest.TestCase):

    def setUp(self):
        self.node   = CarlaADASInterface("test_col")
        self.events = []
        self.node.on_collision(self.events.append)

    def _send(self, payload: dict):
        cb, _ = self.node._subs[T.COLLISION]
        cb(_String(data=json.dumps(payload)))

    def test_actor_and_impulse_delivered(self):
        self._send({"actor": "vehicle.tesla.model3", "impulse": [1.0, -2.0, 0.5]})
        self.assertEqual(len(self.events), 1)
        self.assertEqual(self.events[0]["actor"],  "vehicle.tesla.model3")
        self.assertEqual(self.events[0]["impulse"], [1.0, -2.0, 0.5])

    def test_invalid_json_does_not_crash(self):
        cb, _ = self.node._subs[T.COLLISION]
        cb(_String(data="not-json{{{"))  # must not raise

    def test_multiple_events_accumulated(self):
        self._send({"actor": "a", "impulse": [0, 0, 0]})
        self._send({"actor": "b", "impulse": [1, 1, 1]})
        self.assertEqual(len(self.events), 2)
        self.assertEqual(self.events[1]["actor"], "b")


# ═══════════════════════════════════════════════════════════════════════════════
# Tests: lane invasion parsing
# ═══════════════════════════════════════════════════════════════════════════════

class TestLaneInvasionParsing(unittest.TestCase):

    def setUp(self):
        self.node   = CarlaADASInterface("test_lane")
        self.events = []
        self.node.on_lane_invasion(self.events.append)

    def _send(self, lane_types: list):
        cb, _ = self.node._subs[T.LANE_INVASION]
        cb(_String(data=json.dumps({"lane_types": lane_types})))

    def test_single_solid_line(self):
        self._send(["Solid"])
        self.assertEqual(self.events[-1], ["Solid"])

    def test_multiple_lane_types(self):
        self._send(["Broken", "Solid"])
        self.assertEqual(self.events[-1], ["Broken", "Solid"])

    def test_empty_lane_types(self):
        self._send([])
        self.assertEqual(self.events[-1], [])

    def test_invalid_json_does_not_crash(self):
        cb, _ = self.node._subs[T.LANE_INVASION]
        cb(_String(data="}}}}"))  # must not raise


# ═══════════════════════════════════════════════════════════════════════════════
# Tests: periodic task
# ═══════════════════════════════════════════════════════════════════════════════

class TestPeriodicTask(unittest.TestCase):

    def test_timer_created_with_correct_period(self):
        node = CarlaADASInterface("test_timer")
        node.create_periodic_task(0.1, lambda: None)
        self.assertEqual(len(node._timers), 1)
        period, _, _ = node._timers[0]
        self.assertAlmostEqual(period, 0.1)

    def test_multiple_timers_at_different_rates(self):
        node = CarlaADASInterface("test_timers")
        node.create_periodic_task(0.1, lambda: None)
        node.create_periodic_task(0.5, lambda: None)
        self.assertEqual(len(node._timers), 2)
        periods = {t[0] for t in node._timers}
        self.assertEqual(periods, {0.1, 0.5})


if __name__ == "__main__":
    unittest.main(verbosity=2)
