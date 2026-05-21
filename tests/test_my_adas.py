"""
Unit tests for MyADAS — validates the functions students must implement.
========================================================================
No ROS installation required — all ROS and display dependencies are mocked.

Tests cover
-----------
  detect_lanes(image)        → correct return type, required keys, value ranges
  compute_control(speed_kmh) → correct return type, value ranges
  Framework wiring           → callbacks call student functions and handle bad output

How to run (from inside student_adas/)
---------------------------------------
    python -m pytest tests/test_my_adas.py -v

Reading the results
--------------------
  PASSED  — your function returned a valid output for this test case.
  FAILED  — your function returned something incorrect; read the error message.
  SKIPPED — the function raises NotImplementedError (not yet implemented).
            Implement the function and re-run.
"""

import sys
import types
import unittest
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np

# ─── Inject ROS mock modules (same pattern as test_interface.py) ──────────────
#
# All ROS packages are replaced with lightweight stubs so the tests run
# without a ROS installation.

class _Twist:
    def __init__(self):
        self.linear  = types.SimpleNamespace(x=0.0, y=0.0, z=0.0)
        self.angular = types.SimpleNamespace(x=0.0, y=0.0, z=0.0)

class _CompressedImage:
    def __init__(self):
        self.data = b""

class _Imu:          pass
class _NavSatFix:    pass

class _Float32:
    def __init__(self, data=0.0): self.data = data

class _String:
    def __init__(self, data=""): self.data = data


class _FakeNode:
    """Minimal stand-in for rclpy.node.Node."""

    def __init__(self, name: str):
        self._node_name = name
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


class _FakeViewer:
    """Lightweight stand-in for adas.FrameViewer."""

    def __init__(self, port=8080): pass
    def start(self):              pass
    def stop(self):               pass
    def push(self, frame):        pass
    def push_telemetry(self, **kwargs): pass
    def push_ui_state(self, state): pass
    def push_hud_event(self, event): pass


def _mock(name, **attrs):
    m = MagicMock(name=name)
    for k, v in attrs.items():
        setattr(m, k, v)
    return m


_geom_msg   = _mock("geometry_msgs.msg", Twist=_Twist)
_sens_msg   = _mock("sensor_msgs.msg",
                    CompressedImage=_CompressedImage, Imu=_Imu,
                    NavSatFix=_NavSatFix)
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

# ─── Import the real CarlaADASInterface (with mocked ROS) ────────────────────
sys.path.insert(0, str(Path(__file__).parent.parent))
from adas.interface import CarlaADASInterface          # noqa: E402
from adas import topics as T                           # noqa: E402

# Patch the `adas` package so MyADAS gets the real interface + fake viewer
_adas_pkg = _mock("adas",
                  CarlaADASInterface=CarlaADASInterface,
                  FrameViewer=_FakeViewer)
sys.modules["adas"] = _adas_pkg

# ─── Import the student module ────────────────────────────────────────────────
from solution.my_adas import MyADAS                    # noqa: E402

# ─── Helpers ──────────────────────────────────────────────────────────────────
_H, _W = 720, 1280                                     # typical CARLA bag size


def _make_image(h: int = _H, w: int = _W) -> np.ndarray:
    """Return a black BGR frame of the requested size."""
    return np.zeros((h, w, 3), dtype=np.uint8)


def _make_node() -> MyADAS:
    """Return a fresh MyADAS instance with mocked sensors."""
    return MyADAS()


# =============================================================================
# Tests: detect_lanes()
# =============================================================================

class TestDetectLanes(unittest.TestCase):
    """
    Contract for  detect_lanes(image: np.ndarray) → dict | None

    The function must either return None (no lanes found) or a dict that
    satisfies all of the following requirements:

      "steer_offset"  float in [-1.0, +1.0]   REQUIRED
                       0.0  = car is centred
                      +0.3  = steer right by 30 %
                      -0.3  = steer left  by 30 %

      "annotated"     np.ndarray, same shape as image   OPTIONAL
                      Your drawing; shown in the browser.
    """

    @classmethod
    def setUpClass(cls):
        cls.node = _make_node()

    # ── helper ────────────────────────────────────────────────────────────────

    def _call(self, image=None):
        """Call detect_lanes and skip the test if not yet implemented."""
        if image is None:
            image = _make_image()
        try:
            return self.node.detect_lanes(image)
        except NotImplementedError:
            self.skipTest(
                "detect_lanes() is not yet implemented.\n"
                "  → Open solution/my_adas.py and replace "
                "'raise NotImplementedError(...)' with your code."
            )

    # ── return type ───────────────────────────────────────────────────────────

    def test_returns_dict_or_none(self):
        """detect_lanes() must return a dict or None — nothing else."""
        result = self._call()
        self.assertIsInstance(
            result, (dict, type(None)),
            f"Expected dict or None, got {type(result).__name__!r}.\n"
            "  → Return a dict like {'steer_offset': 0.0} or return None.",
        )

    # ── required key: steer_offset ────────────────────────────────────────────

    def test_steer_offset_key_present(self):
        """The returned dict must contain the key 'steer_offset'."""
        result = self._call()
        if result is None:
            self.skipTest(
                "detect_lanes() returned None on the test image.\n"
                "  → This is allowed when no lanes are detected, but you should "
                "also handle frames where lanes are visible."
            )
        self.assertIn(
            "steer_offset", result,
            "The returned dict is missing the required key 'steer_offset'.\n"
            "  → Example: return {'steer_offset': 0.0}",
        )

    def test_steer_offset_is_float(self):
        """'steer_offset' must be an int or float."""
        result = self._call()
        if result is None:
            self.skipTest("detect_lanes() returned None.")
        offset = result.get("steer_offset")
        self.assertIsInstance(
            offset, (int, float),
            f"'steer_offset' must be a number, got {type(offset).__name__!r}: {offset!r}.\n"
            "  → Example: return {'steer_offset': 0.25}",
        )

    def test_steer_offset_in_range(self):
        """'steer_offset' must be in the range [-1.0, +1.0]."""
        result = self._call()
        if result is None:
            self.skipTest("detect_lanes() returned None.")
        offset = float(result.get("steer_offset", 0))
        self.assertGreaterEqual(
            offset, -1.0,
            f"'steer_offset' = {offset:.4f} is below -1.0 (maximum left).\n"
            "  → Clamp your value: offset = max(-1.0, min(1.0, raw_offset))",
        )
        self.assertLessEqual(
            offset, 1.0,
            f"'steer_offset' = {offset:.4f} is above +1.0 (maximum right).\n"
            "  → Clamp your value: offset = max(-1.0, min(1.0, raw_offset))",
        )

    # ── optional key: annotated ───────────────────────────────────────────────

    def test_annotated_is_ndarray_if_present(self):
        """If 'annotated' is returned it must be a np.ndarray."""
        result = self._call()
        if result is None or "annotated" not in result:
            return   # optional key — nothing to check
        ann = result["annotated"]
        self.assertIsInstance(
            ann, np.ndarray,
            f"'annotated' must be np.ndarray, got {type(ann).__name__!r}.\n"
            "  → Example: annotated = image.copy(); cv2.line(annotated, ...)",
        )

    def test_annotated_same_shape_as_input(self):
        """If 'annotated' is returned, its shape must match the input image."""
        image = _make_image()
        result = self._call(image)
        if result is None or "annotated" not in result:
            return
        ann = result["annotated"]
        self.assertEqual(
            ann.shape, image.shape,
            f"'annotated' shape {ann.shape} does not match "
            f"input image shape {image.shape}.\n"
            "  → Start from a copy of the input: annotated = image.copy()",
        )

    # ── robustness ────────────────────────────────────────────────────────────

    def test_black_frame_does_not_crash(self):
        """detect_lanes() must not raise an exception on a fully black image."""
        try:
            self.node.detect_lanes(_make_image())
        except NotImplementedError:
            self.skipTest("Not yet implemented.")
        except Exception as exc:
            self.fail(
                f"detect_lanes() raised {type(exc).__name__} on a black image: {exc}\n"
                "  → Make sure your code handles images with no visible features."
            )

    def test_various_image_sizes_do_not_crash(self):
        """detect_lanes() should handle different image sizes without crashing."""
        for h, w in [(360, 640), (480, 854), (720, 1280)]:
            with self.subTest(size=f"{h}×{w}"):
                try:
                    result = self.node.detect_lanes(_make_image(h, w))
                except NotImplementedError:
                    self.skipTest("Not yet implemented.")
                except Exception as exc:
                    self.fail(
                        f"detect_lanes() raised {type(exc).__name__} "
                        f"on a {h}×{w} image: {exc}"
                    )
                else:
                    if result is not None and "annotated" in result:
                        self.assertEqual(
                            result["annotated"].shape[:2], (h, w),
                            f"'annotated' shape mismatch for {h}×{w} input.\n"
                            "  → Use annotated = image.copy() to start.",
                        )


# =============================================================================
# Tests: compute_control()
# =============================================================================

class TestComputeControl(unittest.TestCase):
    """
    Contract for  compute_control(speed_kmh: float) → (throttle, brake, steer) | None

    If a tuple/list is returned it must satisfy:
      throttle  float  [0.0, 1.0]     gas pedal
      brake     float  [0.0, 1.0]     brake pedal
      steer     float [-1.0, +1.0]    steering angle

    Returning None is allowed (sends no command that cycle).
    """

    @classmethod
    def setUpClass(cls):
        cls.node = _make_node()

    # ── helper ────────────────────────────────────────────────────────────────

    def _call(self, speed_kmh: float = 30.0):
        """Call compute_control and skip the test if not yet implemented."""
        try:
            return self.node.compute_control(speed_kmh)
        except NotImplementedError:
            self.skipTest(
                "compute_control() is not yet implemented.\n"
                "  → Open solution/my_adas.py and replace "
                "'raise NotImplementedError(...)' with your code."
            )

    # ── return type ───────────────────────────────────────────────────────────

    def test_returns_tuple_or_none(self):
        """compute_control() must return a 3-tuple/list or None."""
        result = self._call()
        if result is None:
            return   # returning None (send no command) is valid
        self.assertIsInstance(
            result, (tuple, list),
            f"Expected a 3-tuple (throttle, brake, steer) or None, "
            f"got {type(result).__name__!r}.\n"
            "  → Example: return (0.4, 0.0, 0.0)",
        )

    def test_returns_exactly_3_elements(self):
        """The returned sequence must have exactly 3 elements."""
        result = self._call()
        if result is None:
            return
        self.assertEqual(
            len(result), 3,
            f"Expected exactly 3 values (throttle, brake, steer), "
            f"got {len(result)}: {result!r}.\n"
            "  → Return a tuple with exactly three floats.",
        )

    def test_all_values_are_numeric(self):
        """throttle, brake, and steer must be int or float."""
        result = self._call()
        if result is None or len(result) != 3:
            return
        for name, val in zip(("throttle", "brake", "steer"), result):
            self.assertIsInstance(
                val, (int, float),
                f"'{name}' must be a number, got {type(val).__name__!r}: {val!r}.\n"
                "  → Make sure all three values are Python floats.",
            )

    # ── value ranges ──────────────────────────────────────────────────────────

    def test_throttle_in_range(self):
        """throttle must be in [0.0, 1.0] for all tested speeds."""
        for speed in [0.0, 15.0, 30.0, 60.0, 100.0]:
            with self.subTest(speed_kmh=speed):
                result = self._call(speed)
                if result is None or len(result) != 3:
                    continue
                throttle = float(result[0])
                self.assertGreaterEqual(
                    throttle, 0.0,
                    f"throttle={throttle} < 0.0 at speed={speed} km/h.\n"
                    "  → Throttle cannot be negative.",
                )
                self.assertLessEqual(
                    throttle, 1.0,
                    f"throttle={throttle} > 1.0 at speed={speed} km/h.\n"
                    "  → Throttle maximum is 1.0 (full gas).",
                )

    def test_brake_in_range(self):
        """brake must be in [0.0, 1.0] for all tested speeds."""
        for speed in [0.0, 15.0, 30.0, 60.0, 100.0]:
            with self.subTest(speed_kmh=speed):
                result = self._call(speed)
                if result is None or len(result) != 3:
                    continue
                brake = float(result[1])
                self.assertGreaterEqual(
                    brake, 0.0,
                    f"brake={brake} < 0.0 at speed={speed} km/h.\n"
                    "  → Brake cannot be negative.",
                )
                self.assertLessEqual(
                    brake, 1.0,
                    f"brake={brake} > 1.0 at speed={speed} km/h.\n"
                    "  → Brake maximum is 1.0 (full brake).",
                )

    def test_steer_in_range(self):
        """steer must be in [-1.0, +1.0] for all tested speeds."""
        for speed in [0.0, 15.0, 30.0, 60.0, 100.0]:
            with self.subTest(speed_kmh=speed):
                result = self._call(speed)
                if result is None or len(result) != 3:
                    continue
                steer = float(result[2])
                self.assertGreaterEqual(
                    steer, -1.0,
                    f"steer={steer} < -1.0 at speed={speed} km/h.\n"
                    "  → Full left is -1.0; clamp values below that.",
                )
                self.assertLessEqual(
                    steer, 1.0,
                    f"steer={steer} > +1.0 at speed={speed} km/h.\n"
                    "  → Full right is +1.0; clamp values above that.",
                )

    # ── safety checks ─────────────────────────────────────────────────────────

    def test_throttle_and_brake_not_both_nonzero(self):
        """Throttle and brake must not both be > 0 at the same time."""
        for speed in [0.0, 30.0, 60.0]:
            with self.subTest(speed_kmh=speed):
                result = self._call(speed)
                if result is None or len(result) != 3:
                    continue
                throttle, brake = float(result[0]), float(result[1])
                self.assertFalse(
                    throttle > 0.0 and brake > 0.0,
                    f"At speed={speed} km/h: throttle={throttle:.3f} and "
                    f"brake={brake:.3f} are both > 0.\n"
                    "  → Applying throttle and brake simultaneously is "
                    "contradictory — use one or the other.",
                )

    # ── speed-dependent behaviour ─────────────────────────────────────────────

    def test_accelerates_from_standstill(self):
        """At 0 km/h the car should apply throttle > 0 to start moving."""
        result = self._call(speed_kmh=0.0)
        if result is None or len(result) != 3:
            return
        throttle = float(result[0])
        self.assertGreater(
            throttle, 0.0,
            "At 0 km/h throttle should be > 0.0 — the car needs to start moving.\n"
            "  → Example: if speed_kmh < target: return (0.4, 0.0, 0.0)",
        )

    def test_does_not_accelerate_at_very_high_speed(self):
        """At 100 km/h the car should NOT apply throttle (already fast enough)."""
        result = self._call(speed_kmh=100.0)
        if result is None or len(result) != 3:
            return
        throttle = float(result[0])
        self.assertEqual(
            throttle, 0.0,
            f"At 100 km/h throttle={throttle:.3f} should be 0.0 — "
            "the car is already too fast.\n"
            "  → Example: if speed_kmh > target: return (0.0, 0.3, 0.0)",
        )

    # ── robustness ────────────────────────────────────────────────────────────

    def test_does_not_crash_at_zero_speed(self):
        """compute_control(0.0) must not raise an exception."""
        try:
            self.node.compute_control(0.0)
        except NotImplementedError:
            self.skipTest("Not yet implemented.")
        except Exception as exc:
            self.fail(
                f"compute_control(0.0) raised {type(exc).__name__}: {exc}\n"
                "  → Make sure your code handles speed = 0."
            )

    def test_does_not_crash_at_high_speed(self):
        """compute_control(120.0) must not raise an exception."""
        try:
            self.node.compute_control(120.0)
        except NotImplementedError:
            self.skipTest("Not yet implemented.")
        except Exception as exc:
            self.fail(
                f"compute_control(120.0) raised {type(exc).__name__}: {exc}\n"
                "  → Make sure your code handles high speeds."
            )


# =============================================================================
# Tests: framework wiring
# =============================================================================

class TestFrameworkWiring(unittest.TestCase):
    """
    Verify that the framework callbacks (process_image, on_speed) correctly
    call the student functions and handle unexpected output gracefully.

    These tests use spy functions to replace detect_lanes / compute_control,
    so they pass regardless of whether you have implemented those functions yet.
    """

    def setUp(self):
        self.node = _make_node()

    # ── process_image → detect_lanes ─────────────────────────────────────────

    def test_process_image_calls_detect_lanes(self):
        """process_image() must delegate to detect_lanes()."""
        called = []
        self.node.detect_lanes = lambda img: (called.append(img), None)[1]
        self.node.process_image(_make_image())
        self.assertTrue(
            len(called) > 0,
            "process_image() did not call detect_lanes().\n"
            "  → The framework callback should call self.detect_lanes(image).",
        )

    def test_process_image_passes_image_to_detect_lanes(self):
        """process_image() must pass the full image to detect_lanes()."""
        received = []
        self.node.detect_lanes = lambda img: (received.append(img.shape), None)[1]
        image = _make_image()
        self.node.process_image(image)
        if received:
            self.assertEqual(
                received[0], image.shape,
                f"detect_lanes() received image shape {received[0]}, "
                f"expected {image.shape}.",
            )

    def test_annotated_frame_is_used_when_returned(self):
        """If detect_lanes() returns 'annotated', it should be pushed to the viewer."""
        pushed = []
        self.node._viewer.push = lambda f: pushed.append(f.shape)

        annotated = np.ones((_H, _W, 3), dtype=np.uint8) * 128
        self.node.detect_lanes = lambda img: {
            "steer_offset": 0.0,
            "annotated": annotated,
        }
        self.node.process_image(_make_image())
        self.assertTrue(
            len(pushed) > 0 and pushed[0] == annotated.shape,
            "When detect_lanes() returns 'annotated', the viewer should display "
            "that frame instead of the raw image.",
        )

    def test_lane_info_stored_after_detect_lanes(self):
        """A valid detect_lanes() result should be stored in self._lane_info."""
        self.node.detect_lanes = lambda img: {"steer_offset": 0.15}
        self.node.process_image(_make_image())
        self.assertIsNotNone(
            self.node._lane_info,
            "self._lane_info should be set to the last detect_lanes() result "
            "so compute_control() can use it.",
        )
        self.assertAlmostEqual(self.node._lane_info["steer_offset"], 0.15)

    # ── on_speed → compute_control ────────────────────────────────────────────

    def test_on_speed_calls_compute_control(self):
        """on_speed() must delegate to compute_control()."""
        called_with = []
        self.node.compute_control = lambda s: (called_with.append(s), None)[1]
        self.node.on_speed(42.0)
        self.assertTrue(
            len(called_with) > 0,
            "on_speed() did not call compute_control().\n"
            "  → The framework callback should call self.compute_control(speed_kmh).",
        )

    def test_on_speed_passes_correct_speed(self):
        """on_speed() must pass the correct speed value to compute_control()."""
        called_with = []
        self.node.compute_control = lambda s: (called_with.append(s), None)[1]
        self.node.on_speed(55.5)
        if called_with:
            self.assertAlmostEqual(
                called_with[0], 55.5,
                msg=f"compute_control received {called_with[0]}, expected 55.5.",
            )

    def test_compute_control_result_is_sent(self):
        """A valid (throttle, brake, steer) return should reach send_control()."""
        self.node.compute_control = lambda s: (0.5, 0.0, -0.2)
        pub = self.node._pubs.get("/carla/hero/cmd_vel_ext")
        if pub is None:
            self.skipTest("Control publisher not found in mocked node.")
        self.node.on_speed(20.0)
        self.assertTrue(
            pub.publish.called,
            "on_speed() did not publish a control command even though "
            "compute_control() returned a valid tuple.",
        )

    # ── error handling ────────────────────────────────────────────────────────

    def test_out_of_range_detect_lanes_does_not_crash(self):
        """process_image() must not propagate ValueError from bad detect_lanes output."""
        self.node.detect_lanes = lambda img: {"steer_offset": 99.0}
        try:
            self.node.process_image(_make_image())
        except Exception as exc:
            self.fail(
                f"process_image() crashed on invalid detect_lanes output: {exc}\n"
                "  → The framework should catch and log the error, not propagate it."
            )

    def test_out_of_range_compute_control_does_not_crash(self):
        """on_speed() must not propagate ValueError from bad compute_control output."""
        self.node.compute_control = lambda s: (5.0, -1.0, 99.0)
        try:
            self.node.on_speed(30.0)
        except Exception as exc:
            self.fail(
                f"on_speed() crashed on invalid compute_control output: {exc}\n"
                "  → The framework should catch and log the error, not propagate it."
            )

    def test_not_implemented_detect_lanes_does_not_crash(self):
        """process_image() must swallow NotImplementedError from detect_lanes()."""
        self.node.detect_lanes = lambda img: (_ for _ in ()).throw(NotImplementedError)
        try:
            self.node.process_image(_make_image())
        except NotImplementedError as exc:
            self.fail(
                f"process_image() let NotImplementedError escape: {exc}\n"
                "  → The framework should ignore it silently."
            )

    def test_not_implemented_compute_control_does_not_crash(self):
        """on_speed() must swallow NotImplementedError from compute_control()."""
        self.node.compute_control = lambda s: (_ for _ in ()).throw(NotImplementedError)
        try:
            self.node.on_speed(30.0)
        except NotImplementedError as exc:
            self.fail(
                f"on_speed() let NotImplementedError escape: {exc}\n"
                "  → The framework should ignore it silently."
            )


# =============================================================================

if __name__ == "__main__":
    unittest.main(verbosity=2)
