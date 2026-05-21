#!/usr/bin/env python3
"""
HSHL ADAS Student Lab
=====================
A human student drives the car manually (steering wheel / keyboard).
Your ADAS module runs in parallel. Implement the three functions below:

    detect_traffic_light(image)  — called for every camera frame (~15 fps)
    detect_objects(image)        — called for every camera frame (~15 fps)
    compute_control(speed_kmh)   — called for every speed update (~45 Hz)

─────────────────────────────────────────────────────────────────────────────
Quick-start
─────────────────────────────────────────────────────────────────────────────
1. Start the bag:   docker compose --profile bag up --build
2. Open browser:    http://localhost:8080   ← live camera + sensor dashboard
3. Implement the three functions below.
4. Validate locally (no simulator needed):
       python -m pytest tests/test_my_adas.py -v

─────────────────────────────────────────────────────────────────────────────
INPUTS  (what you receive)
─────────────────────────────────────────────────────────────────────────────
Camera frame  →  detect_traffic_light(image)  and  detect_objects(image)
  image         np.ndarray, shape (720, 1280, 3), BGR colour order

Speed update  →  compute_control(speed_kmh)
  speed_kmh     float, current speed in km/h  (0 … ~120)

─────────────────────────────────────────────────────────────────────────────
OUTPUTS
─────────────────────────────────────────────────────────────────────────────
detect_traffic_light(image)  →  dict | None
    {"state": "red"|"yellow"|"green"|"unknown",   # REQUIRED
     "annotated": np.ndarray}                      # OPTIONAL

detect_objects(image)  →  dict | None
    {"pedestrians": [{"bbox": [x1,y1,x2,y2], "confidence": float}, ...],
     "vehicles":    [{"bbox": [x1,y1,x2,y2], "confidence": float}, ...],
     "annotated": np.ndarray}                      # OPTIONAL

compute_control(speed_kmh)  →  (throttle, brake, steer) | None
    Return a tuple to INTERVENE — your ADAS takes over the controls temporarily.
    Return None   to let the HUMAN DRIVER stay in control (no intervention).
    NOTE: never set throttle > 0 and brake > 0 simultaneously.

─────────────────────────────────────────────────────────────────────────────
HUD — show messages on the simulation screen (driver and instructor see these)
─────────────────────────────────────────────────────────────────────────────
    self.show_notification(text, duration=3.0)   white  — general info
    self.show_warning(text, duration=5.0)        yellow — caution
    self.show_alert(text, duration=5.0)          red    — critical

Other helpers:
    self.send_control(throttle=0.0, brake=0.0, steer=0.0)
    self.create_periodic_task(period_sec, callback)
    self.current_speed   # float, always up to date
"""
import cv2          # type: ignore
import numpy as np  # type: ignore
import rclpy        # type: ignore

from ultralytics import YOLO
from adas import CarlaADASInterface, FrameViewer


class MyADAS(CarlaADASInterface):

    def __init__(self):
        super().__init__("my_adas")

        # Last detection results — available inside compute_control()
        self._traffic_light_info = None   # set by detect_traffic_light()
        self._object_info        = None   # set by detect_objects()
        self._lane_info = None
        self._yolo = YOLO("yolov8n.pt")

        # ── Live frame viewer ──────────────────────────────────────────────────
        # Open  http://localhost:8080  in your browser to watch the camera feed.
        self._viewer = FrameViewer(port=8080)
        self._viewer.start()
        self.register_viewer(self._viewer)

        # ── Register sensor callbacks ──────────────────────────────────────────
        self.on_camera_image(self.process_image)
        self.on_speed_update(self.on_speed)

        # Uncomment to enable optional sensors:
        # self.on_imu_update(self.on_imu)
        # self.on_gnss_update(self.on_gnss)
        # self.on_collision(self.on_collision_event)
        # self.on_lane_invasion(self.on_lane_event)

        self.get_logger().info("MyADAS started — waiting for sensor data...")

    # =========================================================================
    # Framework callbacks — route sensor data to your functions.
    # You do NOT need to modify these.
    # =========================================================================

    def process_image(self, image: np.ndarray):
        """Called for every camera frame (~15 fps). Runs both detection tasks."""
        annotated = image
        # ── Lane detection ──────────────────────────────────────────
        try:
            lane_result = self.detect_lanes(image)

            if lane_result is not None:
                self._lane_info = lane_result

                if "annotated" in lane_result:
                    annotated = lane_result["annotated"]

        except NotImplementedError:
            pass

        except Exception as exc:
            self.get_logger().error(
                f"detect_lanes() raised {type(exc).__name__}: {exc}"
            )
        # ── Traffic light detection ──────────────────────────────────────────
        try:
            tl_result = self.detect_traffic_light(image)
            if tl_result is not None:
                self._traffic_light_info = tl_result
                if "annotated" in tl_result:
                    annotated = tl_result["annotated"]
        except NotImplementedError:
            pass
        except Exception as exc:
            self.get_logger().error(f"detect_traffic_light() raised {type(exc).__name__}: {exc}")

        # ── Object detection ─────────────────────────────────────────────────
        try:
            obj_result = self.detect_objects(image)
            if obj_result is not None:
                self._object_info = obj_result
                # object annotated frame takes priority if provided
                if "annotated" in obj_result:
                    annotated = obj_result["annotated"]
        except NotImplementedError:
            pass
        except Exception as exc:
            self.get_logger().error(f"detect_objects() raised {type(exc).__name__}: {exc}")

        self._viewer.push(annotated)

    def on_speed(self, speed_kmh: float):
        """Called for every speed update (~45 Hz). Runs compute_control()."""
        try:
            result = self.compute_control(speed_kmh)
            if result is not None:
                throttle, brake, steer = self._check_compute_control(result)
                self.send_control(throttle, brake, steer)
        except NotImplementedError:
            pass
        except (ValueError, TypeError) as exc:
            msg = f"compute_control() bad output: {exc}"
            self.get_logger().error(msg)
            self.show_alert(msg, duration=4.0)
        except Exception as exc:
            self.get_logger().error(f"compute_control() raised {type(exc).__name__}: {exc}")

    # =========================================================================
    # TODO ①  detect_traffic_light
    # =========================================================================

    def detect_traffic_light(self, image: np.ndarray):
        """
        Detect the traffic light state from the camera frame.

        INPUT
        ─────
        image : np.ndarray
            BGR colour frame, shape (H, W, 3).

        HINTS
        ─────
        Focus on the upper portion of the image — traffic lights hang above the road:
            h, w = image.shape[:2]
            roi  = image[:h // 3, :]

        Use HSV colour masking to isolate light colours:
            hsv  = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            red1 = cv2.inRange(hsv_roi, np.array([0,   120, 100]), np.array([10,  255, 255]))
            red2 = cv2.inRange(hsv_roi, np.array([160, 120, 100]), np.array([180, 255, 255]))
            red  = cv2.bitwise_or(red1, red2)
            green  = cv2.inRange(hsv_roi, np.array([40, 100, 100]), np.array([90, 255, 255]))
            yellow = cv2.inRange(hsv_roi, np.array([15, 100, 100]), np.array([40, 255, 255]))

        Count pixels and pick the dominant colour:
            counts = {c: cv2.countNonZero(mask) for c, mask in ...}
            state  = max(counts, key=counts.get) if max(counts.values()) > 50 else "unknown"

        OUTPUT
        ──────
        Return a dict — or None if you cannot determine the state:
            {
                "state":    "red" | "yellow" | "green" | "unknown",  # REQUIRED
                "annotated": np.ndarray,   # OPTIONAL — same shape as image
            }
        """

        h, w = image.shape[:2]

        roi = image[:h // 3, :]

        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

        red1 = cv2.inRange(hsv, np.array([0, 120, 100]), np.array([10, 255, 255]))
        red2 = cv2.inRange(hsv, np.array([160, 120, 100]), np.array([180, 255, 255]))
        red = cv2.bitwise_or(red1, red2)

        yellow = cv2.inRange(hsv, np.array([15, 100, 100]), np.array([40, 255, 255]))

        green = cv2.inRange(hsv, np.array([40, 100, 100]), np.array([90, 255, 255]))

        counts = {
            "red": cv2.countNonZero(red),
            "yellow": cv2.countNonZero(yellow),
            "green": cv2.countNonZero(green)
        }

        max_color = max(counts, key=counts.get)

        if counts[max_color] < 50:
            state = "unknown"
        else:
            state = max_color

        annotated = image.copy()

        cv2.putText(
            annotated,
            f"Traffic Light: {state}",
            (30, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2
        )

        return {
            "state": state,
            "annotated": annotated
}
        # ── TODO: replace this line with your implementation ──────────────────
        # raise NotImplementedError("Implement detect_traffic_light() above ↑")

    # =========================================================================
    # TODO ②  detect_objects
    # =========================================================================

    def detect_objects(self, image: np.ndarray):
        """
        Detect pedestrians and vehicles in the camera frame.

        INPUT
        ─────
        image : np.ndarray
            BGR colour frame, shape (H, W, 3).

        HINTS — using YOLOv8 (add 'ultralytics' to requirements.txt first)
        ────────────────────────────────────────────────────────────────────
        In __init__:
            from ultralytics import YOLO
            self._yolo = YOLO("yolov8n.pt")   # downloads automatically on first run

        In detect_objects:
            results = self._yolo(image, verbose=False)[0]
            for box in results.boxes:
                cls  = int(box.cls[0])   # COCO class id
                conf = float(box.conf[0])
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                # cls == 0  → person
                # cls in (2, 3, 5, 7)  → car / motorcycle / bus / truck

        OUTPUT
        ──────
        Return a dict — or None:
            {
                "pedestrians": [{"bbox": [x1,y1,x2,y2], "confidence": float}, ...],
                "vehicles":    [{"bbox": [x1,y1,x2,y2], "confidence": float}, ...],
                "annotated":   np.ndarray,   # OPTIONAL — same shape as image
            }
        Both lists may be empty (return them as []) if nothing is detected.
        """
        results = self._yolo(image, verbose=False)[0]

        pedestrians = []
        vehicles = []

        for box in results.boxes:

            cls = int(box.cls[0])
            conf = float(box.conf[0])

            x1, y1, x2, y2 = map(int, box.xyxy[0])

            entry = {
                "bbox": [x1, y1, x2, y2],
                "confidence": conf
            }

            # Person
            if cls == 0:
                pedestrians.append(entry)

            # Vehicle classes
            elif cls in (2, 3, 5, 7):
                vehicles.append(entry)

        annotated = results.plot()

        return {
            "pedestrians": pedestrians,
            "vehicles": vehicles,
            "annotated": annotated
        }
        # ── TODO: replace this line with your implementation ──────────────────
        #raise NotImplementedError("Implement detect_objects() above ↑")

    # =========================================================================
    # TODO ③  compute_control
    # =========================================================================


    def detect_lanes(self, image: np.ndarray):

        annotated = image.copy()

        steer_offset = 0.0

        cv2.putText(
            annotated,
            "Lane Detection Active",
            (30, 100),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (255, 0, 0),
            2
        )

        return {
            "steer_offset": float(steer_offset),
            "annotated": annotated
        }
    
    def compute_control(self, speed_kmh: float):
        """
        Decide whether to notify the driver or temporarily take over the controls.

        CONTEXT
        ───────
        A human student is driving the car manually (steering wheel or keyboard).
        Your ADAS runs in parallel. You have two tools:

            A) Notify the driver  — call show_notification / show_warning / show_alert.
               The message appears on screen immediately. Controls are NOT affected.

            B) Intervene          — return (throttle, brake, steer).
               The simulator hands control to your ADAS immediately.
               As long as you keep returning commands (at least every 500 ms),
               your ADAS stays in control.
               When you return None, the car returns to the human driver.

        Design principle: only intervene when necessary.
        Return None when there is no hazard — keep the driver in control.

        INPUT
        ─────
        speed_kmh : float
            Current vehicle speed in km/h (typically 0 … 120).

        You can also read:
            self.current_speed          — same value, always up to date
            self._traffic_light_info    — last dict from detect_traffic_light(), or None
            self._object_info           — last dict from detect_objects(), or None

        ACCESSING DETECTION RESULTS
        ────────────────────────────
            tl       = self._traffic_light_info
            tl_state = tl["state"] if tl else "unknown"

            obj         = self._object_info
            pedestrians = obj["pedestrians"] if obj else []
            vehicles    = obj["vehicles"]    if obj else []

        SUGGESTED BEHAVIOUR
        ───────────────────
            Red light     → return (0.0, 0.8, 0.0)  +  show_alert("RED LIGHT")
            Yellow light  → return (0.0, 0.4, 0.0)  +  show_warning("Yellow — slowing")
            Green light   → return None              +  show_notification("Green — go")
            Pedestrian    → return (0.0, 1.0, 0.0)  +  show_alert("PEDESTRIAN AHEAD")
            Vehicle close → return (0.0, 0.3, 0.0)  +  show_warning("Vehicle ahead")
            No hazards    → return None  (driver stays in full control)

        OUTPUT
        ──────
        Return a 3-tuple to intervene, or None to let the driver drive:
            throttle : float  [0.0, 1.0]    gas pedal   (0 = coast, 1 = full gas)
            brake    : float  [0.0, 1.0]    brake pedal (0 = none,  1 = full brake)
            steer    : float [-1.0, +1.0]   steering    (-1 = full left, +1 = full right)

        NOTE: do NOT apply throttle and brake simultaneously.
        """
        tl = self._traffic_light_info
        tl_state = tl["state"] if tl else "unknown"

        obj = self._object_info

        pedestrians = obj["pedestrians"] if obj else []
        vehicles = obj["vehicles"] if obj else []

        # Emergency brake for pedestrians
        if len(pedestrians) > 0:

            self.show_alert("PEDESTRIAN AHEAD", duration=3.0)

            return (0.0, 1.0, 0.0)

        # Red light braking
        if tl_state == "red":

            self.show_alert("RED LIGHT - STOPPING", duration=3.0)

            return (0.0, 0.8, 0.0)

        # Yellow light slow down
        if tl_state == "yellow":

            self.show_warning("YELLOW LIGHT - SLOWING", duration=3.0)

            return (0.0, 0.4, 0.0)

        # Green light
        if tl_state == "green":

            self.show_notification("GREEN LIGHT - GO", duration=2.0)

            return None

        # Vehicle ahead warning
        if len(vehicles) > 0:

            self.show_warning("VEHICLE AHEAD", duration=2.0)

        return None

        # ── TODO: replace this line with your implementation ──────────────────
        #raise NotImplementedError("Implement compute_control() above ↑")

    # =========================================================================
    # Optional sensor callbacks (uncomment the on_* call in __init__ first)
    # =========================================================================

    # def on_imu(self, msg):
    #     ax = msg.linear_acceleration.x   # forward/backward m/s²  (+ = forward)
    #     ay = msg.linear_acceleration.y   # left/right       m/s²  (+ = left)
    #     az = msg.linear_acceleration.z   # up/down          m/s²  (+ = up)
    #     gz = msg.angular_velocity.z      # yaw rate         rad/s (+ = turning left)

    # def on_gnss(self, msg):
    #     lat = msg.latitude    # degrees north
    #     lon = msg.longitude   # degrees east
    #     alt = msg.altitude    # metres above sea level

    # def on_collision_event(self, info: dict):
    #     actor     = info["actor"]           # what was hit
    #     impulse   = info["impulse"]         # [x, y, z] in N·s
    #     magnitude = sum(v**2 for v in impulse) ** 0.5
    #     self.show_alert(f"Collision! {magnitude:.1f} N·s")

    # def on_lane_event(self, lane_types: list):
    #     if "Solid" in lane_types:
    #         self.show_warning("Solid line crossed!")

    # =========================================================================
    # Validation helpers — used by the framework, not by you
    # =========================================================================

    @staticmethod
    def _check_compute_control(result):
        if not (isinstance(result, (tuple, list)) and len(result) == 3):
            raise TypeError(
                f"compute_control() must return a 3-tuple or None, "
                f"got {type(result).__name__!r} of length {len(result) if hasattr(result, '__len__') else '?'}."
            )
        throttle, brake, steer = float(result[0]), float(result[1]), float(result[2])
        if not (0.0 <= throttle <= 1.0):
            raise ValueError(f"throttle={throttle} out of [0.0, 1.0].")
        if not (0.0 <= brake <= 1.0):
            raise ValueError(f"brake={brake} out of [0.0, 1.0].")
        if not (-1.0 <= steer <= 1.0):
            raise ValueError(f"steer={steer} out of [-1.0, +1.0].")
        if throttle > 0.0 and brake > 0.0:
            raise ValueError("throttle and brake must not both be > 0 simultaneously.")
        return throttle, brake, steer


def main():
    rclpy.init()
    node = MyADAS()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
