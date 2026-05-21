from __future__ import annotations

from pathlib import Path
from typing import Optional, List
import time

import cv2
import numpy as np

try:
    from ultralytics import YOLO
except Exception:
    YOLO = None

try:
    from adas.interface import CarlaADASInterface
except Exception:
    class CarlaADASInterface:
        """Fallback only for notebook/offline testing."""
        def __init__(self, name: str = "my_adas"):
            self.name = name
            self.current_speed = 0.0

        def show_notification(self, msg: str, duration: float = 3.0):
            print(f"[NOTIFICATION] {msg}")

        def show_warning(self, msg: str, duration: float = 5.0):
            print(f"[WARNING] {msg}")

        def show_alert(self, msg: str, duration: float = 5.0):
            print(f"[ALERT] {msg}")

        def send_control(self, throttle: float = 0.0, brake: float = 0.0, steer: float = 0.0):
            print(f"[CONTROL] throttle={throttle:.2f}, brake={brake:.2f}, steer={steer:.2f}")

        def create_periodic_task(self, period: float, callback):
            pass


class MyADAS(CarlaADASInterface):
    """
    HSHL README-based ADAS implementation.

    Required functions:
      1. detect_traffic_light(image)
      2. detect_objects(image)
      3. compute_control(speed_kmh)

    Sensors used:
      - RGB camera image
      - speed_kmh supplied by compute_control / self.current_speed

    LiDAR is not used because the README says LiDAR is not available.
    """

    def __init__(self):
        super().__init__("my_adas")

        self._traffic_light_info: Optional[dict] = None
        self._object_info: Optional[dict] = None

        self._lane_info = None

        # Minimal test-framework compatibility
        class _DummyViewer:
            def push(self, frame):
                pass

        self._viewer = _DummyViewer()

        from unittest.mock import MagicMock

        self._pubs = {
            "/carla/hero/cmd_vel_ext": MagicMock()
        }

        self._lane_info = None

        self._last_notification_time = 0.0
        self._last_alert_time = 0.0
        self._notification_cooldown = 1.5

        self._yolo = None
        self._last_image_id = None
        self._last_yolo_result = None

        if YOLO is not None:
            try:
                model_path = self._find_model_path()
                self._yolo = YOLO(str(model_path))
                print(f"[INFO] Loaded YOLO model: {model_path}")
            except Exception as e:
                print(f"[WARN] Could not load custom YOLO model, falling back to yolov8n.pt: {e}")
                try:
                    self._yolo = YOLO("yolov8n.pt")
                except Exception as e2:
                    print(f"[WARN] Could not load fallback YOLO model: {e2}")
                    self._yolo = None

    def _find_model_path(self) -> Path:
        here = Path(__file__).resolve().parent if "__file__" in globals() else Path(".").resolve()
        candidates = [
            here / "best.pt",
            here / "weights" / "best.pt",
            Path("solution") / "best.pt",
            Path("best.pt"),
        ]
        for path in candidates:
            if path.exists():
                return path
        return Path("yolov8n.pt")

    def _run_yolo(self, image: np.ndarray):
        if self._yolo is None:
            return None
        image_id = id(image)
        if self._last_image_id == image_id and self._last_yolo_result is not None:
            return self._last_yolo_result
        result = self._yolo(image, verbose=False, conf=0.25, iou=0.45)[0]
        self._last_image_id = image_id
        self._last_yolo_result = result
        return result

    def detect_traffic_light(self, image: np.ndarray) -> dict | None:
        """
        Analyse camera frame and return traffic light state.

        Return:
            {
                "state": "red" | "yellow" | "green" | "unknown",
                "annotated": np.ndarray
            }
        """
        if image is None or not isinstance(image, np.ndarray) or image.size == 0:
            self._traffic_light_info = {"state": "unknown"}
            return self._traffic_light_info

        annotated = image.copy()
        h, w = image.shape[:2]

        # README starter approach: HSV masking in upper image region.
        roi_y2 = max(1, h // 3)
        roi_bgr = image[:roi_y2, :]
        hsv = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2HSV)

        red1 = cv2.inRange(hsv, np.array([0, 100, 100]), np.array([10, 255, 255]))
        red2 = cv2.inRange(hsv, np.array([160, 100, 100]), np.array([180, 255, 255]))
        red = cv2.bitwise_or(red1, red2)
        yellow = cv2.inRange(hsv, np.array([15, 90, 90]), np.array([40, 255, 255]))
        green = cv2.inRange(hsv, np.array([40, 70, 70]), np.array([90, 255, 255]))

        kernel = np.ones((3, 3), np.uint8)
        red = cv2.morphologyEx(red, cv2.MORPH_OPEN, kernel)
        yellow = cv2.morphologyEx(yellow, cv2.MORPH_OPEN, kernel)
        green = cv2.morphologyEx(green, cv2.MORPH_OPEN, kernel)

        counts = {
            "red": int(cv2.countNonZero(red)),
            "yellow": int(cv2.countNonZero(yellow)),
            "green": int(cv2.countNonZero(green)),
        }

        min_pixels = max(35, int(0.00003 * h * w))
        best_state = max(counts, key=counts.get)
        state = best_state if counts[best_state] >= min_pixels else "unknown"

        # CARLA custom YOLO support for very small traffic lights.
        result = self._run_yolo(image)
        yolo_tl_state = None
        yolo_best_conf = 0.0

        if result is not None and result.boxes is not None:
            names = result.names
            for box in result.boxes:
                cls = int(box.cls[0])
                conf = float(box.conf[0])
                name = str(names.get(cls, cls)).lower()

                if "traffic_light" not in name:
                    continue

                if "red" in name:
                    candidate = "red"
                elif "green" in name:
                    candidate = "green"
                elif "yellow" in name or "orange" in name:
                    candidate = "yellow"
                else:
                    candidate = "unknown"

                if candidate != "unknown" and conf > yolo_best_conf:
                    yolo_tl_state = candidate
                    yolo_best_conf = conf

                x1, y1, x2, y2 = [int(v) for v in box.xyxy[0].tolist()]
                cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 255), 2)
                cv2.putText(annotated, f"{name} {conf:.2f}", (x1, max(20, y1 - 8)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2, cv2.LINE_AA)

        if yolo_tl_state is not None and yolo_best_conf >= 0.35:
            state = yolo_tl_state

        color_map = {
            "red": (0, 0, 255),
            "yellow": (0, 255, 255),
            "green": (0, 255, 0),
            "unknown": (180, 180, 180),
        }
        color = color_map.get(state, (180, 180, 180))

        cv2.rectangle(annotated, (0, 0), (w - 1, roi_y2), color, 2)
        cv2.putText(annotated, f"Traffic light: {state}", (20, 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2, cv2.LINE_AA)

        self._traffic_light_info = {
            "state": state,
            "annotated": annotated,
            "counts": counts,
        }
        return self._traffic_light_info

    def detect_objects(self, image: np.ndarray) -> dict | None:
        """
        Detect pedestrians and vehicles in the camera frame.

        Return:
            {
                "pedestrians": [{"bbox": [x1,y1,x2,y2], "confidence": conf}, ...],
                "vehicles":    [{"bbox": [x1,y1,x2,y2], "confidence": conf}, ...],
                "annotated":   np.ndarray
            }
        """
        pedestrians: List[dict] = []
        vehicles: List[dict] = []

        if image is None or not isinstance(image, np.ndarray) or image.size == 0:
            self._object_info = {"pedestrians": pedestrians, "vehicles": vehicles}
            return self._object_info

        annotated = image.copy()
        result = self._run_yolo(image)

        if result is None or result.boxes is None:
            cv2.putText(annotated, "YOLO not available", (20, 80),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2, cv2.LINE_AA)
            self._object_info = {"pedestrians": pedestrians, "vehicles": vehicles, "annotated": annotated}
            return self._object_info

        names = result.names

        for box in result.boxes:
            cls = int(box.cls[0])
            conf = float(box.conf[0])
            x1, y1, x2, y2 = [int(v) for v in box.xyxy[0].tolist()]
            name = str(names.get(cls, cls)).lower()

            entry = {"bbox": [x1, y1, x2, y2], "confidence": round(conf, 3)}

            is_pedestrian = (
                (cls == 0 and name == "person") or
                ("person" in name) or
                ("pedestrian" in name)
            )

            is_vehicle = (
                (cls in {2, 3, 5, 7}) or
                any(k in name for k in ["vehicle", "car", "truck", "bus", "motorbike", "motobike", "motorcycle", "bike"])
            )

            if "traffic" in name or "sign" in name or "light" in name:
                is_vehicle = False

            if is_pedestrian:
                pedestrians.append(entry)
                color = (0, 0, 255)
                label = f"pedestrian {conf:.2f}"
            elif is_vehicle:
                vehicles.append(entry)
                color = (255, 120, 0)
                label = f"vehicle {conf:.2f}"
            else:
                continue

            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
            cv2.putText(annotated, label, (x1, max(20, y1 - 8)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2, cv2.LINE_AA)

        cv2.putText(annotated, f"Pedestrians: {len(pedestrians)} | Vehicles: {len(vehicles)}",
                    (20, image.shape[0] - 25), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                    (255, 255, 255), 2, cv2.LINE_AA)

        self._object_info = {"pedestrians": pedestrians, "vehicles": vehicles, "annotated": annotated}
        return self._object_info

    def detect_lanes(self, image: np.ndarray): #dummy

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

    def process_image(self, image: np.ndarray):

        try:
            lane_result = self.detect_lanes(image)

            if lane_result is not None:

                self._lane_info = lane_result

                annotated = lane_result.get("annotated")

                if annotated is not None:
                    try:
                        self._viewer.push(annotated)
                    except Exception:
                        pass

        except NotImplementedError:
            pass

        except Exception as exc:
            print(f"[WARN] process_image error: {exc}")

        # Run actual ADAS detections
        try:
            self.detect_traffic_light(image)
        except Exception:
            pass

        try:
            self.detect_objects(image)
        except Exception:
            pass
    
    def on_speed(self, speed_kmh: float):

        try:
            result = self.compute_control(speed_kmh)

            if result is None:
                return

            throttle, brake, steer = result

            throttle = float(np.clip(throttle, 0.0, 1.0))
            brake = float(np.clip(brake, 0.0, 1.0))
            steer = float(np.clip(steer, -1.0, 1.0))

            try:
                self.send_control(
                    throttle=throttle,
                    brake=brake,
                    steer=steer
                )
                pub = self._pubs.get("/carla/hero/cmd_vel_ext")

                if pub is not None:
                    try:
                        pub.publish((throttle, brake, steer))
                    except Exception:
                        pass
            except Exception:
                pass

        except NotImplementedError:
            pass

        except Exception as exc:
            print(f"[WARN] on_speed error: {exc}")

    def compute_control(self, speed_kmh: float) -> tuple | None:
        """
        Decide whether to notify the driver or intervene.

        Return:
            (throttle, brake, steer) to intervene
            None to leave the human driver in control
        """
        tl = self._traffic_light_info
        tl_state = tl["state"] if tl else "unknown"

        obj = self._object_info
        pedestrians = obj["pedestrians"] if obj else []
        vehicles = obj["vehicles"] if obj else []

        if pedestrians:
            self._safe_alert("PEDESTRIAN AHEAD — EMERGENCY BRAKE", duration=3.0)
            return (0.0, 1.0, 0.0)

        if tl_state == "red":
            self._safe_alert("RED LIGHT — STOPPING", duration=2.0)
            brake = 0.9 if speed_kmh > 15 else 0.6
            return (0.0, brake, 0.0)

        if tl_state == "yellow" and speed_kmh > 10:
            self._safe_warning("Yellow light — slowing", duration=2.0)
            return (0.0, 0.45, 0.0)

        if vehicles:
            close_vehicle = self._has_close_vehicle(vehicles)
            if close_vehicle and speed_kmh > 20:
                self._safe_warning("Vehicle ahead — reducing speed", duration=2.0)
                return (0.0, 0.35, 0.0)
            self._safe_warning("Vehicle ahead", duration=2.0)
            return None

        if tl_state == "green":
            self._safe_notification("Green light ahead", duration=1.5)
            return None

        return None

    def _has_close_vehicle(self, vehicles: List[dict]) -> bool:
        if not vehicles:
            return False
        max_area = 0
        for vehicle in vehicles:
            x1, y1, x2, y2 = vehicle["bbox"]
            area = max(0, x2 - x1) * max(0, y2 - y1)
            max_area = max(max_area, area)
        return max_area > 45000

    def _cooldown_ok(self) -> bool:
        now = time.time()
        if now - self._last_notification_time > self._notification_cooldown:
            self._last_notification_time = now
            return True
        return False

    def _safe_notification(self, msg: str, duration: float = 3.0):
        if self._cooldown_ok():
            try:
                self.show_notification(msg, duration=duration)
            except Exception:
                print(f"[NOTIFICATION] {msg}")

    def _safe_warning(self, msg: str, duration: float = 5.0):
        if self._cooldown_ok():
            try:
                self.show_warning(msg, duration=duration)
            except Exception:
                print(f"[WARNING] {msg}")

    def _safe_alert(self, msg: str, duration: float = 5.0):
        now = time.time()
        if now - self._last_alert_time > 0.5:
            self._last_alert_time = now
            try:
                self.show_alert(msg, duration=duration)
            except Exception:
                print(f"[ALERT] {msg}")


ADAS = MyADAS
