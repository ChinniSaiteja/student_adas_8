# HSHL ADAS Student Lab

Welcome to the **Hamm-Lippstadt University of Applied Sciences (HSHL) ADAS Student Lab**.

You will implement an **Advanced Driver Assistance System (ADAS)** that runs alongside a **human driver** in a CARLA simulation. A student drives the vehicle manually using a steering wheel or keyboard. Your ADAS module runs in parallel, watching the sensors, and either warns the driver via on-screen notifications or briefly takes over vehicle control when it detects a hazard.

---

## Table of Contents

1. [The Big Picture](#the-big-picture)
2. [Your Tasks](#your-tasks)
3. [How Intervention Works](#how-intervention-works)
4. [Repository Structure](#repository-structure)
5. [Development Workflow](#development-workflow)
6. [Sensor API Reference](#sensor-api-reference)
7. [HUD & Control API](#hud--control-api)
8. [Adding Python Libraries](#adding-python-libraries)
9. [Docker Command Reference](#docker-command-reference)
10. [Tips & Best Practices](#tips--best-practices)

---

## The Big Picture

```
┌──────────────────────────────────────────────┐
│             CARLA Simulation (lab PC)         │
│                                               │
│   Human Driver ──► steering wheel / keyboard  │
│                         │                     │
│                    RUNNING_MANUAL             │
│                         │                     │
│       ◄── override ─────┤◄─── your ADAS      │
│                    RUNNING_REMOTE             │
│                         │                     │
│       returns to MANUAL when ADAS stops       │
└──────────────────────────────────────────────┘
```

**Normal operation:** a student drives the car manually. Your ADAS module watches the camera and sensors in the background.

**Notification:** when you detect something worth flagging (upcoming red light, nearby pedestrian), call `show_warning()` or `show_alert()`. The message appears on the simulation screen immediately, without touching the controls.

**Intervention:** when you detect a hazard that requires action, call `send_control()`. The simulator hands control to your ADAS for as long as you keep sending commands (at least once every 500 ms). When you stop, the simulator returns control to the human driver automatically.

> **Design principle:** only intervene when necessary. Your ADAS should enhance the driver, not replace them.

---

## Your Tasks

Your implementation lives in **`solution/my_adas.py`**. You must implement **three functions**:

| # | Function | What it does |
|---|---|---|
| ① | `detect_traffic_light(image)` | Detect the current traffic light state from the camera frame |
| ② | `detect_objects(image)` | Detect pedestrians and other vehicles in the scene |
| ③ | `compute_control(speed_kmh)` | Decide whether to notify the driver or intervene |

---

### Task ① — Traffic Light Detection

```python
def detect_traffic_light(self, image: np.ndarray) -> dict | None:
```

Analyse the camera frame and return the traffic light state.

**Return value:**

```python
{
    "state":    "red" | "yellow" | "green" | "unknown",  # REQUIRED
    "annotated": np.ndarray,                              # OPTIONAL — your drawings
}
```

**Hints:**
- Use HSV colour masking (`cv2.inRange`) to isolate red/yellow/green regions
- Focus on the upper portion of the image — traffic lights hang above the road
- Account for CARLA's lighting: red is a bright saturated region near `H≈0` or `H≈170`

---

### Task ② — Pedestrian & Vehicle Detection

```python
def detect_objects(self, image: np.ndarray) -> dict | None:
```

Detect other road users in the camera frame.

**Return value:**

```python
{
    "pedestrians": [ {"bbox": [x1,y1,x2,y2], "confidence": 0.9}, ... ],  # REQUIRED
    "vehicles":    [ {"bbox": [x1,y1,x2,y2], "confidence": 0.8}, ... ],  # REQUIRED
    "annotated":   np.ndarray,                                            # OPTIONAL
}
```

`bbox` is in pixel coordinates `[x1, y1, x2, y2]` (top-left to bottom-right).
Both lists may be empty — return `{"pedestrians": [], "vehicles": []}` if nothing is detected.

**Hints:**
- A pre-trained YOLO model (`ultralytics`) works well — add it to `requirements.txt`
- COCO class IDs: `0` = person, `2` = car, `3` = motorcycle, `5` = bus, `7` = truck

---

### Task ③ — Driver Assistance (Notify or Intervene)

```python
def compute_control(self, speed_kmh: float) -> tuple | None:
```

Use your detection results to decide what to do. You have two tools:

**Tool A — Notify the driver (no control change):**

```python
self.show_notification("Green light ahead",        duration=3.0)   # white
self.show_warning("Pedestrian on the right",       duration=5.0)   # yellow
self.show_alert("RED LIGHT — STOPPING",            duration=5.0)   # red
```

**Tool B — Override the controls temporarily:**

```python
return (throttle, brake, steer)
#        ────────  ─────  ─────
#   float [0,1]  [0,1]  [-1,+1]
```

Return `None` (or just don't return) to **not intervene** this cycle — the human driver stays in control.

> **Rule:** never set throttle and brake both > 0 at the same time.
> **Rule:** if you return a control command, keep doing so at least once every 500 ms or the car returns to the driver.

**Suggested behaviour:**

| Situation | Recommended action |
|---|---|
| Traffic light = **red** | Override: brake to a stop. Alert: "RED LIGHT — STOPPING" |
| Traffic light = **yellow** | Override: decelerate. Warning: "Yellow light — slowing" |
| Traffic light = **green** | Notification: "Green — go" and return `None` (driver decides) |
| **Pedestrian** detected | Override: emergency brake. Alert: "PEDESTRIAN AHEAD" |
| **Vehicle** too close | Warning: "Vehicle ahead" and optionally override to reduce speed |
| No hazards | Return `None` — leave the driver in control |

**Accessing your detection results inside `compute_control`:**

```python
def compute_control(self, speed_kmh: float):
    tl       = self._traffic_light_info        # dict from detect_traffic_light(), or None
    tl_state = tl["state"] if tl else "unknown"

    obj         = self._object_info            # dict from detect_objects(), or None
    pedestrians = obj["pedestrians"] if obj else []
    vehicles    = obj["vehicles"]    if obj else []

    # Intervene on red light
    if tl_state == "red":
        self.show_alert("RED LIGHT — STOPPING", duration=2.0)
        return (0.0, 0.8, 0.0)   # brake hard

    # Warn on pedestrian, override to emergency stop
    if pedestrians:
        self.show_alert("PEDESTRIAN AHEAD", duration=3.0)
        return (0.0, 1.0, 0.0)   # full brake

    # No hazard — let the driver drive
    return None
```

---

## How Intervention Works

When you return a `(throttle, brake, steer)` tuple from `compute_control`, the following happens:

1. The simulator switches from `RUNNING_MANUAL` → `RUNNING_REMOTE`
2. Your control values are applied to the vehicle
3. As long as your ADAS keeps returning commands (at least once every 500 ms), the simulator stays in `RUNNING_REMOTE`
4. When you return `None`, the 500 ms timeout expires and the simulator returns to `RUNNING_MANUAL` — the human driver is back in control

You can use `create_periodic_task` to send sustained control commands at a fixed rate during an intervention:

```python
def __init__(self):
    super().__init__("my_adas")
    self._braking = False
    self.create_periodic_task(0.1, self._send_intervention)   # 10 Hz

def _send_intervention(self):
    if self._braking:
        self.send_control(throttle=0.0, brake=0.8, steer=0.0)
    # If not braking, we send nothing — driver retakes control automatically
```

---

## Repository Structure

```
student_adas/
│
├── solution/
│   └── my_adas.py          ← YOUR FILE — implement the three functions here
│
├── adas/
│   ├── interface.py        ← base class (do not modify)
│   ├── viewer.py           ← browser camera viewer (do not modify)
│   └── topics.py           ← ROS topic name constants (do not modify)
│
├── tests/
│   └── test_my_adas.py     ← run this to validate your outputs locally
│
├── bags/                   ← place instructor bag folders here (home testing)
│
├── dev_tools/
│   ├── play_bag.sh         ← bag-replay entry point
│   └── control_logger/     ← prints your control commands to the terminal
│
├── Dockerfile              ← add apt packages here
├── requirements.txt        ← add pip packages here
├── docker-compose.yaml     ← run configuration
└── README.md               ← this file
```

> **Rule of thumb:** the only file you ever need to edit is `solution/my_adas.py`.
> When adding dependencies, also edit `requirements.txt` (pip) or `Dockerfile` (apt).

---

## Development Workflow

### Step 1 — Set up the environment

Clone this repository, then build and start the home-testing stack:

```bash
docker compose --profile bag up --build
```

Open **http://localhost:8080** in your browser — you will see the live camera feed from the bag.

---

### Step 2 — Implement `detect_traffic_light`

Find the `detect_traffic_light` method in `solution/my_adas.py` and replace the `raise NotImplementedError` line.

**Starter approach using HSV colour masking:**

```python
def detect_traffic_light(self, image: np.ndarray):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    h, w = image.shape[:2]
    roi  = hsv[:h // 3, :]   # upper third — traffic lights hang above

    red1   = cv2.inRange(roi, np.array([0,   120, 100]), np.array([10,  255, 255]))
    red2   = cv2.inRange(roi, np.array([160, 120, 100]), np.array([180, 255, 255]))
    red    = cv2.bitwise_or(red1, red2)
    green  = cv2.inRange(roi, np.array([40, 100, 100]), np.array([90, 255, 255]))
    yellow = cv2.inRange(roi, np.array([15, 100, 100]), np.array([40, 255, 255]))

    counts = {"red": cv2.countNonZero(red),
              "green": cv2.countNonZero(green),
              "yellow": cv2.countNonZero(yellow)}

    state = max(counts, key=counts.get) if max(counts.values()) > 50 else "unknown"
    return {"state": state}
```

---

### Step 3 — Implement `detect_objects`

Find the `detect_objects` method and replace the `raise NotImplementedError` line.

**Starter approach using YOLOv8** (add `ultralytics` to `requirements.txt` first):

```python
from ultralytics import YOLO

class MyADAS(CarlaADASInterface):
    def __init__(self):
        super().__init__("my_adas")
        self._yolo = YOLO("yolov8n.pt")
        ...

    def detect_objects(self, image: np.ndarray):
        results = self._yolo(image, verbose=False)[0]
        pedestrians, vehicles = [], []
        for box in results.boxes:
            cls  = int(box.cls[0])
            conf = float(box.conf[0])
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            entry = {"bbox": [x1, y1, x2, y2], "confidence": conf}
            if cls == 0:
                pedestrians.append(entry)
            elif cls in (2, 3, 5, 7):
                vehicles.append(entry)
        annotated = results.plot()
        return {"pedestrians": pedestrians, "vehicles": vehicles, "annotated": annotated}
```

---

### Step 4 — Implement `compute_control`

Decide when to notify the driver and when to intervene. See the [Task ③ section](#task--driver-assistance-notify-or-intervene) for examples.

Start simple: return `None` always, then add one notification, then add one override, and verify each step with the ctrl-logger.

---

### Step 5 — Test locally

```bash
cd student_adas
python -m pytest tests/test_my_adas.py -v
```

---

### Step 6 — Run with bag replay

Get the session bag from your instructor and place it in `bags/`:

```
student_adas/bags/session_2026-xx-xx/
    ├── metadata.yaml
    └── session_2026-xx-xx_0.db3
```

Start the home-testing stack:

```bash
docker compose --profile bag up --build
```

In a second terminal, watch your control commands in real time:

```bash
docker compose --profile bag logs -f ctrl-logger
```

Open **http://localhost:8080** for the camera view.

---

### Step 7 — Run in the lab

Once your code works with the bag, connect to the live simulation where a student is driving:

```bash
docker compose up --build
```

Your ADAS module runs in parallel with the human driver from this point on.

---

## Sensor API Reference

### Always available

| Sensor | Registration | Callback signature | Rate |
|---|---|---|---|
| Camera | `self.on_camera_image(cb)` | `cb(image: np.ndarray)` — BGR `(720, 1280, 3)` | ~15 fps |
| Speed | `self.on_speed_update(cb)` | `cb(speed_kmh: float)` | ~45 Hz |

`self.current_speed` is always up to date and can be read from any method.

### Optional sensors (uncomment in `__init__` to activate)

| Sensor | Registration | What you receive |
|---|---|---|
| IMU | `self.on_imu_update(cb)` | `msg.linear_acceleration.{x,y,z}` m/s²; `msg.angular_velocity.z` rad/s |
| GNSS | `self.on_gnss_update(cb)` | `msg.latitude`, `msg.longitude`, `msg.altitude` |
| Collision | `self.on_collision(cb)` | `info["actor"]`, `info["impulse"]` [x,y,z] N·s |
| Lane invasion | `self.on_lane_invasion(cb)` | `lane_types` e.g. `["Solid"]` |

---

## HUD & Control API

### Notify the driver (no control change)

```python
self.show_notification("Green light ahead",       duration=3.0)   # white  — info
self.show_warning("Pedestrian on the right",      duration=5.0)   # yellow — caution
self.show_alert("RED LIGHT — STOPPING",           duration=5.0)   # red    — critical
```

These appear on the simulation screen immediately. The instructor and driver can see them. Use them to make your ADAS logic visible.

### Override the controls

```python
# Return from compute_control() to take over:
return (throttle, brake, steer)

# Or call directly from any callback / periodic task:
self.send_control(throttle=0.0, brake=0.8, steer=0.0)
```

Return `None` from `compute_control` (or call `send_control` less often than every 500 ms) to release control back to the driver.

### Periodic Tasks

Run a function at a fixed rate — useful for sustained interventions or status updates:

```python
def __init__(self):
    super().__init__("my_adas")
    self.create_periodic_task(0.5, self.status_update)   # every 500 ms

def status_update(self):
    self.show_notification(f"Speed: {self.current_speed:.1f} km/h")
```

---

## Adding Python Libraries

Add pip packages to `requirements.txt`, one per line:

```
ultralytics
filterpy
```

Then rebuild: `docker compose --profile bag up --build`

### Already included

| Library | Import |
|---|---|
| `numpy` | `import numpy as np` |
| `opencv` | `import cv2` |
| `scikit-learn` | `from sklearn import ...` |
| `scipy` | `from scipy import ...` |

For system-level (`apt`) packages, add them to the `Dockerfile` and rebuild.

---

## Docker Command Reference

### At home (bag replay)

| Task | Command |
|---|---|
| Build and start | `docker compose --profile bag up --build` |
| Stop everything | `docker compose --profile bag down` |
| Follow control logger | `docker compose --profile bag logs -f ctrl-logger` |
| Follow ADAS node logs | `docker logs -f adas_student` |
| Open a shell inside | `docker exec -it adas_student bash` |

### In the lab (live, alongside human driver)

| Task | Command |
|---|---|
| Build and start | `docker compose up --build` |
| Stop | `docker compose down` |
| Follow logs | `docker logs -f adas_student` |

---

## Tips & Best Practices

**Return `None` from `compute_control` when there is no hazard.** This gives control back to the human driver. Only intervene when your detections are confident and the situation actually requires it.

**Use HUD messages for everything your ADAS is doing.** The instructor watches the HUD during the lab session. `show_notification` (white), `show_warning` (yellow), `show_alert` (red) all appear on screen immediately.

**Keep detection callbacks fast.** The camera fires at ~15 fps (~65 ms budget). If your model is slow, run it in a `create_periodic_task` at a lower rate and cache the result.

**Avoid `time.sleep()` inside callbacks.** ROS 2 uses a cooperative executor — sleeping blocks all other callbacks. Use `create_periodic_task` for timed work.

**Test incrementally.** Start with `compute_control` always returning `None` (driver in full control). Add one notification. Add one override. Verify each step with the ctrl-logger before moving on.

**Commit often.** Use `git commit` to save working checkpoints before experimenting.

**LiDAR is not available.** Use the camera, IMU, GNSS, and the collision / lane invasion detectors.
