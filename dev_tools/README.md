# Home Testing Guide

This folder contains developer tools for testing your ADAS system **without access to the CARLA simulation**.

---

## How it works

When you are at home, you replace the live simulation with a **pre-recorded bag file** provided by your instructor. The bag contains real sensor data captured during a simulation session and replays it in a loop. Your code receives the same callbacks — `process_image`, `on_speed`, `on_imu_update`, etc. — as if the simulation were running.

```
                         ┌─────────────────────────┐
Instructor's bag file ──►│  bag-replay (ros2 bag)  │──► /carla/hero/camera  ──► your on_camera_image()
                         └─────────────────────────┘──► /carla/hero/speed   ──► your on_speed_update()
                                                     ──► /carla/hero/imu    ──► your on_imu_update()
                                                     └── ...
```

---

## Quick start

### Step 1 — Get a bag file from your instructor

Your instructor will provide a `.zip` (or `.tar.gz`) file containing a bag folder. The folder contains a `metadata.yaml` file and one or more data files.

### Step 2 — Place the bag inside `bags/`

Unzip the archive so your directory looks like this:

```
student_adas/
└── bags/
    └── session_2025-03-01_14-30/   ← the bag folder (name will vary)
        ├── metadata.yaml
        └── session_2025-03-01_14-30_0.db3
```

### Step 3 — Start with the bag profile

```bash
docker compose --profile bag up --build
```

This starts four containers:

| Container | What it does |
|-----------|-------------|
| `zenoh_router_student` | Local message broker (replaces the lab PC's router) |
| `bag_replay` | Replays the bag on a loop |
| `adas_student` | Your ADAS code |
| `ctrl_logger` | Prints your control commands to the terminal |

### Step 4 — Watch the output

In a second terminal:

```bash
# Your ADAS node logs (perception messages, warnings, etc.)
docker logs -f adas_student

# Your control commands and HUD events
docker compose --profile bag logs -f ctrl-logger
```

### Step 5 — Stop

```bash
docker compose --profile bag down
```

---

## Choosing a specific bag

If you have more than one bag in `bags/`, set `BAG_NAME` in a `.env` file next to `docker-compose.yaml`:

```
# student_adas/.env
BAG_NAME=session_2025-03-01_14-30
```

Then restart normally. The `.env` file is read automatically by Docker Compose.

---

## Control Logger output

The `ctrl-logger` service prints every `send_control()` call and every HUD event:

```
=== HSHL ADAS Control Logger ===
  Listening for control commands on  /carla/hero/cmd_vel_ext
  ...

[CTRL]  throttle=+0.350  brake=0.000  steer=-0.082  | speed=43.2 km/h
[CTRL]  throttle=+0.350  brake=0.000  steer=-0.043  | speed=44.8 km/h
[HUD WARNING ]  Vehicle ahead!  (5.0s)
[CTRL]  throttle=+0.000  brake=0.400  steer=+0.000  | speed=44.1 km/h
[HUD ALERT   ]  COLLISION IMMINENT  (5.0s)
```

Out-of-range values (throttle outside [0,1], steer outside [-1,1]) are highlighted in red.

---

## Folder structure (do not modify)

```
dev_tools/
├── README.md           ← this file
├── play_bag.sh         ← entrypoint used by the bag-replay container
└── control_logger/
    └── node.py         ← ROS 2 node that prints your control output
```

---

## Troubleshooting

**"No bag found in bags/"**
→ Make sure the bag folder is directly inside `student_adas/bags/`, not inside a subfolder of a subfolder.
→ The folder must contain a file named `metadata.yaml`.

**"Bag not found" with BAG_NAME set**
→ Check the spelling — `BAG_NAME` must match the exact folder name inside `bags/`.
→ Run `ls bags/` to see what folders are available.

**My callbacks are not firing**
→ Check that the bag was recorded with `ROLE_NAME=hero`. The topics must match `/carla/hero/...`.
→ Run `docker exec -it bag_replay ros2 bag info /bags/<name>` to inspect the bag's topics.

**Replay is too fast or too slow**
→ Use `ros2 bag play ... --rate 0.5` (half speed) or `--rate 2.0` (double speed).
→ Edit `dev_tools/play_bag.sh` and add `--rate <value>` after `--loop`.
