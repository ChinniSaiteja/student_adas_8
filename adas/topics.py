"""
ROS 2 topic names shared between the CARLA simulation and student ADAS systems.
All names must match the simulation side (ros_client/common/config.py).
"""
import os

ROLE_NAME = os.getenv("ROLE_NAME", "hero")
_PREFIX   = f"/carla/{ROLE_NAME}"

# ── Inputs: always available (subscribe) ──────────────────────────────────────
CAMERA_IMAGE  = f"{_PREFIX}/camera/image/compressed"  # sensor_msgs/CompressedImage  – BGR 1280×720 @ ~30 fps
VEHICLE_SPEED = f"{_PREFIX}/speed"                    # std_msgs/Float32             – km/h

# ── Inputs: optional sensors (subscribe if needed) ────────────────────────────
# Enable these via on_imu_update / on_gnss_update / on_collision / on_lane_invasion
# in your CarlaADASInterface subclass.  The simulation publishes all of them.
IMU           = f"{_PREFIX}/imu"            # sensor_msgs/Imu     – linear_acceleration, angular_velocity
GNSS          = f"{_PREFIX}/gnss"           # sensor_msgs/NavSatFix – latitude, longitude, altitude (m)
COLLISION     = f"{_PREFIX}/collision"      # std_msgs/String     – JSON  {"actor": str, "impulse": [x,y,z]}
LANE_INVASION = f"{_PREFIX}/lane_invasion"  # std_msgs/String     – JSON  {"lane_types": [...]}
UI_STATE      = f"{_PREFIX}/ui_state"       # std_msgs/String     – JSON runtime display state from Machine A

# NOTE: LiDAR is intentionally excluded from the student API.

# ── Outputs (publish) ─────────────────────────────────────────────────────────
CMD_VEL_EXT = f"{_PREFIX}/cmd_vel_ext"  # geometry_msgs/Twist  → throttle / brake / steer
HUD_EVENT   = f"{_PREFIX}/hud_event"   # std_msgs/String      → JSON HUD notification
