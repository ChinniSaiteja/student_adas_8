#!/usr/bin/env python3
"""
HSHL ADAS Student Lab — Bag Validator
======================================
Checks that a ROS 2 bag folder is complete and suitable for student home use.
Works without a ROS installation by reading the SQLite3 database directly.

Usage
-----
    # Auto-detect from student_adas/bags/
    python tests/validate_bag.py

    # Validate a specific folder
    python tests/validate_bag.py /path/to/bag_folder

    # Custom vehicle role name
    python tests/validate_bag.py /path/to/bag --role my_car

Exit codes
----------
    0  all required checks passed
    1  one or more required checks failed
"""

import argparse
import json
import os
import sqlite3
import struct
import sys
from pathlib import Path

try:
    import yaml
    _HAS_YAML = True
except ImportError:
    _HAS_YAML = False

try:
    import cv2
    import numpy as np
    _HAS_CV2 = True
except ImportError:
    _HAS_CV2 = False

# ─── Console colours (auto-disabled on non-TTY) ───────────────────────────────
_TTY    = sys.stdout.isatty()
_GREEN  = "\033[92m" if _TTY else ""
_RED    = "\033[91m" if _TTY else ""
_YELLOW = "\033[93m" if _TTY else ""
_WHITE  = "\033[97m" if _TTY else ""
_RESET  = "\033[0m"  if _TTY else ""
_BOLD   = "\033[1m"  if _TTY else ""

# ─── Validation thresholds ────────────────────────────────────────────────────
MIN_DURATION_SEC    = 10      # seconds
MIN_CAMERA_MSGS     = 50      # at least 50 camera frames
MIN_SPEED_MSGS      = 20      # at least 20 speed readings
CAMERA_SAMPLE_COUNT = 10      # frames to decode-test
SPEED_SAMPLE_COUNT  = 100     # speed messages to sanity-check

# ─── Expected topics (filled at runtime with actual role name) ────────────────
REQUIRED_TOPICS = {
    "{prefix}/camera/image/compressed": "sensor_msgs/msg/CompressedImage",
    "{prefix}/speed":                   "std_msgs/msg/Float32",
}
OPTIONAL_TOPICS = {
    "{prefix}/imu":           "sensor_msgs/msg/Imu",
    "{prefix}/gnss":          "sensor_msgs/msg/NavSatFix",
    "{prefix}/collision":     "std_msgs/msg/String",
    "{prefix}/lane_invasion": "std_msgs/msg/String",
}

# ─── Result tags ──────────────────────────────────────────────────────────────
PASS, FAIL, WARN, INFO = "PASS", "FAIL", "WARN", "INFO"

def _tag(result: str) -> str:
    colour = {PASS: _GREEN, FAIL: _RED, WARN: _YELLOW, INFO: _WHITE}.get(result, "")
    return f"{colour}[{result:<4}]{_RESET}"

def _row(result: str, msg: str):
    print(f"  {_tag(result)} {msg}")


# ─── CDR deserialisation helpers ──────────────────────────────────────────────
# ROS 2 bags store messages in CDR format.  The layout for simple types is:
#   bytes 0-3 : encapsulation header  (byte[1] == 0x01 → little-endian)
#   bytes 4.. : message payload

def _cdr_is_le(data: bytes) -> bool:
    return len(data) >= 2 and data[1] == 0x01


def decode_float32_cdr(data: bytes):
    """Return float from a CDR-serialised std_msgs/Float32, or None on error."""
    if len(data) < 8:
        return None
    try:
        fmt = "<f" if _cdr_is_le(data) else ">f"
        return struct.unpack_from(fmt, data, 4)[0]
    except struct.error:
        return None


def decode_jpeg_from_cdr(data: bytes):
    """
    Find the JPEG SOI marker (FF D8 FF) inside raw CDR bytes and decode it.
    Returns a numpy BGR frame, or None.
    """
    if not _HAS_CV2:
        return None
    idx = data.find(b"\xFF\xD8\xFF")
    if idx == -1:
        return None
    arr   = np.frombuffer(data[idx:], dtype=np.uint8)
    frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    return frame


def decode_json_from_cdr(data: bytes):
    """
    Extract the first JSON object from CDR-serialised std_msgs/String.
    The string payload is: 4-byte CDR header + 4-byte length + null-terminated UTF-8.
    """
    raw   = bytes(data)
    start = raw.find(b"{")
    if start == -1:
        return None
    chunk = raw[start:].split(b"\x00")[0]
    try:
        return json.loads(chunk.decode("utf-8"))
    except Exception:
        return None


# ─── Metadata helpers ─────────────────────────────────────────────────────────

def _read_metadata(bag_dir: Path) -> dict:
    path = bag_dir / "metadata.yaml"
    if not path.exists():
        return {}
    text = path.read_text(encoding="utf-8")
    if _HAS_YAML:
        return yaml.safe_load(text) or {}
    # Minimal fallback — grep for the two fields we care about
    result: dict = {}
    for line in text.splitlines():
        if "storage_identifier" in line:
            result["_storage"] = line.split(":", 1)[-1].strip()
        if "nanoseconds" in line and "duration" in line:
            try:
                result["_duration_ns"] = int(line.split(":", 1)[-1].strip())
            except ValueError:
                pass
    return result


def _duration_from_meta(meta: dict):
    info = meta.get("rosbag2_bagfile_information", meta)
    dur  = info.get("duration", {})
    if isinstance(dur, dict):
        return dur.get("nanoseconds", 0) / 1e9
    if "_duration_ns" in meta:
        return meta["_duration_ns"] / 1e9
    return None


def _storage_id(meta: dict) -> str:
    info = meta.get("rosbag2_bagfile_information", meta)
    return info.get("storage_identifier", meta.get("_storage", "?"))


# ─── SQLite3 helpers ──────────────────────────────────────────────────────────

def _find_db3(bag_dir: Path):
    for p in bag_dir.iterdir():
        if p.suffix == ".db3":
            return p
    return None


def _load_topic_stats(conn) -> dict:
    """Return {topic_name: {id, type, count, duration}} from the bag database."""
    stats = {}
    for row_id, name, tp in conn.execute("SELECT id, name, type FROM topics"):
        count = conn.execute(
            "SELECT COUNT(*) FROM messages WHERE topic_id=?", (row_id,)
        ).fetchone()[0]
        ts = conn.execute(
            "SELECT MIN(timestamp), MAX(timestamp) FROM messages WHERE topic_id=?",
            (row_id,),
        ).fetchone()
        min_ts, max_ts = ts if ts else (None, None)
        dur = ((max_ts - min_ts) / 1e9
               if (min_ts and max_ts and max_ts != min_ts) else 0.0)
        stats[name] = {"id": row_id, "type": tp, "count": count, "duration": dur}
    return stats


# ─── Main validation ──────────────────────────────────────────────────────────

def validate_bag(bag_path: Path, role: str = "hero") -> bool:
    """
    Validate *bag_path* and print a human-readable report.
    Returns True if all required checks pass, False otherwise.
    """
    prefix   = f"/carla/{role}"
    required = {k.format(prefix=prefix): v for k, v in REQUIRED_TOPICS.items()}
    optional = {k.format(prefix=prefix): v for k, v in OPTIONAL_TOPICS.items()}

    failures = 0
    W = 56

    print(f"\n{_BOLD}{'='*W}{_RESET}")
    print(f"{_BOLD} HSHL ADAS Bag Validator{_RESET}")
    print(f"{_BOLD}{'='*W}{_RESET}")
    print(f"  Bag  : {bag_path}")
    print(f"  Role : {role}")
    print(f"{_BOLD}{'='*W}{_RESET}\n")

    # ── 1. File structure ─────────────────────────────────────────────────────
    print(f"{_BOLD}File structure:{_RESET}")

    if not bag_path.is_dir():
        _row(FAIL, f"Directory not found: {bag_path}")
        _print_summary(W, failures + 1)
        return False
    _row(PASS, "Bag directory found")

    has_meta = (bag_path / "metadata.yaml").exists()
    _row(PASS if has_meta else FAIL, "metadata.yaml present")
    if not has_meta:
        failures += 1

    meta    = _read_metadata(bag_path)
    storage = _storage_id(meta)
    _row(INFO, f"Storage format : {storage}")

    db3 = None
    if storage == "mcap":
        _row(WARN, "MCAP format detected — SQLite checks skipped")
        _row(WARN, "Run  ros2 bag info <bag>  inside Docker to inspect MCAP bags")
    else:
        db3 = _find_db3(bag_path)
        _row(PASS if db3 else FAIL,
             f"Database (.db3) : {db3.name if db3 else 'NOT FOUND'}")
        if not db3:
            failures += 1

    # ── 2. Duration ───────────────────────────────────────────────────────────
    print(f"\n{_BOLD}Duration:{_RESET}")
    duration = _duration_from_meta(meta)
    if duration is not None:
        ok = duration >= MIN_DURATION_SEC
        _row(PASS if ok else FAIL,
             f"{duration:.1f} s  (minimum: {MIN_DURATION_SEC} s)")
        if not ok:
            failures += 1
    else:
        _row(WARN, "Could not determine duration from metadata.yaml")

    if db3 is None:
        _print_summary(W, failures)
        return failures == 0

    with sqlite3.connect(db3) as conn:
        topic_stats = _load_topic_stats(conn)

        # ── 3. Required topics ────────────────────────────────────────────────
        print(f"\n{_BOLD}Required topics:{_RESET}")
        for topic, exp_type in required.items():
            min_count = MIN_CAMERA_MSGS if "camera" in topic else MIN_SPEED_MSGS

            if topic not in topic_stats:
                _row(FAIL, f"{topic}")
                _row(FAIL, f"  └─ MISSING (required)")
                failures += 1
                continue

            info  = topic_stats[topic]
            ok    = info["count"] >= min_count
            label = (f"{topic:<54}  {info['count']:>5} msgs  "
                     f"{info['duration']:.1f}s")

            actual = info["type"]
            if exp_type not in actual and actual not in exp_type:
                label += f"  {_YELLOW}(type: {actual}){_RESET}"

            _row(PASS if ok else FAIL, label)
            if not ok:
                _row(FAIL if info["count"] == 0 else WARN,
                     f"  └─ {info['count']} messages < required minimum {min_count}")
                if info["count"] == 0:
                    failures += 1

        # ── 4. Optional topics ────────────────────────────────────────────────
        print(f"\n{_BOLD}Optional topics:{_RESET}")
        for topic in optional:
            if topic in topic_stats:
                info = topic_stats[topic]
                _row(PASS, f"{topic:<54}  {info['count']:>5} msgs  "
                           f"{info['duration']:.1f}s")
            else:
                _row(WARN, f"{topic:<54}  not present (optional — ok)")

        # ── 5. Camera frame decode ────────────────────────────────────────────
        cam_topic = f"{prefix}/camera/image/compressed"
        if cam_topic in topic_stats:
            print(f"\n{_BOLD}Camera frame sampling:{_RESET}")
            if not _HAS_CV2:
                _row(WARN, "cv2 unavailable — skipping frame decode check")
            else:
                cam_id = topic_stats[cam_topic]["id"]
                total  = topic_stats[cam_topic]["count"]
                step   = max(1, total // CAMERA_SAMPLE_COUNT)

                rows = conn.execute(
                    "SELECT data FROM messages WHERE topic_id=? ORDER BY timestamp",
                    (cam_id,)
                ).fetchall()
                samples = rows[::step][:CAMERA_SAMPLE_COUNT]

                decoded    = 0
                shape_set  = set()
                for (raw,) in samples:
                    frame = decode_jpeg_from_cdr(bytes(raw))
                    if frame is not None and frame.size > 0:
                        decoded += 1
                        shape_set.add(frame.shape[:2])  # (H, W)

                shapes = ", ".join(f"{h}×{w}" for h, w in sorted(shape_set)) or "n/a"
                status = PASS if decoded == len(samples) else (
                    WARN if decoded > 0 else FAIL
                )
                _row(status,
                     f"Decoded {decoded}/{len(samples)} sampled frames  "
                     f"(resolution: {shapes})")
                if decoded == 0:
                    failures += 1

        # ── 6. Speed value sanity ─────────────────────────────────────────────
        spd_topic = f"{prefix}/speed"
        if spd_topic in topic_stats:
            print(f"\n{_BOLD}Speed value sanity:{_RESET}")
            spd_id = topic_stats[spd_topic]["id"]
            rows   = conn.execute(
                "SELECT data FROM messages WHERE topic_id=? "
                "ORDER BY RANDOM() LIMIT ?",
                (spd_id, SPEED_SAMPLE_COUNT)
            ).fetchall()
            values = [
                v for (raw,) in rows
                if (v := decode_float32_cdr(bytes(raw))) is not None
            ]
            if values:
                lo, hi = min(values), max(values)
                range_ok = -1.0 <= lo and hi <= 300.0
                _row(PASS if range_ok else WARN,
                     f"Speed range {lo:.1f} – {hi:.1f} km/h  "
                     f"({len(values)} samples)")
                if not range_ok:
                    _row(WARN, "Speed values are outside the normal range — "
                               "verify the bag")
            else:
                _row(WARN, "Could not decode speed values from CDR")

        # ── 7. Event message JSON integrity ───────────────────────────────────
        for topic_key, label in [
            (f"{prefix}/collision",     "Collision events"),
            (f"{prefix}/lane_invasion", "Lane-invasion events"),
        ]:
            info = topic_stats.get(topic_key)
            if not info or info["count"] == 0:
                continue
            print(f"\n{_BOLD}{label}:{_RESET}")
            rows = conn.execute(
                "SELECT data FROM messages WHERE topic_id=? LIMIT 5",
                (info["id"],)
            ).fetchall()
            ok_n = sum(
                1 for (raw,) in rows
                if decode_json_from_cdr(bytes(raw)) is not None
            )
            _row(PASS if ok_n == len(rows) else WARN,
                 f"{info['count']} total  |  {ok_n}/{len(rows)} sampled "
                 f"messages parse as valid JSON")

    _print_summary(W, failures)
    return failures == 0


def _print_summary(width: int, failures: int):
    print(f"\n{_BOLD}{'='*width}{_RESET}")
    if failures == 0:
        print(f"{_GREEN}{_BOLD} Result: PASS — bag is ready for student use{_RESET}")
    else:
        print(f"{_RED}{_BOLD} Result: FAIL — {failures} check(s) failed{_RESET}")
        print(f"  Fix the errors above before distributing this bag.")
    print(f"{_BOLD}{'='*width}{_RESET}\n")


# ─── Auto-detect helper ───────────────────────────────────────────────────────

def _find_default_bag():
    """Look for the first bag in student_adas/bags/."""
    bags_dir = Path(__file__).parent.parent / "bags"
    if not bags_dir.is_dir():
        return None
    for p in bags_dir.iterdir():
        if p.is_dir() and (p / "metadata.yaml").exists():
            return p
    return None


# ─── pytest integration ───────────────────────────────────────────────────────
# When run via pytest (test_validate_bag.py imports this module), these
# functions expose individual checks as separate test functions.

def _get_default_bag_for_test():
    bag = _find_default_bag()
    if bag is None:
        import pytest
        pytest.skip("No bag found in bags/ — place one there to run bag tests")
    return bag


def pytest_bag_structure():
    """pytest: bag directory and metadata.yaml exist."""
    bag = _get_default_bag_for_test()
    assert bag.is_dir(), f"Bag dir missing: {bag}"
    assert (bag / "metadata.yaml").exists(), "metadata.yaml missing"


def pytest_bag_has_db3():
    """pytest: .db3 database file is present."""
    bag  = _get_default_bag_for_test()
    meta = _read_metadata(bag)
    if _storage_id(meta) == "mcap":
        import pytest; pytest.skip("MCAP bag — db3 check skipped")
    assert _find_db3(bag) is not None, "No .db3 file found in bag"


def pytest_bag_duration():
    """pytest: bag is at least MIN_DURATION_SEC long."""
    bag      = _get_default_bag_for_test()
    meta     = _read_metadata(bag)
    duration = _duration_from_meta(meta)
    assert duration is not None, "Cannot read duration from metadata.yaml"
    assert duration >= MIN_DURATION_SEC, (
        f"Bag too short: {duration:.1f}s < {MIN_DURATION_SEC}s"
    )


def pytest_bag_required_topics():
    """pytest: all required topics are present with enough messages."""
    bag    = _get_default_bag_for_test()
    role   = os.getenv("ROLE_NAME", "hero")
    prefix = f"/carla/{role}"
    db3    = _find_db3(bag)
    if db3 is None:
        import pytest; pytest.skip("No .db3 found")

    with sqlite3.connect(db3) as conn:
        stats = _load_topic_stats(conn)

    missing  = []
    too_few  = []
    for tmpl, _ in REQUIRED_TOPICS.items():
        topic = tmpl.format(prefix=prefix)
        min_n = MIN_CAMERA_MSGS if "camera" in topic else MIN_SPEED_MSGS
        if topic not in stats:
            missing.append(topic)
        elif stats[topic]["count"] < min_n:
            too_few.append(f"{topic} ({stats[topic]['count']} < {min_n})")

    assert not missing,  f"Missing required topics: {missing}"
    assert not too_few,  f"Too few messages: {too_few}"


def pytest_bag_camera_decodable():
    """pytest: sampled camera frames decode as valid JPEG/BGR images."""
    if not _HAS_CV2:
        import pytest; pytest.skip("cv2 not available")

    bag    = _get_default_bag_for_test()
    role   = os.getenv("ROLE_NAME", "hero")
    db3    = _find_db3(bag)
    if db3 is None:
        import pytest; pytest.skip("No .db3 found")

    cam_topic = f"/carla/{role}/camera/image/compressed"
    with sqlite3.connect(db3) as conn:
        stats = _load_topic_stats(conn)
        assert cam_topic in stats, f"Camera topic not found: {cam_topic}"

        cam_id = stats[cam_topic]["id"]
        total  = stats[cam_topic]["count"]
        step   = max(1, total // CAMERA_SAMPLE_COUNT)
        rows   = conn.execute(
            "SELECT data FROM messages WHERE topic_id=? ORDER BY timestamp",
            (cam_id,)
        ).fetchall()

    samples = rows[::step][:CAMERA_SAMPLE_COUNT]
    decoded = sum(
        1 for (raw,) in samples
        if (f := decode_jpeg_from_cdr(bytes(raw))) is not None and f.size > 0
    )
    assert decoded > 0, "No camera frames could be decoded"
    assert decoded == len(samples), (
        f"Only {decoded}/{len(samples)} sampled frames decoded successfully"
    )


def pytest_bag_speed_values():
    """pytest: sampled speed values are in the physically plausible range."""
    bag  = _get_default_bag_for_test()
    role = os.getenv("ROLE_NAME", "hero")
    db3  = _find_db3(bag)
    if db3 is None:
        import pytest; pytest.skip("No .db3 found")

    spd_topic = f"/carla/{role}/speed"
    with sqlite3.connect(db3) as conn:
        stats = _load_topic_stats(conn)
        assert spd_topic in stats, f"Speed topic not found: {spd_topic}"

        spd_id = stats[spd_topic]["id"]
        rows   = conn.execute(
            "SELECT data FROM messages WHERE topic_id=? "
            "ORDER BY RANDOM() LIMIT ?",
            (spd_id, SPEED_SAMPLE_COUNT)
        ).fetchall()

    values = [
        v for (raw,) in rows
        if (v := decode_float32_cdr(bytes(raw))) is not None
    ]
    assert values, "Could not decode any speed values from CDR"
    lo, hi = min(values), max(values)
    assert -1.0 <= lo, f"Speed below -1 km/h: {lo}"
    assert hi <= 300.0, f"Speed above 300 km/h: {hi}"


def pytest_bag_event_json():
    """pytest: collision and lane-invasion messages contain valid JSON."""
    bag  = _get_default_bag_for_test()
    role = os.getenv("ROLE_NAME", "hero")
    db3  = _find_db3(bag)
    if db3 is None:
        import pytest; pytest.skip("No .db3 found")

    with sqlite3.connect(db3) as conn:
        stats = _load_topic_stats(conn)
        for topic_key in [
            f"/carla/{role}/collision",
            f"/carla/{role}/lane_invasion",
        ]:
            info = stats.get(topic_key)
            if not info or info["count"] == 0:
                continue
            rows = conn.execute(
                "SELECT data FROM messages WHERE topic_id=? LIMIT 5",
                (info["id"],)
            ).fetchall()
            ok_n = sum(
                1 for (raw,) in rows
                if decode_json_from_cdr(bytes(raw)) is not None
            )
            assert ok_n == len(rows), (
                f"{topic_key}: {ok_n}/{len(rows)} messages failed JSON parse"
            )


# ─── CLI entry point ──────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Validate a ROS 2 bag for HSHL ADAS student home use.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "bag_path",
        nargs="?",
        help="Bag folder path (default: auto-detect from bags/)",
    )
    parser.add_argument(
        "--role",
        default=os.getenv("ROLE_NAME", "hero"),
        help="CARLA vehicle role name (default: hero)",
    )
    args = parser.parse_args()

    if args.bag_path:
        bag = Path(args.bag_path)
    else:
        bag = _find_default_bag()
        if bag is None:
            print("No bag found.  Provide a path or place a bag in bags/.")
            print("Usage: python tests/validate_bag.py /path/to/bag_folder")
            sys.exit(1)

    sys.exit(0 if validate_bag(bag, role=args.role) else 1)


if __name__ == "__main__":
    main()
