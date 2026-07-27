"""
Calibration profile registry: isolates lens calibration, physical scale,
usable ROI (crop), detection thresholds, and background reference *per
camera* -- so switching between cameras/sources never reads or
overwrites another one's calibration. This now covers not just distinct
source types (USB vs VDO.Ninja) but distinct *physical USB cameras* too
(e.g. the monochrome U20CAM and a second color/fisheye ELP camera) --
see the "active-profile keys" note below.

This module owns no calibration MATH at all. distortion.py/scale.py/
circle_config.py/background_reference.py (and fisheye_calibration.py)
are untouched -- each already accepts an optional `path=` override for
its save/load functions. A profile is just a named, identity-tagged
bundle of *paths* into a per-profile subfolder (profiles/<id>/), plus a
small amount of its own metadata persisted in one small registry file.

Storage layout:
    calibration_profiles.json          -- the registry (this module's own file)
    profiles/<id>/calibration_data.json    -- distortion.py's or fisheye_calibration.py's own format, untouched
    profiles/<id>/scale_calibration.json   -- scale.py's own format, untouched
    profiles/<id>/circle_config.json       -- circle_config.py's own format, untouched
    profiles/<id>/background_reference.npy -- background_reference.py's own format, untouched

Active-profile keys: VDO.Ninja uses the plain key "vdo_ninja". USB
cameras use "usb:<device_path>" (or "usb:name:<friendly name>" for the
rare device with no DevicePath -- see camera_identity.py) since OpenCV's
index alone can't tell two physical USB cameras apart reliably, and a
single "usb" key can't either once there's more than one. profile_key()
computes the right key for a given profile dict.

Schema backfill: profiles created before camera_role/calibration_model/
detector_type/etc. existed (i.e. the real profile already on disk from
before this) are transparently backfilled with safe, inferred defaults
on every load (see _backfill_profile_schema) -- never written back
automatically, just corrected in memory every time, so an old file on
disk is never in a "broken until manually migrated" state.
"""

from __future__ import annotations

import datetime
import json
import shutil
import uuid
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).parent
DEFAULT_REGISTRY_PATH = PROJECT_ROOT / "calibration_profiles.json"
PROFILES_DIR = PROJECT_ROOT / "profiles"

# Pre-profile-system flat files -- read only by ensure_default_usb_profile()
# for one-time migration, never written to again afterward.
_LEGACY_DISTORTION_PATH = PROJECT_ROOT / "calibration_data.json"
_LEGACY_SCALE_PATH = PROJECT_ROOT / "scale_calibration.json"
_LEGACY_CIRCLE_CONFIG_PATH = PROJECT_ROOT / "circle_config.json"
_LEGACY_BACKGROUND_REFERENCE_PATH = PROJECT_ROOT / "background_reference.npy"
_LEGACY_APP_CONFIG_PATH = PROJECT_ROOT / "app_config.json"

SOURCE_TYPES = ("usb", "vdo_ninja")
CAMERA_ROLES = ("monochrome_ftir", "color_ftir")


def _empty_registry() -> dict[str, Any]:
    return {"profiles": [], "active_by_source": {}}


def _backfill_profile_schema(profile: dict) -> dict:
    """
    Fills in fields that didn't exist in earlier versions of this
    registry with safe, inferred defaults -- applied in memory on every
    load (see _load_registry), never as a one-time destructive rewrite.
    A profile already containing these fields is returned unchanged.
    """
    profile.setdefault("device_path", None)
    profile.setdefault("device_name", profile.get("name"))
    profile.setdefault("fourcc", "MJPG" if profile.get("source_type") == "usb" else None)
    # Requested connect resolution/fps -- None means "use the role's
    # ROLE_CONNECT_PARAMS default" (live_camera_tab.py). Distinct from
    # `resolution` below, which reflects whatever frame size a
    # *calibration* was actually run at -- these two can legitimately
    # differ if the connect format changes after calibrating.
    profile.setdefault("connect_width", None)
    profile.setdefault("connect_height", None)
    profile.setdefault("connect_fps", None)
    profile.setdefault("calibration_model", "pinhole")
    profile.setdefault(
        "detector_type",
        "color" if profile.get("source_type") == "vdo_ninja" else "grayscale",
    )
    profile.setdefault("profile_version", 1)

    if "camera_role" not in profile:
        # Only the one pre-role USB migration profile (name == "USB
        # Camera", created by ensure_default_usb_profile() before camera
        # roles existed) can be confidently inferred as the monochrome
        # U20CAM -- anything else is left unset rather than guessed.
        profile["camera_role"] = (
            "monochrome_ftir"
            if profile.get("source_type") == "usb" and profile.get("name") == "USB Camera"
            else None
        )

    if "alias" not in profile:
        profile["alias"] = (
            "Monochrome FTIR Camera" if profile["camera_role"] == "monochrome_ftir" else profile.get("name", "")
        )

    return profile


def profile_key(profile: dict) -> str:
    """The active-profile lookup key for a given profile dict -- see the
    module docstring's "Active-profile keys" note."""
    if profile["source_type"] == "usb":
        identity = profile.get("device_path") or f"name:{profile.get('device_name') or profile['name']}"
        return f"usb:{identity}"

    return profile["source_type"]


def _load_registry(path: Path = DEFAULT_REGISTRY_PATH) -> dict[str, Any]:
    path = Path(path)

    if not path.exists():
        return _empty_registry()

    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except (json.JSONDecodeError, OSError):
        return _empty_registry()

    data.setdefault("profiles", [])
    data.setdefault("active_by_source", {})
    data["profiles"] = [_backfill_profile_schema(p) for p in data["profiles"]]

    # Before multiple physical USB cameras existed, active_by_source used
    # the bare key "usb" (one USB camera, one active profile, no identity
    # needed to tell them apart). That flat key doesn't match any
    # profile_key()-shaped lookup now, so a registry saved under the old
    # scheme would look like it has no active USB profile at all. Remaps
    # it to whatever key that profile's now-backfilled identity resolves
    # to -- in memory only, same as the rest of this backfill.
    if "usb" in data["active_by_source"]:
        old_profile_id = data["active_by_source"].pop("usb")

        if old_profile_id is not None:
            matching_profile = next((p for p in data["profiles"] if p["id"] == old_profile_id), None)

            if matching_profile is not None:
                data["active_by_source"][profile_key(matching_profile)] = old_profile_id

    return data


def _save_registry(data: dict[str, Any], path: Path = DEFAULT_REGISTRY_PATH) -> None:
    with open(Path(path), "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def list_profiles(source_type: str | None = None, path: Path = DEFAULT_REGISTRY_PATH) -> list[dict]:
    registry = _load_registry(path)

    if source_type is None:
        return list(registry["profiles"])

    return [p for p in registry["profiles"] if p["source_type"] == source_type]


def get_profile(profile_id: str, path: Path = DEFAULT_REGISTRY_PATH) -> dict | None:
    registry = _load_registry(path)

    for profile in registry["profiles"]:
        if profile["id"] == profile_id:
            return profile

    return None


def create_profile(
    source_type: str,
    name: str,
    resolution: tuple[int, int] | None = None,
    orientation: str = "normal",
    alias: str | None = None,
    camera_role: str | None = None,
    device_path: str | None = None,
    device_name: str | None = None,
    fourcc: str | None = None,
    connect_width: int | None = None,
    connect_height: int | None = None,
    connect_fps: float | None = None,
    calibration_model: str = "pinhole",
    detector_type: str | None = None,
    path: Path = DEFAULT_REGISTRY_PATH,
) -> str:
    """
    Creates a new profile, makes it the active one for its resolved key
    (profile_key() -- an explicit "New Profile"/first-connect action
    means you intend to use it right away), and returns its id.

    detector_type defaults from source_type when not given ("color" for
    vdo_ninja, "grayscale" for usb) -- override for a color USB camera
    (e.g. camera_role="color_ftir").
    """
    if source_type not in SOURCE_TYPES:
        raise ValueError(f"Unknown source_type {source_type!r}, expected one of {SOURCE_TYPES}")

    if camera_role is not None and camera_role not in CAMERA_ROLES:
        raise ValueError(f"Unknown camera_role {camera_role!r}, expected one of {CAMERA_ROLES}")

    if detector_type is None:
        detector_type = "color" if source_type == "vdo_ninja" else "grayscale"

    registry = _load_registry(path)

    profile_id = f"{source_type}_{uuid.uuid4().hex[:10]}"
    profile = {
        "id": profile_id,
        "source_type": source_type,
        "name": name,
        "alias": alias if alias is not None else name,
        "camera_role": camera_role,
        "device_path": device_path,
        "device_name": device_name if device_name is not None else name,
        "fourcc": fourcc,
        "connect_width": connect_width,
        "connect_height": connect_height,
        "connect_fps": connect_fps,
        "calibration_model": calibration_model,
        "detector_type": detector_type,
        "profile_version": 1,
        "resolution": list(resolution) if resolution is not None else None,
        "orientation": orientation,
        "crop_percentages": {"top": 0, "bottom": 0, "left": 0, "right": 0},
        "created_at": datetime.datetime.now().isoformat(timespec="seconds"),
    }

    registry["profiles"].append(profile)
    registry["active_by_source"][profile_key(profile)] = profile_id
    _save_registry(registry, path)

    profile_dir(profile_id).mkdir(parents=True, exist_ok=True)

    return profile_id


def delete_profile(profile_id: str, path: Path = DEFAULT_REGISTRY_PATH) -> None:
    """
    Removes a profile, its active-by-source pointer (if it was the active
    one for its key), and its on-disk profiles/<id>/ folder -- e.g. to
    correct a profile mistakenly created against the wrong physical
    device. Raises ValueError if the id doesn't exist.
    """
    registry = _load_registry(path)
    matching = next((p for p in registry["profiles"] if p["id"] == profile_id), None)

    if matching is None:
        raise ValueError(f"No such profile: {profile_id!r}")

    registry["profiles"] = [p for p in registry["profiles"] if p["id"] != profile_id]

    key = profile_key(matching)
    if registry["active_by_source"].get(key) == profile_id:
        del registry["active_by_source"][key]

    _save_registry(registry, path)

    directory = PROFILES_DIR / profile_id
    if directory.exists():
        shutil.rmtree(directory)


def set_active_profile(key: str, profile_id: str | None, path: Path = DEFAULT_REGISTRY_PATH) -> None:
    """
    `key` is a profile_key()-shaped string ("vdo_ninja", or
    "usb:<device_path>") -- see the module docstring's "Active-profile
    keys" note. Kept as a plain string parameter (not re-deriving it from
    profile_id) since callers set the active profile for a *key* they
    already know (e.g. the identity just connected to), independent of
    which specific profile ends up assigned to it.
    """
    registry = _load_registry(path)

    if profile_id is not None and get_profile(profile_id, path) is None:
        raise ValueError(f"No such profile: {profile_id!r}")

    registry["active_by_source"][key] = profile_id
    _save_registry(registry, path)


def get_active_profile(key: str, path: Path = DEFAULT_REGISTRY_PATH) -> dict | None:
    registry = _load_registry(path)
    active_id = registry["active_by_source"].get(key)

    if active_id is None:
        return None

    return get_profile(active_id, path)


def update_profile_crop(profile_id: str, crop_percentages: dict, path: Path = DEFAULT_REGISTRY_PATH) -> None:
    registry = _load_registry(path)

    for profile in registry["profiles"]:
        if profile["id"] == profile_id:
            profile["crop_percentages"] = dict(crop_percentages)
            _save_registry(registry, path)
            return

    raise ValueError(f"No such profile: {profile_id!r}")


def update_profile_connect_format(
    profile_id: str,
    width: int,
    height: int,
    fps: float,
    fourcc: str,
    path: Path = DEFAULT_REGISTRY_PATH,
) -> None:
    """
    Remembers the actual (width, height, fps, fourcc) requested at the
    last successful connect, so re-opening this camera later reselects
    the same format instead of always falling back to its role's
    ROLE_CONNECT_PARAMS default -- lets a user's format choice (e.g. the
    ELP at a resolution other than its role default) stick across
    reconnects/restarts.
    """
    registry = _load_registry(path)

    for profile in registry["profiles"]:
        if profile["id"] == profile_id:
            profile["connect_width"] = width
            profile["connect_height"] = height
            profile["connect_fps"] = fps
            profile["fourcc"] = fourcc
            _save_registry(registry, path)
            return

    raise ValueError(f"No such profile: {profile_id!r}")


# ---- USB camera identity lookups ----


def find_profile_by_device_path(device_path: str, path: Path = DEFAULT_REGISTRY_PATH) -> dict | None:
    for profile in list_profiles(source_type="usb", path=path):
        if profile.get("device_path") == device_path:
            return profile

    return None


def find_profile_by_name(name: str, path: Path = DEFAULT_REGISTRY_PATH) -> dict | None:
    """The documented fallback when a device exposes no DevicePath (see
    camera_identity.py) -- two identical unnamed devices can't be told
    apart this way, a known limitation of the fallback tier."""
    for profile in list_profiles(source_type="usb", path=path):
        if profile.get("device_name") == name or profile.get("name") == name:
            return profile

    return None


def update_profile_device_path(profile_id: str, device_path: str, path: Path = DEFAULT_REGISTRY_PATH) -> None:
    """
    Self-healing: a profile that was originally matched (or migrated)
    without a device_path -- found by name instead, or backfilled from
    before device_path existed -- gets it filled in the first time it's
    actually resolved during a real connect, so future lookups are the
    more reliable device_path match instead of the name fallback. If the
    profile was active under its old (name-based) key, the active
    pointer moves to the new key too, so it isn't orphaned.
    """
    registry = _load_registry(path)

    for profile in registry["profiles"]:
        if profile["id"] == profile_id:
            old_key = profile_key(profile)
            profile["device_path"] = device_path
            new_key = profile_key(profile)

            if old_key != new_key and registry["active_by_source"].get(old_key) == profile_id:
                registry["active_by_source"][new_key] = profile_id
                del registry["active_by_source"][old_key]

            _save_registry(registry, path)
            return

    raise ValueError(f"No such profile: {profile_id!r}")


# ---- per-profile file paths (the whole point of this module) ----


def profile_dir(profile_id: str) -> Path:
    directory = PROFILES_DIR / profile_id
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def distortion_path(profile_id: str) -> Path:
    return profile_dir(profile_id) / "calibration_data.json"


def scale_path(profile_id: str) -> Path:
    return profile_dir(profile_id) / "scale_calibration.json"


def circle_config_path(profile_id: str) -> Path:
    return profile_dir(profile_id) / "circle_config.json"


def background_reference_path(profile_id: str) -> Path:
    return profile_dir(profile_id) / "background_reference.npy"


# ---- one-time migration of the pre-profile-system flat files ----


def ensure_default_usb_profile(path: Path = DEFAULT_REGISTRY_PATH) -> str | None:
    """
    If the profile registry doesn't exist yet, this is the first run
    since profiles were introduced. If any of the old flat calibration
    files exist from before, migrate them into a new "USB Camera"
    profile so nothing already captured is lost -- copies the files
    (never deletes/moves the originals, which are simply no longer read
    directly afterward). Returns the new profile's id, or None if there
    was nothing to migrate (fresh install) or the registry already
    exists (already migrated or already in use).
    """
    path = Path(path)

    if path.exists():
        return None  # already initialized -- never re-migrate over real usage

    legacy_files = {
        "distortion": _LEGACY_DISTORTION_PATH,
        "scale": _LEGACY_SCALE_PATH,
        "circle_config": _LEGACY_CIRCLE_CONFIG_PATH,
        "background_reference": _LEGACY_BACKGROUND_REFERENCE_PATH,
    }
    present = {key: p for key, p in legacy_files.items() if p.exists()}

    if not present:
        return None  # fresh install, nothing to migrate -- caller creates profiles as needed

    resolution = None
    if _LEGACY_DISTORTION_PATH.exists():
        try:
            with open(_LEGACY_DISTORTION_PATH, "r", encoding="utf-8") as f:
                resolution = tuple(json.load(f)["image_size"])
        except (json.JSONDecodeError, OSError, KeyError):
            resolution = None

    # Explicit, not relying on _backfill_profile_schema()'s inference --
    # that inference exists as a safety net for a registry that already
    # existed before these fields did (the real case this shipped
    # against), not as the primary path for a brand-new migration.
    profile_id = create_profile(
        "usb",
        "USB Camera",
        resolution=resolution,
        alias="Monochrome FTIR Camera",
        camera_role="monochrome_ftir",
        fourcc="MJPG",
        calibration_model="pinhole",
        detector_type="grayscale",
        path=path,
    )

    destinations = {
        "distortion": distortion_path(profile_id),
        "scale": scale_path(profile_id),
        "circle_config": circle_config_path(profile_id),
        "background_reference": background_reference_path(profile_id),
    }
    for key, src in present.items():
        shutil.copyfile(src, destinations[key])

    if _LEGACY_APP_CONFIG_PATH.exists():
        try:
            with open(_LEGACY_APP_CONFIG_PATH, "r", encoding="utf-8") as f:
                legacy_config = json.load(f)
            crop = {
                "top": int(legacy_config.get("undistort_crop_top_pct", 0)),
                "bottom": int(legacy_config.get("undistort_crop_bottom_pct", 0)),
                "left": int(legacy_config.get("undistort_crop_left_pct", 0)),
                "right": int(legacy_config.get("undistort_crop_right_pct", 0)),
            }
            update_profile_crop(profile_id, crop, path)
        except (json.JSONDecodeError, OSError, KeyError):
            pass

    return profile_id
