"""Resolve explicit tracker settings without silently accepting unused knobs."""
import math


def configure_tracker(defaults, config, tracker_name):
    settings = dict(defaults)
    overrides = config.get("tracker_overrides", {})
    if not isinstance(overrides, dict):
        raise ValueError("tracker_overrides must map tracker names to settings")
    if set(overrides) - {"botsort", "tracktrack", "deepocsort"}:
        raise ValueError("Unsupported tracker name in tracker_overrides")
    changes = overrides.get(tracker_name, {})
    if not isinstance(changes, dict):
        raise ValueError(f"tracker_overrides.{tracker_name} must be an object")
    managed = {"tracker_type", "with_reid", "model", "gmc_method", "track_buffer"}
    for key, value in changes.items():
        if key not in defaults or key in managed:
            raise ValueError(f"Unsupported or centrally managed {tracker_name} override: {key}")
        expected = type(defaults[key])
        if expected in {int, float}:
            if type(value) not in {int, float} or not math.isfinite(value) or value < 0:
                raise ValueError(f"{key} must be a finite nonnegative number")
            if expected is int and type(value) is not int:
                raise ValueError(f"{key} must be an integer")
        elif type(value) is not expected:
            raise ValueError(f"{key} must have type {expected.__name__}")
        settings[key] = value
    settings.update(with_reid=config.get("with_reid", False), gmc_method=config["gmc_method"])
    if "track_buffer" in config:
        settings["track_buffer"] = config["track_buffer"]
    if settings["with_reid"]:
        settings["model"] = config.get("reid_model", "yolo26n-cls.pt")
    return settings
