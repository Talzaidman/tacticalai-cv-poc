"""Video-set helpers shared by single- and multi-clip pipeline stages."""

from pathlib import Path


LEGACY_CLIPS = ("attack_clip", "attack_clip2")
VIDEO_SUFFIXES = (".mp4", ".mov", ".mkv", ".avi")


def resolve_clips(video_root: Path, clips=None):
    """Return ordered ``(clip_name, video_path)`` pairs without renaming inputs.

    Existing two-clip reproductions keep their historical default. New workflows
    pass ``--clips`` explicitly, which avoids accidentally including unrelated
    footage from the same directory.
    """
    video_root = Path(video_root)
    names = tuple(clips) if clips else LEGACY_CLIPS
    if not names:
        raise ValueError("At least one clip is required")
    if len(names) != len(set(names)):
        raise ValueError("Clip names must be unique")

    result = []
    for name in names:
        raw = Path(name)
        if raw.name != name or raw.suffix:
            raise ValueError(f"Clip must be a filename stem, not a path: {name}")
        matches = [video_root / f"{name}{suffix}" for suffix in VIDEO_SUFFIXES]
        matches = [path for path in matches if path.is_file()]
        if len(matches) != 1:
            raise FileNotFoundError(
                f"Expected exactly one video for clip {name!r} in {video_root}; "
                f"found {[str(path) for path in matches]}"
            )
        result.append((name, matches[0]))
    return tuple(result)
