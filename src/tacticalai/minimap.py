"""Top-left court inset for frame-synchronous playback of observed tracks."""
import cv2
import numpy as np

from .calibration import transform
from .court import floor_lines


def apply_roles(rows, annotations, frame_idx):
    """Optional manual, run-specific role intervals; never repair track identity."""
    result = []
    for row in rows:
        row = dict(row)
        matches = [entry for entry in annotations.get("intervals", [])
                   if entry["track_id"] == row["track_id"]
                   and entry["start_frame"] <= frame_idx <= entry["end_frame"]]
        if len(matches) > 1:
            raise ValueError("Overlapping manual role intervals")
        row["role"] = matches[0]["role"] if matches else row.get("role", "unknown")
        row["role_source"] = "manual_visual_review" if matches else "unknown"
        result.append(row)
    return result


def track_color(track_id):
    # Display colors encode run-local track IDs, not teams or physical identities.
    hsv = np.uint8([[[int(track_id * 67) % 180, 160, 255]]])
    return tuple(int(v) for v in cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0, 0])


def text(image, value, point, scale=.45, color=(245, 245, 245)):
    cv2.putText(image, value, point, cv2.FONT_HERSHEY_SIMPLEX, scale, (8, 10, 10), 3, cv2.LINE_AA)
    cv2.putText(image, value, point, cv2.FONT_HERSHEY_SIMPLEX, scale, color, 1, cv2.LINE_AA)


def compose(frame, rows, court, quality, timestamp_s, players_only=False, flip_x=False):
    """720p playback layout. Missing measurements disappear, never freeze in place."""
    height = 720
    width = round(frame.shape[1] * height / frame.shape[0])
    width += width % 2
    output = cv2.resize(frame, (width, height))
    sx, sy = width/frame.shape[1], height/frame.shape[0]
    inset_width, inset_height = 370, 250
    inset = np.full((inset_height, inset_width, 3), (28, 38, 32), dtype=np.uint8)
    scale = min((inset_width-32)/court["length_m"], (inset_height-70)/court["width_m"])
    ox = (inset_width-scale*court["length_m"])/2
    matrix = np.array([[(-1 if flip_x else 1)*scale, 0,
                        inset_width-ox if flip_x else ox], [0, scale, 36], [0, 0, 1]], float)
    for line in floor_lines(court):
        pixels = np.rint(transform(line, matrix)).astype(np.int32)
        cv2.polylines(inset, [pixels], False, (170, 185, 175), 1, cv2.LINE_AA)
    shown = []
    for row in rows:
        if players_only and row["role"] != "player":
            continue
        if row["court_xy_m"] is None or row["court_region"] != "inside":
            continue
        color = track_color(row["track_id"])
        xy = tuple(np.rint(transform([row["court_xy_m"]], matrix)[0]).astype(int))
        cv2.circle(inset, xy, 5, (8, 12, 10), -1, cv2.LINE_AA)
        cv2.circle(inset, xy, 4, color, -1, cv2.LINE_AA)
        text(inset, str(row["track_id"]), (xy[0]+5, xy[1]-4), .36, color)
        foot = tuple(np.rint(np.array(row["footpoint_image_xy"])*[sx, sy]).astype(int))
        cv2.circle(output, foot, 5, color, 2, cv2.LINE_AA)
        text(output, str(row["track_id"]), (foot[0]+6, foot[1]-8), .5, color)
        shown.append(row["track_id"])
    text(inset, "COURT MAP | approximate", (12, 21), .47)
    text(inset, f"{len(shown)} observed {'players' if players_only else 'people'} | {timestamp_s:04.2f}s", (12, 225), .4)
    text(inset, "Colors = track IDs; missing tracks hidden", (12, 242), .35)
    if quality["method"] == "unavailable":
        text(inset, "Court calibration unavailable", (30, 115), .5, (0, 180, 255))
    cv2.rectangle(inset, (0, 0), (inset_width-1, inset_height-1), (130, 150, 138), 1)
    output[12:12+inset_height, 12:12+inset_width] = inset
    # Return the exact displayed population for auditing against exported tracks.
    return output, shown
