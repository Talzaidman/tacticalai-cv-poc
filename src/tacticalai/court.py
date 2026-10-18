"""Explicit FIBA-style floor coordinates, shared by manual and future auto anchors.

Origin: left baseline / far sideline on the diagram. X runs along the court,
Y across it. This orientation is a convention, not a team/attack direction.
Line-centre approximations ignore the 5 cm paint thickness in this POC.
"""
import numpy as np


def landmarks(court):
    length, width = court["length_m"], court["width_m"]
    depth, lane = court["key_depth_m"], court["key_width_m"]
    low, high = (width - lane) / 2, (width + lane) / 2
    corner = court.get("three_point_corner_offset_m", .9)
    points = {"centre": [length / 2, width / 2],
              "half_far": [length / 2, 0], "half_near": [length / 2, width]}
    for end, base, free in [("left", 0, depth), ("right", length, length-depth)]:
        points.update({f"{end}_corner_far": [base, 0],
                       f"{end}_corner_near": [base, width],
                       f"{end}_key_base_far": [base, low],
                       f"{end}_key_base_near": [base, high],
                       f"{end}_key_free_far": [free, low],
                       f"{end}_key_free_near": [free, high],
                       f"{end}_three_base_far": [base, corner],
                       f"{end}_three_base_near": [base, width-corner],
                       f"{end}_free_mid": [free, width / 2]})
    return points


def floor_lines(court):
    """Polylines for diagnostic overlay; no elevated basket/backboard geometry."""
    length, width = court["length_m"], court["width_m"]
    depth, lane = court["key_depth_m"], court["key_width_m"]
    low, high = (width-lane)/2, (width+lane)/2
    lines = [[[0, 0], [length, 0], [length, width], [0, width], [0, 0]],
             [[length/2, 0], [length/2, width]]]
    theta = np.linspace(0, 2*np.pi, 121)
    for x in (length/2, depth, length-depth):
        lines.append(np.column_stack((x + 1.8*np.cos(theta), width/2 + 1.8*np.sin(theta))))
    for base, free in ((0, depth), (length, length-depth)):
        lines.append([[base, low], [free, low], [free, high], [base, high]])
    corner = court.get("three_point_corner_offset_m", .9)
    radius, basket_offset = 6.75, 1.575
    angle = np.arcsin((width/2-corner)/radius)
    theta = np.linspace(-angle, angle, 121)
    arc = np.column_stack((basket_offset+radius*np.cos(theta), width/2+radius*np.sin(theta)))
    three = np.vstack(([0, corner], arc, [0, width-corner]))
    lines.extend((three, three*np.array([-1, 1])+np.array([length, 0])))
    return [np.asarray(line, dtype=np.float64) for line in lines]
