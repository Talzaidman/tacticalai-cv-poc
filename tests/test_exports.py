import csv
import io
import tempfile
import unittest
from pathlib import Path

from tacticalai.export import csv_writer, records, sha256, write_rows


class Array:
    def __init__(self, values):
        self.values = values

    def tolist(self):
        return self.values


class Boxes:
    xyxy = Array([[1, 2, 30, 40]])
    conf = Array([0.8])
    cls = Array([0])
    id = Array([7])
    is_track = True


class ExportTests(unittest.TestCase):
    def test_coordinates_timestamp_ids_and_unknown_identity(self):
        row = records(Boxes(), 25, 25)[0]
        self.assertEqual(row["timestamp_s"], 1.0)
        self.assertEqual(row["bbox_xyxy"], [1, 2, 30, 40])
        self.assertEqual(row["track_id"], 7)
        self.assertIsNone(row["jersey_number"])
        self.assertIsNone(row["court_xy_m"])

    def test_detection_is_not_an_identity(self):
        boxes = Boxes()
        boxes.is_track, boxes.id = False, None
        rows = records(boxes, 0, 25)
        self.assertIsNone(rows[0]["track_id"])
        stream = io.StringIO()
        write_rows(csv_writer(stream), rows)
        data = list(csv.DictReader(io.StringIO(stream.getvalue())))[0]
        self.assertEqual(data["track_id"], "")
        self.assertEqual(data["x2"], "30.0")

    def test_empty_frames_and_source_hash(self):
        self.assertEqual(records(None, 0, 25), [])
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "input"
            path.write_bytes(b"abc")
            self.assertEqual(sha256(path), "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad")
