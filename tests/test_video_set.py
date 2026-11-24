import tempfile
import unittest
from pathlib import Path

from tacticalai.video_set import LEGACY_CLIPS, resolve_clips


class VideoSetTests(unittest.TestCase):
    def test_explicit_single_clip_preserves_name_and_extension(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            video = root / "80.mp4"
            video.touch()
            self.assertEqual(resolve_clips(root, ["80"]), (("80", video),))

    def test_default_is_legacy_pair(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name in LEGACY_CLIPS:
                (root / f"{name}.mp4").touch()
            self.assertEqual([name for name, _ in resolve_clips(root)], list(LEGACY_CLIPS))

    def test_rejects_paths_and_ambiguous_extensions(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with self.assertRaises(ValueError):
                resolve_clips(root, ["folder/80"])
            (root / "80.mp4").touch()
            (root / "80.mov").touch()
            with self.assertRaises(FileNotFoundError):
                resolve_clips(root, ["80"])


if __name__ == "__main__":
    unittest.main()
