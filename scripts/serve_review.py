"""Serve local review files with byte ranges so MP4 frame seeking works."""

from __future__ import annotations

import argparse
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import re


class ReviewHandler(SimpleHTTPRequestHandler):
    def send_head(self):
        self.remaining = None
        requested = self.headers.get("Range")
        path = Path(self.translate_path(self.path))
        if not requested or not path.is_file():
            return super().send_head()
        match = re.fullmatch(r"bytes=(\d*)-(\d*)", requested)
        if not match or not any(match.groups()):
            self.send_error(400, "Unsupported byte range")
            return None
        try:
            stream = path.open("rb")
        except OSError:
            self.send_error(404, "File not found")
            return None
        size = path.stat().st_size
        first, last = match.groups()
        start = int(first) if first else max(0, size - int(last))
        end = min(int(last), size - 1) if first and last else size - 1
        if start >= size or start > end:
            stream.close()
            self.send_response(416)
            self.send_header("Content-Range", f"bytes */{size}")
            self.send_header("Content-Length", "0")
            self.end_headers()
            return None
        self.remaining = end - start + 1
        stream.seek(start)
        self.send_response(206)
        self.send_header("Content-Type", self.guess_type(str(path)))
        self.send_header("Content-Length", str(self.remaining))
        self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
        self.send_header("Accept-Ranges", "bytes")
        self.end_headers()
        return stream

    def end_headers(self):
        self.send_header("Cache-Control", "no-cache")
        super().end_headers()

    def copyfile(self, source, outputfile):
        if self.remaining is None:
            return super().copyfile(source, outputfile)
        try:
            while self.remaining:
                chunk = source.read(min(self.remaining, 128 * 1024))
                if not chunk:
                    break
                outputfile.write(chunk)
                self.remaining -= len(chunk)
        except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
            pass  # Browsers cancel requests when seeking to another frame.


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--port", type=int, default=8886)
    args = parser.parse_args()
    directory = args.directory.resolve(strict=True)
    server = ThreadingHTTPServer(
        ("127.0.0.1", args.port), partial(ReviewHandler, directory=str(directory))
    )
    print(f"Review: http://127.0.0.1:{args.port}/", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
