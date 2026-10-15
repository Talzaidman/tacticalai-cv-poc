"""CPU-only, synchronized H.264 comparisons of already-rendered tracking runs."""
from pathlib import Path
import subprocess


def comparison_grid(paths, labels, output, width=640):
    import cv2
    import numpy as np
    import imageio_ffmpeg

    if len(paths) != 4 or len(labels) != 4:
        raise ValueError('Expected four videos and labels')
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    caps = [cv2.VideoCapture(str(p)) for p in paths]
    process = None
    try:
        if not all(c.isOpened() for c in caps):
            raise ValueError('Cannot open every comparison input')
        fps = caps[0].get(cv2.CAP_PROP_FPS)
        expected = int(caps[0].get(cv2.CAP_PROP_FRAME_COUNT))
        size = tuple(int(caps[0].get(p)) for p in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT))
        for cap in caps:
            if abs(cap.get(cv2.CAP_PROP_FPS) - fps) > 1e-6 or int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) != expected:
                raise ValueError('Input FPS or frame counts differ')
            if tuple(int(cap.get(p)) for p in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT)) != size:
                raise ValueError('Input resolutions differ')
        height = round(size[1] * width / size[0])
        height += height % 2
        panel_h = height + 44
        # Stream annotated frames directly to FFmpeg; no extra MP4 intermediate.
        process = subprocess.Popen([
            imageio_ffmpeg.get_ffmpeg_exe(), '-y', '-loglevel', 'error',
            '-f', 'rawvideo', '-pix_fmt', 'bgr24', '-s', f'{width*2}x{panel_h*2}',
            '-r', str(fps), '-i', '-', '-an', '-c:v', 'libx264', '-crf', '19',
            '-pix_fmt', 'yuv420p', '-movflags', '+faststart', str(output)], stdin=subprocess.PIPE)
        count = 0
        while True:
            frames = [cap.read() for cap in caps]
            if not any(ok for ok, _ in frames):
                break
            if not all(ok for ok, _ in frames):
                raise ValueError('Inputs decoded to unequal lengths')
            canvas = np.zeros((panel_h*2, width*2, 3), dtype=np.uint8)
            for i, ((_, frame), label) in enumerate(zip(frames, labels)):
                x, y = (i % 2)*width, (i // 2)*panel_h
                canvas[y+44:y+panel_h, x:x+width] = cv2.resize(frame, (width, height))
                cv2.putText(canvas, f'{label} | {count/fps:.2f}s', (x+12, y+29),
                            cv2.FONT_HERSHEY_SIMPLEX, .62, (255, 255, 255), 1, cv2.LINE_AA)
            process.stdin.write(canvas.tobytes())
            if count == min(round(2*fps), expected-1):
                cv2.imwrite(str(output.with_suffix('.jpg')), canvas)
            count += 1
        process.stdin.close()
        if process.wait() != 0:
            raise RuntimeError('FFmpeg encoding failed')
        if count != expected:
            raise ValueError(f'Decoded {count}, expected {expected}')
        check = cv2.VideoCapture(str(output))
        decoded = 0
        try:
            while check.read()[0]:
                decoded += 1
        finally:
            check.release()
        if decoded != expected:
            raise ValueError('Comparison output frame count mismatch')
        return dict(path=str(output), frames=decoded, fps=fps)
    finally:
        for cap in caps:
            cap.release()
        if process is not None and process.poll() is None:
            process.kill()
            process.wait()
