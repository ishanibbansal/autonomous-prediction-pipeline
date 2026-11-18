#!/usr/bin/env python3
"""Convert numbered BEV SVG frames into one opaque looping GIF.

Requires cairosvg and Pillow. This script is not part of the C++ build.
"""

import argparse
import io
import math
import re
import sys
from pathlib import Path

FRAME_PATTERN = re.compile(r"^frame_(\d+)\.svg$")


def parse_args(argv):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", required=True, type=Path, help="directory of frame_####.svg files")
    parser.add_argument("--output", required=True, type=Path, help="output .gif path")
    parser.add_argument("--fps", required=True, type=float, help="frames per second, must be finite and > 0")
    parser.add_argument("--width", required=True, type=int, help="output raster width in pixels, must be > 0")
    return parser.parse_args(argv)


def frame_number(path):
    match = FRAME_PATTERN.match(path.name)
    if match is None:
        return None
    return int(match.group(1))


def main(argv):
    args = parse_args(argv)
    if not math.isfinite(args.fps) or args.fps <= 0.0:
        print("make_gif: --fps must be finite and strictly positive", file=sys.stderr)
        return 1
    if args.width <= 0:
        print("make_gif: --width must be strictly positive", file=sys.stderr)
        return 1
    if not args.input_dir.is_dir():
        print(f"make_gif: input is not a directory: {args.input_dir}", file=sys.stderr)
        return 1

    frames = [path for path in args.input_dir.iterdir() if frame_number(path) is not None]
    frames.sort(key=frame_number)
    if not frames:
        print(f"make_gif: no frame_####.svg files in {args.input_dir}", file=sys.stderr)
        return 1

    try:
        import cairosvg
        from PIL import Image
    except ImportError as error:
        print(f"make_gif: missing optional dependency ({error})", file=sys.stderr)
        return 1

    duration_ms = int(round(1000.0 / args.fps))
    if duration_ms <= 0:
        print("make_gif: --fps is too large to represent as a GIF delay", file=sys.stderr)
        return 1

    images = []
    for path in frames:
        png = cairosvg.svg2png(url=str(path), output_width=args.width)
        with Image.open(io.BytesIO(png)) as decoded:
            rgba = decoded.convert("RGBA")
            rgba.load()
        opaque = Image.new("RGB", rgba.size, (11, 18, 32))
        opaque.paste(rgba, mask=rgba.getchannel("A"))
        images.append(opaque)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    images[0].save(
        args.output,
        save_all=True,
        append_images=images[1:],
        duration=duration_ms,
        loop=0,
        disposal=2,
    )
    print(f"make_gif: wrote {len(images)} frames to {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
