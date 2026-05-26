# -*- coding: utf-8 -*-
"""DullRazor implementation for removing hair artifacts."""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2


def apply_dullrazor_image(image):
	"""Apply DullRazor to an in-memory BGR image."""
	if image is None:
		raise ValueError("Input image is None")

	gray_scale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	kernel = cv2.getStructuringElement(1, (9, 9))
	blackhat = cv2.morphologyEx(gray_scale, cv2.MORPH_BLACKHAT, kernel)
	bhg = cv2.GaussianBlur(blackhat, (3, 3), cv2.BORDER_DEFAULT)
	_, mask = cv2.threshold(bhg, 10, 255, cv2.THRESH_BINARY)
	dst = cv2.inpaint(image, mask, 6, cv2.INPAINT_TELEA)
	return dst


def apply_dullrazor_path(input_path: str | Path, output_path: str | Path | None = None) -> Path:
	"""Apply DullRazor to an image on disk and save the result."""
	input_path = Path(input_path)
	if output_path is None:
		output_path = input_path
	output_path = Path(output_path)

	image = cv2.imread(str(input_path), cv2.IMREAD_COLOR)
	if image is None:
		raise ValueError(f"Failed to read image: {input_path}")

	processed = apply_dullrazor_image(image)
	output_path.parent.mkdir(parents=True, exist_ok=True)
	if not cv2.imwrite(str(output_path), processed):
		raise RuntimeError(f"Failed to write image: {output_path}")
	return output_path


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(description="Apply DullRazor to a single image.")
	parser.add_argument("--input", required=True, help="Input image path")
	parser.add_argument("--output", default=None, help="Output image path (default: overwrite input)")
	return parser.parse_args()


def main() -> None:
	args = parse_args()
	output_path = apply_dullrazor_path(args.input, args.output)
	print(f"Saved: {output_path}")


if __name__ == "__main__":
	main()


