"""Face detection and a reproducible, local face descriptor.

The descriptor is intentionally transparent: a detected face is normalized to a
canonical grayscale crop and encoded as an L2-normalized pixel vector. It is
useful for matching the same person/photo across an image-search result, but it
is not presented as an identity database or a production biometric system.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import cv2
import numpy as np

from .models import FaceScan


class FacePipelineError(RuntimeError):
    """Raised when an input image cannot produce a face scan."""


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def load_image(path: Path) -> np.ndarray:
    if not path.exists():
        raise FacePipelineError(f"Image not found: {path}")
    raw = path.read_bytes()
    if not raw:
        raise FacePipelineError(f"Image is empty: {path}")
    image = cv2.imdecode(np.frombuffer(raw, dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise FacePipelineError(
            f"OpenCV could not decode {path}. Use a JPEG, PNG, or WebP image."
        )
    return image


def decode_image(raw: bytes) -> np.ndarray:
    image = cv2.imdecode(np.frombuffer(raw, dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise FacePipelineError("Downloaded candidate was not a readable image.")
    return image


def detect_faces(image: np.ndarray) -> list[tuple[int, int, int, int]]:
    grayscale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    cascade_path = Path(cv2.data.haarcascades) / "haarcascade_frontalface_default.xml"
    detector = cv2.CascadeClassifier(str(cascade_path))
    if detector.empty():
        raise FacePipelineError("OpenCV's bundled face detector could not be loaded.")
    boxes = detector.detectMultiScale(
        grayscale,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(48, 48),
    )
    return [tuple(int(value) for value in box) for box in boxes]


def _canonical_crop(
    image: np.ndarray, box: tuple[int, int, int, int]
) -> np.ndarray:
    x, y, width, height = box
    padding_x = int(width * 0.18)
    padding_y = int(height * 0.22)
    left = max(0, x - padding_x)
    top = max(0, y - padding_y)
    right = min(image.shape[1], x + width + padding_x)
    bottom = min(image.shape[0], y + height + padding_y)
    crop = image[top:bottom, left:right]
    grayscale = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    normalized = cv2.resize(grayscale, (64, 64), interpolation=cv2.INTER_AREA)
    normalized = cv2.equalizeHist(normalized)
    return normalized.astype(np.float32) / 255.0


def encode_face(image: np.ndarray, box: tuple[int, int, int, int]) -> np.ndarray:
    vector = _canonical_crop(image, box).reshape(-1)
    norm = float(np.linalg.norm(vector))
    if norm == 0:
        raise FacePipelineError("The detected face crop has no usable visual signal.")
    return vector / norm


def encoding_digest(encoding: np.ndarray) -> str:
    rounded = np.round(encoding, decimals=6).astype(np.float32)
    return hashlib.sha256(rounded.tobytes()).hexdigest()


def analyze_image(path: Path) -> tuple[FaceScan, np.ndarray, np.ndarray]:
    raw = path.read_bytes()
    image = load_image(path)
    boxes = detect_faces(image)
    if not boxes:
        raise FacePipelineError(
            "No face detected. Use a clear, front-facing image with one visible face."
        )
    # The largest face is the least surprising default for a face scan.
    box = max(boxes, key=lambda item: item[2] * item[3])
    encoding = encode_face(image, box)
    scan = FaceScan(
        image_sha256=sha256_bytes(raw),
        width=int(image.shape[1]),
        height=int(image.shape[0]),
        face_box=box,
        encoding_digest=encoding_digest(encoding),
        encoding_dimensions=int(encoding.size),
    )
    return scan, image, encoding


def best_face_similarity(
    reference_encoding: np.ndarray, candidate_image: np.ndarray
) -> tuple[float, tuple[int, int, int, int] | None]:
    boxes = detect_faces(candidate_image)
    if not boxes:
        return 0.0, None
    scores = [
        (float(np.dot(reference_encoding, encode_face(candidate_image, box))), box)
        for box in boxes
    ]
    return max(scores, key=lambda item: item[0])


def average_hash(image: np.ndarray) -> int:
    grayscale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(grayscale, (8, 8), interpolation=cv2.INTER_AREA)
    threshold = float(resized.mean())
    bits = (resized >= threshold).flatten()
    return sum((1 << index) for index, bit in enumerate(bits) if bit)


def hamming_distance(left: int, right: int) -> int:
    return (left ^ right).bit_count()