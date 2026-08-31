"""Shared data structures for the FaceChain pipeline."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class FaceScan:
    image_sha256: str
    width: int
    height: int
    face_box: tuple[int, int, int, int]
    encoding_digest: str
    encoding_dimensions: int

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["face_box"] = list(self.face_box)
        return data


@dataclass
class SearchResult:
    title: str
    url: str
    source: str
    snippet: str = ""
    image_url: str | None = None
    match_score: float | None = None
    match_method: str | None = None
    candidate_image_sha256: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)