"""Live reverse-image search using public TinEye and Google Lens flows."""

from __future__ import annotations

import json
import hashlib
import re
from pathlib import Path
from urllib.parse import urljoin, urlparse

import requests
from bs4 import BeautifulSoup

from .face import (
    FacePipelineError,
    average_hash,
    best_face_similarity,
    decode_image,
    hamming_distance,
)
from .models import SearchResult


class SearchError(RuntimeError):
    """Raised when a live search cannot be completed."""


class TinEyeSearch:
    """Search TinEye's public web endpoint and normalize its result shape."""

    endpoint = "https://tineye.com/api/v1/result_json/"
    user_agent = "Mozilla/5.0 FaceChain/0.1"

    def __init__(self, timeout_seconds: int = 30) -> None:
        self.session = requests.Session()
        self.session.headers.update(
            {
                "User-Agent": self.user_agent,
                "Accept": "application/json, text/plain, */*",
                "Origin": "https://tineye.com",
                "Referer": "https://tineye.com/",
            }
        )
        self.timeout_seconds = timeout_seconds

    def search(self, image_path: Path) -> list[SearchResult]:
        try:
            with image_path.open("rb") as image_file:
                response = self.session.post(
                    self.endpoint,
                    files={"image": (image_path.name, image_file, "image/png")},
                    timeout=self.timeout_seconds,
                )
            response.raise_for_status()
            payload = response.json()
        except (OSError, requests.RequestException, ValueError) as error:
            raise SearchError(f"TinEye request failed: {error}") from error

        results: list[SearchResult] = []
        for match in payload.get("matches", []):
            if not isinstance(match, dict):
                continue
            candidate_image = self._first_string(
                match, ("image_url", "image", "thumbnail", "thumb_url")
            )
            backlinks = match.get("backlinks", [])
            if not isinstance(backlinks, list):
                backlinks = []
            for backlink in backlinks:
                if not isinstance(backlink, dict):
                    continue
                url = self._first_string(backlink, ("backlink", "url", "page_url"))
                if not url:
                    continue
                results.append(
                    SearchResult(
                        title=self._first_string(
                            backlink, ("title", "domain", "url")
                        )
                        or "TinEye match",
                        url=url,
                        source=self._first_string(
                            backlink, ("domain", "domain_name")
                        )
                        or urlparse(url).netloc.removeprefix("www."),
                        snippet=self._first_string(
                            backlink, ("description", "snippet")
                        )
                        or "",
                        image_url=candidate_image,
                    )
                )
        return results[:30]

    @staticmethod
    def _first_string(value: dict, keys: tuple[str, ...]) -> str | None:
        for key in keys:
            item = value.get(key)
            if isinstance(item, str) and item.strip():
                return item.strip()
        return None

    def resolve_candidate_image(self, result: SearchResult) -> bytes | None:
        if result.image_url:
            raw = self._download(result.image_url)
            if raw:
                return raw
        try:
            response = self.session.get(
                result.url,
                headers={"Accept": "text/html,application/xhtml+xml"},
                timeout=self.timeout_seconds,
            )
            response.raise_for_status()
        except requests.RequestException:
            return None
        soup = BeautifulSoup(response.text, "html.parser")
        for selector in (
            'meta[property="og:image"]',
            'meta[name="twitter:image"]',
            'meta[itemprop="image"]',
        ):
            tag = soup.select_one(selector)
            if tag and tag.get("content"):
                image_url = urljoin(result.url, tag["content"])
                raw = self._download(image_url)
                if raw:
                    result.image_url = image_url
                    return raw
        return None

    def _download(self, url: str) -> bytes | None:
        try:
            response = self.session.get(
                url,
                headers={"Accept": "image/avif,image/webp,image/*,*/*;q=0.8"},
                timeout=self.timeout_seconds,
            )
            response.raise_for_status()
            if "image" not in response.headers.get("content-type", "").lower():
                return None
            return response.content
        except requests.RequestException:
            return None

    def find_matching_post(
        self,
        image_path: Path,
        reference_encoding,
        *,
        threshold: float = 0.72,
        max_candidates: int = 12,
    ) -> tuple[SearchResult, list[SearchResult]]:
        return _evaluate_provider(
            self,
            image_path,
            reference_encoding,
            threshold=threshold,
            max_candidates=max_candidates,
        )


def _evaluate_provider(
    provider,
    image_path: Path,
    reference_encoding,
    *,
    threshold: float,
    max_candidates: int,
) -> tuple[SearchResult, list[SearchResult]]:
    discovered = provider.search(image_path)
    if not discovered:
        raise SearchError("Live provider returned no indexed image matches.")
    image = decode_image(image_path.read_bytes())
    reference_hash = average_hash(image)
    evaluated: list[SearchResult] = []

    for result in discovered[:max_candidates]:
        raw = provider.resolve_candidate_image(result)
        if not raw:
            continue
        try:
            candidate = decode_image(raw)
            score, _ = best_face_similarity(reference_encoding, candidate)
        except FacePipelineError:
            continue
        result.match_score = round(score, 4)
        result.candidate_image_sha256 = hashlib.sha256(raw).hexdigest()
        candidate_hash = average_hash(candidate)
        distance = hamming_distance(reference_hash, candidate_hash)
        if score >= threshold:
            result.match_method = f"face cosine similarity ({score:.3f})"
            evaluated.append(result)
        elif distance <= 8:
            result.match_method = f"near-duplicate image hash (distance {distance})"
            result.match_score = round(max(score, 1 - distance / 64), 4)
            evaluated.append(result)

    if not evaluated:
        raise SearchError(
            "Live search returned pages, but none exposed a candidate image "
            f"that passed the face-match threshold ({threshold:.2f}). "
            "Review the search result pages or try a better crop."
        )
    selected = max(evaluated, key=lambda item: item.match_score or 0)
    return selected, discovered


class GoogleLensSearch:
    """Upload a local image and extract external visual-match pages."""

    endpoint = "https://lens.google.com/v3/upload?hl=en"
    user_agent = (
        "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/131.0 Safari/537.36 FaceChain/0.1"
    )

    def __init__(self, timeout_seconds: int = 30) -> None:
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": self.user_agent})
        self.timeout_seconds = timeout_seconds

    def search(self, image_path: Path) -> list[SearchResult]:
        try:
            response = self.session.post(
                self.endpoint,
                files={
                    "encoded_image": (
                        image_path.name,
                        image_path.read_bytes(),
                        "application/octet-stream",
                    )
                },
                data={"image_content": ""},
                allow_redirects=True,
                timeout=self.timeout_seconds,
            )
            response.raise_for_status()
        except requests.RequestException as error:
            raise SearchError(f"Google Lens request failed: {error}") from error

        results = self._parse_results(response.text, response.url)
        if not results:
            raise SearchError(
                "Google Lens returned no parseable external matches. "
                "Try a larger, clearer image or run again later."
            )
        return results

    def _parse_results(self, html: str, base_url: str) -> list[SearchResult]:
        soup = BeautifulSoup(html, "html.parser")
        results: list[SearchResult] = []
        seen_urls: set[str] = set()

        def add_result(
            title: str, url: str, snippet: str = "", image_url: str | None = None
        ) -> None:
            url = urljoin(base_url, url).strip()
            parsed = urlparse(url)
            if parsed.scheme not in {"http", "https"}:
                return
            host = parsed.netloc.lower()
            if not host or host.endswith("google.com") or host.endswith("googleusercontent.com"):
                return
            if url in seen_urls:
                return
            seen_urls.add(url)
            results.append(
                SearchResult(
                    title=self._clean_text(title) or host,
                    url=url,
                    source=host.removeprefix("www."),
                    snippet=self._clean_text(snippet),
                    image_url=image_url,
                )
            )

        for meta in soup.select('meta[property="og:url"], meta[name="twitter:url"]'):
            target = meta.get("content")
            image = soup.select_one('meta[property="og:image"]')
            if target:
                add_result(
                    soup.title.get_text(" ", strip=True) if soup.title else "",
                    target,
                    image_url=image.get("content") if image else None,
                )

        for anchor in soup.select("a[href]"):
            href = anchor.get("href", "")
            if href.startswith("/url?q="):
                href = href.split("/url?q=", 1)[1].split("&", 1)[0]
            if not href.startswith(("http://", "https://")):
                continue
            title = anchor.get_text(" ", strip=True)
            parent_text = anchor.parent.get_text(" ", strip=True) if anchor.parent else ""
            image = anchor.find("img")
            image_url = None
            if image:
                image_url = (
                    image.get("data-iurl")
                    or image.get("data-src")
                    or image.get("src")
                )
            add_result(title, href, parent_text, image_url)

        # Lens sometimes serializes visual matches in JSON rather than anchors.
        for script in soup.select('script[type="application/ld+json"]'):
            try:
                payload = json.loads(script.string or script.get_text())
            except (json.JSONDecodeError, TypeError):
                continue
            self._add_json_results(payload, add_result)

        # Keep the result set bounded and remove obvious Google internal links.
        return results[:30]

    def _add_json_results(self, payload: object, add_result) -> None:
        if isinstance(payload, list):
            for item in payload:
                self._add_json_results(item, add_result)
            return
        if not isinstance(payload, dict):
            return
        url = payload.get("url") or payload.get("contentUrl")
        if isinstance(url, str):
            image_url = payload.get("image")
            if isinstance(image_url, dict):
                image_url = image_url.get("url")
            add_result(
                str(payload.get("name") or payload.get("headline") or ""),
                url,
                str(payload.get("description") or ""),
                image_url if isinstance(image_url, str) else None,
            )
        for value in payload.values():
            if isinstance(value, (dict, list)):
                self._add_json_results(value, add_result)

    @staticmethod
    def _clean_text(value: str) -> str:
        return re.sub(r"\s+", " ", value).strip()

    def resolve_candidate_image(self, result: SearchResult) -> bytes | None:
        if result.image_url:
            raw = self._download(result.image_url)
            if raw:
                return raw
        page_html = self._download(result.url, accept_html=True)
        if not page_html:
            return None
        soup = BeautifulSoup(page_html.decode("utf-8", errors="ignore"), "html.parser")
        for selector in (
            'meta[property="og:image"]',
            'meta[name="twitter:image"]',
            'meta[itemprop="image"]',
        ):
            tag = soup.select_one(selector)
            if tag and tag.get("content"):
                raw = self._download(urljoin(result.url, tag["content"]))
                if raw:
                    result.image_url = urljoin(result.url, tag["content"])
                    return raw
        return None

    def _download(
        self, url: str, *, accept_html: bool = False
    ) -> bytes | None:
        try:
            response = self.session.get(
                url,
                headers={
                    "Accept": "text/html,application/xhtml+xml"
                    if accept_html
                    else "image/avif,image/webp,image/apng,image/*,*/*;q=0.8"
                },
                timeout=self.timeout_seconds,
                allow_redirects=True,
            )
            response.raise_for_status()
            content_type = response.headers.get("content-type", "").lower()
            if accept_html and "html" not in content_type:
                return None
            if not accept_html and "image" not in content_type:
                return None
            return response.content
        except requests.RequestException:
            return None

    def find_matching_post(
        self,
        image_path: Path,
        reference_encoding,
        *,
        threshold: float = 0.72,
        max_candidates: int = 12,
    ) -> tuple[SearchResult, list[SearchResult]]:
        return _evaluate_provider(
            self,
            image_path,
            reference_encoding,
            threshold=threshold,
            max_candidates=max_candidates,
        )


class CompositeImageSearch:
    """Try independent live providers without hiding provider failures."""

    def __init__(self, timeout_seconds: int = 30) -> None:
        self.providers = [
            TinEyeSearch(timeout_seconds),
            GoogleLensSearch(timeout_seconds),
        ]

    def find_matching_post(
        self,
        image_path: Path,
        reference_encoding,
        *,
        threshold: float = 0.72,
        max_candidates: int = 12,
    ) -> tuple[SearchResult, list[SearchResult]]:
        failures: list[str] = []
        all_discovered: list[SearchResult] = []
        for provider in self.providers:
            try:
                selected, discovered = provider.find_matching_post(
                    image_path,
                    reference_encoding,
                    threshold=threshold,
                    max_candidates=max_candidates,
                )
                return selected, discovered
            except SearchError as error:
                failures.append(str(error))
        detail = " | ".join(failures)
        raise SearchError(
            "No provider returned a verifiable matching image. "
            "The search was live, but the result cannot be trusted without "
            f"image evidence. Provider details: {detail}"
        )