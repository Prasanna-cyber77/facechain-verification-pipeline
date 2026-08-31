"""A small persisted blockchain-style ledger for tamper-evident records."""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from typing import Any


class LedgerError(RuntimeError):
    """Raised when a chain cannot be written or verified."""


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def digest(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


class LocalBlockchain:
    """Append-only hash chain; the allowed local/simulated chain for the demo."""

    def __init__(self, path: Path) -> None:
        self.path = path

    def _load(self) -> list[dict[str, Any]]:
        if not self.path.exists():
            return []
        try:
            content = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise LedgerError(f"Cannot read ledger {self.path}: {error}") from error
        if not isinstance(content, list):
            raise LedgerError(f"Ledger must contain a JSON array: {self.path}")
        return content

    def _write(self, chain: list[dict[str, Any]]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.path.with_suffix(self.path.suffix + ".tmp")
        temporary.write_text(
            json.dumps(chain, indent=2, ensure_ascii=True) + "\n", encoding="utf-8"
        )
        temporary.replace(self.path)

    @staticmethod
    def _block_hash(block: dict[str, Any]) -> str:
        unsigned = {key: value for key, value in block.items() if key != "hash"}
        return digest(unsigned)

    def _ensure_genesis(self, chain: list[dict[str, Any]]) -> None:
        if chain:
            return
        genesis = {
            "index": 0,
            "timestamp": "2026-01-01T00:00:00Z",
            "previous_hash": "0" * 64,
            "data": {"type": "genesis", "network": "facechain-local"},
        }
        genesis["hash"] = self._block_hash(genesis)
        chain.append(genesis)

    def append(self, data: dict[str, Any]) -> dict[str, Any]:
        chain = self._load()
        self._ensure_genesis(chain)
        previous = chain[-1]
        block = {
            "index": len(chain),
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "previous_hash": previous["hash"],
            "data": data,
        }
        block["hash"] = self._block_hash(block)
        chain.append(block)
        self._write(chain)
        return block

    def verify(self) -> tuple[bool, list[str]]:
        chain = self._load()
        if not chain:
            return False, ["Ledger is empty."]
        errors: list[str] = []
        for index, block in enumerate(chain):
            expected_hash = self._block_hash(block)
            if block.get("hash") != expected_hash:
                errors.append(f"Block {index}: hash does not match its contents.")
            if block.get("index") != index:
                errors.append(f"Block {index}: index is inconsistent.")
            if index == 0:
                if block.get("previous_hash") != "0" * 64:
                    errors.append("Genesis block: previous_hash is invalid.")
            elif block.get("previous_hash") != chain[index - 1].get("hash"):
                errors.append(f"Block {index}: previous_hash link is broken.")
        return not errors, errors

    def verify_record(self, record_id: str) -> tuple[bool, list[str], dict[str, Any] | None]:
        valid, errors = self.verify()
        chain = self._load()
        found = None
        for block in chain:
            if block.get("data", {}).get("record_id") == record_id:
                found = block
                break
        if found is None:
            errors.append(f"Record not found: {record_id}")
        return valid and found is not None and not errors, errors, found