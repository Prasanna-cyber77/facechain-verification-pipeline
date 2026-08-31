"""Command-line interface for the end-to-end FaceChain demonstration."""

from __future__ import annotations

import argparse
import sys
import uuid
from pathlib import Path

from .blockchain import LocalBlockchain, digest
from .face import FacePipelineError, analyze_image
from .search import CompositeImageSearch, SearchError


def heading(label: str) -> None:
    print(f"\n{'=' * 72}\n{label}\n{'=' * 72}")


def run_pipeline(args: argparse.Namespace) -> int:
    image_path = Path(args.image).expanduser().resolve()
    ledger_path = Path(args.ledger).expanduser().resolve()

    heading("FACECHAIN / END-TO-END VERIFICATION")
    print(f"Input scan: {image_path}")

    try:
        scan, _, encoding = analyze_image(image_path)
    except (OSError, FacePipelineError) as error:
        print(f"\n[FACE DETECTION FAILED] {error}", file=sys.stderr)
        return 2

    print("\n[1/3] FACE DETECTION + ENCODING")
    print(f"  Face box:          x={scan.face_box[0]}, y={scan.face_box[1]}, "
          f"w={scan.face_box[2]}, h={scan.face_box[3]}")
    print(f"  Encoding size:     {scan.encoding_dimensions} dimensions")
    print(f"  Encoding digest:   {scan.encoding_digest[:24]}...")
    print(f"  Input SHA-256:     {scan.image_sha256[:24]}...")

    print("\n[2/3] LIVE WEB / SOCIAL SEARCH")
    print("  Providers:         TinEye + Google Lens live upload endpoints")
    print("  Hardcoded result:  no")
    searcher = CompositeImageSearch(timeout_seconds=args.timeout)
    try:
        selected, discovered = searcher.find_matching_post(
            image_path,
            encoding,
            threshold=args.threshold,
            max_candidates=args.max_candidates,
        )
    except (OSError, SearchError, FacePipelineError) as error:
        print(f"\n[LIVE SEARCH FAILED] {error}", file=sys.stderr)
        return 3

    print(f"  Pages discovered:  {len(discovered)}")
    for result in discovered[:3]:
        print(f"    - {result.source}: {result.title[:80]}")
    print("\n  Matching post selected:")
    print(f"    Source:          {selected.source}")
    print(f"    Title:           {selected.title[:120]}")
    print(f"    URL:             {selected.url}")
    print(f"    Match evidence:  {selected.match_method}")

    record_id = f"fc-{uuid.uuid4().hex[:12]}"
    record = {
        "record_id": record_id,
        "type": "face-search-verification",
        "chain": "facechain-local",
        "input": scan.to_dict(),
        "discovered_post": selected.to_dict(),
        "post_fingerprint": digest(
            {
                "url": selected.url,
                "title": selected.title,
                "snippet": selected.snippet,
                "image_sha256": selected.candidate_image_sha256,
            }
        ),
        "method": {
            "face_detector": "OpenCV Haar cascade",
            "face_encoding": "64x64 normalized grayscale descriptor",
            "web_search": "TinEye + Google Lens live upload",
            "blockchain": "local append-only SHA-256 hash chain",
        },
    }

    print("\n[3/3] BLOCKCHAIN UPLOAD + RE-VERIFICATION")
    blockchain = LocalBlockchain(ledger_path)
    try:
        block = blockchain.append(record)
        valid, errors, _ = blockchain.verify_record(record_id)
    except Exception as error:
        print(f"\n[BLOCKCHAIN FAILED] {error}", file=sys.stderr)
        return 4
    if not valid:
        print("\n[RE-VERIFICATION FAILED]", file=sys.stderr)
        for error in errors:
            print(f"  - {error}", file=sys.stderr)
        return 5

    print(f"  Ledger:            {ledger_path}")
    print(f"  Block index:       {block['index']}")
    print(f"  Block hash:        {block['hash'][:32]}...")
    print(f"  Previous hash:     {block['previous_hash'][:32]}...")
    print(f"  Record ID:         {record_id}")
    print("  Re-verification:   PASS — chain links and record hash are intact")
    print("\nPIPELINE COMPLETE")
    print(f"To re-verify later: python -m facechain verify --ledger {ledger_path} "
          f"--record-id {record_id}")
    return 0


def verify_ledger(args: argparse.Namespace) -> int:
    ledger_path = Path(args.ledger).expanduser().resolve()
    blockchain = LocalBlockchain(ledger_path)
    valid, errors, record = blockchain.verify_record(args.record_id)
    heading("FACECHAIN / LEDGER RE-VERIFICATION")
    print(f"Ledger:     {ledger_path}")
    print(f"Record ID:  {args.record_id}")
    if record:
        print(f"Block:      {record.get('index')}")
        print(f"Block hash: {str(record.get('hash', ''))[:32]}...")
    if valid:
        print("\nPASS — the hash chain and requested record are valid.")
        return 0
    print("\nFAIL — tampering or a missing record was detected:")
    for error in errors:
        print(f"  - {error}")
    return 5


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="facechain",
        description="Face scan -> live reverse-image search -> local blockchain verification",
    )
    commands = parser.add_subparsers(dest="command", required=True)

    run = commands.add_parser("run", help="execute the full pipeline")
    run.add_argument("--image", required=True, help="path to the input face scan")
    run.add_argument(
        "--ledger",
        default="data/ledger.json",
        help="JSON ledger path (default: data/ledger.json)",
    )
    run.add_argument(
        "--threshold",
        type=float,
        default=0.72,
        help="minimum face cosine similarity (default: 0.72)",
    )
    run.add_argument(
        "--max-candidates",
        type=int,
        default=12,
        help="maximum live result images to evaluate (default: 12)",
    )
    run.add_argument(
        "--timeout",
        type=int,
        default=30,
        help="network timeout in seconds (default: 30)",
    )
    run.set_defaults(handler=run_pipeline)

    verify = commands.add_parser("verify", help="re-verify a stored blockchain record")
    verify.add_argument("--ledger", default="data/ledger.json")
    verify.add_argument("--record-id", required=True)
    verify.set_defaults(handler=verify_ledger)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    return int(args.handler(args))


if __name__ == "__main__":
    raise SystemExit(main())