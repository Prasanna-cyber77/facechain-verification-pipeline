# FaceChain Verification Pipeline

FaceChain is an end-to-end demonstration for the HH Goa 2026 shortlisting task:

```text
face scan → face detection + encoding → live web/social reverse-image search
→ matching post → blockchain-style upload → tamper-evident re-verification
```

It is intentionally a command-line project so the entire pipeline can be
recorded in one terminal window. It does not use a hardcoded post. Each run
uploads the supplied image to TinEye's public reverse-image search flow and
falls back to Google Lens, evaluates candidate images exposed by the live
results, and anchors the selected result in a local SHA-256 hash chain.

## What is implemented

1. **Face identification stage** — OpenCV detects the largest face using its
   bundled Haar cascade when available, or downloads the small YuNet detector
   model for newer OpenCV wheels. A normalized 64×64 grayscale face descriptor
   is generated and summarized by a SHA-256 digest.
2. **Genuine search stage** — the input is uploaded to TinEye at runtime, with
   Google Lens as an independent fallback. External result pages are parsed,
   candidate images are downloaded, and each candidate is checked for face
   similarity (or a near-duplicate image hash).
3. **Blockchain verification stage** — the discovered post metadata, candidate
   image fingerprint, and face-scan fingerprint are appended to
   `data/ledger.json`. Each block includes its own SHA-256 hash and the previous
   block hash. The run immediately re-verifies the record.

The brief explicitly allows a local/simulated chain, so the demo uses a
transparent local chain rather than requiring a wallet, gas, or a secret key.

## Run it

### Recommended: uv

```bash
uv sync
uv run python -m facechain run --image /path/to/face-scan.jpg
```

### Standard Python environment

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -m facechain run --image /path/to/face-scan.jpg
```

The input should be a clear JPEG, PNG, or WebP containing one front-facing
face. The image must be publicly searchable for the live search step to return
a useful matching post. The first run on an OpenCV 5 environment downloads the
YuNet detector model into `.cache/facechain/`. A run creates
`data/ledger.json`.

## Re-verify a previous record

The full run prints the record ID. Re-run verification with:

```bash
uv run python -m facechain verify \
  --ledger data/ledger.json \
  --record-id fc-XXXXXXXXXXXX
```

To demonstrate tamper detection during a recording, change a character in
`data/ledger.json`, run the command again, and show the `FAIL` result. Restore
the file afterward.

## Screen-recording script

1. Start the terminal in this repository.
2. Run the `facechain run` command with a real face image that also appears
   publicly on the web.
3. Pause on the three numbered stages and the matching URL.
4. Copy the printed `verify` command and run it.
5. Optionally edit one ledger value and rerun verification to show tamper
   detection, then restore the ledger.

## Known limitations and responsible use

- The local descriptor is intentionally lightweight and explainable. It is not
  a production-grade biometric recognition model and should not be used for
  access control, surveillance, or decisions about people.
- TinEye and Google Lens result markup and rate limits can change. If both
  public upload endpoints are unavailable, rerun later or adapt
  `facechain/search.py` to an authorized reverse-image-search API.
- Public social pages may block automated image retrieval. The pipeline fails
  explicitly rather than claiming a match without image evidence.
- Only use images and web content you have permission to process. Avoid
  uploading sensitive face scans.

## Project layout

```text
facechain/
  cli.py          terminal orchestration and recording-friendly output
  face.py         OpenCV detection, encoding, and face comparison
  search.py       live TinEye/Google Lens upload and result parsing
  blockchain.py   local append-only hash chain and verification
  models.py       typed records shared by all stages
```