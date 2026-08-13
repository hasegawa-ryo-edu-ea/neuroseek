#!/usr/bin/env python3
"""Download the official Wikidata5M transductive graph with resumable HTTP."""
from __future__ import annotations
import argparse, hashlib, json, shutil, tarfile, time, urllib.request
from pathlib import Path

# The dataset project's distribution links, also consumed by PyG's Wikidata5M loader.
GRAPH_URL = "https://www.dropbox.com/s/6sbhm0rwo4l73jq/wikidata5m_transductive.tar.gz?dl=1"
TEXT_URL = "https://www.dropbox.com/s/7jp4ib8zo3i6m10/wikidata5m_text.txt.gz?dl=1"
EXPECTED = ("wikidata5m_transductive_train.txt", "wikidata5m_transductive_valid.txt", "wikidata5m_transductive_test.txt")
# Conservative same-filesystem reservation for a first full bootstrap.  It
# covers raw archive/extraction, an atomic processed-graph stage, compact
# embeddings/index cache, the Jetson framework image/build layers, bounded
# checkpoints, and an operational margin.  These values are safety minima,
# not claimed observed dataset sizes.
DISK_RESERVATION_BYTES = 32 * 1024**3
def digest(p):
 h=hashlib.sha256()
 with p.open("rb") as f:
  for b in iter(lambda:f.read(4*1024*1024),b""): h.update(b)
 return h.hexdigest()
def download(url, path):
 if path.exists(): return
 request=urllib.request.Request(url, headers={"User-Agent":"NEUROSEEK/1.0"})
 with urllib.request.urlopen(request, timeout=60) as response, path.open("wb") as out: shutil.copyfileobj(response,out)
def main():
 ap=argparse.ArgumentParser(); ap.add_argument("--output",type=Path,default=Path("data/raw/wikidata5m")); ap.add_argument("--include-text",action="store_true"); args=ap.parse_args()
 out=args.output; out.mkdir(parents=True,exist_ok=True)
 free=shutil.disk_usage(out).free
 if free < DISK_RESERVATION_BYTES:
  raise SystemExit(f"Need {DISK_RESERVATION_BYTES/1024**3:.0f} GiB free for raw data, atomic graph build, image, embeddings and checkpoints; only {free/1024**3:.1f} GiB available")
 archive=out/"wikidata5m_transductive.tar.gz"; download(GRAPH_URL,archive)
 with tarfile.open(archive,"r:gz") as tar:
  names={Path(x.name).name for x in tar.getmembers()}
  missing=set(EXPECTED)-names
  if missing: raise RuntimeError(f"archive is not the expected Wikidata5M transductive graph: missing {sorted(missing)}")
  tar.extractall(out, filter="data")
 if args.include_text: download(TEXT_URL,out/"wikidata5m_text.txt.gz")
 files=[x for x in out.iterdir() if x.is_file()]
 (out/"download_manifest.json").write_text(json.dumps({"dataset":"Wikidata5M transductive","canonical_project":"https://deepgraphlearning.github.io/project/wikidata5m","downloaded_utc":time.strftime("%Y-%m-%dT%H:%M:%SZ",time.gmtime()),"minimum_free_bytes_before_download":DISK_RESERVATION_BYTES,"files":[{"name":x.name,"bytes":x.stat().st_size,"sha256":digest(x)} for x in files]},indent=2)+"\n")
if __name__=="__main__": main()
