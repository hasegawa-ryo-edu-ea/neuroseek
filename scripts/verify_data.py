#!/usr/bin/env python3
"""Validate raw Wikidata5M triples or an immutable processed graph."""
from __future__ import annotations
import argparse, hashlib, json
from pathlib import Path
def digest(p):
 h=hashlib.sha256()
 with p.open("rb") as f:
  for b in iter(lambda:f.read(4*1024*1024),b""): h.update(b)
 return h.hexdigest()
def main():
 ap=argparse.ArgumentParser(); ap.add_argument("path",type=Path); args=ap.parse_args(); root=args.path
 manifest=root/"manifest.json"
 if manifest.exists():
  m=json.loads(manifest.read_text())
  for name, meta in m["files"].items():
   p=root/name
   if not p.is_file() or p.stat().st_size != meta["bytes"] or digest(p)!=meta["sha256"]: raise SystemExit(f"processed integrity failure: {name}")
  n,e=m["entity_count"],m["original_triples"]
  expected={"forward_offsets.u64":(n+1)*8,"reverse_offsets.u64":(n+1)*8,"forward_neighbors.u32":e*4,"reverse_neighbors.u32":e*4,"forward_relations.u16":e*2,"reverse_relations.u16":e*2}
  for name,size in expected.items():
   if (root/name).stat().st_size != size: raise SystemExit(f"wrong binary length for {name}")
  print(json.dumps({"ok":True,"kind":"processed","entities":n,"relations":m["relation_count"],"original_triples":e})); return
 paths=sorted(root.glob("wikidata5m_*_*.txt"))
 if not paths: raise SystemExit("no raw split TSV files found")
 counts={}
 for path in paths:
  count=0
  with path.open(encoding="utf-8") as f:
   for line_no,line in enumerate(f,1):
    p=line.rstrip("\n\r").split("\t")
    if len(p)!=3 or not all(p) or not p[0].startswith("Q") or not p[1].startswith("P") or not p[2].startswith("Q"): raise SystemExit(f"malformed triple {path}:{line_no}")
    count+=1
  counts[path.name]=count
 print(json.dumps({"ok":True,"kind":"raw","splits":counts,"triples":sum(counts.values())}))
if __name__=="__main__": main()
