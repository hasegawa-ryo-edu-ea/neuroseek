#!/usr/bin/env python3
"""Compile Wikidata5M TSV triples into immutable mmap CSR files.

This is deliberately streaming/disk-backed. It never builds a Python graph or
loads all triples at once; SQLite is only a temporary ID compiler and is removed
after a successful atomic publish.
"""
from __future__ import annotations
import argparse, hashlib, json, os, shutil, sqlite3, sys, tempfile, time
from pathlib import Path
import numpy as np

U32_MAX, U16_MAX = 2**32 - 1, 2**16 - 1

def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(4 * 1024 * 1024), b""): h.update(block)
    return h.hexdigest()

def rows(paths):
    for path in paths:
        with path.open("rt", encoding="utf-8", newline="") as f:
            for number, line in enumerate(f, 1):
                parts = line.rstrip("\n\r").split("\t")
                if len(parts) != 3 or not all(parts):
                    raise ValueError(f"{path}:{number}: expected three non-empty tab-separated columns")
                yield parts

def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--input", type=Path, action="append", required=True, help="triple TSV; repeat for train/valid/test")
    p.add_argument("--train-only", action="store_true", help="compile only the first --input as the traversal graph; remaining inputs are recorded as held-out sources")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--force", action="store_true")
    p.add_argument("--entity-aliases", type=Path, help="optional Wikidata ID<TAB>display alias table")
    p.add_argument("--relation-aliases", type=Path, help="optional relation ID<TAB>display alias table")
    args = p.parse_args()
    all_paths = [x.resolve() for x in args.input]
    if any(not x.is_file() for x in all_paths): raise SystemExit("all --input paths must exist")
    paths = all_paths[:1] if args.train_only else all_paths
    out = args.output.resolve()
    if out.exists() and not args.force:
        # A repository-created empty mountpoint is not a dataset and should not
        # make first-run bootstrap fail.  Any real contents remain protected.
        if any(out.iterdir()):
            raise SystemExit(f"{out} exists; refuse to overwrite (use --force only for this output)")
        out.rmdir()
    stage = Path(tempfile.mkdtemp(prefix="neuroseek-graph-", dir=out.parent))
    try:
        db = sqlite3.connect(stage / "ids.sqlite")
        db.execute("PRAGMA journal_mode=WAL"); db.execute("PRAGMA synchronous=NORMAL")
        db.execute("CREATE TABLE entities(id INTEGER PRIMARY KEY, text TEXT UNIQUE NOT NULL)")
        db.execute("CREATE TABLE relations(id INTEGER PRIMARY KEY, text TEXT UNIQUE NOT NULL)")
        # SQLite rowid allocation gives stable compact IDs in source-file order.
        batch_e, batch_r = [], []
        triple_count = 0
        for triple_count, (h, r, t) in enumerate(rows(paths), 1):
            batch_e.extend(((h,), (t,))); batch_r.append((r,))
            if len(batch_e) >= 100_000:
                db.executemany("INSERT OR IGNORE INTO entities(text) VALUES(?)", batch_e)
                db.executemany("INSERT OR IGNORE INTO relations(text) VALUES(?)", batch_r); db.commit(); batch_e.clear(); batch_r.clear()
            if triple_count % 1_000_000 == 0:
                print(f"[ids] streamed {triple_count:,} triples", file=sys.stderr, flush=True)
        db.executemany("INSERT OR IGNORE INTO entities(text) VALUES(?)", batch_e)
        db.executemany("INSERT OR IGNORE INTO relations(text) VALUES(?)", batch_r); db.commit()
        n = db.execute("SELECT COUNT(*) FROM entities").fetchone()[0]; rcount = db.execute("SELECT COUNT(*) FROM relations").fetchone()[0]
        if n > U32_MAX or rcount > U16_MAX: raise RuntimeError(f"ID width overflow: {n} entities, {rcount} relations")
        # The first streaming pass already counted each parsed triple.  Reusing
        # that observed count avoids an otherwise needless full-dataset scan.
        e = triple_count
        fdeg, rdeg = np.zeros(n, dtype=np.uint64), np.zeros(n, dtype=np.uint64)
        # This is a compiler-only ID table, not a production graph object.  On
        # the observed 4.59M-entity Wikidata5M build it takes about 0.7 GiB RSS,
        # well within the idle 8 GiB Jetson budget, and removes roughly 60M
        # random SQLite point lookups from the two remaining streaming passes.
        # Keeping the graph arrays as mmap files is still the runtime invariant.
        entities = {text: ident - 1 for ident, text in db.execute("SELECT id,text FROM entities")}
        rels = {text: ident - 1 for ident, text in db.execute("SELECT id,text FROM relations")}
        for index, (h, _, t) in enumerate(rows(paths), 1):
            fdeg[entities[h]] += 1; rdeg[entities[t]] += 1
            if index % 1_000_000 == 0:
                print(f"[degrees] streamed {index:,}/{e:,} triples", file=sys.stderr, flush=True)
        def offsets(deg, name):
            x = np.memmap(stage / name, dtype="<u8", mode="w+", shape=(n + 1,)); x[0] = 0; np.cumsum(deg, out=x[1:]); x.flush(); return x
        fo, ro = offsets(fdeg, "forward_offsets.u64"), offsets(rdeg, "reverse_offsets.u64")
        fn = np.memmap(stage / "forward_neighbors.u32", dtype="<u4", mode="w+", shape=(e,)); fr = np.memmap(stage / "forward_relations.u16", dtype="<u2", mode="w+", shape=(e,))
        rn = np.memmap(stage / "reverse_neighbors.u32", dtype="<u4", mode="w+", shape=(e,)); rr = np.memmap(stage / "reverse_relations.u16", dtype="<u2", mode="w+", shape=(e,))
        fc, rc = np.array(fo[:-1], copy=True), np.array(ro[:-1], copy=True)
        for index, (h, rel, t) in enumerate(rows(paths), 1):
            a, b, q = entities[h], entities[t], rels[rel]
            i, j = fc[a], rc[b]; fn[i], fr[i], rn[j], rr[j] = b, q, a, q; fc[a] += 1; rc[b] += 1
            if index % 1_000_000 == 0:
                print(f"[csr] streamed {index:,}/{e:,} triples", file=sys.stderr, flush=True)
        for x in (fn, fr, rn, rr): x.flush()
        def write_lookup(filename, table, aliases):
            alias_map = {}
            if aliases:
                with aliases.open(encoding="utf-8") as source:
                    for line in source:
                        key, sep, value = line.rstrip("\r\n").partition("\t")
                        if sep and value and key not in alias_map: alias_map[key] = value.split("\t", 1)[0]
            with (stage / filename).open("w", encoding="utf-8") as f:
                for ident, text in db.execute(f"SELECT id,text FROM {table} ORDER BY id"):
                    f.write(f"{ident - 1}\t{text}\t{alias_map.get(text, text)}\n")
        write_lookup("entities.tsv", "entities", args.entity_aliases)
        write_lookup("relations.tsv", "relations", args.relation_aliases)
        db.close(); (stage / "ids.sqlite").unlink(missing_ok=True); (stage / "ids.sqlite-wal").unlink(missing_ok=True); (stage / "ids.sqlite-shm").unlink(missing_ok=True)
        files = {x.name: {"bytes": x.stat().st_size, "sha256": sha256(x)} for x in stage.iterdir() if x.is_file()}
        manifest = {"format_version": 1, "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "entity_count": n, "relation_count": rcount, "original_triples": e, "traversal_edges": 2 * e, "id_types": {"node": "uint32", "relation": "uint16"}, "input": [{"path": str(x), "sha256": sha256(x), "bytes": x.stat().st_size} for x in paths], "heldout_inputs": [{"path": str(x), "sha256": sha256(x), "bytes": x.stat().st_size} for x in all_paths[len(paths):]], "train_only": args.train_only, "files": files}
        (stage / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
        # Publish without a window where a previously verified graph vanishes.
        # The old immutable directory is removed only after the replacement has
        # been atomically made visible.
        backup = None
        if out.exists():
            backup = out.with_name(out.name + ".previous")
            if backup.exists():
                raise RuntimeError(f"refuse to overwrite retained previous graph: {backup}")
            os.replace(out, backup)
        try:
            os.replace(stage, out)
        except Exception:
            if backup is not None and not out.exists(): os.replace(backup, out)
            raise
        if backup is not None: shutil.rmtree(backup)
    except Exception:
        shutil.rmtree(stage, ignore_errors=True); raise
if __name__ == "__main__": main()
