# Data and model release assets

The project source is kept in Git.  The Wikidata5M corpus, processed graph,
and semantic embedding model are attached to the repository's
`initial-data-and-models` GitHub Release because several files exceed
GitHub's regular 100 MB Git-file limit.

Download both assets and extract them from the repository root:

```bash
tar --zstd -xf neuroseek-wikidata5m-raw.tar.zst
tar --zstd -xf neuroseek-processed-data-and-models.tar.zst
```

This restores `data/raw/wikidata5m/` and `data/processed/`, including the
full `semantic_full` float16 embedding model and its manifest.  Runtime
caches, build output, training runs, logs, and probe artifacts are
intentionally not distributed.
