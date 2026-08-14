# Data and model assets

The source code is stored in Git. The Wikidata5M corpus, processed graph, and
semantic model are published as assets of the `initial-data-and-models` GitHub
Release because several files exceed GitHub's regular file-size limit.

From the repository root, download both archives and extract them:

```bash
tar --zstd -xf neuroseek-wikidata5m-raw.tar.zst
tar --zstd -xf neuroseek-processed-data-and-models.tar.zst
```

This restores:

- `data/raw/wikidata5m/`: the source Wikidata5M files
- `data/processed/`: the processed graph, `semantic_full`, and its manifest

Runtime caches, build output, training runs, logs, and probe artifacts are not
included in the release.
