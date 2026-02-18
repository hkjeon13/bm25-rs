# bm25-rs

A high-performance [BM25](https://en.wikipedia.org/wiki/Okapi_BM25) search engine written in Rust with Python bindings via [PyO3](https://pyo3.rs/). Designed as a drop-in replacement for pure-Python BM25 libraries like `rank_bm25`, delivering **3-16x faster** search with minimal API changes.

## Features

- **Rust-powered core** — all indexing and scoring happens in Rust for maximum throughput
- **Python-first API** — use it like any other Python library, no Rust knowledge required
- **Frozen index mode** — pre-compute BM25 scores once, search many times at near-zero cost
- **Batch search** — parallel query execution via [Rayon](https://github.com/rayon-rs/rayon), with Python GIL released
- **Multiple serialization formats** — JSON, [bincode](https://github.com/bincode-org/bincode), and [MessagePack](https://msgpack.org/) for saving/loading indices
- **Tunable parameters** — adjust BM25 `k1` and `b` parameters at any time

## Installation

### Requirements

- Python 3.8+
- Rust toolchain (stable) — install via [rustup](https://rustup.rs/)

### From git

```bash
pip install setuptools-rust
pip install git+https://github.com/user/bm25-rs.git
```

### From source

```bash
git clone https://github.com/user/bm25-rs.git
cd bm25-rs
pip install -e .
```

## Quick Start

```python
from bm25 import BM25

# Create an index
bm25 = BM25()

# Add documents: (id, tokens, original_text)
bm25.add_document("doc1", ["hello", "world"], "hello world")
bm25.add_document("doc2", ["hello", "rust"], "hello rust")
bm25.add_document("doc3", ["foo", "bar"], "foo bar")

# Freeze the index (pre-compute BM25 scores)
bm25.freeze()

# Search — returns list of (score, doc_id, text)
results = bm25.search(["hello"], n=2)
for score, doc_id, text in results:
    print(f"  {doc_id}: {score:.4f} — {text}")
```

## Usage Guide

### Adding Documents

Each document requires three arguments: a unique ID, a list of tokens, and the original text.

```python
bm25 = BM25()

# Add a single document
bm25.add_document("doc1", ["natural", "language", "processing"], "natural language processing")

# Add multiple documents at once (more efficient)
documents = [
    ("doc1", ["natural", "language", "processing"], "natural language processing"),
    ("doc2", ["information", "retrieval"], "information retrieval"),
    ("doc3", ["machine", "learning", "model"], "machine learning model"),
]
bm25.add_documents(documents)
```

> **Note:** Tokenization is your responsibility. Use any tokenizer you prefer — whitespace splitting, spaCy, MeCab, Hugging Face tokenizers, etc.

### Searching

There are two search modes:

#### Frozen search (recommended for repeated queries)

Call `freeze()` once to pre-compute all BM25 scores. Subsequent searches are extremely fast since they only need to look up pre-computed values.

```python
bm25.freeze()

# Returns top-n results as list of (score, doc_id, text)
results = bm25.search(["query", "tokens"], n=10)
```

#### Instance search (no freeze required)

Computes BM25 scores on-the-fly. Useful when the index is frequently updated.

```python
results = bm25.search_instance(["query", "tokens"], n=10)
```

### Batch Search

Process multiple queries in parallel using all available CPU cores. The Python GIL is released during computation, so other Python threads can run concurrently.

```python
queries = [
    ["natural", "language"],
    ["machine", "learning"],
    ["information", "retrieval"],
]

# Frozen batch search
bm25.freeze()
all_results = bm25.batch_search(queries, n=5)

# Instance batch search (no freeze)
all_results = bm25.batch_search_instance(queries, n=5)
```

### Removing Documents

```python
bm25.remove_document("doc1")
```

> After adding or removing documents, the index is automatically unfrozen. Call `freeze()` again before using `search()` or `batch_search()`.

### Tuning BM25 Parameters

```python
bm25.set_k1(1.2)   # Term frequency saturation (default: 1.5)
bm25.set_b(0.8)    # Document length normalization (default: 0.75)
```

- `k1` (> 0): Controls how quickly term frequency saturates. Higher values give more weight to term frequency.
- `b` (0.0 - 1.0): Controls document length normalization. `b=1.0` fully normalizes by document length; `b=0.0` ignores document length.

### Saving & Loading

Three serialization formats are supported:

```python
# JSON — human-readable, good for debugging
bm25.save("index.json")
bm25 = BM25.load("index.json")

# MessagePack — fast serialization, compact files, cross-language compatible
bm25.save_msgpack("index.msgpack")
bm25 = BM25.load_msgpack("index.msgpack")

# bincode — fast Rust-native binary format
bm25.save_bin("index.bin")
bm25 = BM25.load_bin("index.bin")
```

#### Format Comparison (1,000 documents)

| Format | Save | Load | File Size |
|---|---|---|---|
| JSON | 6.9 ms | 26.0 ms | 335 KB |
| MessagePack | 2.2 ms | 8.9 ms | 320 KB |
| bincode | 2.1 ms | 10.2 ms | 378 KB |

**Recommendation:** Use **MessagePack** for production workloads — it offers the best balance of speed, file size, and cross-language compatibility. Use **JSON** when you need to inspect or debug the index manually.

> **Note:** After loading, the index is in an unfrozen state. Call `freeze()` before using `search()` or `batch_search()`.

### Utility Methods

```python
bm25.doc_count()              # Number of documents in the index
bm25.contains_document("id")  # Check if a document exists
```

### Inspecting Internal State

```python
bm25.get_index_map()     # Token → {doc_id: term_frequency}
bm25.get_doc_len_map()   # doc_id → document length
bm25.get_doc_texts()     # doc_id → original text
bm25.get_freeze_map()    # Token → {doc_id: pre-computed BM25 score}
```

## Benchmarks

Compared against [rank_bm25](https://github.com/dorianbrown/rank_bm25) (pure Python BM25Okapi).

> Candidate pool: 10,000 wiki documents / Query: ~900 tokens (similar document retrieval)

| Method | Description | Time (ms) | Std (ms) |
|---|---|---|---|
| `BM25.search` (ours) | Pre-computed scores (frozen) | **307** | 7.47 |
| `BM25.search_instance` (ours) | On-the-fly scoring | **1,120** | 13.1 |
| `BM25Okapi.get_scores` | Pure Python scoring | 4,940 | 117 |

## API Reference

### Constructor

| Method | Description |
|---|---|
| `BM25()` | Create a new empty BM25 index |

### Document Management

| Method | Description |
|---|---|
| `add_document(id, tokens, text)` | Add a single document |
| `add_documents([(id, tokens, text), ...])` | Add multiple documents at once |
| `remove_document(id)` | Remove a document by ID |
| `doc_count()` | Get the number of indexed documents |
| `contains_document(id)` | Check if a document exists |

### Search

| Method | Description |
|---|---|
| `freeze()` | Pre-compute BM25 scores for fast search |
| `search(tokens, n)` | Search the frozen index (requires `freeze()`) |
| `search_instance(tokens, n)` | Search without freezing (on-the-fly scoring) |
| `batch_search(token_lists, n)` | Parallel search over multiple queries (requires `freeze()`) |
| `batch_search_instance(token_lists, n)` | Parallel on-the-fly search over multiple queries |

### Persistence

| Method | Description |
|---|---|
| `save(path)` / `BM25.load(path)` | JSON format |
| `save_msgpack(path)` / `BM25.load_msgpack(path)` | MessagePack format |
| `save_bin(path)` / `BM25.load_bin(path)` | bincode format |

### Parameters

| Method | Description |
|---|---|
| `set_k1(value)` | Set k1 parameter (must be > 0, default: 1.5) |
| `set_b(value)` | Set b parameter (must be in [0, 1], default: 0.75) |

## Architecture

```
Python (your code)
  │
  ├── bm25/__init__.py      # Python package entry point
  └── bm25/lib.rs           # Rust core (PyO3 bindings)
        ├── Indexing          — HashMap<token, HashMap<doc_id, tf>>
        ├── Scoring           — BM25 Okapi formula
        ├── Top-N selection   — BinaryHeap (O(D log N) vs O(D log D))
        ├── Freeze            — Pre-computed scores with Rayon parallelism
        ├── Batch search      — Rayon parallel queries, GIL released
        └── Serialization     — serde (JSON / MessagePack / bincode)
```

## License

MIT
