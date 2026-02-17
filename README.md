# bm25-rs
BM-25 알고리즘을 Rust로 구현 및 Python으로 사용할 수 있게끔 하는 라이브러리

# Quick Start
```python
from bm25 import BM25

bm25 = BM25()
bm25.add_document("doc1", ["나는", "밥을", "먹는다"], "나는 밥을 먹는다")
bm25.freeze()
print(bm25.search(["밥"], 1))
```

# Installation
## Requirements
- Rust
- Python 3.6+
- python3-pip

## Requirements for Python
```
pip install -r requirements.txt
```

## Test with Python BM25
PyPI에 배포되어 있는 BM25모듈인 rank_bm25.BM25Okapi 와 속도 비교(examples/speed_test.py). 
> Candidates Pool: 1만건의 위키 문서 / Query: 약 900토큰의 문서 (유사 문서 검색)

|방법|설명|속도(ms)|오차(ms)|
|---|---|---|---|
|BM25.search (ours) |미리 bm25점수를 계산하고 검색 |307 |7.47|
|BM25.search_instance (ours) |쿼리 입력 시 계산 |1120 |13.1|
|BM25Okapi.get_scores |전체 점수 계산 |4940|117|

## Install 
```angular2html
pip install git+https://ldccai.lotte.net/gitlab/psyche/bm25-rs.git
```
