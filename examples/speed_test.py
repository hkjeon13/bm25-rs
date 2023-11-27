from datasets import load_dataset
from transformers import AutoTokenizer
from bm25 import BM25
from rank_bm25 import BM25Okapi # bm25 module implemented with Python
import time
from tqdm import tqdm


def main():
    tokenizer =  AutoTokenizer.from_pretrained("klue/bert-base")
    dataset = load_dataset("psyche/kowiki", "20230801.process", split="train")
    sample_dataset = dataset.select(range(50000))
    query = tokenizer.tokenize(dataset["text"][20000])
    def tokenize(examples):
        examples['tokens'] = [tokenizer.tokenize(text) for text in examples["text"]]
        return examples

    sample_dataset = sample_dataset.map(tokenize, batched=True, remove_columns=dataset.column_names)

    rs_bm25 = BM25()
    for i, candidate in tqdm(enumerate(sample_dataset)):
        rs_bm25.add_document(f"{i}", candidate["tokens"])
        
    print("[rs_bm25] Indexing has been finished!")
    rs_bm25.freeze()
    tokens = [d["tokens"] for d in sample_dataset]
    py_bm25 = BM25Okapi(tokens)
    print("[py_bm25] Indexing has been finished!")
    
    start = time.time()
    for _ in range(100):
        rs_bm25.search(query, 100)
    print("Rust BM25(ours, precalculated):", (time.time()-start)/100)
    
    start = time.time()
    for _ in range(100):
        py_bm25.get_scores(query)
    print("Python BM25(pypi BM25 module):", (time.time()-start)/100)


if __name__ == "__main__":
    main()

