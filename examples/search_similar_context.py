from typing import Optional
from dataclasses import dataclass, field
from transformers import HfArgumentParser
from datasets import load_dataset
from multiprocessing import cpu_count
from tqdm import tqdm
from bm25 import BM25


@dataclass
class DataParams:
    data_name_or_path: str = field(
        default="korean-corpus/namu_wiki",
        metadata={"help": "데이터셋의 이름 또는 경로를 입력합니다."}
    )

    data_auth_token: str = field(
        default="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",
        metadata={"help": "비공개 데이터셋을 다운로드하기 위한 토큰을 설정합니다."}
    )

    data_id_column: str = field(
        default="id",
        metadata = {"help":"데이터 셋의 ID 컬럼의 이름을 입력합니다."}
    )

    data_text_column: str = field(
        default="text",
        metadata = {"help":"데이터 셋의 텍스트 컬럼의 이름을 입력합니다."}
    )

    pre_tokenized_data_name_or_path: Optional[str] = field(
        default=None,
        metadata={"help": "토큰화된 데이터셋의 이름 또는 경로를 입력합니다."}
    )

    pre_tokenized_data_auth_token: str = field(
        default="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",
        metadata={"help": "비공개 토큰화된 데이터셋을 다운로드하기 위한 토큰을 설정합니다."}
    )

    split_name: str = field(
        default="train",
        metadata={"help": "데이터셋의 split 이름을 입력합니다."}
    )

    target_dataset_name_or_path: str = field(
        default="KETI-AIR/korquad,v1.0",
        metadata={"help": "타깃 데이터셋의 이름 또는 경로를 입력합니다."}
    )

    target_dataset_auth_token: str = field(
        default="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",
        metadata={"help": "비공개 타깃 데이터셋을 다운로드하기 위한 토큰을 설정합니다."}
    )

    target_text_column: str = field(
        default="context",
        metadata={"help": "비공개 타깃 데이터셋을 다운로드하기 위한 토큰을 설정합니다."}
    )

    output_dir:str = field(
        default = "similar_text",
        metadata = {"help":"데이터가 저장될 경로를 입력합니다."}
    )



def main():
    parser = HfArgumentParser((DataParams,))
    data_args = parser.parse_args()
    if data_args.pre_tokenized_data_name_or_path is None:
        from konlpy.tag import Mecab
        tokenizer = Mecab()

        dataset = load_dataset(data_args.data_name_or_path, use_auth_token=data_args.data_auth_token,
                               split=data_args.split_name)

        def example_function(examples):
            return {
                "doc_id": examples[data_args.data_id_column], 
                "tokens": [tokenizer.nouns(text) for text in examples[data_args.data_text_column]]
            }

        dataset = dataset.map(example_function, batched=True, num_proc=cpu_count(), remove_columns=[data_args.data_text_column])
    else:
        dataset = load_dataset(
            data_args.pre_tokenized_data_name_or_path,
            use_auth_token=data_args.pre_tokenized_data_auth_token,
            split=data_args.split_name
        )

    
    bm25 = BM25()
    for d in tqdm(dataset):
        bm25.add_document(d["doc_id"], d["tokens"])
    

    target_dataset = load_dataset(
        data_args.target_dataset_name_or_path,
        use_auth_token=data_args.target_dataset_auth_token,
        split=data_args.split_name
    )

    def example_function(examples):
        tokens = [tokenizer.nouns(text) for text in examples[target_text_column]]
        return {"tokens": tokens}

    target_dataset = target_dataset.map(example_function, batched=True, num_proc=cpu_count())

    def get_similar_context(examples):
        similar_context = [[v for v, _ in bm25.search(target_tokens, 100)] for target_tokens in examples["tokens"]]
        return {"similar_context": similar_context}

    target_dataset = target_dataset.map(get_similar_context, batched=True, remove_columns=["tokens"])
    target_dataset.save_to_disk(data_args.output_dir)


if __name__ == "__main__":
    main()
