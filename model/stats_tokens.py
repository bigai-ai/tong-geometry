r"""Compute number of tokens in datasets"""
import tqdm
import transformers
from torch.utils.data import DataLoader
from utils import DataCollatorForLM, LMDatasetWithHFTokenizer, worker_init_fn

tokenizer = transformers.AutoTokenizer.from_pretrained(
    "deepseek-ai/deepseek-coder-1.3b-instruct",
    model_max_length=512,
    padding_side="right",
    use_fast=True)
DatasetClass = LMDatasetWithHFTokenizer

train_dataset = DatasetClass(txt_path="/mnt/data/tg/data/dedup_final.dt",
                             loc_path="/mnt/data/tg/data/dedup_final.tell",
                             pretrain=True,
                             add_dst=True,
                             replace_comma=True,
                             ignore_context=True,
                             split="train",
                             tokenizer=tokenizer,
                             max_num_workers=8)
dev_dataset = DatasetClass(txt_path="/mnt/data/tg/data/dedup_final.dt",
                           loc_path="/mnt/data/tg/data/dedup_final.tell",
                           pretrain=True,
                           add_dst=True,
                           replace_comma=True,
                           ignore_context=True,
                           split="dev",
                           tokenizer=tokenizer,
                           max_num_workers=8)
test_dataset = DatasetClass(txt_path="/mnt/data/tg/data/dedup_final.dt",
                            loc_path="/mnt/data/tg/data/dedup_final.tell",
                            pretrain=True,
                            add_dst=True,
                            replace_comma=True,
                            ignore_context=True,
                            split="test",
                            tokenizer=tokenizer,
                            max_num_workers=8)
for dataset in [train_dataset, dev_dataset, test_dataset]:
    total_tokens = 0
    dataloader = DataLoader(
        dataset,
        num_workers=8,
        batch_size=32,
        shuffle=False,
        pin_memory=False,
        drop_last=False,
        worker_init_fn=worker_init_fn,
        collate_fn=DataCollatorForLM(pad_token_id=tokenizer.pad_token_id))
    for batch in tqdm.tqdm(dataloader):
        total_tokens += (
            batch["input_ids"] == tokenizer.pad_token_id).int().argmax(
                dim=1).sum().item()
        total_tokens += len(batch["input_ids"])
    print(total_tokens)

train_dataset = DatasetClass(txt_path="/mnt/data/tg/data/filter.ft",
                             loc_path="/mnt/data/tg/data/filter.tell",
                             pretrain=True,
                             add_dst=True,
                             replace_comma=True,
                             ignore_context=True,
                             split="train",
                             tokenizer=tokenizer,
                             max_num_workers=8)
dev_dataset = DatasetClass(txt_path="/mnt/data/tg/data/filter.ft",
                           loc_path="/mnt/data/tg/data/filter.tell",
                           pretrain=True,
                           add_dst=True,
                           replace_comma=True,
                           ignore_context=True,
                           split="dev",
                           tokenizer=tokenizer,
                           max_num_workers=8)
test_dataset = DatasetClass(txt_path="/mnt/data/tg/data/filter.ft",
                            loc_path="/mnt/data/tg/data/filter.tell",
                            pretrain=True,
                            add_dst=True,
                            replace_comma=True,
                            ignore_context=True,
                            split="test",
                            tokenizer=tokenizer,
                            max_num_workers=8)
for dataset in [train_dataset, dev_dataset, test_dataset]:
    total_tokens = 0
    dataloader = DataLoader(
        dataset,
        num_workers=8,
        batch_size=32,
        shuffle=False,
        pin_memory=False,
        drop_last=False,
        worker_init_fn=worker_init_fn,
        collate_fn=DataCollatorForLM(pad_token_id=tokenizer.pad_token_id))
    for batch in tqdm.tqdm(dataloader):
        total_tokens += (
            batch["input_ids"] == tokenizer.pad_token_id).int().argmax(
                dim=1).sum().item()
        total_tokens += len(batch["input_ids"])
    print(total_tokens)
