r"""Training LLM models"""
import os
import random
from dataclasses import dataclass, field
from typing import Optional

import evaluate
import numpy as np
import torch
import transformers
from pynvml import (nvmlDeviceGetHandleByIndex, nvmlDeviceGetMemoryInfo,
                    nvmlInit)
from transformers import Trainer
from utils import (IGNORE_INDEX, SPECIAL_TOKENS, DataCollatorForLM,
                   DataCollatorForPRM, LMDatasetWithHFTokenizer,
                   LMDatasetWithTGTokenizer, PRMDatasetWithHFTokenizer,
                   PRMDatasetWithTGTokenizer, TongGeoTokenizer, monkey_patch)

ACCURACY = evaluate.load("accuracy")


def print_gpu_utilization():
    """Print GPU memory utilization"""
    nvmlInit()
    handle = nvmlDeviceGetHandleByIndex(0)
    info = nvmlDeviceGetMemoryInfo(handle)
    print(f"GPU memory occupied: {info.used//1024**2} MB.")


def print_summary(result):
    """Print total training time and throughput"""
    print(f"Time: {result.metrics['train_runtime']:.2f}")
    print(f"Samples/second: {result.metrics['train_samples_per_second']:.2f}")
    print_gpu_utilization()


def preprocess_logits_for_metrics_lm(logits, labels):
    """Preprocess logits to avoid memory issue."""
    # Convert logits to tensor if they are not
    if not isinstance(logits, torch.Tensor):
        logits = torch.tensor(logits)

    if not isinstance(labels, torch.Tensor):
        labels = torch.tensor(labels)

    # Shift logits and labels for causal language modeling
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()

    # Flatten the tokens for calculating loss
    shift_logits = shift_logits.view(-1, shift_logits.size(-1))
    shift_labels = shift_labels.view(-1)

    # Calculate the loss
    loss = torch.nn.functional.cross_entropy(shift_logits,
                                             shift_labels,
                                             ignore_index=IGNORE_INDEX)
    numel = shift_labels.ne(IGNORE_INDEX).sum()
    return loss, numel


def compute_metrics_lm(eval_pred):
    """Compute the perplexity for LM learning."""
    (loss, numel), _ = eval_pred
    eval_loss = 0.0
    total_numel = 0
    for loss_i, numel_i in zip(loss, numel):
        eval_loss += loss_i * numel_i
        total_numel += numel_i
    eval_loss /= total_numel
    perplexity = np.exp(eval_loss)
    return {'eval_loss': eval_loss, 'perplexity': perplexity}


def preprocess_logits_for_metrics_prm(logits, labels):
    """Preprocess logits to avoid memory issue."""
    return torch.argmax(logits, dim=-1)


def compute_metrics_prm(eval_pred):
    """Compute the accuracy for PRM learning."""
    predictions, labels = eval_pred
    return ACCURACY.compute(predictions=predictions, references=labels)


@dataclass
class ModelArguments:
    """Arguments regarding model selection."""
    mode: Optional[str] = field(default="lm",
                                metadata={"help": "either lm or prm"})
    weigh_classes: Optional[bool] = field(
        default=False, metadata={"help": "whether to weigh classes in prm"})
    debug_sys: Optional[bool] = field(
        default=False,
        metadata={"help": "whether to debug system performance"})
    num_labels: Optional[int] = field(
        default=24, metadata={"help": "number of labels in prm"})
    model_name_or_path: Optional[str] = field(
        default="deepseek-ai/deepseek-coder-1.3b-instruct",
        metadata={"help": "model architecture path"})
    customized_tokenizer: Optional[bool] = field(
        default=False,
        metadata={"help": "whether to use customized tokenizer"})
    vocab_file: Optional[str] = field(
        default=None, metadata={"help": "customized tokenizer's vocab file"})
    from_scratch: Optional[bool] = field(
        default=False,
        metadata={"help": "whether to retrain the model from scratch"})


@dataclass
class DataArguments:
    """Arguments regarding the input data path."""
    txt_path: str = field(metadata={"help": "path to the txt data"})
    loc_path: str = field(metadata={"help": "path to the loc data"})
    pretrain: Optional[bool] = field(
        default=True, metadata={"help": "whether to train without facts"})
    add_dst: Optional[bool] = field(
        default=True, metadata={"help": "whether to add destination in data"})
    replace_comma: Optional[bool] = field(
        default=True,
        metadata={"help": "whether to remove comma in action call"})
    ignore_context: Optional[bool] = field(
        default=True,
        metadata={"help": "whether to ignore the context actions"})
    aux_path: Optional[str] = field(default="",
                                    metadata={"help": "path to the aux data"})


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    """Arguments regarding training runs."""
    cache_dir: Optional[str] = field(default=None)
    optim: Optional[str] = field(default="adamw_torch")
    model_max_length: Optional[int] = field(
        default=512,
        metadata={
            "help":
            "Maximum sequence length. Sequences will be right padded " +
            "and possibly truncated."
        },
    )
    gradient_checkpointing_kwargs: Optional[dict] = field(
        default_factory=lambda: {"use_reentrant": False})


def safe_save_model_for_hf_trainer(trainer: transformers.Trainer,
                                   output_dir: str):
    """Collects the state dict and dump to disk."""
    state_dict = trainer.model.state_dict()
    if trainer.args.should_save:
        cpu_state_dict = {
            key: value.cpu()
            for key, value in state_dict.items()
        }
        del state_dict
        trainer._save(output_dir, state_dict=cpu_state_dict)  # pylint: disable=protected-access


def train():
    """Training"""
    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments))
    # pylint: disable=unbalanced-tuple-unpacking
    (model_args, data_args,
     training_args) = parser.parse_args_into_dataclasses()

    mode = model_args.mode.lower()

    assert mode in ["lm", "prm"
                    ], f"""mode must be either "lm" or "prm", but {mode}"""

    if training_args.local_rank == 0:
        print('=' * 100)
        print("Model args:")
        print(model_args)
        print('=' * 100)
        print("Data args:")
        print(data_args)
        print('=' * 100)
        print("Training args:")
        print(training_args)

    if training_args.local_rank == 0:
        tokenizer_path = ("local" if model_args.customized_tokenizer else
                          model_args.model_name_or_path)
        print(f"Loading tokenizer from {tokenizer_path}")

    if model_args.customized_tokenizer:
        tokenizer = TongGeoTokenizer(vocab_file=model_args.vocab_file,
                                     special_tokens=SPECIAL_TOKENS,
                                     cls_token=SPECIAL_TOKENS["cls_token"],
                                     sep_token=SPECIAL_TOKENS["sep_token"],
                                     fact_token=SPECIAL_TOKENS["fact_token"],
                                     eos_token=SPECIAL_TOKENS["eos_token"],
                                     pad_token=SPECIAL_TOKENS["pad_token"],
                                     max_len=training_args.model_max_length,
                                     padding=False,
                                     return_tensor=True)
        if training_args.local_rank == 0:
            print("PAD Token:", tokenizer.pad_token, tokenizer.pad_token_id)
            print("CLS Token:", tokenizer.cls_token, tokenizer.cls_token_id)
            print("EOS Token:", tokenizer.eos_token, tokenizer.eos_token_id)
    else:
        tokenizer = transformers.AutoTokenizer.from_pretrained(
            model_args.model_name_or_path,
            model_max_length=training_args.model_max_length,
            padding_side="right",
            use_fast=True)
        if training_args.local_rank == 0:
            print("PAD Token:", tokenizer.pad_token, tokenizer.pad_token_id)
            print("BOS Token", tokenizer.bos_token, tokenizer.bos_token_id)
            print("EOS Token", tokenizer.eos_token, tokenizer.eos_token_id)

    if training_args.local_rank == 0:
        print(f"Loading model from {model_args.model_name_or_path}")

    if mode == "lm":
        DatasetClass = (LMDatasetWithTGTokenizer
                        if model_args.customized_tokenizer else
                        LMDatasetWithHFTokenizer)
        if model_args.from_scratch:
            model_config = transformers.AutoConfig.from_pretrained(
                model_args.model_name_or_path)
            model = transformers.AutoModelForCausalLM.from_config(
                model_config, torch_dtype=torch.bfloat16)
        else:
            model = transformers.AutoModelForCausalLM.from_pretrained(
                model_args.model_name_or_path, torch_dtype=torch.bfloat16)
    else:
        DatasetClass = (PRMDatasetWithTGTokenizer
                        if model_args.customized_tokenizer else
                        PRMDatasetWithHFTokenizer)
        if model_args.from_scratch:
            model_config = transformers.AutoConfig.from_pretrained(
                model_args.model_name_or_path)
            model_config.num_labels = model_args.num_labels
            model = transformers.AutoModelForSequenceClassification.from_config(
                model_config, torch_dtype=torch.bfloat16)
        else:
            model = transformers.AutoModelForSequenceClassification.from_pretrained(
                model_args.model_name_or_path,
                num_labels=model_args.num_labels,
                torch_dtype=torch.bfloat16)
        model.config.pad_token_id = tokenizer.pad_token_id
        if model_args.weigh_classes:
            monkey_patch(model)

    if training_args.local_rank == 0:
        print("Original model:")
        print(model)

    if model_args.customized_tokenizer:
        model.config.vocab_size = len(tokenizer.word_to_index)
        model.model.embed_tokens = torch.nn.Embedding(
            len(tokenizer.word_to_index),
            model.model.embed_tokens.embedding_dim,
            dtype=torch.bfloat16)
        if model_args.mode == "lm":
            model.lm_head = torch.nn.Linear(model.lm_head.in_features,
                                            len(tokenizer.word_to_index),
                                            dtype=torch.bfloat16)
        if training_args.local_rank == 0:
            print("After surgery:")
            print(model)

    model_size = sum(t.numel() for t in model.parameters())

    if training_args.local_rank == 0:
        print(f"Model size: {model_size/1024**3:.1f}B parameters")

    if training_args.local_rank == 0:
        print("Loading datasets")

    if model_args.mode == "lm":
        train_dataset = DatasetClass(
            txt_path=data_args.txt_path,
            loc_path=data_args.loc_path,
            pretrain=data_args.pretrain,
            add_dst=data_args.add_dst,
            replace_comma=data_args.replace_comma,
            ignore_context=data_args.ignore_context,
            split="train",
            tokenizer=tokenizer,
            max_num_workers=training_args.dataloader_num_workers,
            aux_path=data_args.aux_path,
            max_aux=model_args.num_labels)
        dev_dataset = DatasetClass(
            txt_path=data_args.txt_path,
            loc_path=data_args.loc_path,
            pretrain=data_args.pretrain,
            add_dst=data_args.add_dst,
            replace_comma=data_args.replace_comma,
            ignore_context=data_args.ignore_context,
            split="dev",
            tokenizer=tokenizer,
            max_num_workers=training_args.dataloader_num_workers,
            aux_path=data_args.aux_path,
            max_aux=model_args.num_labels)
        # test_dataset = DatasetClass(
        #     txt_path=data_args.txt_path,
        #     loc_path=data_args.loc_path,
        #     pretrain=data_args.pretrain,
        #     add_dst=data_args.add_dst,
        #     replace_comma=data_args.replace_comma,
        #     ignore_context=data_args.ignore_context,
        #     split="test",
        #     tokenizer=tokenizer,
        #     max_num_workers=training_args.dataloader_num_workers)
        data_collator = DataCollatorForLM(pad_token_id=tokenizer.pad_token_id)
        preprocess_logits_for_metrics = preprocess_logits_for_metrics_lm
        compute_metrics = compute_metrics_lm
    else:
        train_dataset = DatasetClass(  # pylint: disable=unexpected-keyword-arg
            txt_path=data_args.txt_path,
            loc_path=data_args.loc_path,
            pretrain=data_args.pretrain,
            add_dst=data_args.add_dst,
            replace_comma=data_args.replace_comma,
            ignore_context=data_args.ignore_context,
            split="train",
            tokenizer=tokenizer,
            max_num_workers=training_args.dataloader_num_workers,
            aux_path=data_args.aux_path,
            max_aux=model_args.num_labels)
        dev_dataset = DatasetClass(  # pylint: disable=unexpected-keyword-arg
            txt_path=data_args.txt_path,
            loc_path=data_args.loc_path,
            pretrain=data_args.pretrain,
            add_dst=data_args.add_dst,
            replace_comma=data_args.replace_comma,
            ignore_context=data_args.ignore_context,
            split="dev",
            tokenizer=tokenizer,
            max_num_workers=training_args.dataloader_num_workers,
            aux_path=data_args.aux_path,
            max_aux=model_args.num_labels)
        # test_dataset = DatasetClass(
        #     txt_path=data_args.txt_path,
        #     loc_path=data_args.loc_path,
        #     pretrain=data_args.pretrain,
        #     add_dst=data_args.add_dst,
        #     replace_comma=data_args.replace_comma,
        #     ignore_context=data_args.ignore_context,
        #     split="test",
        #     tokenizer=tokenizer,
        #     max_num_workers=training_args.dataloader_num_workers)
        data_collator = DataCollatorForPRM(pad_token_id=tokenizer.pad_token_id)
        preprocess_logits_for_metrics = preprocess_logits_for_metrics_prm
        compute_metrics = compute_metrics_prm

    os.environ["TOKENIZERS_PARALLELISM"] = "true"

    if training_args.local_rank == 0:
        print("Training dataset samples:", len(train_dataset))
        for index in random.sample(range(len(train_dataset)), 3):
            print(f"Sample {index} of the training set: " +
                  f"{train_dataset[index]['input_ids']}, " +
                  f"{train_dataset[index]['labels']}.")
            print(
                f"Sample {index} of the training set: " +
                f"{tokenizer.decode(train_dataset[index]['input_ids'].tolist())}."
            )

    data_module = {
        "train_dataset": train_dataset,
        "eval_dataset": dev_dataset,
        "data_collator": data_collator
    }
    trainer = Trainer(
        model=model,
        args=training_args,
        preprocess_logits_for_metrics=preprocess_logits_for_metrics,
        compute_metrics=compute_metrics,
        **data_module)

    results = trainer.train(
        resume_from_checkpoint=training_args.resume_from_checkpoint)
    if model_args.debug_sys:
        if training_args.local_rank == 0:
            print_summary(results)
    else:
        trainer.save_state()
        safe_save_model_for_hf_trainer(trainer=trainer,
                                       output_dir=training_args.output_dir)


if __name__ == "__main__":
    train()
