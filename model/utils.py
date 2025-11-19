r"""Create the causal LM dataset and process supervsion dataset."""
# pylint: disable=too-few-public-methods
import copy
import math
import pickle
import random
from collections import Counter, OrderedDict
from dataclasses import dataclass
from types import MethodType
from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
from torch.utils.data import DataLoader, Dataset, random_split
from tqdm import tqdm
from transformers import (AutoTokenizer, PreTrainedTokenizer,
                          PreTrainedTokenizerFast)
from transformers.cache_utils import Cache
from transformers.modeling_outputs import SequenceClassifierOutputWithPast

from tonggeometry.constructor import AllConstructors

CONSTRUCTOR_TO_LEN = {
    constructor.__name__: constructor.to_names_len
    for constructor in AllConstructors
}
CONSTRUCTOR_TO_LEN["BaseAcuteTriangle"] = 3

IGNORE_INDEX = -100

SPECIAL_TOKENS = OrderedDict([
    ("sep_token", "[SEP]"),
    ("eos_token", "[EOS]"),
    ("fact_token", "[FACT]"),
    ("cls_token", "[CLS]"),
    ("unk_token", "[UNK]"),
    ("pad_token", "[PAD]"),
])

RAW_STATS = """% 1	44232966	5307202267
% 2	11683725	992981372
% 3	3386347	228664541
% 4	1416149	86442659
% 5	639396	32718726
% 6	339097	15319316
% 7	197721	8253773
% 8	126629	5110073
% 9	85320	3415459
% 10	59344	2362506
% 11	42415	1679235
% 12	32264	1272722
% 13	24211	957561
% 14	18918	697758
% 15	14014	500532
% 16	9964	339900
% 17	6516	209872
% 18	3700	113027
% 19	1627	49724
% 20	545	16501
% 21	98	2588
% 22	10	277
% 23	2	14"""


def get_class_weights():
    """Get class weights from raw stats to balance label distribution"""
    lines = RAW_STATS.split('\n')
    data = OrderedDict()
    total_conf = 0
    for line in lines:
        parts = line.split(' ')[1].split('\t')
        num_aux, num_conf, num_rule = list(map(int, parts))
        data[num_aux] = (num_conf, num_rule)
        total_conf += num_conf

    k = max(data.keys())

    rate_of_label = [0.0] * k
    for num_aux, (num_conf, num_rule) in data.items():
        rate_of_aux = num_conf / total_conf
        uniform_rate = 1.0 / num_aux
        for label in range(num_aux):
            rate_of_label[label] += rate_of_aux * uniform_rate

    class_weights = []
    for label, rate in enumerate(rate_of_label):
        weight = 1.0 / (k * rate)
        class_weights.append(weight)
    return class_weights


def forward(
    self,
    input_ids: Optional[torch.LongTensor] = None,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_values: Optional[Union[Cache, List[torch.FloatTensor]]] = None,
    inputs_embeds: Optional[torch.FloatTensor] = None,
    labels: Optional[torch.LongTensor] = None,
    use_cache: Optional[bool] = None,
    output_attentions: Optional[bool] = None,
    output_hidden_states: Optional[bool] = None,
    return_dict: Optional[bool] = None,
) -> Union[Tuple, SequenceClassifierOutputWithPast]:
    r"""Copied from huggingface transformers' llama modeling code"""
    return_dict = return_dict if return_dict is not None else self.config.use_return_dict

    transformer_outputs = self.model(
        input_ids,
        attention_mask=attention_mask,
        position_ids=position_ids,
        past_key_values=past_key_values,
        inputs_embeds=inputs_embeds,
        use_cache=use_cache,
        output_attentions=output_attentions,
        output_hidden_states=output_hidden_states,
        return_dict=return_dict,
    )
    hidden_states = transformer_outputs[0]
    logits = self.score(hidden_states)

    if input_ids is not None:
        batch_size = input_ids.shape[0]
    else:
        batch_size = inputs_embeds.shape[0]

    if self.config.pad_token_id is None and batch_size != 1:
        raise ValueError(
            "Cannot handle batch sizes > 1 if no padding token is defined.")
    if self.config.pad_token_id is None:
        sequence_lengths = -1
    else:
        if input_ids is not None:
            # if no pad token found, use modulo instead of reverse indexing for ONNX compatibility
            sequence_lengths = torch.eq(
                input_ids, self.config.pad_token_id).int().argmax(-1) - 1
            sequence_lengths = sequence_lengths % input_ids.shape[-1]
            sequence_lengths = sequence_lengths.to(logits.device)
        else:
            sequence_lengths = -1

    pooled_logits = logits[torch.arange(batch_size, device=logits.device),
                           sequence_lengths]

    loss = None
    if labels is not None:
        labels = labels.to(logits.device)
        if self.config.problem_type is None:
            if self.num_labels == 1:
                self.config.problem_type = "regression"
            elif self.num_labels > 1 and (labels.dtype == torch.long  # pylint: disable=consider-using-in
                                          or labels.dtype == torch.int):
                self.config.problem_type = "single_label_classification"
            else:
                self.config.problem_type = "multi_label_classification"

        if self.config.problem_type == "regression":
            loss_fct = MSELoss()
            if self.num_labels == 1:
                loss = loss_fct(pooled_logits.squeeze(), labels.squeeze())
            else:
                loss = loss_fct(pooled_logits, labels)
        elif self.config.problem_type == "single_label_classification":
            loss_fct = CrossEntropyLoss(weight=self.class_weights)
            loss = loss_fct(pooled_logits.view(-1, self.num_labels),
                            labels.view(-1))
        elif self.config.problem_type == "multi_label_classification":
            loss_fct = BCEWithLogitsLoss()
            loss = loss_fct(pooled_logits, labels)
    if not return_dict:
        output = (pooled_logits, ) + transformer_outputs[1:]
        return ((loss, ) + output) if loss is not None else output

    return SequenceClassifierOutputWithPast(
        loss=loss,
        logits=pooled_logits,
        past_key_values=transformer_outputs.past_key_values,
        hidden_states=transformer_outputs.hidden_states,
        attentions=transformer_outputs.attentions,
    )


def monkey_patch(model):
    """Monkey patch the model to use class weights for loss computation"""
    class_weights = get_class_weights()
    if model.num_labels > len(
            class_weights):  # num_labels must be >= len(class_weights)
        class_weights += [0] * (model.num_labels - len(class_weights))
    model.register_buffer("class_weights",
                          torch.tensor(class_weights, dtype=model.dtype))
    model.forward = MethodType(forward, model)
    return model


def worker_init_fn(worker_id: int):
    """Initialize worker seed"""
    seed = torch.initial_seed() % 2**32
    random.seed(seed)
    np.random.seed(seed)


@dataclass
class DataCollatorForLM:
    """Collate examples for language modeling"""
    pad_token_id: int
    ignore_token_id: int = IGNORE_INDEX
    input_tensor: bool = True

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids, labels = tuple([instance[key] for instance in instances]
                                  for key in ("input_ids", "labels"))
        if not self.input_tensor:
            input_ids = [torch.tensor(x) for x in input_ids]
            labels = [torch.tensor(x) for x in labels]
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids, batch_first=True, padding_value=self.pad_token_id)
        labels = torch.nn.utils.rnn.pad_sequence(
            labels, batch_first=True, padding_value=self.ignore_token_id)

        return {
            "input_ids": input_ids,
            "labels": labels,
            "attention_mask": input_ids.ne(self.pad_token_id),
        }


@dataclass
class DataCollatorForPRM:
    """Collate examples for process-based reward modeling"""
    pad_token_id: int
    ignore_token_id: int = IGNORE_INDEX
    input_tensor: bool = True

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids, labels = tuple([instance[key] for instance in instances]
                                  for key in ("input_ids", "labels"))
        if not self.input_tensor:
            input_ids = [torch.tensor(x) for x in input_ids]
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids, batch_first=True, padding_value=self.pad_token_id)
        labels = torch.tensor(labels)

        return {
            "input_ids": input_ids,
            "labels": labels,
            "attention_mask": input_ids.ne(self.pad_token_id),
        }


class TongGeoTokenizer:
    """TongGeometry tokenizer"""

    def __init__(self,
                 vocab_file: str,
                 special_tokens: dict,
                 cls_token: str,
                 sep_token: str,
                 fact_token: str,
                 eos_token: str,
                 pad_token: str,
                 max_len: int,
                 padding: bool = False,
                 return_tensor: bool = True):
        self.word_to_index = {}
        self.index_to_word = {}
        self.idx = 0
        with open(vocab_file, 'r', encoding="utf-8") as file:
            for line in file:
                token = line.strip()
                if token == "\\n":
                    token = '\n'
                self.add_token(token)
        for special_token in special_tokens.values():
            self.add_token(special_token)
        self.cls_token = cls_token
        self.cls_token_id = self.word_to_index[self.cls_token]
        self.sep_token = sep_token
        self.sep_token_id = self.word_to_index[self.sep_token]
        self.fact_token = fact_token
        self.fact_token_id = self.word_to_index[self.fact_token]
        self.eos_token = eos_token
        self.eos_token_id = self.word_to_index[self.eos_token]
        self.pad_token = pad_token
        self.pad_token_id = self.word_to_index[self.pad_token]
        self.max_len = max_len
        self.padding = padding
        self.return_tensor = return_tensor

    def add_token(self, token: str):
        """Add a token to the vocab"""
        self.word_to_index[token] = self.idx
        self.index_to_word[self.idx] = token
        self.idx += 1

    def pad(self, input_ids: List[int]):
        """Pad the input_ids to max_len"""
        if len(input_ids) > self.max_len:
            return input_ids[:self.max_len]
        new_input_ids = input_ids + [self.pad_token_id
                                     ] * (self.max_len - len(input_ids))
        if self.return_tensor:
            return torch.tensor(new_input_ids)
        return new_input_ids

    def encode(self, input_string: str):
        """Encode the input string into token ids"""
        input_ids = []
        parts = input_string.split(' ')
        for part in parts:
            if part in self.word_to_index:
                input_ids.append(self.word_to_index[part])
            else:
                input_ids += [self.word_to_index[char] for char in part]
        if len(input_ids) > self.max_len:
            input_ids = input_ids[:self.max_len]
        if self.padding:
            input_ids = input_ids + [self.pad_token_id
                                     ] * (self.max_len - len(input_ids))
        if self.return_tensor:
            return torch.tensor(input_ids)
        return input_ids

    def encode_pretokenized(self, pretokenized: List[str]):
        """Encode the pretokenized string into token ids"""
        input_ids = []
        parts = pretokenized
        for part in parts:
            if part in self.word_to_index:
                input_ids.append(self.word_to_index[part])
            else:
                input_ids += [self.word_to_index[char] for char in part]
        if len(input_ids) > self.max_len:
            input_ids = input_ids[:self.max_len]
        if self.padding:
            input_ids = input_ids + [self.pad_token_id
                                     ] * (self.max_len - len(input_ids))
        if self.return_tensor:
            return torch.tensor(input_ids)
        return input_ids

    def decode(self, input_ids: List[int], literal: bool = True):
        """Decode from token ids"""
        output_string = []
        for input_id in input_ids:
            word = self.index_to_word[input_id]
            output_string.append(word)
            if not literal and input_id in [
                    self.pad_token_id, self.eos_token_id
            ]:
                break
        return ''.join(output_string)


class TongGeoDataset(Dataset):
    """TongGeometry dataset"""

    def __init__(self, txt_path: str, loc_path: str, pretrain: bool,
                 add_dst: bool, replace_comma: bool, ignore_context: bool,
                 split: str, tokenizer: Union[TongGeoTokenizer,
                                              PreTrainedTokenizer,
                                              PreTrainedTokenizerFast],
                 max_num_workers: int):
        self.txt_path = txt_path
        self.loc_path = loc_path
        self.pretrain = pretrain
        self.add_dst = add_dst
        self.replace_comma = replace_comma
        self.ignore_context = ignore_context
        self.split = split
        self.tokenizer = tokenizer
        self.max_num_workers = max_num_workers

        assert split in [
            "train", "dev", "test"
        ], f"""split must be train, dev or test, but {split}"""

        with open(loc_path, 'r', encoding="utf-8") as file:
            all_lines = file.read()[:-1]  # remove last '\n'
        generator = torch.Generator().manual_seed(42)
        seek_loc = [0] + list(map(int, all_lines.split('\n')))[:-1]
        _, dev_set, test_set = random_split(seek_loc, [0.99, 0.005, 0.005],
                                            generator=generator)
        if split == "train":
            self.seek_loc = seek_loc
        elif split == "dev":
            self.seek_loc = dev_set
        else:
            self.seek_loc = test_set
        self.len = len(self.seek_loc)
        self.file_handlers = [
            open(self.txt_path, 'r', encoding="utf-8")  # pylint: disable=consider-using-with
            for _ in range(max_num_workers + 1)
        ]
        self.alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"

    def __len__(self):
        return self.len

    def __del__(self):
        for file_handler in self.file_handlers:
            file_handler.close()

    def __getitem__(self, idx: int):
        worker_info = torch.utils.data.get_worker_info()
        if worker_info:
            worker_id = worker_info.id
        else:
            worker_id = 0
        if self.pretrain:
            return self._getitem_pretrain(idx, worker_id)
        return self._getitem_finetune(idx, worker_id)

    def _getitem_pretrain(self, idx: int, worker_id: int):
        raise NotImplementedError()

    def _getitem_finetune(self, idx: int, worker_id: int):
        raise NotImplementedError()


class LMDatasetWithTGTokenizer(TongGeoDataset):
    """TongGeometry's language modeling dataset"""

    def __init__(self,
                 txt_path: str,
                 loc_path: str,
                 pretrain: bool,
                 add_dst: bool,
                 replace_comma: bool,
                 ignore_context: bool,
                 split: str,
                 tokenizer: Union[TongGeoTokenizer, PreTrainedTokenizer,
                                  PreTrainedTokenizerFast],
                 max_num_workers: int,
                 aux_path: str = None,
                 max_aux: int = 14):
        super().__init__(txt_path, loc_path, pretrain, add_dst, replace_comma,
                         ignore_context, split, tokenizer, max_num_workers)
        self.aux_path = aux_path
        self.max_aux = max_aux
        if aux_path and self.split == "train":
            # i_sample = 1, any i
            with open(aux_path, "rb") as f:
                self.aux_group = pickle.load(f)
            to_pop = []
            for key in self.aux_group:
                if key > self.max_aux:
                    to_pop.append(key)
            for key in to_pop:
                self.aux_group[self.max_aux] += self.aux_group.pop(key)
            self.aux_group_len = {}
            for key, val in self.aux_group.items():
                self.aux_group_len[key] = len(val)
            self.total_sample = max_aux
            self.len = len(self.aux_group[1]) * self.total_sample

    def _loc_map(self, idx):
        if self.aux_path and self.split == "train":
            rounds, offset = divmod(idx, self.total_sample)
            group_idx = offset + 1
            data_id = rounds % self.aux_group_len[group_idx]
            return self.aux_group[group_idx][data_id]
        return self.seek_loc[idx]

    def _tokenize(self, source: List[str], target: List[str]):
        source_ids = self.tokenizer.encode_pretokenized(source)
        target_ids = self.tokenizer.encode_pretokenized(target)
        source_len = len(source_ids)
        if self.tokenizer.return_tensor:
            input_ids = torch.cat((source_ids, target_ids))
        else:
            input_ids = source_ids + target_ids
        label = copy.deepcopy(input_ids)
        if self.ignore_context:
            if self.tokenizer.return_tensor:
                label[:source_len] = IGNORE_INDEX
            else:
                for i in range(source_len):
                    label[i] = IGNORE_INDEX
        return {"input_ids": input_ids, "labels": label}

    def _getitem_pretrain(self, idx: int, worker_id: int) -> str:
        self.file_handlers[worker_id].seek(self._loc_map(idx))
        contents = []
        while True:
            line = self.file_handlers[worker_id].readline()
            if line == "---\n":
                break
            contents.append(line)
        delimiter = 0
        pt = 0
        ctx_parts = []
        aux_parts = []
        for line in contents:
            if line == '\n':
                delimiter += 1
                continue
            if delimiter == 0:
                if self.replace_comma:
                    line = line.replace(',', '')
                action_sep_idx = line.index('(')
                action_type = line[:action_sep_idx]
                if self.add_dst:
                    to_names_len = CONSTRUCTOR_TO_LEN[action_type]
                    if to_names_len > 0:
                        dst = f"{self.alphabet[pt:pt+to_names_len]}="
                        pt += to_names_len
                        ctx_parts.append(dst)
                ctx_parts += [action_type, line[action_sep_idx:]]
            elif delimiter == 1:
                if self.replace_comma:
                    line = line.replace(',', '')
                action_sep_idx = line.index('(')
                action_type = line[:action_sep_idx]
                if self.add_dst:
                    to_names_len = CONSTRUCTOR_TO_LEN[action_type]
                    if to_names_len > 0:
                        dst = f"{self.alphabet[pt:pt+to_names_len]}="
                        pt += to_names_len
                        aux_parts.append(dst)
                aux_parts += [action_type, line[action_sep_idx:]]
            elif delimiter == 2:
                break
        ctx_parts += [
            self.tokenizer.sep_token, self.tokenizer.fact_token, "()\n"
        ]
        aux_parts.append(self.tokenizer.eos_token)
        return self._tokenize(ctx_parts, aux_parts)

    def _getitem_finetune(self, idx: int, worker_id: int) -> str:
        self.file_handlers[worker_id].seek(self._loc_map(idx))
        contents = []
        while True:
            line = self.file_handlers[worker_id].readline()
            if line == "---\n":
                break
            contents.append(line)
        delimiter = 0
        pt = 0
        ctx_parts = []
        aux_parts = []
        all_facts = []
        local_counter = Counter()
        for line in contents:
            if line == '\n':
                delimiter += 1
                continue
            if delimiter == 0:
                if self.replace_comma:
                    line = line.replace(',', '')
                action_sep_idx = line.index('(')
                action_type = line[:action_sep_idx]
                if self.add_dst:
                    to_names_len = CONSTRUCTOR_TO_LEN[action_type]
                    if to_names_len > 0:
                        dst = f"{self.alphabet[pt:pt+to_names_len]}="
                        pt += to_names_len
                        ctx_parts.append(dst)
                ctx_parts += [action_type, line[action_sep_idx:]]
            elif delimiter == 1:
                if self.replace_comma:
                    line = line.replace(',', '')
                action_sep_idx = line.index('(')
                action_type = line[:action_sep_idx]
                if self.add_dst:
                    to_names_len = CONSTRUCTOR_TO_LEN[action_type]
                    if to_names_len > 0:
                        dst = f"{self.alphabet[pt:pt+to_names_len]}="
                        pt += to_names_len
                        aux_parts.append(dst)
                aux_parts += [action_type, line[action_sep_idx:]]
            elif delimiter == 2:
                parts = line.strip().split(' ')
                fact_type = parts[0]
                fact_body_parts = parts[1:-7]
                if fact_body_parts[0] == "(None,":
                    fact_body = [
                        '(', "None", ',', fact_body_parts[1], "None", ',',
                        fact_body_parts[3]
                    ]
                else:
                    fact_body = [''.join(fact_body_parts)]
                all_facts.append((fact_type, fact_body))
                local_counter[fact_type] += 1
        weights = [1 / local_counter[elem[0]] for elem in all_facts]
        fact_choice = random.choices(all_facts, weights=weights, k=1)[0]
        ctx_parts += [self.tokenizer.sep_token, fact_choice[0]
                      ] + fact_choice[1] + ['\n']
        aux_parts.append(self.tokenizer.eos_token)
        return self._tokenize(ctx_parts, aux_parts)


class PRMDatasetWithTGTokenizer(TongGeoDataset):
    """TongGeometry's process-based reward dataset"""

    def __init__(self,
                 txt_path: str,
                 loc_path: str,
                 pretrain: bool,
                 add_dst: bool,
                 replace_comma: bool,
                 ignore_context: bool,
                 split: str,
                 tokenizer: Union[TongGeoTokenizer, PreTrainedTokenizer,
                                  PreTrainedTokenizerFast],
                 max_num_workers: int,
                 aux_path: str = None,
                 max_aux: int = 5):
        super().__init__(txt_path, loc_path, pretrain, add_dst, replace_comma,
                         ignore_context, split, tokenizer, max_num_workers)
        self.max_aux = max_aux
        if self.split == "train":
            self.lcm = math.lcm(*list(range(1, max_aux + 1)))
            self.i_weight = [self.lcm // k for k in range(max_aux, 0, -1)]
            self.i_sample = []
            cumsum = 0
            for i_weight in self.i_weight:
                cumsum += i_weight
                self.i_sample.append(cumsum)
            self.boundary = []
            cumsum = 0
            for i_sample in self.i_sample:
                cumsum += i_sample
                self.boundary.append(cumsum)
            with open(aux_path, "rb") as f:
                self.aux_group = pickle.load(f)
            to_pop = []
            for key in self.aux_group:
                if key > self.max_aux:
                    to_pop.append(key)
            for key in to_pop:
                self.aux_group[self.max_aux] += self.aux_group.pop(key)
            self.aux_group_len = {}
            for key, val in self.aux_group.items():
                self.aux_group_len[key] = len(val)
            self.total_sample = self.boundary[-1]
            rounds = len(self.aux_group[1]) // self.i_sample[0]
            self.len = rounds * self.total_sample + len(
                self.aux_group[1]) - rounds * self.i_sample[0]

    def _loc_map(self, idx):
        if self.split == "train":
            rounds, offset = divmod(idx, self.total_sample)
            for b_idx, boundary in enumerate(self.boundary):
                if offset < boundary:
                    group_idx = b_idx + 1
                    break
            if b_idx != 0:
                offset -= self.boundary[b_idx - 1]
            num_data_passed = rounds * self.i_sample[b_idx] + offset
            data_id = num_data_passed % self.aux_group_len[group_idx]
            return self.aux_group[group_idx][data_id]
        return self.seek_loc[idx]

    def _tokenize(self, source: List[str], target: int):
        input_ids = self.tokenizer.encode_pretokenized(source)
        return {"input_ids": input_ids, "labels": target}

    def _getitem_pretrain(self, idx: int, worker_id: int) -> str:
        self.file_handlers[worker_id].seek(self._loc_map(idx))
        contents = []
        while True:
            line = self.file_handlers[worker_id].readline()
            if line == "---\n":
                break
            contents.append(line)
        delimiter = 0
        pt = 0
        ctx_parts = []
        aux_parts = []
        for line in contents:
            if line == '\n':
                delimiter += 1
                continue
            if delimiter == 0:
                if self.replace_comma:
                    line = line.replace(',', '')
                action_sep_idx = line.index('(')
                action_type = line[:action_sep_idx]
                if self.add_dst:
                    to_names_len = CONSTRUCTOR_TO_LEN[action_type]
                    if to_names_len > 0:
                        dst = f"{self.alphabet[pt:pt+to_names_len]}="
                        pt += to_names_len
                        ctx_parts.append(dst)
                ctx_parts += [action_type, line[action_sep_idx:]]
            elif delimiter == 1:
                if self.replace_comma:
                    line = line.replace(',', '')
                action_sep_idx = line.index('(')
                action_type = line[:action_sep_idx]
                step = [action_type, line[action_sep_idx:]]
                if self.add_dst:
                    to_names_len = CONSTRUCTOR_TO_LEN[action_type]
                    if to_names_len > 0:
                        dst = f"{self.alphabet[pt:pt+to_names_len]}="
                        pt += to_names_len
                        step.insert(0, dst)
                aux_parts.append(step)
            elif delimiter == 2:
                break
        len_aux_parts = len(aux_parts)
        if self.split == "train":
            if len_aux_parts == 1:
                i = 0
            elif len_aux_parts <= self.max_aux:
                weight = self.i_weight[:len_aux_parts]
                i = random.choices(range(len_aux_parts), weights=weight,
                                   k=1)[0]
            else:
                diff = len_aux_parts - self.max_aux + 1
                weight = self.i_weight[:self.max_aux - 1] + [
                    self.i_weight[self.max_aux - 1] / diff
                ] * diff
                i = random.choices(range(len_aux_parts), weights=weight,
                                   k=1)[0]
        else:
            i = random.randrange(len_aux_parts)
        ctx_parts += [
            self.tokenizer.sep_token, self.tokenizer.fact_token, "()\n"
        ]
        return self._tokenize(
            sum(aux_parts[:len(aux_parts) - i], ctx_parts) +
            [self.tokenizer.cls_token], min(i, self.max_aux - 1))

    def _getitem_finetune(self, idx: int, worker_id: int) -> str:
        self.file_handlers[worker_id].seek(self._loc_map(idx))
        contents = []
        while True:
            line = self.file_handlers[worker_id].readline()
            if line == "---\n":
                break
            contents.append(line)
        delimiter = 0
        pt = 0
        ctx_parts = []
        aux_parts = []
        all_facts = []
        local_counter = Counter()
        for line in contents:
            if line == '\n':
                delimiter += 1
                continue
            if delimiter == 0:
                if self.replace_comma:
                    line = line.replace(',', '')
                action_sep_idx = line.index('(')
                action_type = line[:action_sep_idx]
                if self.add_dst:
                    to_names_len = CONSTRUCTOR_TO_LEN[action_type]
                    if to_names_len > 0:
                        dst = f"{self.alphabet[pt:pt+to_names_len]}="
                        pt += to_names_len
                        ctx_parts.append(dst)
                ctx_parts += [action_type, line[action_sep_idx:]]
            elif delimiter == 1:
                if self.replace_comma:
                    line = line.replace(',', '')
                action_sep_idx = line.index('(')
                action_type = line[:action_sep_idx]
                step = [action_type, line[action_sep_idx:]]
                if self.add_dst:
                    to_names_len = CONSTRUCTOR_TO_LEN[action_type]
                    if to_names_len > 0:
                        dst = f"{self.alphabet[pt:pt+to_names_len]}="
                        pt += to_names_len
                        step.insert(0, dst)
                aux_parts.append(step)
            elif delimiter == 2:
                parts = line.strip().split(' ')
                fact_type = parts[0]
                fact_body_parts = parts[1:-7]
                if fact_body_parts[0] == "(None,":
                    fact_body = [
                        '(', "None", ',', fact_body_parts[1], "None", ',',
                        fact_body_parts[3]
                    ]
                else:
                    fact_body = [''.join(fact_body_parts)]
                all_facts.append((fact_type, fact_body))
                local_counter[fact_type] += 1
        weights = [1 / local_counter[elem[0]] for elem in all_facts]
        fact_choice = random.choices(all_facts, weights=weights, k=1)[0]
        len_aux_parts = len(aux_parts)
        if self.split == "train":
            if len_aux_parts == 1:
                i = 0
            elif len_aux_parts <= self.max_aux:
                weight = self.i_weight[:len_aux_parts]
                i = random.choices(range(len_aux_parts), weights=weight,
                                   k=1)[0]
            else:
                diff = len_aux_parts - self.max_aux + 1
                weight = self.i_weight[:self.max_aux - 1] + [
                    self.i_weight[self.max_aux - 1] / diff
                ] * diff
                i = random.choices(range(len_aux_parts), weights=weight,
                                   k=1)[0]
        else:
            i = random.randrange(len_aux_parts)
        ctx_parts += [self.tokenizer.sep_token, fact_choice[0]
                      ] + fact_choice[1] + ['\n']
        return self._tokenize(
            sum(aux_parts[:len(aux_parts) - i], ctx_parts) +
            [self.tokenizer.cls_token], min(i, self.max_aux - 1))


class LMDatasetWithHFTokenizer(TongGeoDataset):
    """TongGeometry's language modeling dataset"""

    def __init__(self,
                 txt_path: str,
                 loc_path: str,
                 pretrain: bool,
                 add_dst: bool,
                 replace_comma: bool,
                 ignore_context: bool,
                 split: str,
                 tokenizer: Union[TongGeoTokenizer, PreTrainedTokenizer,
                                  PreTrainedTokenizerFast],
                 max_num_workers: int,
                 aux_path: str = None,
                 max_aux: int = 14):
        super().__init__(txt_path, loc_path, pretrain, add_dst, replace_comma,
                         ignore_context, split, tokenizer, max_num_workers)
        self.aux_path = aux_path
        self.max_aux = max_aux
        if aux_path and self.split == "train":
            # i_sample = 1, any i
            with open(aux_path, "rb") as f:
                self.aux_group = pickle.load(f)
            to_pop = []
            for key in self.aux_group:
                if key > self.max_aux:
                    to_pop.append(key)
            for key in to_pop:
                self.aux_group[self.max_aux] += self.aux_group.pop(key)
            self.aux_group_len = {}
            for key, val in self.aux_group.items():
                self.aux_group_len[key] = len(val)
            self.total_sample = max_aux
            self.len = len(self.aux_group[1]) * self.total_sample

    def _loc_map(self, idx):
        if self.aux_path and self.split == "train":
            rounds, offset = divmod(idx, self.total_sample)
            group_idx = offset + 1
            data_id = rounds % self.aux_group_len[group_idx]
            return self.aux_group[group_idx][data_id]
        return self.seek_loc[idx]

    def _tokenize(self, source: str, target: str):
        source_ids = self.tokenizer(source, return_tensors="pt",
                                    padding=False)["input_ids"][0]
        target_ids = self.tokenizer(target, return_tensors="pt",
                                    padding=False)["input_ids"][0]
        input_ids = torch.cat(
            (source_ids, target_ids[1:]
             ))[:self.tokenizer.model_max_length]  # auto added BOS
        label = copy.deepcopy(input_ids)
        if self.ignore_context:
            source_len = len(source_ids)
            label[:source_len] = IGNORE_INDEX
        return {"input_ids": input_ids, "labels": label}

    def _getitem_pretrain(self, idx: int, worker_id: int) -> str:
        self.file_handlers[worker_id].seek(self._loc_map(idx))
        contents = []
        while True:
            line = self.file_handlers[worker_id].readline()
            if line == "---\n":
                break
            contents.append(line)
        delimiter = 0
        pt = 0
        ctx_parts = []
        aux_parts = []
        for line in contents:
            if line == '\n':
                delimiter += 1
                continue
            if delimiter == 0:
                if self.replace_comma:
                    line = line.replace(',', ", ")
                action_sep_idx = line.index('(')
                action_type = line[:action_sep_idx]
                if self.add_dst:
                    to_names_len = CONSTRUCTOR_TO_LEN[action_type]
                    if to_names_len > 0:
                        dst = ", ".join(
                            self.alphabet[pt:pt + to_names_len]) + " = "
                        pt += to_names_len
                        ctx_parts.append(dst)
                ctx_parts.append(line)
            elif delimiter == 1:
                if self.replace_comma:
                    line = line.replace(',', ", ")
                action_sep_idx = line.index('(')
                action_type = line[:action_sep_idx]
                if self.add_dst:
                    to_names_len = CONSTRUCTOR_TO_LEN[action_type]
                    if to_names_len > 0:
                        dst = ", ".join(
                            self.alphabet[pt:pt + to_names_len]) + " = "
                        pt += to_names_len
                        aux_parts.append(dst)
                aux_parts.append(line)
            elif delimiter == 2:
                break
        ctx_parts.append("# Prove fact()\n")
        aux_parts.append(self.tokenizer.eos_token)
        return self._tokenize(''.join(ctx_parts), ''.join(aux_parts))

    def _getitem_finetune(self, idx: int, worker_id: int) -> str:
        self.file_handlers[worker_id].seek(self._loc_map(idx))
        contents = []
        while True:
            line = self.file_handlers[worker_id].readline()
            if line == "---\n":
                break
            contents.append(line)
        delimiter = 0
        pt = 0
        ctx_parts = []
        aux_parts = []
        all_facts = []
        local_counter = Counter()
        for line in contents:
            if line == '\n':
                delimiter += 1
                continue
            if delimiter == 0:
                if self.replace_comma:
                    line = line.replace(',', ", ")
                action_sep_idx = line.index('(')
                action_type = line[:action_sep_idx]
                if self.add_dst:
                    to_names_len = CONSTRUCTOR_TO_LEN[action_type]
                    if to_names_len > 0:
                        dst = ", ".join(
                            self.alphabet[pt:pt + to_names_len]) + " = "
                        pt += to_names_len
                        ctx_parts.append(dst)
                ctx_parts.append(line)
            elif delimiter == 1:
                if self.replace_comma:
                    line = line.replace(',', ", ")
                action_sep_idx = line.index('(')
                action_type = line[:action_sep_idx]
                if self.add_dst:
                    to_names_len = CONSTRUCTOR_TO_LEN[action_type]
                    if to_names_len > 0:
                        dst = ", ".join(
                            self.alphabet[pt:pt + to_names_len]) + " = "
                        pt += to_names_len
                        aux_parts.append(dst)
                aux_parts.append(line)
            elif delimiter == 2:
                parts = line.strip().split(' ')
                fact_type = parts[0]
                fact_body = ' '.join(parts[1:-7])
                all_facts.append((fact_type, fact_body))
                local_counter[fact_type] += 1
        weights = [1 / local_counter[elem[0]] for elem in all_facts]
        fact_choice = random.choices(all_facts, weights=weights, k=1)[0]
        ctx_parts.append(f"# Prove {fact_choice[0]}{fact_choice[1]}\n")
        aux_parts.append(self.tokenizer.eos_token)
        return self._tokenize(''.join(ctx_parts), ''.join(aux_parts))


class PRMDatasetWithHFTokenizer(TongGeoDataset):
    """TongGeometry's process-based reward dataset"""

    def __init__(self,
                 txt_path: str,
                 loc_path: str,
                 pretrain: bool,
                 add_dst: bool,
                 replace_comma: bool,
                 ignore_context: bool,
                 split: str,
                 tokenizer: Union[TongGeoTokenizer, PreTrainedTokenizer,
                                  PreTrainedTokenizerFast],
                 max_num_workers: int,
                 aux_path: str = None,
                 max_aux: int = 5):
        super().__init__(txt_path, loc_path, pretrain, add_dst, replace_comma,
                         ignore_context, split, tokenizer, max_num_workers)
        self.max_aux = max_aux
        if self.split == "train":
            self.lcm = math.lcm(*list(range(1, max_aux + 1)))
            self.i_weight = [self.lcm // k for k in range(max_aux, 0, -1)]
            self.i_sample = []
            cumsum = 0
            for i_weight in self.i_weight:
                cumsum += i_weight
                self.i_sample.append(cumsum)
            self.boundary = []
            cumsum = 0
            for i_sample in self.i_sample:
                cumsum += i_sample
                self.boundary.append(cumsum)
            with open(aux_path, "rb") as f:
                self.aux_group = pickle.load(f)
            to_pop = []
            for key in self.aux_group:
                if key > self.max_aux:
                    to_pop.append(key)
            for key in to_pop:
                self.aux_group[self.max_aux] += self.aux_group.pop(key)
            self.aux_group_len = {}
            for key, val in self.aux_group.items():
                self.aux_group_len[key] = len(val)
            self.total_sample = self.boundary[-1]
            rounds = len(self.aux_group[1]) // self.i_sample[0]
            self.len = rounds * self.total_sample + len(
                self.aux_group[1]) - rounds * self.i_sample[0]

    def _loc_map(self, idx):
        if self.split == "train":
            rounds, offset = divmod(idx, self.total_sample)
            for b_idx, boundary in enumerate(self.boundary):
                if offset < boundary:
                    group_idx = b_idx + 1
                    break
            if b_idx != 0:
                offset -= self.boundary[b_idx - 1]
            num_data_passed = rounds * self.i_sample[b_idx] + offset
            data_id = num_data_passed % self.aux_group_len[group_idx]
            return self.aux_group[group_idx][data_id]
        return self.seek_loc[idx]

    def _tokenize(self, source: str, target: int):
        input_ids = self.tokenizer(source,
                                   return_tensors="pt",
                                   padding=False,
                                   max_length=self.tokenizer.model_max_length,
                                   truncation=True)["input_ids"][0]
        return {"input_ids": input_ids, "labels": target}

    def _getitem_pretrain(self, idx: int, worker_id: int) -> str:
        self.file_handlers[worker_id].seek(self._loc_map(idx))
        contents = []
        while True:
            line = self.file_handlers[worker_id].readline()
            if line == "---\n":
                break
            contents.append(line)
        delimiter = 0
        pt = 0
        ctx_parts = []
        aux_parts = []
        for line in contents:
            if line == '\n':
                delimiter += 1
                continue
            if delimiter == 0:
                if self.replace_comma:
                    line = line.replace(',', ", ")
                action_sep_idx = line.index('(')
                action_type = line[:action_sep_idx]
                if self.add_dst:
                    to_names_len = CONSTRUCTOR_TO_LEN[action_type]
                    if to_names_len > 0:
                        dst = ", ".join(
                            self.alphabet[pt:pt + to_names_len]) + " = "
                        pt += to_names_len
                        ctx_parts.append(dst)
                ctx_parts.append(line)
            elif delimiter == 1:
                if self.replace_comma:
                    line = line.replace(',', ", ")
                action_sep_idx = line.index('(')
                action_type = line[:action_sep_idx]
                step = [line]
                if self.add_dst:
                    to_names_len = CONSTRUCTOR_TO_LEN[action_type]
                    if to_names_len > 0:
                        dst = ", ".join(
                            self.alphabet[pt:pt + to_names_len]) + " = "
                        pt += to_names_len
                        step.insert(0, dst)
                aux_parts.append(step)
            elif delimiter == 2:
                break
        len_aux_parts = len(aux_parts)
        if self.split == "train":
            if len_aux_parts == 1:
                i = 0
            elif len_aux_parts <= self.max_aux:
                weight = self.i_weight[:len_aux_parts]
                i = random.choices(range(len_aux_parts), weights=weight,
                                   k=1)[0]
            else:
                diff = len_aux_parts - self.max_aux + 1
                weight = self.i_weight[:self.max_aux - 1] + [
                    self.i_weight[self.max_aux - 1] / diff
                ] * diff
                i = random.choices(range(len_aux_parts), weights=weight,
                                   k=1)[0]
        else:
            i = random.randrange(len_aux_parts)
        ctx_parts.append("# Prove fact()\n")
        return self._tokenize(
            ''.join(
                sum(aux_parts[:len(aux_parts) - i], ctx_parts) + ["label = "]),
            min(i, self.max_aux - 1))

    def _getitem_finetune(self, idx: int, worker_id: int) -> str:
        self.file_handlers[worker_id].seek(self._loc_map(idx))
        contents = []
        while True:
            line = self.file_handlers[worker_id].readline()
            if line == "---\n":
                break
            contents.append(line)
        delimiter = 0
        pt = 0
        ctx_parts = []
        aux_parts = []
        all_facts = []
        local_counter = Counter()
        for line in contents:
            if line == '\n':
                delimiter += 1
                continue
            if delimiter == 0:
                if self.replace_comma:
                    line = line.replace(',', ", ")
                action_sep_idx = line.index('(')
                action_type = line[:action_sep_idx]
                if self.add_dst:
                    to_names_len = CONSTRUCTOR_TO_LEN[action_type]
                    if to_names_len > 0:
                        dst = ", ".join(
                            self.alphabet[pt:pt + to_names_len]) + " = "
                        pt += to_names_len
                        ctx_parts.append(dst)
                ctx_parts.append(line)
            elif delimiter == 1:
                if self.replace_comma:
                    line = line.replace(',', ", ")
                action_sep_idx = line.index('(')
                action_type = line[:action_sep_idx]
                step = [line]
                if self.add_dst:
                    to_names_len = CONSTRUCTOR_TO_LEN[action_type]
                    if to_names_len > 0:
                        dst = ", ".join(
                            self.alphabet[pt:pt + to_names_len]) + " = "
                        pt += to_names_len
                        step.insert(0, dst)
                aux_parts.append(step)
            elif delimiter == 2:
                parts = line.strip().split(' ')
                fact_type = parts[0]
                fact_body = ' '.join(parts[1:-7])
                all_facts.append((fact_type, fact_body))
                local_counter[fact_type] += 1
        weights = [1 / local_counter[elem[0]] for elem in all_facts]
        fact_choice = random.choices(all_facts, weights=weights, k=1)[0]
        len_aux_parts = len(aux_parts)
        if self.split == "train":
            if len_aux_parts == 1:
                i = 0
            elif len_aux_parts <= self.max_aux:
                weight = self.i_weight[:len_aux_parts]
                i = random.choices(range(len_aux_parts), weights=weight,
                                   k=1)[0]
            else:
                diff = len_aux_parts - self.max_aux + 1
                weight = self.i_weight[:self.max_aux - 1] + [
                    self.i_weight[self.max_aux - 1] / diff
                ] * diff
                i = random.choices(range(len_aux_parts), weights=weight,
                                   k=1)[0]
        else:
            i = random.randrange(len_aux_parts)
        ctx_parts.append(f"# Prove {fact_choice[0]}{fact_choice[1]}\n")
        return self._tokenize(
            ''.join(
                sum(aux_parts[:len(aux_parts) - i], ctx_parts) + ["label = "]),
            min(i, self.max_aux - 1))


def test_tokenizer():
    """Test tokenizer"""
    test_case1 = (
        "ABC = BaseAcuteTriangle ()\nD= InCenter (A,B,C)\nE= Perpendicular " +
        "(D,A,C)\n [SEP] eqline (IM,KM)\nF= CircumscribedCircle (A,C,D)\nG " +
        "= CircumscribedCircle(B,C,D)\n [EOS]")
    test_case2 = test_case1.split(' ')

    tokenizer = TongGeoTokenizer(vocab_file="./model/vocab.txt",
                                 special_tokens=SPECIAL_TOKENS,
                                 sep_token=SPECIAL_TOKENS["sep_token"],
                                 pad_token=SPECIAL_TOKENS["pad_token"],
                                 eos_token=SPECIAL_TOKENS["eos_token"],
                                 fact_token=SPECIAL_TOKENS["fact_token"],
                                 cls_token=SPECIAL_TOKENS["cls_token"],
                                 max_len=512,
                                 padding=False,
                                 return_tensor=False)
    print(tokenizer.word_to_index)
    print(tokenizer.cls_token, tokenizer.cls_token_id)
    print(tokenizer.sep_token, tokenizer.sep_token_id)
    print(tokenizer.eos_token, tokenizer.eos_token_id)
    print(tokenizer.pad_token, tokenizer.pad_token_id)
    print(tokenizer.fact_token, tokenizer.fact_token_id)
    input_ids = tokenizer.encode(test_case1)
    print(input_ids)
    decoded = tokenizer.decode(input_ids, True)
    print(repr(decoded))
    assert decoded == test_case1.replace(' ', '')
    input_ids = tokenizer.encode_pretokenized(test_case2)
    print(input_ids)
    decoded = tokenizer.decode(input_ids, True)
    print(repr(decoded))
    assert decoded == test_case1.replace(' ', '')


def test_lmtg_dataset():
    """Test LM dataset"""
    tokenizer = TongGeoTokenizer(vocab_file="./model/vocab.txt",
                                 special_tokens=SPECIAL_TOKENS,
                                 sep_token=SPECIAL_TOKENS["sep_token"],
                                 pad_token=SPECIAL_TOKENS["pad_token"],
                                 eos_token=SPECIAL_TOKENS["eos_token"],
                                 fact_token=SPECIAL_TOKENS["fact_token"],
                                 cls_token=SPECIAL_TOKENS["cls_token"],
                                 max_len=512,
                                 padding=False,
                                 return_tensor=True)

    dataset = LMDatasetWithTGTokenizer(
        txt_path="/mnt/data/tg/data/filter.ft",
        loc_path="/mnt/data/tg/data/filter.tell",
        pretrain=True,
        add_dst=True,
        replace_comma=True,
        ignore_context=True,
        split="train",
        tokenizer=tokenizer,
        max_num_workers=8)

    data = dataset[0]
    print(data)
    words = []
    for token_id in data["input_ids"].tolist():
        words.append(tokenizer.index_to_word[token_id])
    print(repr(words))
    print(tokenizer.decode(data["input_ids"].tolist()))

    dataset = LMDatasetWithTGTokenizer(
        txt_path="/mnt/data/tg/data/filter.ft",
        loc_path="/mnt/data/tg/data/filter.tell",
        pretrain=True,
        add_dst=True,
        replace_comma=True,
        ignore_context=False,
        split="train",
        tokenizer=tokenizer,
        max_num_workers=8)

    data = dataset[0]
    print(data)
    words = []
    for token_id in data["input_ids"].tolist():
        words.append(tokenizer.index_to_word[token_id])
    print(repr(words))
    print(tokenizer.decode(data["input_ids"].tolist()))

    dataset = LMDatasetWithTGTokenizer(
        txt_path="/mnt/data/tg/data/filter.ft",
        loc_path="/mnt/data/tg/data/filter.tell",
        pretrain=False,
        add_dst=True,
        replace_comma=True,
        ignore_context=True,
        split="train",
        tokenizer=tokenizer,
        max_num_workers=8)

    data = dataset[0]
    print(data)
    words = []
    for token_id in data["input_ids"].tolist():
        words.append(tokenizer.index_to_word[token_id])
    print(repr(words))
    print(tokenizer.decode(data["input_ids"].tolist()))

    dataset = LMDatasetWithTGTokenizer(
        txt_path="/mnt/data/tg/data/filter.ft",
        loc_path="/mnt/data/tg/data/filter.tell",
        pretrain=False,
        add_dst=True,
        replace_comma=True,
        ignore_context=False,
        split="train",
        tokenizer=tokenizer,
        max_num_workers=8)

    data = dataset[0]
    print(data)
    words = []
    for token_id in data["input_ids"].tolist():
        words.append(tokenizer.index_to_word[token_id])
    print(repr(words))
    print(tokenizer.decode(data["input_ids"].tolist()))


def test_lmhf_dataset():
    """Test LM dataset"""
    tokenizer = AutoTokenizer.from_pretrained(
        "deepseek-ai/deepseek-coder-1.3b-instruct",
        model_max_length=512,
        padding_side="right",
        use_fast=True)

    dataset = LMDatasetWithHFTokenizer(
        txt_path="/mnt/data/tg/data/filter.ft",
        loc_path="/mnt/data/tg/data/filter.tell",
        pretrain=True,
        add_dst=True,
        replace_comma=True,
        ignore_context=True,
        split="train",
        tokenizer=tokenizer,
        max_num_workers=8)

    data = dataset[0]
    print(data)
    print(tokenizer.convert_ids_to_tokens(data["input_ids"].tolist()))
    print(tokenizer.decode(data["input_ids"].tolist()))

    dataset = LMDatasetWithHFTokenizer(
        txt_path="/mnt/data/tg/data/filter.ft",
        loc_path="/mnt/data/tg/data/filter.tell",
        pretrain=True,
        add_dst=True,
        replace_comma=True,
        ignore_context=False,
        split="train",
        tokenizer=tokenizer,
        max_num_workers=8)

    data = dataset[0]
    print(data)
    print(tokenizer.convert_ids_to_tokens(data["input_ids"].tolist()))
    print(tokenizer.decode(data["input_ids"].tolist()))

    dataset = LMDatasetWithHFTokenizer(
        txt_path="/mnt/data/tg/data/filter.ft",
        loc_path="/mnt/data/tg/data/filter.tell",
        pretrain=False,
        add_dst=True,
        replace_comma=True,
        ignore_context=True,
        split="train",
        tokenizer=tokenizer,
        max_num_workers=8)

    data = dataset[0]
    print(data)
    print(tokenizer.convert_ids_to_tokens(data["input_ids"].tolist()))
    print(tokenizer.decode(data["input_ids"].tolist()))

    dataset = LMDatasetWithHFTokenizer(
        txt_path="/mnt/data/tg/data/filter.ft",
        loc_path="/mnt/data/tg/data/filter.tell",
        pretrain=False,
        add_dst=True,
        replace_comma=True,
        ignore_context=False,
        split="train",
        tokenizer=tokenizer,
        max_num_workers=8)

    data = dataset[0]
    print(data)
    print(tokenizer.convert_ids_to_tokens(data["input_ids"].tolist()))
    print(tokenizer.decode(data["input_ids"].tolist()))


def test_lmtg_l_dataset():
    """Test LM dataset"""
    tokenizer = TongGeoTokenizer(vocab_file="./model/vocab.txt",
                                 special_tokens=SPECIAL_TOKENS,
                                 sep_token=SPECIAL_TOKENS["sep_token"],
                                 pad_token=SPECIAL_TOKENS["pad_token"],
                                 eos_token=SPECIAL_TOKENS["eos_token"],
                                 fact_token=SPECIAL_TOKENS["fact_token"],
                                 cls_token=SPECIAL_TOKENS["cls_token"],
                                 max_len=512,
                                 padding=False,
                                 return_tensor=True)

    dataset = LMDatasetWithTGTokenizer(
        txt_path="/mnt/data/tg/data/filter.ft",
        loc_path="/mnt/data/tg/data/filter.tell",
        pretrain=False,
        add_dst=True,
        replace_comma=True,
        ignore_context=True,
        split="train",
        tokenizer=tokenizer,
        max_num_workers=8,
        aux_path='',
        max_aux=14)
    print(dataset.aux_path)
    if dataset.aux_path:
        print("This line shouldn't be printed")

    dataset = LMDatasetWithTGTokenizer(
        txt_path="/mnt/data/tg/data/filter.ft",
        loc_path="/mnt/data/tg/data/filter.tell",
        pretrain=False,
        add_dst=True,
        replace_comma=True,
        ignore_context=True,
        split="train",
        tokenizer=tokenizer,
        max_num_workers=8,
        aux_path="/mnt/data/tg/data/aux.pkl",
        max_aux=14)
    print(dataset.aux_path)
    if dataset.aux_path:
        print("This line should be printed")

    for i in range(28):
        print(tokenizer.decode(dataset[i]["input_ids"].tolist()))


def test_lmhf_l_dataset():
    """Test LM dataset"""
    tokenizer = AutoTokenizer.from_pretrained(
        "deepseek-ai/deepseek-coder-1.3b-instruct",
        model_max_length=512,
        padding_side="right",
        use_fast=True)

    dataset = LMDatasetWithHFTokenizer(
        txt_path="/mnt/data/tg/data/filter.ft",
        loc_path="/mnt/data/tg/data/filter.tell",
        pretrain=False,
        add_dst=True,
        replace_comma=True,
        ignore_context=True,
        split="train",
        tokenizer=tokenizer,
        max_num_workers=8,
        aux_path='',
        max_aux=14)
    print(dataset.aux_path)
    if dataset.aux_path:
        print("This line shouldn't be printed")

    dataset = LMDatasetWithHFTokenizer(
        txt_path="/mnt/data/tg/data/filter.ft",
        loc_path="/mnt/data/tg/data/filter.tell",
        pretrain=False,
        add_dst=True,
        replace_comma=True,
        ignore_context=True,
        split="train",
        tokenizer=tokenizer,
        max_num_workers=8,
        aux_path="/mnt/data/tg/data/aux.pkl",
        max_aux=14)
    print(dataset.aux_path)
    if dataset.aux_path:
        print("This line should be printed")

    for i in range(28):
        print(tokenizer.decode(dataset[i]["input_ids"].tolist()))


def test_prmtg_dataset():
    """Test PRM dataset"""
    tokenizer = TongGeoTokenizer(vocab_file="./model/vocab.txt",
                                 special_tokens=SPECIAL_TOKENS,
                                 sep_token=SPECIAL_TOKENS["sep_token"],
                                 pad_token=SPECIAL_TOKENS["pad_token"],
                                 eos_token=SPECIAL_TOKENS["eos_token"],
                                 fact_token=SPECIAL_TOKENS["fact_token"],
                                 cls_token=SPECIAL_TOKENS["cls_token"],
                                 max_len=512,
                                 padding=False,
                                 return_tensor=True)

    dataset = PRMDatasetWithTGTokenizer(
        txt_path="/mnt/data/tg/data/filter.ft",
        loc_path="/mnt/data/tg/data/filter.tell",
        pretrain=True,
        add_dst=True,
        replace_comma=True,
        ignore_context=True,
        split="train",
        tokenizer=tokenizer,
        max_num_workers=8,
        aux_path="/mnt/data/tg/data/aux.pkl",
        max_aux=5)

    data = dataset[0]
    print(data)
    words = []
    for token_id in data["input_ids"].tolist():
        words.append(tokenizer.index_to_word[token_id])
    print(repr(words))
    print(tokenizer.decode(data["input_ids"].tolist()))

    dataset = PRMDatasetWithTGTokenizer(
        txt_path="/mnt/data/tg/data/filter.ft",
        loc_path="/mnt/data/tg/data/filter.tell",
        pretrain=False,
        add_dst=True,
        replace_comma=True,
        ignore_context=True,
        split="train",
        tokenizer=tokenizer,
        max_num_workers=8,
        aux_path="/mnt/data/tg/data/aux.pkl",
        max_aux=5)

    data = dataset[0]
    print(data)
    words = []
    for token_id in data["input_ids"].tolist():
        words.append(tokenizer.index_to_word[token_id])
    print(repr(words))
    print(tokenizer.decode(data["input_ids"].tolist()))

    label_dist = {}
    for i in random.choices(range(len(dataset)), k=2048):
        label = dataset[i]["labels"]
        if label in label_dist:
            label_dist[label] += 1
        else:
            label_dist[label] = 1
    print(label_dist)


def test_prmhf_dataset():
    """Test PRM dataset"""
    tokenizer = AutoTokenizer.from_pretrained(
        "deepseek-ai/deepseek-coder-1.3b-instruct",
        model_max_length=512,
        padding_side="right",
        use_fast=True)

    dataset = PRMDatasetWithHFTokenizer(
        txt_path="/mnt/data/tg/data/filter.ft",
        loc_path="/mnt/data/tg/data/filter.tell",
        pretrain=True,
        add_dst=True,
        replace_comma=True,
        ignore_context=True,
        split="train",
        tokenizer=tokenizer,
        max_num_workers=8,
        aux_path="/mnt/data/tg/data/aux.pkl",
        max_aux=5)

    data = dataset[0]
    print(data)
    print(tokenizer.convert_ids_to_tokens(data["input_ids"].tolist()))
    print(tokenizer.decode(data["input_ids"].tolist()))

    dataset = PRMDatasetWithHFTokenizer(
        txt_path="/mnt/data/tg/data/filter.ft",
        loc_path="/mnt/data/tg/data/filter.tell",
        pretrain=False,
        add_dst=True,
        replace_comma=True,
        ignore_context=True,
        split="train",
        tokenizer=tokenizer,
        max_num_workers=8,
        aux_path="/mnt/data/tg/data/aux.pkl",
        max_aux=5)

    data = dataset[0]
    print(data)
    print(tokenizer.convert_ids_to_tokens(data["input_ids"].tolist()))
    print(tokenizer.decode(data["input_ids"].tolist()))

    label_dist = {}
    for i in random.choices(range(len(dataset)), k=2048):
        label = dataset[i]["labels"]
        if label in label_dist:
            label_dist[label] += 1
        else:
            label_dist[label] = 1
    print(label_dist)


def test_lmtg_dataloder_speed():
    """Test LM dataloader speed"""
    torch.manual_seed(123456)
    tokenizer = TongGeoTokenizer(vocab_file="./model/vocab.txt",
                                 special_tokens=SPECIAL_TOKENS,
                                 sep_token=SPECIAL_TOKENS["sep_token"],
                                 pad_token=SPECIAL_TOKENS["pad_token"],
                                 eos_token=SPECIAL_TOKENS["eos_token"],
                                 fact_token=SPECIAL_TOKENS["fact_token"],
                                 cls_token=SPECIAL_TOKENS["cls_token"],
                                 max_len=512,
                                 padding=False,
                                 return_tensor=True)

    dataset = LMDatasetWithTGTokenizer(
        txt_path="/mnt/data/tg/data/filter.ft",
        loc_path="/mnt/data/tg/data/filter.tell",
        pretrain=False,
        add_dst=True,
        replace_comma=True,
        ignore_context=True,
        split="train",
        tokenizer=tokenizer,
        max_num_workers=8)
    dataloader = DataLoader(
        dataset,
        num_workers=8,
        batch_size=32,
        shuffle=True,
        pin_memory=True,
        drop_last=True,
        worker_init_fn=worker_init_fn,
        collate_fn=DataCollatorForLM(pad_token_id=tokenizer.pad_token_id))

    counter = 0
    for batch in tqdm(dataloader):
        counter += 1
        if counter == 1:
            print(batch)
        if counter == 30000:
            break


def test_lmhf_dataloder_speed():
    """Test LM dataloader speed"""
    torch.manual_seed(123456)
    tokenizer = AutoTokenizer.from_pretrained(
        "deepseek-ai/deepseek-coder-1.3b-instruct",
        model_max_length=512,
        padding_side="right",
        use_fast=True)

    dataset = LMDatasetWithHFTokenizer(
        txt_path="/mnt/data/tg/data/filter.ft",
        loc_path="/mnt/data/tg/data/filter.tell",
        pretrain=False,
        add_dst=True,
        replace_comma=True,
        ignore_context=True,
        split="train",
        tokenizer=tokenizer,
        max_num_workers=8)
    dataloader = DataLoader(
        dataset,
        num_workers=8,
        batch_size=32,
        shuffle=True,
        pin_memory=True,
        drop_last=True,
        worker_init_fn=worker_init_fn,
        collate_fn=DataCollatorForLM(pad_token_id=tokenizer.pad_token_id))

    counter = 0
    for batch in tqdm(dataloader):
        counter += 1
        if counter == 1:
            print(batch)
        if counter == 30000:
            break


def test_lmtg_l_dataloder_speed():
    """Test LM dataloader speed"""
    torch.manual_seed(123456)
    tokenizer = TongGeoTokenizer(vocab_file="./model/vocab.txt",
                                 special_tokens=SPECIAL_TOKENS,
                                 sep_token=SPECIAL_TOKENS["sep_token"],
                                 pad_token=SPECIAL_TOKENS["pad_token"],
                                 eos_token=SPECIAL_TOKENS["eos_token"],
                                 fact_token=SPECIAL_TOKENS["fact_token"],
                                 cls_token=SPECIAL_TOKENS["cls_token"],
                                 max_len=512,
                                 padding=False,
                                 return_tensor=True)

    dataset = LMDatasetWithTGTokenizer(
        txt_path="/mnt/data/tg/data/filter.ft",
        loc_path="/mnt/data/tg/data/filter.tell",
        pretrain=False,
        add_dst=True,
        replace_comma=True,
        ignore_context=True,
        split="train",
        tokenizer=tokenizer,
        max_num_workers=8,
        aux_path="/mnt/data/tg/data/aux.pkl",
        max_aux=14)
    dataloader = DataLoader(
        dataset,
        num_workers=8,
        batch_size=32,
        shuffle=True,
        pin_memory=True,
        drop_last=True,
        worker_init_fn=worker_init_fn,
        collate_fn=DataCollatorForLM(pad_token_id=tokenizer.pad_token_id))

    counter = 0
    for batch in tqdm(dataloader):
        counter += 1
        if counter == 1:
            print(batch)
        if counter == 30000:
            break


def test_lmhf_l_dataloder_speed():
    """Test LM dataloader speed"""
    torch.manual_seed(123456)
    tokenizer = AutoTokenizer.from_pretrained(
        "deepseek-ai/deepseek-coder-1.3b-instruct",
        model_max_length=512,
        padding_side="right",
        use_fast=True)

    dataset = LMDatasetWithHFTokenizer(
        txt_path="/mnt/data/tg/data/filter.ft",
        loc_path="/mnt/data/tg/data/filter.tell",
        pretrain=False,
        add_dst=True,
        replace_comma=True,
        ignore_context=True,
        split="train",
        tokenizer=tokenizer,
        max_num_workers=8,
        aux_path="/mnt/data/tg/data/aux.pkl",
        max_aux=14)
    dataloader = DataLoader(
        dataset,
        num_workers=8,
        batch_size=32,
        shuffle=True,
        pin_memory=True,
        drop_last=True,
        worker_init_fn=worker_init_fn,
        collate_fn=DataCollatorForLM(pad_token_id=tokenizer.pad_token_id))

    counter = 0
    for batch in tqdm(dataloader):
        counter += 1
        if counter == 1:
            print(batch)
        if counter == 30000:
            break


def test_prmtg_dataloder_speed():
    """Test PRM dataloader speed"""
    torch.manual_seed(123456)
    tokenizer = TongGeoTokenizer(vocab_file="./model/vocab.txt",
                                 special_tokens=SPECIAL_TOKENS,
                                 sep_token=SPECIAL_TOKENS["sep_token"],
                                 pad_token=SPECIAL_TOKENS["pad_token"],
                                 eos_token=SPECIAL_TOKENS["eos_token"],
                                 fact_token=SPECIAL_TOKENS["fact_token"],
                                 cls_token=SPECIAL_TOKENS["cls_token"],
                                 max_len=512,
                                 padding=False,
                                 return_tensor=True)

    dataset = PRMDatasetWithTGTokenizer(
        txt_path="/mnt/data/tg/data/filter.ft",
        loc_path="/mnt/data/tg/data/filter.tell",
        pretrain=False,
        add_dst=True,
        replace_comma=True,
        ignore_context=True,
        split="train",
        tokenizer=tokenizer,
        max_num_workers=8,
        aux_path="/mnt/data/tg/data/aux.pkl",
        max_aux=5)
    dataloader = DataLoader(
        dataset,
        num_workers=8,
        batch_size=32,
        shuffle=True,
        pin_memory=True,
        drop_last=True,
        worker_init_fn=worker_init_fn,
        collate_fn=DataCollatorForPRM(pad_token_id=tokenizer.pad_token_id))

    counter = 0
    for batch in tqdm(dataloader):
        counter += 1
        if counter == 1:
            print(batch)
        if counter == 30000:
            break


def test_prmhf_dataloder_speed():
    """Test PRM dataloader speed"""
    torch.manual_seed(123456)
    tokenizer = AutoTokenizer.from_pretrained(
        "deepseek-ai/deepseek-coder-1.3b-instruct",
        model_max_length=512,
        padding_side="right",
        use_fast=True)

    dataset = PRMDatasetWithHFTokenizer(
        txt_path="/mnt/data/tg/data/filter.ft",
        loc_path="/mnt/data/tg/data/filter.tell",
        pretrain=False,
        add_dst=True,
        replace_comma=True,
        ignore_context=True,
        split="train",
        tokenizer=tokenizer,
        max_num_workers=8,
        aux_path="/mnt/data/tg/data/aux.pkl",
        max_aux=5)
    dataloader = DataLoader(
        dataset,
        num_workers=8,
        batch_size=32,
        shuffle=True,
        pin_memory=True,
        drop_last=True,
        worker_init_fn=worker_init_fn,
        collate_fn=DataCollatorForPRM(pad_token_id=tokenizer.pad_token_id))

    counter = 0
    for batch in tqdm(dataloader):
        counter += 1
        if counter == 1:
            print(batch)
        if counter == 30000:
            break


if __name__ == "__main__":
    test_tokenizer()
    test_lmtg_dataset()
    test_lmhf_dataset()
    test_lmtg_l_dataset()
    test_lmhf_l_dataset()
    test_prmtg_dataset()
    test_prmhf_dataset()
    test_lmtg_dataloder_speed()
    test_lmhf_dataloder_speed()
    test_lmtg_l_dataloder_speed()
    test_lmhf_l_dataloder_speed()
    test_prmtg_dataloder_speed()
    test_prmhf_dataloder_speed()
