r"""Inference process with neural models using beam search"""

import argparse
import multiprocessing
import os

import torch
from transformers import (AutoModelForCausalLM,
                          AutoModelForSequenceClassification, AutoTokenizer,
                          set_seed)

from tonggeometry.action import Action
from tonggeometry.constructor import (AllConstructors, BaseAcuteTriangle,
                                      ConstructorIndex)
from tonggeometry.diagram import Diagram
from tonggeometry.inference_engine.predicate import string_to_fact

CONSTRUCTOR_TO_LEN = {
    constructor.__name__: constructor.to_names_len
    for constructor in AllConstructors
}
CONSTRUCTOR_TO_LEN["BaseAcuteTriangle"] = 3


def process_one_path(lm_string,
                     tokenizer,
                     model_lm,
                     model_cls,
                     top_p,
                     temperature,
                     num_samples,
                     accumulation_steps,
                     base_steps=1):
    """Process one tree path"""
    inputs = tokenizer(lm_string, return_tensors="pt").to(model_lm.device)

    input_length = inputs.input_ids.shape[1]
    all_values = []
    all_steps_to_go = []
    all_generated_lines = []
    unique_lines = set()
    newline_token_id = tokenizer('\n').input_ids[1]  # auto added BOS

    for _ in range(accumulation_steps):
        outputs = model_lm.generate(**inputs,
                                    use_cache=True,
                                    max_length=512,
                                    do_sample=True,
                                    return_dict_in_generate=True,
                                    output_scores=True,
                                    top_p=top_p,
                                    temperature=temperature,
                                    num_return_sequences=num_samples,
                                    pad_token_id=tokenizer.eos_token_id)

        transition_scores = model_lm.compute_transition_scores(
            outputs.sequences, outputs.scores, normalize_logits=True)

        generated_tokens = outputs.sequences[:, input_length:]
        generated_lines = []
        values = []
        for idx in range(num_samples):
            sum_logprob = 0.0
            run_steps = 0
            for end, (tok, score) in enumerate(
                    zip(generated_tokens[idx], transition_scores[idx])):
                # token {tok:5d}
                # token string {tokenizer.decode(tok):8s}
                # logits {score.numpy():.4f}
                # probability {np.exp(score.numpy()):.2%}
                sum_logprob += score
                if tok == newline_token_id:
                    run_steps += 1
                    if run_steps == base_steps:
                        break
                if tok in [tokenizer.pad_token_id, tokenizer.eos_token_id]:
                    break
            if run_steps < base_steps:
                continue
            one_line = tokenizer.decode(generated_tokens[idx, :end + 1],
                                        skip_special_tokens=True)
            if len(one_line) <= 1 or one_line in unique_lines:
                continue
            unique_lines.add(one_line)
            generated_lines.append(one_line)
            values.append(sum_logprob / (end + 1))

        if len(generated_lines) == 0:
            all_values.append(torch.tensor([]).to(model_cls.device))
            all_steps_to_go.append(torch.tensor([]).to(model_cls.device))
            all_generated_lines += generated_lines
            continue

        values = torch.tensor(values).to(model_cls.device)
        inputs_cls = tokenizer([
            lm_string + one_line + "label = " for one_line in generated_lines
        ],
                               return_tensors="pt",
                               padding=True,
                               max_length=tokenizer.model_max_length,
                               truncation=True).to(model_cls.device)
        with torch.no_grad():
            cls_outputs = model_cls(**inputs_cls)
            logits = cls_outputs.logits
            steps_to_go = logits.argmax(dim=-1)

        all_values.append(values)
        all_steps_to_go.append(steps_to_go)
        all_generated_lines += generated_lines

    return torch.cat(all_values), torch.cat(
        all_steps_to_go), all_generated_lines


def take_one_step(d_action_id):
    """Take one step"""
    d, action, id = d_action_id
    return d.apply_action(action), id


def take_one_seq(d_actions_id):
    """Take one step"""
    d, actions, id = d_actions_id
    for action in actions:
        d = d.apply_action(action)
    return d, id


def string_to_action(text_action, sep=','):
    """Action repr to the action"""
    text_action = text_action.strip()
    if '(' not in text_action:
        return Action(BaseAcuteTriangle, '')
    if ')' not in text_action:
        return Action(BaseAcuteTriangle, '')
    left = text_action.index('(')
    right = text_action.index(')')
    constructor = text_action[:left]
    if constructor == "BaseAcuteTriangle":
        return Action(BaseAcuteTriangle, '')
    from_names = text_action[left + 1:right]
    from_names = ''.join(from_names.split(sep))
    if constructor in ConstructorIndex:
        return Action(AllConstructors[ConstructorIndex[constructor]],
                      from_names)
    return Action(BaseAcuteTriangle, '')


def format_lm(contents,
              replace_comma=True,
              add_dst=True,
              use_placeholder=False):
    """Format the input file lines"""
    alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
    delimiter = 0
    pt = 0
    ctx_parts = []
    for line in contents:
        if line == '\n':
            delimiter += 1
            continue
        if delimiter == 0:
            if replace_comma:
                line = line.replace(',', ", ")
            action_sep_idx = line.index('(')
            action_type = line[:action_sep_idx]
            if add_dst:
                to_names_len = CONSTRUCTOR_TO_LEN[action_type]
                if to_names_len > 0:
                    dst = ", ".join(alphabet[pt:pt + to_names_len]) + " = "
                    pt += to_names_len
                    ctx_parts.append(dst)
            ctx_parts.append(line)
        elif delimiter == 1:
            parts = line.strip().split(' ')
            fact_type = parts[0]
            fact_body = ' '.join(parts[1:])
    if use_placeholder:
        ctx_parts.append("# Prove fact()\n")
    else:
        ctx_parts.append(f"# Prove {fact_type}{fact_body}\n")  # pylint: disable=used-before-assignment
    return ''.join(ctx_parts)


def solve():
    """Entry point for distributed problem solve."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--problem",
                        type=str,
                        required=True,
                        help="path to the problem to be solved")
    parser.add_argument("--tokenizer",
                        type=str,
                        required=True,
                        help="path to the tokenizer")
    parser.add_argument("--lm-s",
                        type=str,
                        required=True,
                        help="path to the policy model")
    parser.add_argument("--lm-l",
                        type=str,
                        required=True,
                        help="path to the policy model")
    parser.add_argument("--cls",
                        type=str,
                        required=True,
                        help="path to the value model")
    parser.add_argument("--use-placeholder",
                        action="store_true",
                        help="whether to use placeholder for proving")
    parser.add_argument("--seed",
                        type=int,
                        default=42,
                        help="random seed for models")
    parser.add_argument("--beam-size",
                        type=int,
                        default=32,
                        help="beam size of the search")
    parser.add_argument("--num-samples",
                        type=int,
                        default=128,
                        help="number of generations for a state")
    parser.add_argument("--accumulation-steps",
                        type=int,
                        default=1,
                        help="accumulation steps to increase num samples")
    parser.add_argument("--max-iters",
                        type=int,
                        default=11,
                        help="maximum number of iterations for search")
    parser.add_argument("--top-p",
                        type=float,
                        default=0.95,
                        help="top-p value")
    parser.add_argument("--temperature",
                        type=float,
                        default=0.6,
                        help="sampling temperature")
    parser.add_argument("--weight",
                        type=float,
                        default=0.1,
                        help="weight of value")
    parser.add_argument("--decay",
                        type=float,
                        default=1.0,
                        help="decay of previous value")
    parser.add_argument("--short-range",
                        type=int,
                        default=4,
                        help="number of iters to run the short model")
    parser.add_argument("--reset-seed",
                        action="store_true",
                        help="whether to reset seed for the long model")
    args = parser.parse_args()

    set_seed(args.seed)

    os.environ["TOKENIZERS_PARALLELISM"] = "true"

    with open(args.problem, 'r', encoding="utf-8") as f:
        lines = f.readlines()

    lm_string = format_lm(lines, use_placeholder=args.use_placeholder)
    base_len = len(lm_string)
    print(f"Problem: {lm_string}")

    d = Diagram()
    all_actions = [string_to_action(line.strip()) for line in lines[:-2]]
    fact = string_to_fact(lines[-1].strip())

    for action in all_actions:
        d = d.apply_action(action)

    if fact in d.used_facts:
        print(f"{fact} already proved in the context. Problem solved.")
        return

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer,
                                              padding_side="right",
                                              use_fast=True)

    model_lm = AutoModelForCausalLM.from_pretrained(
        args.lm_s, torch_dtype=torch.bfloat16).cuda()

    model_cls = AutoModelForSequenceClassification.from_pretrained(
        args.cls, num_labels=5, torch_dtype=torch.bfloat16).cuda()
    model_cls.eval()

    iteration = 0

    # initial round
    print(f"{iteration}-th iter")
    values, steps_to_go, generated_lines = process_one_path(
        lm_string, tokenizer, model_lm, model_cls, args.top_p,
        args.temperature, args.num_samples, args.accumulation_steps)
    _, ids = (values - args.weight * steps_to_go).topk(
        k=min(len(values), args.beam_size))

    ids = ids.tolist()

    next_actions = []
    for one_line in generated_lines:
        if '=' in one_line:
            string_action = one_line.split('=')[1]
        else:
            string_action = one_line
        next_actions.append(string_to_action(string_action, ", "))

    new_ds = []
    success = False
    with multiprocessing.Pool(processes=multiprocessing.cpu_count()) as pool:
        for result in pool.imap_unordered(take_one_step,
                                          [(d, next_actions[id], id)
                                           for id in ids]):
            new_d, id = result
            if new_d.is_terminal:
                continue
            seq = lm_string + generated_lines[id]
            print(f"Tried:\n{seq[base_len:]}")
            new_ds.append((new_d, seq, values[id]))
            if fact in new_d.used_facts:
                success = True
                print("Success")
                print(new_d.actions[len(d.actions):])
                print(f"{fact} proved. Problem solved.")
    if success:
        return

    iteration += 1
    while iteration <= args.short_range and len(new_ds) > 0:
        print(f"{iteration}-th iter")
        ds = new_ds
        new_ds = []
        all_values = []
        all_steps_to_go = []
        all_generated_lines = []
        base = 0
        id_to_did = {}
        for d_id, (_, new_lm_string, base_value) in enumerate(ds):
            values, steps_to_go, generated_lines = process_one_path(
                new_lm_string, tokenizer, model_lm, model_cls, args.top_p,
                args.temperature, args.num_samples, args.accumulation_steps)
            for i in range(len(values)):
                id_to_did[base + i] = d_id
            base += len(values)
            all_values.append(values + args.decay * base_value)
            all_steps_to_go.append(steps_to_go)
            all_generated_lines += generated_lines
        values = torch.cat(all_values)
        steps_to_go = torch.cat(all_steps_to_go)
        generated_lines = all_generated_lines

        next_actions = []
        for one_line in generated_lines:
            if '=' in one_line:
                string_action = one_line.split('=')[1]
            else:
                string_action = one_line
            next_actions.append(string_to_action(string_action, ", "))

        _, ids = (values - args.weight * steps_to_go).topk(
            k=min(len(values), args.beam_size))

        ids = ids.tolist()

        success = False
        with multiprocessing.Pool(
                processes=multiprocessing.cpu_count()) as pool:
            for result in pool.imap_unordered(
                    take_one_step,
                [(ds[id_to_did[id]][0], next_actions[id], id) for id in ids]):
                new_d, id = result
                if new_d.is_terminal:
                    continue
                seq = ds[id_to_did[id]][1] + generated_lines[id]
                print(f"Tried:\n{seq[base_len:]}")
                new_ds.append((new_d, seq, values[id]))
                if fact in new_d.used_facts:
                    success = True
                    print("Success")
                    print(new_d.actions[len(d.actions):])
                    print(f"{fact} proved. Problem solved.")
        if success:
            return

        iteration += 1

    print("Proving fails using the short policy model.")
    if args.max_iters <= args.short_range:
        print("Process done")
        return

    del model_lm

    if args.reset_seed:
        set_seed(args.seed)

    model_lm = AutoModelForCausalLM.from_pretrained(
        args.lm_l, torch_dtype=torch.bfloat16).cuda()

    # long model initial run
    print(f"{iteration - 1}-th iter from the long model")
    values, steps_to_go, generated_lines = process_one_path(
        lm_string, tokenizer, model_lm, model_cls, args.top_p,
        args.temperature, args.num_samples, args.accumulation_steps,
        args.short_range + 1)
    _, ids = (values - args.weight * steps_to_go).topk(
        k=min(len(values), args.beam_size))

    ids = ids.tolist()

    next_actions = []
    for one_sequence in generated_lines:
        seq = []
        for one_line in one_sequence.strip().split('\n'):
            if '=' in one_line:
                string_action = one_line.split('=')[1]
            else:
                string_action = one_line
            seq.append(string_to_action(string_action, ", "))
        next_actions.append(seq)

    new_ds = []

    success = False
    with multiprocessing.Pool(processes=multiprocessing.cpu_count()) as pool:
        for result in pool.imap_unordered(take_one_seq,
                                          [(d, next_actions[id], id)
                                           for id in ids]):
            new_d, id = result
            if new_d.is_terminal:
                continue
            seq = lm_string + generated_lines[id]
            print(f"Tried:\n{seq[base_len:]}")
            new_ds.append((new_d, seq, values[id]))
            if fact in new_d.used_facts:
                print("Success")
                print(new_d.actions[len(d.actions):])
                print(f"{fact} proved. Problem solved.")
                success = True
    if success:
        return

    while iteration <= args.max_iters and len(new_ds) > 0:
        print(f"{iteration}-th iter")
        ds = new_ds
        new_ds = []
        all_values = []
        all_steps_to_go = []
        all_generated_lines = []
        base = 0
        id_to_did = {}
        for d_id, (_, new_lm_string, base_value) in enumerate(ds):
            values, steps_to_go, generated_lines = process_one_path(
                new_lm_string, tokenizer, model_lm, model_cls, args.top_p,
                args.temperature, args.num_samples, args.accumulation_steps)
            for i in range(len(values)):
                id_to_did[base + i] = d_id
            base += len(values)
            all_values.append(values + args.decay * base_value)
            all_steps_to_go.append(steps_to_go)
            all_generated_lines += generated_lines
        values = torch.cat(all_values)
        steps_to_go = torch.cat(all_steps_to_go)
        generated_lines = all_generated_lines

        next_actions = []
        for one_line in generated_lines:
            if '=' in one_line:
                string_action = one_line.split('=')[1]
            else:
                string_action = one_line
            next_actions.append(string_to_action(string_action, ", "))

        _, ids = (values - args.weight * steps_to_go).topk(
            k=min(len(values), args.beam_size))

        ids = ids.tolist()

        success = False
        with multiprocessing.Pool(
                processes=multiprocessing.cpu_count()) as pool:
            for result in pool.imap_unordered(
                    take_one_step,
                [(ds[id_to_did[id]][0], next_actions[id], id) for id in ids]):
                new_d, id = result
                if new_d.is_terminal:
                    continue
                seq = ds[id_to_did[id]][1] + generated_lines[id]
                print(f"Tried:\n{seq[base_len:]}")
                new_ds.append((new_d, seq, values[id]))
                if fact in new_d.used_facts:
                    print("Success")
                    print(new_d.actions[len(d.actions):])
                    print(f"{fact} proved. Problem solved.")
                    success = True
        if success:
            return

        iteration += 1

    print("Proving fails using the long policy model.")
    print("Process done")


if __name__ == "__main__":
    multiprocessing.set_start_method("forkserver")
    solve()
