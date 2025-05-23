# coding: utf-8
# Copyright 2019 Sinovation Ventures AI Institute
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""PyTorch pretrain for ZEN model."""

import os
import sys
import math

from argparse import ArgumentParser
from pathlib import Path
import torch
import logging
import json
import random
import numpy as np
from collections import namedtuple
import time
import datetime

from torch.utils.data import DataLoader, Dataset, RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ZEN import WEIGHTS_NAME, CONFIG_NAME
from ZEN import ZenConfig, ZenForPreTraining
from ZEN import BertTokenizer
from ZEN import BertAdam, WarmupLinearSchedule
from ZEN import ZenNgramDict

InputFeatures = namedtuple(
    "InputFeatures",
    "input_ids input_mask segment_ids lm_label_ids is_next ngram_ids ngram_masks ngram_positions ngram_starts ngram_lengths ngram_segment_ids")

log_format = '%(asctime)-10s: %(message)s'
logging.basicConfig(level=logging.INFO, format=log_format)

def convert_example_to_features(example, tokenizer, max_seq_length, max_ngram_in_sequence):
    tokens = example["tokens"]
    segment_ids = example["segment_ids"]
    is_random_next = example["is_random_next"]
    masked_lm_positions = example["masked_lm_positions"]
    masked_lm_labels = example["masked_lm_labels"]

    # add ngram level information
    ngram_ids = example["ngram_ids"]
    ngram_positions = example["ngram_positions"]
    ngram_lengths = example["ngram_lengths"]
    ngram_segment_ids = example["ngram_segment_ids"]

    assert len(tokens) == len(segment_ids) <= max_seq_length  # The preprocessed data should be already truncated
    input_ids = tokenizer.convert_tokens_to_ids(tokens)
    masked_label_ids = tokenizer.convert_tokens_to_ids(masked_lm_labels)

    input_array = np.zeros(max_seq_length, dtype=np.int64)
    input_array[:len(input_ids)] = input_ids

    mask_array = np.zeros(max_seq_length, dtype=bool)
    mask_array[:len(input_ids)] = 1

    segment_array = np.zeros(max_seq_length, dtype=bool)
    segment_array[:len(segment_ids)] = segment_ids

    lm_label_array = np.full(max_seq_length, dtype=np.int64, fill_value=-1)
    lm_label_array[masked_lm_positions] = masked_label_ids

    # add ngram pads
    ngram_id_array = np.zeros(max_ngram_in_sequence, dtype=np.int64)
    ngram_id_array[:len(ngram_ids)] = ngram_ids

    # record the masked positions

    # The matrix here take too much space either in disk or in memory, so the usage have to be lazily convert the
    # the start position and length to the matrix at training time.

    ngram_positions_matrix = np.zeros(shape=(max_seq_length, max_ngram_in_sequence), dtype=bool)
    for i in range(len(ngram_ids)):
        ngram_positions_matrix[ngram_positions[i]:ngram_positions[i]+ngram_lengths[i], i] = 1

    ngram_start_array = np.zeros(max_ngram_in_sequence, dtype=np.int32)
    ngram_start_array[:len(ngram_ids)] = ngram_positions

    ngram_length_array = np.zeros(max_ngram_in_sequence, dtype=np.int32)
    ngram_length_array[:len(ngram_ids)] = ngram_lengths

    ngram_mask_array = np.zeros(max_ngram_in_sequence, dtype=bool)
    ngram_mask_array[:len(ngram_ids)] = 1

    ngram_segment_array = np.zeros(max_ngram_in_sequence, dtype=bool)
    ngram_segment_array[:len(ngram_ids)] = ngram_segment_ids
    features = InputFeatures(input_ids=input_array,
                             input_mask=mask_array,
                             segment_ids=segment_array,
                             lm_label_ids=lm_label_array,
                             is_next=is_random_next,
                             ngram_ids=ngram_id_array,
                             ngram_masks=ngram_mask_array,
                             ngram_positions=ngram_positions_matrix,
                             ngram_starts=ngram_start_array,
                             ngram_lengths=ngram_length_array,
                             ngram_segment_ids=ngram_segment_array)
    return features


class PregeneratedDataset(Dataset):
    def __init__(self, training_path, epoch, tokenizer, num_data_epochs, reduce_memory=False, fp16=False):
        self.vocab = tokenizer.vocab
        self.tokenizer = tokenizer
        self.epoch = epoch
        self.data_epoch = epoch % num_data_epochs
        data_file = training_path / f"epoch_{self.data_epoch}.json"
        metrics_file = training_path / f"epoch_{self.data_epoch}_metrics.json"
        assert data_file.is_file() and metrics_file.is_file()
        metrics = json.loads(metrics_file.read_text())
        num_samples = metrics['num_training_examples']
        seq_len = metrics['max_seq_len']
        max_ngram_in_sequence = metrics['max_ngram_in_sequence']
        self.temp_dir = None
        self.working_dir = None
        self.fp16 = fp16
        if reduce_memory:
            self.temp_dir = "/tmp"
            # TemporaryDirectory()
            self.working_dir = Path(self.temp_dir)
            input_ids = np.memmap(filename=self.working_dir / 'input_ids.memmap',
                                  mode='w+', dtype=np.int32, shape=(num_samples, seq_len))
            input_masks = np.memmap(filename=self.working_dir / 'input_masks.memmap',
                                    shape=(num_samples, seq_len), mode='w+', dtype=bool)
            segment_ids = np.memmap(filename=self.working_dir / 'segment_ids.memmap',
                                    shape=(num_samples, seq_len), mode='w+', dtype=bool)
            lm_label_ids = np.memmap(filename=self.working_dir / 'lm_label_ids.memmap',
                                     shape=(num_samples, seq_len), mode='w+', dtype=np.int32)
            lm_label_ids[:] = -1
            is_nexts = np.memmap(filename=self.working_dir / 'is_nexts.memmap',
                                 shape=(num_samples,), mode='w+', dtype=bool)
            # add ngram level features
            ngram_ids = np.memmap(filename=self.working_dir / 'ngram_ids.memmap',
                                 mode='w+', dtype=np.int32, shape=(num_samples, max_ngram_in_sequence))

            ngram_masks = np.memmap(filename=self.working_dir / 'ngram_masks.memmap',
                                   mode='w+', dtype=bool, shape=(num_samples, max_ngram_in_sequence))

            ngram_positions = np.memmap(filename=self.working_dir / 'ngram_positions.memmap',
                                      mode='w+', dtype=bool, shape=(num_samples, seq_len, max_ngram_in_sequence))

            ngram_starts = np.memmap(filename=self.working_dir / 'ngram_starts.memmap',
                                    mode='w+', dtype=np.int32, shape=(num_samples, max_ngram_in_sequence))

            ngram_lengths = np.memmap(filename=self.working_dir / 'ngram_lengths.memmap',
                                     mode='w+', dtype=np.int32, shape=(num_samples, max_ngram_in_sequence))

            ngram_segment_ids = np.memmap(filename=self.working_dir / 'ngram_segment_ids.memmap',
                                         mode='w+', dtype=bool, shape=(num_samples, max_ngram_in_sequence))

        else:
            input_ids = np.zeros(shape=(num_samples, seq_len), dtype=np.int32)
            input_masks = np.zeros(shape=(num_samples, seq_len), dtype=bool)
            segment_ids = np.zeros(shape=(num_samples, seq_len), dtype=bool)
            lm_label_ids = np.full(shape=(num_samples, seq_len), dtype=np.int32, fill_value=-1)
            is_nexts = np.zeros(shape=(num_samples,), dtype=bool)
            # add ngram level features

            ngram_ids = np.zeros(shape=(num_samples, max_ngram_in_sequence), dtype=np.int32)
            ngram_masks = np.zeros(shape=(num_samples, max_ngram_in_sequence), dtype=bool)

            ngram_positions = np.zeros(shape=(num_samples, seq_len, max_ngram_in_sequence), dtype=bool)
            ngram_starts = np.zeros(shape=(num_samples, max_ngram_in_sequence), dtype=np.int32)
            ngram_lengths = np.zeros(shape=(num_samples, max_ngram_in_sequence), dtype=np.int32)

            ngram_segment_ids = np.zeros(shape=(num_samples, max_ngram_in_sequence), dtype=bool)

        logging.info(f"Loading training examples for epoch {epoch}")
        with data_file.open() as f:
            for i, line in enumerate(tqdm(f, total=num_samples, desc="Training examples")):
                line = line.strip()
                example = json.loads(line)
                features = convert_example_to_features(example, tokenizer, seq_len, max_ngram_in_sequence)
                input_ids[i] = features.input_ids
                segment_ids[i] = features.segment_ids
                input_masks[i] = features.input_mask
                lm_label_ids[i] = features.lm_label_ids
                is_nexts[i] = features.is_next
                # add ngram related ids
                ngram_ids[i] = features.ngram_ids
                ngram_masks[i] = features.ngram_masks
                ngram_positions[i] = features.ngram_positions
                ngram_starts[i] = features.ngram_starts
                ngram_lengths[i] = features.ngram_lengths
                ngram_segment_ids[i] = features.ngram_segment_ids

        assert i == num_samples - 1  # Assert that the sample count metric was true
        logging.info("Loading complete!")
        self.num_samples = num_samples
        self.seq_len = seq_len
        self.input_ids = input_ids
        self.input_masks = input_masks
        self.segment_ids = segment_ids
        self.lm_label_ids = lm_label_ids
        self.is_nexts = is_nexts
        self.ngram_ids = ngram_ids
        self.ngram_masks = ngram_masks
        self.ngram_positions = ngram_positions
        self.ngram_segment_ids = ngram_segment_ids
        self.ngram_starts = ngram_starts
        self.ngram_lengths = ngram_lengths

    def __len__(self):
        return self.num_samples

    def __getitem__(self, item):

        position = torch.tensor(self.ngram_positions[item].astype(np.double))
        if self.fp16:
            position = position.half()
        else:
            position = position.float()

        return (torch.tensor(self.input_ids[item].astype(np.int64)),
                torch.tensor(self.input_masks[item].astype(np.int64)),
                torch.tensor(self.segment_ids[item].astype(np.int64)),
                torch.tensor(self.lm_label_ids[item].astype(np.int64)),
                torch.tensor(self.is_nexts[item].astype(np.int64)),
                torch.tensor(self.ngram_ids[item].astype(np.int64)),
                torch.tensor(self.ngram_masks[item].astype(np.int64)),
                position,
                torch.tensor(self.ngram_starts[item].astype(np.int64)),
                torch.tensor(self.ngram_lengths[item].astype(np.int64)),
                torch.tensor(self.ngram_segment_ids[item].astype(np.int64)))


def main():
    parser = ArgumentParser()
    parser.add_argument('--pregenerated_data', type=Path, required=True)
    parser.add_argument('--output_dir', type=Path, required=True)
    parser.add_argument("--bert_model", type=str, required=True,
                        help="Bert pre-trained model selected in the list: bert-base-uncased, "
                             "bert-large-uncased, bert-base-cased, bert-base-multilingual, bert-base-chinese.")
    parser.add_argument("--do_lower_case", action="store_true")
    parser.add_argument("--reduce_memory", action="store_true",
                        help="Store training data as on-disc memmaps to massively reduce memory usage")

    parser.add_argument("--epochs", type=int, default=5, help="Number of epochs to train for")
    parser.add_argument("--local_rank",
                        type=int,
                        default=-1,
                        help="local_rank for distributed training on gpus")
    parser.add_argument("--no_cuda",
                        action='store_true',
                        help="Whether not to use CUDA when available")
    parser.add_argument('--gradient_accumulation_steps',
                        type=int,
                        default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument("--train_batch_size",
                        default=32,
                        type=int,
                        help="Total batch size for training.")
    parser.add_argument('--fp16',
                        action='store_true',
                        help="Whether to use 16-bit float precision instead of 32-bit")
    parser.add_argument('--scratch',
                        action='store_true',
                        help="Whether to train from scratch")
    parser.add_argument('--loss_scale',
                        type=float, default=0,
                        help="Loss scaling to improve fp16 numeric stability. Only used when fp16 set to True.\n"
                             "0 (default value): dynamic loss scaling.\n"
                             "Positive power of 2: static loss scaling value.\n")
    parser.add_argument("--warmup_proportion",
                        default=0.9,
                        type=float,
                        help="Proportion of training to perform linear learning rate warmup for. "
                             "E.g., 0.1 = 10%% of training.")
    parser.add_argument("--learning_rate",
                        default=3e-5,
                        type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument('--seed',
                        type=int,
                        default=42,
                        help="random seed for initialization")
    parser.add_argument('--save_name',
                        type=str,
                        default="zen",
                        help="The prefix used for saving the remote model")
    parser.add_argument('--method', 
                        type=str,
                        default="gate",
                        )
    parser.add_argument('--do_eval',
                        action='store_true',
                        help="Whether to run eval on the dev set.")
    parser.add_argument('--do_train',
                        action='store_true',
                        help="Whether to run eval on the dev set.")
    
    parser.add_argument("--already_trained_epoch",
                        default=0,
                        type=int)

    args = parser.parse_args()

    assert args.pregenerated_data.is_dir(), \
        "--pregenerated_data should point to the folder of files made by pregenerate_training_data.py!"


    samples_per_epoch = []
    for i in range(args.epochs):
        epoch_file = args.pregenerated_data / f"epoch_{i}.json"
        metrics_file = args.pregenerated_data / f"epoch_{i}_metrics.json"
        if epoch_file.is_file() and metrics_file.is_file():
            metrics = json.loads(metrics_file.read_text())
            samples_per_epoch.append(metrics['num_training_examples'])
        else:
            if i == 0:
                exit("No training data was found!")
            print(f"Warning! There are fewer epochs of pregenerated data ({i}) than training epochs ({args.epochs}).")
            print("This script will loop over the available data, but training diversity may be negatively impacted.")
            num_data_epochs = i
            break
    else:
        num_data_epochs = args.epochs

    if args.local_rank == -1 or args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        n_gpu = torch.cuda.device_count()
    else:
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        n_gpu = 1
        # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.distributed.init_process_group(backend='nccl')
    logging.info("device: {} n_gpu: {}, distributed training: {}, 16-bits training: {}".format(
        device, n_gpu, bool(args.local_rank != -1), args.fp16))

    if args.gradient_accumulation_steps < 1:
        raise ValueError("Invalid gradient_accumulation_steps parameter: {}, should be >= 1".format(
            args.gradient_accumulation_steps))

    args.train_batch_size = args.train_batch_size // args.gradient_accumulation_steps

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

    tokenizer = BertTokenizer.from_pretrained(args.bert_model, do_lower_case=args.do_lower_case)
    ngram_dict = ZenNgramDict(args.bert_model, tokenizer=tokenizer)

    total_train_examples = 0
    for i in range(args.epochs):
        # The modulo takes into account the fact that we may loop over limited epochs of data
        total_train_examples += samples_per_epoch[i % len(samples_per_epoch)]

    num_train_optimization_steps = int(
        total_train_examples / args.train_batch_size / args.gradient_accumulation_steps)
    if args.local_rank != -1:
        num_train_optimization_steps = num_train_optimization_steps // torch.distributed.get_world_size()

    if args.scratch:
        config = ZenConfig(21128, 104089)
        model = ZenForPreTraining(config)
    else:
        model = ZenForPreTraining.from_pretrained(args.bert_model)

    if args.fp16:
        model.half()
    model.to(device)
    if args.local_rank != -1:
        try:
            from apex.parallel import DistributedDataParallel as DDP # type: ignore
        except ImportError:
            raise ImportError(
                "Please install apex from https://www.github.com/nvidia/apex to use distributed and fp16 training.")
        model = DDP(model)
    elif n_gpu > 1:
        model = torch.nn.DataParallel(model)

    # Prepare optimizer
    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
         'weight_decay': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]

    if args.fp16:
        try:
            from apex.optimizers import FP16_Optimizer # type: ignore
            from apex.optimizers import FusedAdam # type: ignore
        except ImportError:
            raise ImportError(
                "Please install apex from https://www.github.com/nvidia/apex to use distributed and fp16 training.")

        optimizer = FusedAdam(optimizer_grouped_parameters,
                              lr=args.learning_rate,
                              bias_correction=False,
                              max_grad_norm=1.0)
        if args.loss_scale == 0:
            optimizer = FP16_Optimizer(optimizer, dynamic_loss_scale=True)
        else:
            optimizer = FP16_Optimizer(optimizer, static_loss_scale=args.loss_scale)
        warmup_linear = WarmupLinearSchedule(warmup=args.warmup_proportion,
                                             t_total=num_train_optimization_steps)
    else:
        optimizer = BertAdam(optimizer_grouped_parameters,
                             lr=args.learning_rate,
                             warmup=args.warmup_proportion,
                             t_total=num_train_optimization_steps)

    global_step = 0
    if args.do_train:
        logging.info("***** Running training *****")
        logging.info("  Num examples = %d", total_train_examples)
        logging.info("  Batch size = %d", args.train_batch_size)
        logging.info("  Num steps = %d", num_train_optimization_steps)
        model.train()
        for epoch in range(args.epochs - 1):

            epoch_dataset = PregeneratedDataset(epoch=epoch,
                                                training_path=args.pregenerated_data,
                                                tokenizer=tokenizer,
                                                num_data_epochs=num_data_epochs,
                                                reduce_memory=args.reduce_memory,
                                                fp16=args.fp16)
            if args.local_rank == -1:
                train_sampler = RandomSampler(epoch_dataset)
            else:
                train_sampler = DistributedSampler(epoch_dataset)
            train_dataloader = DataLoader(epoch_dataset, sampler=train_sampler, batch_size=args.train_batch_size)
            tr_loss = 0
            nb_tr_examples, nb_tr_steps = 0, 0
            with tqdm(total=len(train_dataloader), desc=f"Epoch {epoch}") as pbar:
                for step, batch in enumerate(train_dataloader):
                    batch = tuple(t.to(device) for t in batch)
                    input_ids, input_mask, segment_ids, lm_label_ids, is_next, ngram_ids, ngram_masks,  ngram_positions, \
                    ngram_starts, \
                    ngram_lengths, ngram_segment_ids = batch

                    loss = model(input_ids,
                                ngram_ids,
                                ngram_positions,
                                segment_ids,
                                ngram_segment_ids,
                                input_mask,
                                ngram_masks,
                                lm_label_ids,
                                is_next)

                    if n_gpu > 1:
                        loss = loss.mean()  # mean() to average on multi-gpu.
                    if args.gradient_accumulation_steps > 1:
                        loss = loss / args.gradient_accumulation_steps
                    if args.fp16:
                        optimizer.backward(loss)
                    else:
                        loss.backward()
                    tr_loss += loss.item()
                    nb_tr_examples += input_ids.size(0)
                    nb_tr_steps += 1
                    pbar.update(1)
                    mean_loss = tr_loss * args.gradient_accumulation_steps / nb_tr_steps
                    pbar.set_postfix_str(f"Loss: {mean_loss:.5f}")
                    if (step + 1) % args.gradient_accumulation_steps == 0:
                        if args.fp16:
                            # modify learning rate with special warm up BERT uses
                            # if args.fp16 is False, BertAdam is used that handles this automatically
                            lr_this_step = args.learning_rate * warmup_linear.get_lr(global_step, args.warmup_proportion)
                            for param_group in optimizer.param_groups:
                                param_group['lr'] = lr_this_step
                        optimizer.step()
                        optimizer.zero_grad()
                        global_step += 1

            # Save a trained model
            if epoch == (args.epochs-2):
                ts = time.time()
                st = datetime.datetime.fromtimestamp(ts).strftime('%m%d%H%M%S')

                saving_path = args.output_dir

                saving_path = Path(os.path.join(saving_path, args.save_name + st + "_epoch_" + str(epoch + args.already_trained_epoch)))

                if saving_path.is_dir() and list(saving_path.iterdir()):
                    logging.warning(f"Output directory ({ saving_path }) already exists and is not empty!")
                saving_path.mkdir(parents=True, exist_ok=True)

                logging.info("** ** * Saving fine-tuned model ** ** * ")
                model_to_save = model.module if hasattr(model, 'module') else model  # Only save the model it-self

                output_model_file = os.path.join(saving_path, WEIGHTS_NAME)
                output_config_file = os.path.join(saving_path, CONFIG_NAME)
                output_ngram_file = os.path.join(saving_path, "ngram.txt")

                torch.save(model_to_save.state_dict(), output_model_file)
                model_to_save.config.to_json_file(output_config_file)
                tokenizer.save_vocabulary(saving_path)

                ngram_dict.save(output_ngram_file)

    if args.do_eval:
        logging.info("***** Running evaluation *****")
        logging.info("  Eval batch size = %d", args.train_batch_size)

        eval_dataset = PregeneratedDataset(epoch=args.epochs - 1,
                                        training_path=args.pregenerated_data,
                                        tokenizer=tokenizer,
                                        num_data_epochs=1,
                                        reduce_memory=args.reduce_memory,
                                        fp16=args.fp16)

        eval_sampler = SequentialSampler(eval_dataset)
        eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=args.train_batch_size)

        model.eval()
        total_mlm_loss = 0.0
        total_nsp_loss = 0.0
        total_mlm_tokens = 0
        total_correct_mlm = 0
        total_nsp = 0
        total_correct_nsp = 0

        loss_fct = torch.nn.CrossEntropyLoss(ignore_index=-1)

        with tqdm(total=len(eval_dataloader), desc="Evaluating") as pbar:
            for step, batch in enumerate(eval_dataloader):
                batch = tuple(t.to(device) for t in batch)
                input_ids, input_mask, segment_ids, lm_label_ids, is_next, \
                ngram_ids, ngram_masks, ngram_positions, ngram_starts, ngram_lengths, ngram_segment_ids = batch

                with torch.no_grad():
                    sequence_output, pooled_output = model.bert(
                        input_ids=input_ids,
                        input_ngram_ids=ngram_ids,
                        ngram_position_matrix=ngram_positions,
                        token_type_ids=segment_ids,
                        ngram_token_type_ids=ngram_segment_ids,
                        attention_mask=input_mask,
                        ngram_attention_mask=ngram_masks,
                        output_all_encoded_layers=False
                    )

                    prediction_scores, seq_relationship_score = model.cls(sequence_output, pooled_output)

                    # MLM metrics
                    active = lm_label_ids != -1
                    mlm_loss = loss_fct(prediction_scores.view(-1, model.config.vocab_size), lm_label_ids.view(-1))
                    total_mlm_loss += mlm_loss.item() * active.sum().item()

                    mlm_preds = prediction_scores.argmax(dim=-1)
                    total_correct_mlm += (mlm_preds[active] == lm_label_ids[active]).sum().item()
                    total_mlm_tokens += active.sum().item()

                    # NSP metrics
                    nsp_loss = loss_fct(seq_relationship_score.view(-1, 2), is_next.view(-1))
                    total_nsp_loss += nsp_loss.item() * is_next.size(0)

                    nsp_preds = seq_relationship_score.argmax(dim=-1)
                    total_correct_nsp += (nsp_preds == is_next).sum().item()
                    total_nsp += is_next.size(0)

                    pbar.update(1)

        avg_mlm_loss = total_mlm_loss / total_mlm_tokens
        avg_nsp_loss = total_nsp_loss / total_nsp
        mlm_accuracy = total_correct_mlm / total_mlm_tokens
        nsp_accuracy = total_correct_nsp / total_nsp
        mlm_perplexity = math.exp(avg_mlm_loss)

        logging.info("***** Eval Results *****")
        logging.info("  MLM Loss: %.4f", avg_mlm_loss)
        logging.info("  MLM Accuracy: %.4f", mlm_accuracy)
        logging.info("  Perplexity: %.2f", mlm_perplexity)
        logging.info("  NSP Loss: %.4f", avg_nsp_loss)
        logging.info("  NSP Accuracy: %.4f", nsp_accuracy)

if __name__ == '__main__':
    main()
