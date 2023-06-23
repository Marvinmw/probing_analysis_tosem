import argparse
from global_logger import Log
import torch
import logging
import random
import numpy as np
from transformers import AutoTokenizer, AutoConfig
from utils.alignment import normalization_code
from models.classifier import FasterEdgeModel
from torch.utils.data import DataLoader, SequentialSampler, RandomSampler
from transformers import (AdamW, get_linear_schedule_with_warmup)
from tqdm import tqdm
import os
import json
from typing import Counter
from torch.utils.data import Dataset
import pickle
from multiprocessing.pool import Pool

cache_span = None
graph_data = None
code_features = None


class InputExample(object):
    def __init__(
        self,
        id,
        index,
        span_i,
        span_j,
        type,
        code,
        ast_node_type=None,
        cfg_node_type=None,
        cdg_node_type=None,
        ddg_tnode_ype=None
    ) -> None:
        # super(self).__init()
        self.id = id
        self.index = index
        # self.tokens = tokens
        self.span_i = span_i
        self.span_j = span_j
        self.spanarray= None
        self.mask=None
        #  self.name = name
        self.type = type
        self.code = code
        self.ast_node_type = ast_node_type
        self.cfg_node_type = cfg_node_type
        self.cdg_node_type = cdg_node_type
        self.ddg_node_type = ddg_tnode_ype


class SpanExample(object):
    def __init__(
        self,
        id,
        index,
        type,
        code,
        ast_node_type=None,
        cfg_node_type=None,
        cdg_node_type=None,
        ddg_tnode_ype=None
    ) -> None:
        # super(self).__init()
        self.id = id
        self.index = index

        self.type = type
        self.code = code
        self.ast_node_type = ast_node_type
        self.cfg_node_type = cfg_node_type
        self.cdg_node_type = cdg_node_type
        self.ddg_node_type = ddg_tnode_ype

Example=SpanExample
class PairSpanFeatures(object):
    """A single set of features of data."""
    def __init__(self, ex1: SpanExample, ex2: SpanExample, label_id):
        self.ex1 = ex1
        self.ex2 = ex2
        self.label_id = label_id


class InputPairSpanFeatures(object):
    """A single set of features of data."""
    def __init__(self, ex1: InputExample, ex2: InputExample, label_id):
        self.ex1 = ex1
        self.ex2 = ex2
        self.label_id = label_id


def create_example_from_file(example_file):
    global cache_span

    examples = pickle.load(open(example_file, "rb"))
    input_examples = []
    # graph data
    for ex in examples:
        index = ex.ex1.index
        assert ex.ex1.index == ex.ex2.index
        nodes_list = cache_span[index]
        e1 = ex.ex1.id
        e2 = ex.ex2.id
        n1_data = nodes_list[e1]
        n2_data = nodes_list[e2]
        ex1 = InputExample(
            e1, index, n1_data[0], n1_data[1], n1_data[4], n1_data[5]
        )  # (s_i, s_j, n["name"], variable_type[ty], node_type[nt],piece_code, v)
        ex2 = InputExample(
            e2, index, n2_data[0], n2_data[1], n2_data[4], n2_data[5]
        )
        inputexample = InputPairSpanFeatures(ex1, ex2, ex.label_id)
        input_examples.append(inputexample)
    input_examples=remove_overlapping(input_examples)
    random.shuffle(input_examples)
    # for e in input_examples:
    #     print(e.ex1.spanarray.shape)
    #     print(e.ex2.spanarray.shape)
    return input_examples


def imap_unordered_bar(func, args, n_processes=10):
    #print("start")
    p = Pool(n_processes)
    res_list = []
    with tqdm(total=len(args)) as pbar:
        for i, res in tqdm(enumerate(p.imap_unordered(func, args))):
            pbar.update()
            res_list.extend(res)
    pbar.close()
    p.close()
    # print("end")
    p.join()
    return res_list


def remove_overlapping(examples):
    new_examples = []
    for e in tqdm(examples):
        if e.ex1.span_j == e.ex2.span_j and e.ex1.span_i == e.ex2.span_i:
            continue
        if e.ex1.span_j < e.ex2.span_i:
            e.ex1.spanarray=[[e.ex1.span_i, e.ex1.span_j], [e.ex1.span_i, e.ex1.span_j]]
            e.ex1.mask = [1,0]
            e.ex2.spanarray= [[e.ex2.span_i, e.ex2.span_j], [e.ex2.span_i, e.ex2.span_j]]
            e.ex2.mask = [1,0]
            new_examples.append(e)
            continue
        if e.ex2.span_j < e.ex1.span_i:
            e.ex1.spanarray=[[e.ex1.span_i, e.ex1.span_j], [e.ex1.span_i, e.ex1.span_j]]
            e.ex1.mask = [1,0]
            e.ex2.spanarray= [[e.ex2.span_i, e.ex2.span_j], [e.ex2.span_i, e.ex2.span_j]]
            e.ex2.mask = [1,0]
            new_examples.append(e)
            continue
        e1_range = set([i for i in range(e.ex1.span_i, e.ex1.span_j + 1)])
        e2_range = set([j for j in range(e.ex2.span_i, e.ex2.span_j + 1)])
        overlapping_span = e1_range.intersection(e2_range)
        if len(overlapping_span) == 0:
            logger.info(f"({e.ex1.span_i}, {e.ex1.span_j}),  ({e.ex2.span_i}, {e.ex2.span_j}), {e.ex2}")
        omin,omax=min(overlapping_span), max(overlapping_span)

        if len(e1_range) > len(e2_range):
            if omin > e.ex1.span_i and omax <  e.ex1.span_j:
                e.ex1.spanarray=[[e.ex1.span_i, omin-1],[omax+1, e.ex1.span_j]]
                e.ex1.mask=[1, 1]
            else:
                e1_diff = list(e1_range.difference(e2_range))
                e.ex1.spanarray=[[min(e1_diff), max(e1_diff)], [min(e1_diff), max(e1_diff)]]
                e.ex1.mask=[1,0]
            e.ex2.spanarray=[[e.ex2.span_i, e.ex2.span_j], [e.ex2.span_i, e.ex2.span_j]]
            e.ex2.mask=[1,0]
        else:
            if omin > e.ex2.span_i and omax < e.ex2.span_j:
                e.ex2.spanarray = [[e.ex2.span_i, omin-1],[omax+1, e.ex2.span_j]]
                e.ex2.mask=[1,1]
            else:
                e2_diff = list(e2_range.difference(e1_range))
                e.ex2.spanarray=[[min(e2_diff), max(e2_diff)], [min(e2_diff), max(e2_diff)]]
                e.ex2.mask=[1,0]
            e.ex1.spanarray=[[e.ex1.span_i, e.ex1.span_j], [e.ex1.span_i, e.ex1.span_j]]
            e.ex1.mask=[1,0]
        new_examples.append(e)
    return new_examples


class CodeLinkDatasetFromFile(Dataset):
    def __init__(
        self,
        args,
        tokenizer,
        graph_type,
        dataset='train',
        crossing=True,
        source_code="java250/java250-graphs/codebert_java250-source-code.json",
        span_file="poj-104-graph/poj-104-test_token_span.json"
    ) -> None:
        global cache_span
        #  global graph_data
        self.examples = []
        self.args = args
        self.dataset = dataset
        index_filename = source_code
        self.code_cache = {}
        data_folder = args.data_folder

        code_cache_path = os.path.join(
            data_folder, f"{args.model_name}/{dataset}_code_cache.pt"
        )
        if not os.path.isfile(code_cache_path):
            with open(index_filename) as f_code:
                source_code_dic = json.load(f_code)
                for index_id, fcn_code in source_code_dic.items():
                    if args.model_name == "codebert":
                        indexmap, code_func = normalization_code(fcn_code)
                       # assert fcn_code == code_func, f"\n {fcn_code} \n {code_func}"
                    else:
                        code_func = fcn_code
                    source_tokens = tokenizer.tokenize(code_func
                                                      )[:args.max_code_length -
                                                        2]
                    source_tokens = [tokenizer.cls_token
                                    ] + source_tokens + [tokenizer.sep_token]
                    source_ids = tokenizer.convert_tokens_to_ids(source_tokens)
                    source_mask = [1] * (len(source_tokens))
                    padding_length = args.max_code_length - len(source_ids)
                    source_ids += [tokenizer.pad_token_id] * padding_length
                    source_mask += [0] * padding_length
                    self.code_cache[index_id] = (source_ids, source_mask)
            pickle.dump(self.code_cache, open(code_cache_path, "wb"))
        else:
            self.code_cache = pickle.load(open(code_cache_path, "rb"))

        #load data
        data_path = os.path.join(
            data_folder,
            f"{args.model_name}/{dataset}_{graph_type}_{crossing}.pt"
        )
        self.logger = Log.get_logger(
            name="link-prediction", logs_dir=args.output_dir
        )
        self.logger.info(
            "Load Span file and Graph data, data path: {}".format(data_path)
        )
        if not os.path.isfile(data_path):
            self.logger.info(
                "Preprocessing Span file and Graph data".format(data_path)
            )
            if os.path.isfile(span_file):
                cache_span = json.load(open(span_file))
            else:
                assert False, "generate the span file at first"

            train_examples = create_example_from_file(
                f"{data_folder}/train_{graph_type}_{crossing}.pt"
            )
            valid_examples = create_example_from_file(
                f"{data_folder}/valid_{graph_type}_{crossing}.pt"
            )
            test_examples = create_example_from_file(
                f"{data_folder}/test_{graph_type}_{crossing}.pt"
            )

            self.logger.info("save preprocessed data")
            pickle.dump(
                train_examples,
                open(
                    f"{data_folder}/{args.model_name}/{args.model_name}_train_{graph_type}_{crossing}.pt",
                    "wb"
                )
            )
            pickle.dump(
                valid_examples,
                open(
                    f"{data_folder}/{args.model_name}/{args.model_name}_valid_{graph_type}_{crossing}.pt",
                    "wb"
                )
            )
            pickle.dump(
                test_examples,
                open(
                    f"{data_folder}/{args.model_name}/{args.model_name}_test_{graph_type}_{crossing}.pt",
                    "wb"
                )
            )
            self.examples = pickle.load(open(data_path, "rb"))
        else:
            self.examples = pickle.load(open(data_path, "rb"))

        if args.debug:
            self.examples = self.examples[:1000]

        labels = [ex.label_id for ex in self.examples]
        self.logger.info(f"Labels {Counter(labels)}")
       

    def print_example(self, tokenizer):
        for idx, example in enumerate(self.examples[:3]):
            self.logger.info("*** Example ***")
            self.logger.info("idx: {}".format(example.ex1.index))
            self.logger.info("label: {}".format(example.label_id))
            self.logger.info(
                "input_tokens_1: {}".format(
                    [
                        x.replace('\u0120', '_')
                        for x in tokenizer.tokenize(example.ex1.code)
                    ]
                )
            )
            self.logger.info(
                "input_ids_1: {}".format(
                    ' '.join(
                        map(str, [example.ex1.span_i, example.ex1.span_j + 1])
                    )
                )
            )
            self.logger.info(
                "span_arry_1: {}, mask {}".format(
                    ' '.join(
                        map(str, example.ex1.spanarray)
                    ),
                    ' '.join(
                        map(str, example.ex1.mask)
                    )
                )
            )
            self.logger.info(
                "input_tokens_reverse_1: {}".format(
                    ' '.join(
                        tokenizer.convert_ids_to_tokens(
                            self.code_cache[example.ex1.index][0]
                            [example.ex1.span_i:example.ex1.span_j + 1]
                        )
                    )
                )
            )
            self.logger.info(
                "input_tokens_2: {}".format(
                    [
                        x.replace('\u0120', '_')
                        for x in tokenizer.tokenize(example.ex2.code)
                    ]
                )
            )
            self.logger.info(
                "input_ids_2: {}".format(
                    ' '.join(
                        map(str, [example.ex2.span_i, example.ex2.span_j + 1])
                    )
                )
            )
            self.logger.info(
                "span_arry_2: {}, mask {}".format(
                    ' '.join(
                        map(str, example.ex2.spanarray)
                    ),
                    ' '.join(
                        map(str, example.ex2.mask)
                    )
                )
            )
            self.logger.info(
                "input_tokens_reverse_2: {}".format(
                    ' '.join(
                        tokenizer.convert_ids_to_tokens(
                            self.code_cache[example.ex2.index][0]
                            [example.ex2.span_i:example.ex2.span_j + 1]
                        )
                    )
                )
            )

    def __len__(self):
        return len(self.examples)

   
    def __getitem__(self, item):
        #calculate graph-guided masked function
        global code_features
        index = self.examples[item].ex1.index
        # ex1_span, ex2_span, label, source_id, attention_mask
        #print(torch.tensor(self.examples[item].ex1.spanarray).shape)
        #print(torch.tensor(self.examples[item].ex2.spanarray).shape)
        #assert torch.tensor(self.examples[item].ex2.spanarray).shape == torch.tensor(self.examples[item].ex1.spanarray).shape, \
        #  f"{self.examples[item].ex1.spanarray} {self.examples[item].ex2.spanarray}"
        return (
            torch.tensor(
                        self.examples[item].ex1.spanarray
            ),
            torch.tensor(
                self.examples[item].ex1.mask
            ),
            torch.tensor(
                        self.examples[item].ex2.spanarray
 
            ), 
            torch.tensor(
                self.examples[item].ex2.mask
            ),
            torch.tensor(self.examples[item].label_id),
            code_features[index][self.args.layer].squeeze(0)
        )


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)


def train(args, train_dataset, eval_dataset, model):
    """ Train the model """

    #build dataloader
    train_sampler = RandomSampler(train_dataset)
    train_dataloader = DataLoader(
        train_dataset,
        sampler=train_sampler,
        batch_size=args.train_batch_size,
        num_workers=0
    )

    args.max_steps = args.epochs * len(train_dataloader)
    args.save_steps = max(1, len(train_dataloader) // 10)
    args.warmup_steps = args.max_steps // 5
    model.to(args.device)

    # Prepare optimizer and schedule (linear warmup and decay)
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {
            'params':
                [
                    p for n, p in model.named_parameters()
                    if not any(nd in n for nd in no_decay)
                ],
            'weight_decay': args.weight_decay
        }, {
            'params':
                [
                    p for n, p in model.named_parameters()
                    if any(nd in n for nd in no_decay)
                ],
            'weight_decay': 0.0
        }
    ]
    optimizer = AdamW(
        optimizer_grouped_parameters,
        lr=args.learning_rate,
        eps=args.adam_epsilon
    )
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=args.warmup_steps,
        num_training_steps=args.max_steps
    )

    # multi-gpu training
    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)

    # Train!
    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_dataset))
    logger.info("  Num Epochs = %d", args.epochs)
    logger.info(
        "  Instantaneous batch size per GPU = %d",
        args.train_batch_size // max(args.n_gpu, 1)
    )
    logger.info(
        "  Total train batch size = %d",
        args.train_batch_size * args.gradient_accumulation_steps
    )
    logger.info(
        "  Gradient Accumulation steps = %d", args.gradient_accumulation_steps
    )
    logger.info("  Total optimization steps = %d", args.max_steps)

    global_step = 0
    tr_loss, logging_loss, avg_loss, tr_nb, tr_num, train_loss = 0.0, 0.0, 0.0, 0, 0, 0
    best_f1 = 0
    best_mcc = 0
    model.zero_grad()

    for idx in range(args.epochs):
        bar = tqdm(train_dataloader, total=len(train_dataloader))
        tr_num = 0
        train_loss = 0
        for step, batch in enumerate(bar):
            (ex1_span, ex1_span_mask, ex2_span, ex2_span_mask, label, input_features) = [
                x.to(args.device) for x in batch
            ]
            model.train()
            loss, logits = model(input_features, ex1_span,ex1_span_mask, ex2_span,ex2_span_mask, label)

            if args.n_gpu > 1:
                loss = loss.mean()

            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps

            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), args.max_grad_norm
            )

            tr_loss += loss.item()
            tr_num += 1
            train_loss += loss.item()
            if avg_loss == 0:
                avg_loss = tr_loss

            avg_loss = round(train_loss / tr_num, 5)
            bar.set_description("epoch {} loss {}".format(idx, avg_loss))

            if (step + 1) % args.gradient_accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()
                scheduler.step()
                global_step += 1
                output_flag = True
                avg_loss = round(
                    np.exp((tr_loss - logging_loss) / (global_step - tr_nb)), 4
                )

                if global_step % args.save_steps == 0:
                    results = evaluate(
                        args, model, eval_dataset, eval_when_training=True
                    )

                    # Save model checkpoint
                    if results['eval_f1'] > best_f1:
                        best_f1 = results['eval_f1']
                        logger.info("  " + "*" * 20)
                        logger.info("  Best f1:%s", round(best_f1, 4))
                        logger.info("  " + "*" * 20)

                        checkpoint_prefix = 'checkpoint-best-f1'
                        output_dir = os.path.join(
                            args.output_dir, '{}'.format(checkpoint_prefix)
                        )
                        if not os.path.exists(output_dir):
                            os.makedirs(output_dir)
                        model_to_save = model.module if hasattr(
                            model, 'module'
                        ) else model
                        output_dir = os.path.join(
                            output_dir, '{}'.format('model.bin')
                        )
                        torch.save(model_to_save.state_dict(), output_dir)
                        logger.info("Saving model checkpoint to %s", output_dir)

                    # Save model checkpoint
                    if results['eval_mcc'] > best_mcc:
                        best_mcc = results['eval_mcc']
                        logger.info("  " + "*" * 20)
                        logger.info("  Best MCC:%s", round(best_mcc, 4))
                        logger.info("  " + "*" * 20)

                        checkpoint_prefix = 'checkpoint-best-mcc'
                        output_dir = os.path.join(
                            args.output_dir, '{}'.format(checkpoint_prefix)
                        )
                        if not os.path.exists(output_dir):
                            os.makedirs(output_dir)
                        model_to_save = model.module if hasattr(
                            model, 'module'
                        ) else model
                        output_dir = os.path.join(
                            output_dir, '{}'.format('model.bin')
                        )
                        torch.save(model_to_save.state_dict(), output_dir)
                        logger.info("Saving model checkpoint to %s", output_dir)


def evaluate(args, model, eval_dataset, eval_when_training=False):
    #build dataloader
    # eval_dataset = TextDataset(args, examples)
    eval_sampler = SequentialSampler(eval_dataset)
    eval_dataloader = DataLoader(
        eval_dataset,
        sampler=eval_sampler,
        batch_size=args.eval_batch_size,
        num_workers=0
    )

    # multi-gpu evaluate
    if args.n_gpu > 1 and eval_when_training is False:
        model = torch.nn.DataParallel(model)

    # Eval!
    logger.info("***** Running evaluation *****")
    logger.info("  Num examples = %d", len(eval_dataset))
    logger.info("  Batch size = %d", args.eval_batch_size)

    eval_loss = 0.0
    nb_eval_steps = 0
    model.eval()
    logits = []
    y_trues = []
    bar = tqdm(eval_dataloader, total=len(eval_dataloader))
    for step, batch in enumerate(bar):
        (ex1_span, ex1_span_mask, ex2_span, ex2_span_mask,labels, input_features) = [
            x.to(args.device) for x in batch
        ]
        with torch.no_grad():
            lm_loss, logit = model(input_features, ex1_span, ex1_span_mask,ex2_span, ex2_span_mask,labels)
            eval_loss += lm_loss.mean().item()
            logits.append(logit.cpu().numpy())
            y_trues.append(labels.cpu().numpy())
        nb_eval_steps += 1

    #calculate scores
    logits = np.concatenate(logits, 0)
    y_trues = np.concatenate(y_trues, 0)
    best_threshold = 0.5
    best_f1 = 0

    y_preds = logits[:, 1] > best_threshold
    from sklearn.metrics import recall_score
    recall = recall_score(y_trues, y_preds)
    from sklearn.metrics import precision_score
    precision = precision_score(y_trues, y_preds)
    from sklearn.metrics import f1_score
    f1 = f1_score(y_trues, y_preds)
    from sklearn.metrics import matthews_corrcoef
    mcc = matthews_corrcoef(y_trues, y_preds)
    from sklearn.metrics import accuracy_score
    acc = accuracy_score(y_trues, y_preds)
    result = {
        "eval_recall": float(recall),
        "eval_precision": float(precision),
        "eval_f1": float(f1),
        "eval_mcc": float(mcc),
        "eval_acc": float(acc),
        "eval_threshold": best_threshold,
    }

    logger.info("***** Eval results *****")
    for key in sorted(result.keys()):
        logger.info("  %s = %s", key, str(round(result[key], 4)))

    return result


def test(args, model, eval_dataset, best_threshold=0):
    #build dataloader
    # eval_dataset = TextDataset( args,examples)
    eval_sampler = SequentialSampler(eval_dataset)
    eval_dataloader = DataLoader(
        eval_dataset,
        sampler=eval_sampler,
        batch_size=args.eval_batch_size,
        num_workers=0
    )

    # multi-gpu evaluate
    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)

    # Eval!
    logger.info("***** Running Test *****")
    logger.info("  Num examples = %d", len(eval_dataset))
    logger.info("  Batch size = %d", args.eval_batch_size)
    eval_loss = 0.0
    nb_eval_steps = 0
    model.eval()
    logits = []
    y_trues = []
    for batch in eval_dataloader:
        (ex1_span,  ex1_span_mask, ex2_span, ex2_span_mask, labels, input_features) = [
            x.to(args.device) for x in batch
        ]
        with torch.no_grad():
            lm_loss, logit = model(input_features, ex1_span, ex1_span_mask,ex2_span, ex2_span_mask,labels)
            eval_loss += lm_loss.mean().item()
            logits.append(logit.cpu().numpy())
            y_trues.append(labels.cpu().numpy())
        nb_eval_steps += 1

    #output result
    logits = np.concatenate(logits, 0)
    y_trues = np.concatenate(y_trues, 0)
    y_preds = logits[:, 1] > best_threshold

    from sklearn.metrics import recall_score
    recall = recall_score(y_trues, y_preds)
    from sklearn.metrics import precision_score
    precision = precision_score(y_trues, y_preds)
    from sklearn.metrics import f1_score
    f1 = f1_score(y_trues, y_preds)
    from sklearn.metrics import matthews_corrcoef
    mcc = matthews_corrcoef(y_trues, y_preds)
    from sklearn.metrics import accuracy_score
    acc = accuracy_score(y_trues, y_preds)
    result = {
        "eval_recall": float(recall),
        "eval_precision": float(precision),
        "eval_f1": float(f1),
        "eval_mcc": float(mcc),
        "eval_acc": float(acc),
        "eval_threshold": best_threshold,
    }

    logger.info("***** Test results *****")
    for key in sorted(result.keys()):
        logger.info("  %s = %s", key, str(round(result[key], 4)))
    json.dump(
        result,
        open(
            os.path.join(args.output_dir, "predictions_performance_test.json"),
            "w"
        ),
        indent=4
    )
    with open(os.path.join(args.output_dir, "predictions.txt"), 'w') as f:
        for example, pred in zip(eval_dataset.examples, y_preds):
            if pred:
                f.write(str(example.label_id) + '\t' + '1' + '\n')
            else:
                f.write(str(example.label_id) + '\t' + '0' + '\n')


if __name__ == "__main__":
    import os
    os.environ["TOKENIZERS_PARALLELISM"] = "true"

    parser = argparse.ArgumentParser(description='Process some integers.')
    ## Required parameters
    parser.add_argument(
        "--dataset",
        default="",
        type=str,  #required=True,
        help="The input training data file (a text file)."
    )
    parser.add_argument(
        "--output_dir",
        default="poj-train-model-saved",
        type=str,  #required=True,
        help=
        "The output directory where the model predictions and checkpoints will be written."
    )
    parser.add_argument(
        "--data_folder",
        default="dataset/exp1",
        type=str,  #required=True,
        help=
        "The output directory where the model predictions and checkpoints will be written."
    )

    ## Dataset Config
    parser.add_argument(
        '--max_code_length',
        dest='max_code_length',
        type=int,
        default=512,
        help='input maximum length'
    )
    parser.add_argument(
        '--posratio',
        dest='posratio',
        type=float,
        default=0.5,
        help='input maximum length'
    )
    parser.add_argument(
        '--graph_type',
        dest='graph_type',
        type=str,
        default="ast",
        help='tokenizer config'
    )

    ## Pretrain model config
    parser.add_argument(
        "--model_name_or_path",
        default="microsoft/codebert-base",
        type=str,
        help="The model checkpoint for weights initialization."
    )
    parser.add_argument(
        "--model_name", default="codebert", type=str, help="The model name."
    )
    parser.add_argument(
        "--config_name",
        default="microsoft/codebert-base",
        type=str,
        help=
        "Optional pretrained config name or path if not the same as model_name_or_path"
    )
    parser.add_argument(
        '--token_config',
        dest='token_config',
        type=str,
        default="microsoft/codebert-base",
        help='tokenizer config'
    )

    parser.add_argument(
        "--do_train", action='store_true', help="Whether to run training."
    )
    parser.add_argument(
        "--do_random",
        action='store_true',
        help="Whether to use random features."
    )
    parser.add_argument(
        "--debug",
        action='store_true',
        help="debug model with the samll dataset"
    )
    parser.add_argument(
        "--do_crossing",
        action='store_true',
        help="crossing project train and test"
    )
    parser.add_argument(
        "--do_eval",
        action='store_true',
        help="Whether to run eval on the dev set."
    )
    parser.add_argument(
        "--do_test",
        action='store_true',
        help="Whether to run eval on the dev set."
    )
    parser.add_argument(
        "--evaluate_during_training",
        action='store_true',
        help="Run evaluation during training at each logging step."
    )
    #parser.add_argument("--span_rep_dim", default=100, type = int, help="dimension of span tokens")
    parser.add_argument(
        "--layer", default=-1, type=int, help="use which layer for probing "
    )
    parser.add_argument(
        "--train_batch_size",
        default=64,
        type=int,
        help="Batch size per GPU/CPU for training."
    )
    parser.add_argument(
        "--eval_batch_size",
        default=64,
        type=int,
        help="Batch size per GPU/CPU for evaluation."
    )
    parser.add_argument(
        "--class_num", default=2, type=int, help="number of labels"
    )
    parser.add_argument(
        '--gradient_accumulation_steps',
        type=int,
        default=1,
        help=
        "Number of updates steps to accumulate before performing a backward/update pass."
    )
    parser.add_argument(
        "--learning_rate",
        default=1e-5,
        type=float,
        help="The initial learning rate for Adam."
    )
    parser.add_argument(
        "--weight_decay",
        default=0.0,
        type=float,
        help="Weight deay if we apply some."
    )
    parser.add_argument(
        "--adam_epsilon",
        default=1e-8,
        type=float,
        help="Epsilon for Adam optimizer."
    )
    parser.add_argument(
        "--max_grad_norm", default=1.0, type=float, help="Max gradient norm."
    )
    parser.add_argument(
        "--max_steps",
        default=-1,
        type=int,
        help=
        "If > 0: set total number of training steps to perform. Override num_train_epochs."
    )
    parser.add_argument(
        "--warmup_steps",
        default=0,
        type=int,
        help="Linear warmup over warmup_steps."
    )
    parser.add_argument(
        '--seed', type=int, default=42, help="random seed for initialization"
    )
    parser.add_argument(
        '--epochs', type=int, default=10, help="training epochs"
    )

    args = parser.parse_args()
    ## Dataset
    assert args.dataset in ["java250", "poj-104"]
    if args.dataset == "java250":
        normalized_source_code_file = f"../datasets/java250/java250-graphs/{args.model_name}_java250-source-code.json"
        span_file = f"../datasets/java250/java250-graphs/java250_token_span_{args.model_name}.json"
    if args.dataset == "poj-104":
        normalized_source_code_file = "../datasets/poj-104/poj-104/test/formated_source_code_clang.jsonl"
        res = {}
        with open(normalized_source_code_file) as f:
            lines = f.readlines()
            for l in lines:
                l_code = json.loads(l)
                index = l_code["index"]
                code = l_code["code"]
                res[index] = code
        json.dump(res, open("../datasets/poj-104/poj-104//test/formated_source_code_clang.json", "w"))
        normalized_source_code_file = "../datasets/poj-104/poj-104//test/formated_source_code_clang.json"
        span_file=f"../datasets/poj-104/poj-104-graph/poj-104-test_token_span_{args.model_name}.json"

    # Setup CUDA, GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.n_gpu = torch.cuda.device_count()

    args.device = device

    # Setup logging
    logger = Log.get_logger(name="link-prediction", logs_dir=args.output_dir)
    logger.debug('main message')
    logging.basicConfig(
        format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
        datefmt='%m/%d/%Y %H:%M:%S',
        level=logging.INFO
    )
    logger.warning(
        "device: %s, n_gpu: %s",
        device,
        args.n_gpu,
    )

    # Set seed
    set_seed(args)
    config = AutoConfig.from_pretrained(
        args.config_name if args.config_name else args.model_name_or_path,
        return_dict=False,
        output_hidden_states=True
    )
    tokenizer = AutoTokenizer.from_pretrained(args.token_config)
    # create dataset
    logger.info("Load Dataset")
    code_featrues_file = os.path.join(
        args.data_folder,
        f"{args.model_name}/input_encoder_hidden_states_random_{args.do_random}.pt"
    )
    code_features = pickle.load(open(code_featrues_file, "rb"))
    # using cleaned data
    args.do_crossing = "cleaned"
    dataset_train = CodeLinkDatasetFromFile(
        args,
        tokenizer,
        args.graph_type,
        dataset=f'{args.model_name}_train',
        crossing=args.do_crossing,
        source_code=normalized_source_code_file,
        span_file=span_file
    )
    dataset_valid = CodeLinkDatasetFromFile(
        args,
        tokenizer,
        args.graph_type,
        dataset=f'{args.model_name}_valid',
        crossing=args.do_crossing,
        source_code=normalized_source_code_file,
        span_file=span_file
    )
    dataset_test = CodeLinkDatasetFromFile(
        args,
        tokenizer,
        args.graph_type,
        dataset=f'{args.model_name}_test',
        crossing=args.do_crossing,
        source_code=normalized_source_code_file,
        span_file=span_file
    )

    dataset_train.print_example(tokenizer)
    dataset_valid.print_example(tokenizer)
    dataset_test.print_example(tokenizer)

    logger.info("Finishing loading Dataset")

    #create model
    model = FasterEdgeModel(config, args)
    num_params_model = sum(
        p.numel() for p in model.parameters() if p.requires_grad
    )

    # show input shape
    logger.info(f"Probimg Model Parameters {num_params_model}")
    logger.info("Training/evaluation parameters %s", args)
    # Training
    if args.do_train:
        train(args, dataset_train, dataset_valid, model)

    # Evaluation
    # results = {}
    if args.do_eval:
        checkpoint_prefix = 'checkpoint-best-f1/model.bin'
        output_dir = os.path.join(
            args.output_dir, '{}'.format(checkpoint_prefix)
        )
        model.load_state_dict(torch.load(output_dir))
        model.to(args.device)
        results = evaluate(args, model, dataset_valid)

    if args.do_test:
        checkpoint_prefix = 'checkpoint-best-mcc/model.bin'
        output_dir = os.path.join(
            args.output_dir, '{}'.format(checkpoint_prefix)
        )
        model.load_state_dict(torch.load(output_dir))
        model.to(args.device)
        test(args, model, dataset_test, best_threshold=0.5)

    # return results
