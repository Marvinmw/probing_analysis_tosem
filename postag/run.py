import warnings
warnings.filterwarnings('ignore')
from typing import List,Any
from dataclasses import dataclass
from torch.utils.data import Dataset, SequentialSampler, RandomSampler
import itertools
import os
import torch
from models import BERTPoSTagger
import argparse
from global_logger import Log
import torch
import logging
import random
import numpy as np
from transformers import AutoTokenizer, AutoModel, AutoConfig
from transformers import (AdamW, get_linear_schedule_with_warmup)
from tqdm import tqdm
import json
from torch.utils.data import Dataset, DataLoader


def align_tokens_and_annotations_bilou(tokenized, annotations, frequent_labels):
    tokens = tokenized.tokens()
    aligned_labels = ["O"] * len(
        tokens
    )  # Make a list to store our labels the same length as our tokens
    for anno in annotations:
        if anno["label2"] not in frequent_labels:
            continue
        annotation_token_ix_set = (
            set()
        )  # A set that stores the token indices of the annotation
        for char_ix in range(anno["start"], anno["end"]+1):

            token_ix = tokenized.char_to_token(char_ix)
            if token_ix is not None:
                annotation_token_ix_set.add(token_ix)
        if len(annotation_token_ix_set) == 1:
            # If there is only one token
            token_ix = annotation_token_ix_set.pop()
            prefix = (
                "U"  # This annotation spans one token so is prefixed with U for unique
            )
            aligned_labels[token_ix] = f"{prefix}-{anno['label2']}"

        else:

            last_token_in_anno_ix = len(annotation_token_ix_set) - 1
            for num, token_ix in enumerate(sorted(annotation_token_ix_set)):
                if num == 0:
                    prefix = "B"
                elif num == last_token_in_anno_ix:
                    prefix = "L"  # Its the last token
                else:
                    prefix = "I"  # We're inside of a multi token annotation
                aligned_labels[token_ix] = f"{prefix}-{anno['label2']}"
    return aligned_labels

class LabelSet:
    def __init__(self, labels: List[str]):
        self.labels_to_id = {}
        self.labels = labels
        self.ids_to_label = {}
        self.labels_to_id["O"] = 0
        self.ids_to_label[0] = "O"
        num = 0  # in case there are no labels
        # Writing BILU will give us incremntal ids for the labels
        for _num, (label, s) in enumerate(itertools.product(labels, "BILU")):
            num = _num + 1  # skip 0
            l = f"{s}-{label}"
            self.labels_to_id[l] = num
            self.ids_to_label[num] = l
        # Add the OUTSIDE label - no label for the token

    def get_aligned_label_ids_from_annotations(self, tokenized_text, annotations):
        raw_labels = align_tokens_and_annotations_bilou(tokenized_text, annotations, self.labels)
        return list(map(self.labels_to_id.get, raw_labels))

IntList = List[int] # A list of token_ids
IntListList = List[IntList] # A List of List of token_ids, e.g. a Batch

@dataclass
class TrainingExample:
    source_ids: IntList
    source_masks: IntList
    labels: IntList


class PosTagDataset(Dataset):

    def __init__(
        self,
        args,
        label_set: LabelSet,
        name,
        tokenizer
    ):
        self.label_set = label_set
        self.examples =[]
        self.tokenizer = tokenizer
        data_file = os.path.join(args.data_folder, f"{name}_cache.pt")
        raw_file = os.path.join(args.data_folder, f"{name}_pos_tag_True.json")
        #if not os.path.isfile( data_file ):
        data = json.load(open(raw_file))
        for [code, annonations, pid] in data:
            # changes tag key to label
            tokenized_code = self.tokenizer(code,  add_special_tokens=False)
            aligned_label_ids = self.label_set.get_aligned_label_ids_from_annotations(tokenized_code, annonations)
            source_tokens = tokenized_code.tokens()[:args.max_code_length-2]
            aligned_label_ids = aligned_label_ids[:args.max_code_length-2]
            source_tokens = [tokenizer.cls_token] + source_tokens + [tokenizer.sep_token]
            aligned_label_ids = [label_set.labels_to_id["O"]]+aligned_label_ids + [label_set.labels_to_id["O"]]
            source_ids = tokenizer.convert_tokens_to_ids(source_tokens)
            source_mask = [1] * (len(source_tokens))
            padding_length = args.max_code_length - len(source_ids)
            source_ids += [tokenizer.pad_token_id] * padding_length
            source_mask += [0] * padding_length
            aligned_label_ids += [label_set.labels_to_id["O"]]* padding_length
            ex = TrainingExample(
                    source_ids=source_ids,
                    source_masks=source_mask,
                    labels=aligned_label_ids
                )
            self.examples.append(ex)
        torch.save(self.examples, data_file)
        #else:
        #    self.examples = torch.load(open(data_file, "rb"))
        
    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx) -> TrainingExample:
        return ( torch.tensor(self.examples[idx].source_ids), 
                 torch.tensor(self.examples[idx].source_masks), 
                 torch.tensor(self.examples[idx].labels)
                )



def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)



def categorical_accuracy(preds, y, tag_pad_idx):
    """
    Returns accuracy per batch, i.e. if you get 8/10 right, this returns 0.8, NOT 8
    """
    max_preds = preds.argmax(dim = 1, keepdim = True) # get the index of the max probability
    non_pad_elements = (y != tag_pad_idx).nonzero()
    correct = max_preds[non_pad_elements].squeeze(1).eq(y[non_pad_elements])
    return correct.sum() / torch.FloatTensor([y[non_pad_elements].shape[0]]).to(device)

def convert_labels_shape(preds, y, tag_pad_idx):
    max_preds = preds.argmax(dim = 1, keepdim = True) # get the index of the max probability
    non_pad_elements = (y != tag_pad_idx).nonzero()
    return max_preds[non_pad_elements].squeeze(1), y[non_pad_elements]

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
    logger.info(f"save steps {args.save_steps}")
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
    criterion = torch.nn.CrossEntropyLoss(ignore_index = 0 )
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
            (source_ids, source_masks, labels ) = [
                x.to(args.device) for x in batch
            ]
            model.train()
            predictions = model(source_ids, source_masks)
            predictions = predictions.view(-1, predictions.shape[-1])
            labels = labels.view(-1)

            loss = criterion(predictions, labels)       
            #acc = categorical_accuracy(predictions, labels, 0)

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
    criterion = torch.nn.CrossEntropyLoss(ignore_index = 0 )
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
    y_preds = []
    y_trues = []
    bar = tqdm(eval_dataloader, total=len(eval_dataloader))
    for step, batch in enumerate(bar):
        (source_ids, source_masks, labels ) = [
                x.to(args.device) for x in batch
            ]
        with torch.no_grad():
            predictions = model(source_ids, source_masks)
            predictions = predictions.view(-1, predictions.shape[-1])
            labels = labels.view(-1)
            
            lm_loss = criterion(predictions, labels)
            pre_labels, labels = convert_labels_shape(predictions, labels, 0)
            eval_loss += lm_loss.mean().item()
            y_preds.append(pre_labels.cpu().numpy())
            y_trues.append(labels.cpu().numpy())
        nb_eval_steps += 1

    #calculate scores
    y_preds = np.concatenate(y_preds, 0)
    y_trues = np.concatenate(y_trues, 0)
    
    from sklearn.metrics import recall_score
    recall = recall_score(y_trues, y_preds, average="macro")
    from sklearn.metrics import precision_score
    precision = precision_score(y_trues, y_preds, average="macro")
    from sklearn.metrics import f1_score
    f1 = f1_score(y_trues, y_preds, average="macro")
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
    }

    logger.info("***** Eval results *****")
    for key in sorted(result.keys()):
        logger.info("  %s = %s", key, str(round(result[key], 4)))

    return result


def test(args, model, eval_dataset):
    #build dataloader
    # eval_dataset = TextDataset( args,examples)
    eval_sampler = SequentialSampler(eval_dataset)
    eval_dataloader = DataLoader(
        eval_dataset,
        sampler=eval_sampler,
        batch_size=args.eval_batch_size,
        num_workers=0
    )
    criterion = torch.nn.CrossEntropyLoss( ignore_index = 0 )
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
    y_preds = []
    y_trues = []

    bar = tqdm(eval_dataloader, total=len(eval_dataloader))
    for step, batch in enumerate(bar):
        (source_ids, source_masks, labels ) = [
                x.to(args.device) for x in batch
            ]
        with torch.no_grad():
            predictions = model(source_ids, source_masks)
            predictions = predictions.view(-1, predictions.shape[-1])
            labels = labels.view(-1)
            
            lm_loss = criterion(predictions, labels)
            pre_labels, labels = convert_labels_shape(predictions, labels, 0)
            eval_loss += lm_loss.mean().item()
            y_preds.append(pre_labels.cpu().numpy())
            y_trues.append(labels.cpu().numpy())
        nb_eval_steps += 1

    #calculate scores
    y_preds = np.concatenate(y_preds, 0)
    y_trues = np.concatenate(y_trues, 0)


    from sklearn.metrics import recall_score
    recall = recall_score(y_trues, y_preds, average="macro")
    from sklearn.metrics import precision_score
    precision = precision_score(y_trues, y_preds, average="macro")
    from sklearn.metrics import f1_score
    f1 = f1_score(y_trues, y_preds, average="macro")
    from sklearn.metrics import matthews_corrcoef
    mcc = matthews_corrcoef(y_trues, y_preds)
    from sklearn.metrics import accuracy_score
    acc = accuracy_score(y_trues, y_preds)
    result = {
        "eval_recall": float(recall),
        "eval_precision": float(precision),
        "eval_f1": float(f1),
        "eval_mcc": float(mcc),
        "eval_acc": float(acc)
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
    # with open(os.path.join(args.output_dir, "predictions.txt"), 'w') as f:
    #     for example, pred in zip(eval_dataset.examples, y_preds):
    #         if pred:
    #             f.write(str(example.label) + '\t' + '1' + '\n')
    #         else:
    #             f.write(str(example.label) + '\t' + '0' + '\n')


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
    
    # Setup CUDA, GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.n_gpu = torch.cuda.device_count()

    args.device = device

    # Setup logging
    logger = Log.get_logger(name="pos-tag", logs_dir=args.output_dir)
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
    labels_file = json.load( open(os.path.join(args.data_folder, "all_labels.json")) )
    label_set = LabelSet(labels=labels_file)
    num_class = len(label_set.labels_to_id)
    logger.info(f"Num of Class {num_class}")
    dataset_train = PosTagDataset(
        args,
        label_set,
        "train",
        tokenizer
    )
    dataset_valid = PosTagDataset(
       args,
        label_set,
        "valid",
        tokenizer
    )
    dataset_test = PosTagDataset(
        args,
        label_set,
        "test",
        tokenizer
    )

    logger.info("Finishing loading Dataset")

    #create model
    if args.do_random:
        logger.info("Random Initialize the encoder")
        encoder = AutoModel.from_config(config)
    else:
        logger.info("pretrained the encoder")
        encoder = AutoModel.from_pretrained(args.model_name_or_path,config=config)
    for name, param in encoder.named_parameters():
            param.requires_grad = False

    model = BERTPoSTagger(encoder, config, num_class, args.layer)
    #model = model.to(args.device)
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
        import os
        os.remove(output_dir)

    if args.do_test:
        checkpoint_prefix = 'checkpoint-best-mcc/model.bin'
        output_dir = os.path.join(
            args.output_dir, '{}'.format(checkpoint_prefix)
        )
        model.load_state_dict(torch.load(output_dir))
        model.to(args.device)
        test(args, model, dataset_test)
        import os
        os.remove(output_dir)

    # return results
