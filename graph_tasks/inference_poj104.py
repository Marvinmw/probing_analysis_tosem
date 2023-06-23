import argparse
from global_logger import Log
import torch
import logging
import random
import numpy as np
from transformers import AutoTokenizer, AutoModel, AutoConfig
from models.classifier import InferenceModel
from utils.alignment import normalization_code
from torch.utils.data import DataLoader
from tqdm import tqdm
import os
import json
from torch.utils.data import Dataset
import pickle

cache_span = None
graph_data = None
source_code = "../datasets/poj-104/poj-104/test/formated_source_code_clang.jsonl"


def random_pairs_np(number_list):
    return np.random.choice(number_list, 2, replace=False)


class CodeBERTDataset(Dataset):
    def __init__(
        self,
        args,
        logger,
        tokenizer,
        source_code=source_code,
        programlist=None
    ) -> None:
        global cache_span
        #  global graph_data
        self.examples = []
        self.examples_map_index = {}
        self.reverse_examples_map_index = {}
        self.args = args
        self.logger = logger
        index_filename = source_code
        self.code_cache = {}
        data_folder = args.data_folder

        with open(index_filename) as f_google:
            hh = f_google.readlines()
            for i in tqdm(range(len(hh))):
                l_google = hh[i]
                data_google = json.loads(l_google)
                indexmap, code_func = normalization_code(data_google["code"])
                index_id = data_google["index"]
                source_tokens = tokenizer.tokenize(code_func
                                                  )[:args.max_code_length - 2]
                source_tokens = [tokenizer.cls_token
                                ] + source_tokens + [tokenizer.sep_token]
                source_ids = tokenizer.convert_tokens_to_ids(source_tokens)
                source_mask = [1] * (len(source_tokens))
                padding_length = args.max_code_length - len(source_ids)
                source_ids += [tokenizer.pad_token_id] * padding_length
                source_mask += [0] * padding_length
                self.code_cache[index_id] = (source_ids, source_mask)

        #load data
        for i, k in enumerate(list(self.code_cache.keys())):
            is_include = True if programlist is None else k in programlist
            if is_include:
                self.examples.append(k)
                self.examples_map_index[k] = i
                self.reverse_examples_map_index[i] = k

    def print_example(self, tokenizer):
        for idx, index in enumerate(self.examples[:3]):
            self.logger.info("*** Example ***")
            self.logger.info("idx: {}".format(index))
            self.logger.info(
                "input_tokens_reverse_1: {}".format(
                    ' '.join(
                        tokenizer.convert_ids_to_tokens(
                            self.code_cache[index][0]
                        )
                    )
                )
            )

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, item):
        #calculate graph-guided masked function
        index = self.examples[item]
        # ex1_span, ex2_span, label, source_id, attention_mask
        return (
            torch.tensor(self.examples_map_index[index]),
            torch.tensor(self.code_cache[index][0]),
            torch.tensor(self.code_cache[index][1])
        )




def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)


def inference(args, eval_dataset, model):
    #build dataloader
    eval_dataloader = DataLoader(eval_dataset, batch_size=1, num_workers=4)

    # Eval!
    logger.info("***** Running Inference *****")
    logger.info("  Num examples = %d", len(eval_dataset))
    logger.info("  Batch size = %d", 1)

    model.eval()
    bar = tqdm(eval_dataloader, total=len(eval_dataloader))
    res = {}
    for step, batch in enumerate(bar):
        bdata = [x.to(args.device) for x in batch]
        index_id = bdata[0]
        id = index_id.item()
        with torch.no_grad():
            hidden_states = model(bdata[1:])
            res[eval_dataset.reverse_examples_map_index[id]
               ] = hidden_states if args.layer == None else hidden_states[
                   args.layer]
    save_file = f"{args.model_name}/input_encoder_hidden_states_random_{args.do_random}.pt" if args.layer == None else f"{args.model_name}/input_encoder_hidden_states_layer{args.layer}_random_{args.do_random}.pt"
    pickle.dump(res, open(os.path.join(args.data_folder, save_file), "wb"))


if __name__ == "__main__":
    import os
    os.environ["TOKENIZERS_PARALLELISM"] = "true"

    parser = argparse.ArgumentParser(description='Process some integers.')
    ## Required parameters
    parser.add_argument(
        "--dataset",
        default="poj-104",
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
        default="dataset",
        type=str,  #required=True,
        help=
        "The output directory where the model predictions and checkpoints will be written."
    )
    parser.add_argument(
        "--sample_program_file",
        default="",
        type=str,  #required=True,
        help=
        "The output directory where the model predictions and checkpoints will be written."
    )
    parser.add_argument(
        "--do_random",
        action='store_true',
        help="Whether randomly initialize the model."
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
        "--layer", default=None, type=int, help="use which layer for probing "
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
        '--seed', type=int, default=42, help="random seed for initialization"
    )

    args = parser.parse_args()

    ## Dataset
    programlist = json.load(
        open(args.sample_program_file, "r")
    ) if args.sample_program_file else None

    # Setup CUDA, GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.n_gpu = torch.cuda.device_count()
    args.device = device

    # Setup logging
    logger = Log.get_logger(name="inference", logs_dir=args.output_dir)
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
    logger.info(f"args {args}")

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
    if args.model_name == "codebert":
        dataset = CodeBERTDataset(
            args,
            logger,
            tokenizer,
            source_code=source_code,
            programlist=programlist
        )
    
    dataset.print_example(tokenizer)
    logger.info("Finishing loading Dataset")
    if args.do_random:
        logger.info("Random Initialize the encoder")
        encoder = AutoModel.from_config(config)
    else:
        logger.info("pretrained the encoder")
        encoder = AutoModel.from_pretrained(
            args.model_name_or_path, config=config
        )
    #create model
    model = InferenceModel(encoder, config, args)
    model.to(args.device)
    num_params_model = sum(
        p.numel() for p in model.parameters() if p.requires_grad
    )

    # show input shape
    logger.info("Inference parameters %s", args)
    inference(args, dataset, model)
