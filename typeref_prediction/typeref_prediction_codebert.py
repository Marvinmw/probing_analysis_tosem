from collections import defaultdict
import argparse
from global_logger import Log
import torch
import logging
import random
import numpy as np
from transformers import AutoTokenizer, AutoModel, AutoConfig
from models.classifier import FasterTypeRefModel
from torch.utils.data import DataLoader, SequentialSampler, RandomSampler
from transformers import (AdamW, get_linear_schedule_with_warmup)
from tqdm import tqdm
import os
import json
from typing import Counter
from torch.utils.data import Dataset
import pickle
from utils.alignment import normalization_code

code_features=None
mask_token_data=None
graph_data = None
primitive_types = None

class InputExample(object):
    def __init__(self, id, index, token_index, label, input_ids, input_mask,code ) -> None:
        self.id = id
        self.index = index
        self.token_index = token_index
        self.label = label
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.code = code

import random

def create_example_from_file_defaut_list( max_index, tokenizer, code_ids ):
    global mask_token_data
    input_examples = []
    count_examples = {"int":[], "char":[], "float":[], "double":[]}
    # graph data
    programs = list( mask_token_data.keys() )
    #logger.info(f"Num of programs, {len(programs)}")
    for pindex in programs:
        type_example_list = mask_token_data[pindex]    
        if len(list(type_example_list.items())):
            lno, v = random.choice(list(type_example_list.items()))
        # print(type_example_list)
        # print(tokenizer.pad_token_id)
        # for lno, v in type_example_list.items():
            tried = 0
            while not v["token_index"] < max_index and tried < 100:
                lno, v = random.choice(list(type_example_list.items()))
                tried = tried + 1
            (input_ids, input_mask) = code_ids[pindex][0].copy(), code_ids[pindex][1].copy()
            input_ids[v["token_index"]] = tokenizer.mask_token_id
            ex = InputExample( lno, pindex, v["token_index"], primitive_types[v["primitive"]],
                        input_ids, input_mask, v["line_code"] )  
            count_examples[v["primitive"]].append(ex)            
            #input_examples.append( ex )
    
    min_num = min([ len(v) for k,v in count_examples.items()] )
    for k, v in count_examples.items():
        input_examples.extend( random.sample(v, min_num) )
    random.shuffle( input_examples )
    return input_examples

def create_example_from_file(data_list, max_index, tokenizer, code_ids ):
    global mask_token_data
    input_examples = []
   
    #logger.info(f"Num of programs, {len(programs)}")
    for edata in data_list:
        pindex = edata["pindex"]
        lno = edata["lno"]
        type_example_list = mask_token_data[pindex]    
        v = type_example_list[lno]
        (input_ids, input_mask) = code_ids[pindex][0].copy(), code_ids[pindex][1].copy()
        input_ids[v["token_index"]] = tokenizer.mask_token_id
        ex = InputExample( lno, pindex, v["token_index"], primitive_types[v["primitive"]],
                        input_ids, input_mask, v["line_code"] )  
        input_examples.append( ex )
    
    random.shuffle( input_examples )
    return input_examples

class CodeBERTDataset(Dataset):
    def __init__(self,  args, tokenizer,  crossing=True,  source_code_file="poj-104/test/google_style.jsonl", mask_token_file="") -> None:
        global mask_token_data
      #  global graph_data
        self.examples = []
        self.examples_map_index = {}
        self.reverse_examples_map_index = {}
        self.args = args
        index_filename = source_code_file
        self.code_cache = {}
        data_folder = args.data_folder 
        
        with  open(index_filename) as f_google:
            hh = f_google.readlines()
            for i in tqdm(range( len(hh) )):
                l_google =hh[i]
                data_google = json.loads(l_google)
                indexmap, code_func = normalization_code(data_google["code"])
                index_id = data_google["index"]
                source_tokens = tokenizer.tokenize(code_func)[:args.max_code_length-2]
                source_tokens =[tokenizer.cls_token]+source_tokens+[tokenizer.sep_token]
                source_ids =  tokenizer.convert_tokens_to_ids(source_tokens) 
                source_mask = [1] * (len(source_tokens))
                padding_length = args.max_code_length - len(source_ids)
                source_ids+=[tokenizer.pad_token_id]*padding_length
                source_mask+=[0]*padding_length
                self.code_cache[index_id]=( source_ids, source_mask )
                
        #load data 
        self.logger = Log.get_logger(name="Mask Type-prediction", logs_dir=args.output_dir)
        if os.path.isfile( mask_token_file ):
            mask_token_data = json.load(open(mask_token_file))
        else:
            assert False, "generate the span file at first"
        
        data_train = json.load(open( os.path.join(data_folder, "train.json") ))
        data_valid = json.load(open( os.path.join(data_folder, "valid.json") ))
        data_test =  json.load(open( os.path.join(data_folder, "test.json") ))
        train_examples = create_example_from_file( data_train, args.max_code_length-3, tokenizer,self.code_cache )
        valid_examples = create_example_from_file( data_valid,  args.max_code_length-3, tokenizer,self.code_cache )
        test_examples = create_example_from_file( data_test, args.max_code_length-3, tokenizer,self.code_cache )
         
        self.logger.info("save preprocessed data")
        pickle.dump( train_examples, open(f"{data_folder}/{args.model_name}/{args.model_name}_train_{crossing}_typeref.pt", "wb")  )
        pickle.dump( valid_examples, open(f"{data_folder}/{args.model_name}/{args.model_name}_valid_{crossing}_typeref.pt", "wb")  )
        pickle.dump( test_examples, open(f"{data_folder}/{args.model_name}/{args.model_name}_test_{crossing}_typeref.pt", "wb")  )
        self.examples = train_examples + valid_examples + test_examples
        pickle.dump( self.examples, open(f"{data_folder}/{args.model_name}/{args.model_name}_data_{crossing}_typeref.pt", "wb")  )
        
        #load data
        for i, e in enumerate( list( self.examples ) ):
            self.examples_map_index[f"{e.index}_{e.id}_{e.token_index}"] = i
            self.reverse_examples_map_index[i] = f"{e.index}_{e.id}_{e.token_index}"
        
             
    def print_example(self, tokenizer):
        for idx, example in enumerate(self.examples[:3]):
                self.logger.info("*** Example ***")
                self.logger.info("idx: {}".format(example.index))
                self.logger.info("label: {}".format(example.label))
                self.logger.info("input_tokens_id: {}".format([ self.code_cache[example.index][0][example.token_index]])) 
                self.logger.info("input_tokens_reverse_1: {}".format(' '.join( 
                    tokenizer.convert_ids_to_tokens( self.code_cache[example.index][0][example.token_index] ) )))  
              
    def __len__(self):
        return len(self.examples)
        
    def __getitem__(self, item):
        index = f"{self.examples[item].index}_{self.examples[item].id}_{self.examples[item].token_index}"
        return ( torch.tensor(self.examples_map_index[index]), 
                torch.tensor(self.examples[item].token_index),
                torch.tensor(self.examples[item].input_ids),
                torch.tensor(self.examples[item].input_mask) )       

class CodeTypeRefDatasetFromFile(Dataset):
    def __init__(self,  args, tokenizer, dataset='train', crossing=True,  source_code_file="") -> None:
        global mask_token_data
      #  global graph_data
        self.examples = []
        self.args = args
        self.dataset = dataset
        index_filename = source_code_file
        self.code_cache = {}
        data_folder = args.data_folder 

        code_cache_path = os.path.join( data_folder, f"{args.model_name}/{dataset}_code_cache.pt" )   
        if not os.path.isfile(code_cache_path):
            with  open(index_filename) as f_google:
                hh = f_google.readlines()
                for i in tqdm(range( len(hh) )):
                    l_google =hh[i]
                    data_google = json.loads(l_google)
                    indexmap, code_func = normalization_code(data_google["code"])
                    index_id = data_google["index"]
                    source_tokens = tokenizer.tokenize(code_func)[:args.max_code_length-2]
                    source_tokens =[tokenizer.cls_token]+source_tokens+[tokenizer.sep_token]
                    source_ids =  tokenizer.convert_tokens_to_ids(source_tokens) 
                    source_mask = [1] * (len(source_tokens))
                    padding_length = args.max_code_length - len(source_ids)
                    source_ids+=[tokenizer.pad_token_id]*padding_length
                    source_mask+=[0]*padding_length
                    self.code_cache[index_id]=( source_ids, source_mask )
                pickle.dump( self.code_cache, open(code_cache_path, "wb"  ) )
        else:
            self.code_cache = pickle.load( open(code_cache_path, "rb") )

        #load data
        data_path = os.path.join( data_folder, f"{args.model_name}/{dataset}_{crossing}_typeref.pt" )    
        self.logger = Log.get_logger(name="type-ref", logs_dir=args.output_dir)
        self.logger.info("Load Mask Type data, data path: {}".format(data_path)) 
        self.examples = pickle.load( open(data_path, "rb") )
        if args.debug:
            self.examples = self.examples[:100]
            
        labels = [ ex.label for ex in self.examples]
        self.logger.info(f"Labels {Counter(labels)}")
    
    def print_example(self, tokenizer):
        for idx, example in enumerate(self.examples[:3]):
                self.logger.info("*** Example ***")
                self.logger.info("idx: {}".format(example.index))
                self.logger.info("label: {}".format(example.label))
                self.logger.info("input_tokens_id: {}".format([ self.code_cache[example.index][0][example.token_index]])) 
                self.logger.info("input_tokens_reverse_1: {}".format(' '.join( 
                    tokenizer.convert_ids_to_tokens( self.code_cache[example.index][0][example.token_index] ) )))  
            
    def __len__(self):
        return len(self.examples)
        
    def __getitem__(self, item):
        global code_features
        index = f"{self.examples[item].index}_{self.examples[item].id}_{self.examples[item].token_index}"
        return ( torch.tensor(self.examples[item].token_index), 
                torch.tensor(self.examples[item].label), 
                code_features[index][self.args.layer].squeeze(0) )    

def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)



def inference_source_code_feature(encoder, codedataset, code_feature_file):
    encoder.eval()
    eval_sampler = SequentialSampler(codedataset)
    eval_dataloader = DataLoader(codedataset, sampler=eval_sampler,batch_size=1,num_workers=4)
    bar = tqdm(eval_dataloader,total=len(eval_dataloader))
    res = {}
    for step, batch in enumerate(bar):   
        (code_index, token_index, source_id, attention_mask)=[x.to("cpu")  for x in batch]
        code_index = code_index.item()
        token_index = token_index.item()
        with torch.no_grad():
            outputs = encoder(source_id, attention_mask=attention_mask, return_dict=True)
            hidden_states = [ l[:, token_index, :].view(-1, config.hidden_size) for l in outputs.hidden_states ] # rep = input_features.gather( 1, index_token ).squeeze(1) 
        res[codedataset.reverse_examples_map_index[code_index]] = hidden_states #if args.layer==None else hidden_states[args.layer]
    
    logger.info(f"Finish preprocessing the dataset")
    pickle.dump( res, open( code_feature_file, "wb" ))
    return 

def train(args, train_dataset,eval_dataset, model, k_fold=False):
    """ Train the model """
    
    #build dataloader
    if k_fold:
        train_dataloader = train_dataset 
    else:    
        train_sampler = RandomSampler(train_dataset)
        train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.train_batch_size)
    
    args.max_steps=args.epochs*len( train_dataloader)
    args.save_steps=max(1, len( train_dataloader)//10)
    args.warmup_steps=args.max_steps//5
    model.to(args.device)
    
    # Prepare optimizer and schedule (linear warmup and decay)
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
         'weight_decay': args.weight_decay},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=args.warmup_steps,
                                                num_training_steps=args.max_steps)

    # multi-gpu training
    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)

    # Train!
    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_dataset))
    logger.info("  Num Epochs = %d", args.epochs)
    logger.info("  Instantaneous batch size per GPU = %d", args.train_batch_size//max(args.n_gpu, 1))
    logger.info("  Total train batch size = %d",args.train_batch_size*args.gradient_accumulation_steps)
    logger.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)
    logger.info("  Total optimization steps = %d", args.max_steps)
    logger.info("  Save steps = %d", args.save_steps)
    
    global_step=0
    tr_loss, logging_loss,avg_loss,tr_nb,tr_num,train_loss = 0.0, 0.0,0.0,0,0,0
    best_f1=0
    best_mcc=0
    model.zero_grad()
    best_results = {} 
    for idx in range(args.epochs): 
        bar = tqdm(train_dataloader,total=len(train_dataloader))
        tr_num=0
        train_loss=0
        for step, batch in enumerate(bar):
            (token_index, label, input_features )=[x.to(args.device)  for x in batch]
            model.train()
            loss,logits = model(input_features, token_index, label)

            if args.n_gpu > 1:
                loss = loss.mean()
                
            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

            tr_loss += loss.item()
            tr_num+=1
            train_loss+=loss.item()
            if avg_loss==0:
                avg_loss=tr_loss
                
            avg_loss=round(train_loss/tr_num,5)
            bar.set_description("epoch {} loss {}".format(idx,avg_loss))
              
            if (step + 1) % args.gradient_accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()
                scheduler.step()  
                global_step += 1
                output_flag=True
                avg_loss=round(np.exp((tr_loss - logging_loss) /(global_step- tr_nb)),4)

                if global_step % args.save_steps == 0:
                    results = evaluate(args, model,eval_dataset, eval_when_training=True, k_fold=k_fold)    
                    
                    # Save model checkpoint
                    if results['eval_f1']>best_f1:
                        best_results["f1"] = results
                        best_f1=results['eval_f1']
                        logger.info("  "+"*"*20)  
                        logger.info("  Best f1:%s",round(best_f1,4))
                        logger.info("  "+"*"*20)                          
                        
                        checkpoint_prefix = 'checkpoint-best-f1'
                        output_dir = os.path.join(args.output_dir, '{}'.format(checkpoint_prefix))                        
                        if not os.path.exists(output_dir):
                            os.makedirs(output_dir)                        
                        model_to_save = model.module if hasattr(model,'module') else model
                        output_dir = os.path.join(output_dir, '{}'.format('model.bin')) 
                        torch.save(model_to_save.state_dict(), output_dir)
                        logger.info("Saving model checkpoint to %s", output_dir)
                    
                    # Save model checkpoint
                    if results['eval_mcc']>best_mcc:
                        best_results["mcc"] = results
                        best_mcc=results['eval_mcc']
                        logger.info("  "+"*"*20)  
                        logger.info("  Best MCC:%s",round(best_mcc,4))
                        logger.info("  "+"*"*20)                          
                        
                        checkpoint_prefix = 'checkpoint-best-mcc'
                        output_dir = os.path.join(args.output_dir, '{}'.format(checkpoint_prefix))                        
                        if not os.path.exists(output_dir):
                            os.makedirs(output_dir)                        
                        model_to_save = model.module if hasattr(model,'module') else model
                        output_dir = os.path.join(output_dir, '{}'.format('model.bin')) 
                        torch.save(model_to_save.state_dict(), output_dir)
                        logger.info("Saving model checkpoint to %s", output_dir)
    return best_results                    

def evaluate(args, model,eval_dataset, eval_when_training=False, k_fold=False):
    #build dataloader
   # eval_dataset = TextDataset(args, examples)
    if k_fold:
        eval_dataloader = eval_dataset
    else:
        eval_sampler = SequentialSampler(eval_dataset)
        eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler,batch_size=args.eval_batch_size)

    # multi-gpu evaluate
    if args.n_gpu > 1 and eval_when_training is False:
        model = torch.nn.DataParallel(model)

    # Eval!
    #logger.info("***** Running evaluation *****")
    #logger.info("  Num examples = %d", len(eval_dataset))
    #logger.info("  Batch size = %d", args.eval_batch_size)

    eval_loss = 0.0
    nb_eval_steps = 0
    model.eval()
    logits=[]  
    y_trues=[]
   # bar = tqdm(eval_dataloader,total=len(eval_dataloader))
    for step, batch in enumerate(eval_dataloader):   
        (token_index, labels, input_features)=[x.to(args.device)  for x in batch]
        with torch.no_grad():
            lm_loss,logit =   model(input_features, token_index, labels)
            eval_loss += lm_loss.mean().item()
            logits.append(logit.cpu() )
            y_trues.append(labels.cpu() )
        nb_eval_steps += 1
    
    #calculate scores
    logits=torch.cat(logits,0)
    y_trues=torch.cat(y_trues,0)
 
    #  y_preds=logits[:,1]>best_threshold
    _, y_preds = torch.max( logits, dim=1 )
    
    y_preds = y_preds.numpy()
    y_trues = y_trues.numpy()
    #print(y_preds )
    #print(logits.numpy())
    from sklearn.metrics import recall_score
    recall=recall_score(y_trues, y_preds, average="macro")
    from sklearn.metrics import precision_score
    precision=precision_score(y_trues, y_preds, average="macro")   
    from sklearn.metrics import f1_score
    f1=f1_score(y_trues, y_preds, average="macro")  
    from sklearn.metrics import matthews_corrcoef
    mcc = matthews_corrcoef(y_trues, y_preds)        
    from sklearn.metrics import accuracy_score
    acc = accuracy_score(y_trues, y_preds)    
    result = {
        "eval_recall": float(recall),
        "eval_precision": float(precision),
        "eval_f1": float(f1),
        "eval_mcc": float(mcc),
        "eval_acc":float(acc),
       # "eval_threshold":best_threshold,
        
    }
    if not k_fold:
        logger.info("***** Eval results *****")
        for key in sorted(result.keys()):
            logger.info("  %s = %s", key, str(round(result[key],4)))

    return result

def test(args, model,eval_dataset, best_threshold=0):
    eval_sampler = SequentialSampler(eval_dataset)
    eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=args.eval_batch_size)

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
    logits=[]  
    y_trues=[]
    for batch in eval_dataloader: 
        (token_index, labels, input_features)=[x.to(args.device)  for x in batch]
        with torch.no_grad():
            lm_loss,logit =   model(input_features, token_index, labels)
            eval_loss += lm_loss.mean().item()
            logits.append(logit.cpu())
            y_trues.append(labels.cpu())
        nb_eval_steps += 1
    
     #calculate scores
    logits=torch.cat(logits,0)
    y_trues=torch.cat(y_trues,0)
  

    #  y_preds=logits[:,1]>best_threshold
    _, y_preds = torch.max( logits, dim=1 )
    y_preds = y_preds.numpy()
    y_trues = y_trues.numpy()
    #print(y_preds )
    #print(logits.numpy())
    from sklearn.metrics import recall_score
    recall=recall_score(y_trues, y_preds, average="macro")
    from sklearn.metrics import precision_score
    precision=precision_score(y_trues, y_preds, average="macro")   
    from sklearn.metrics import f1_score
    f1=f1_score(y_trues, y_preds, average="macro")  
    from sklearn.metrics import matthews_corrcoef
    mcc = matthews_corrcoef(y_trues, y_preds)           
    from sklearn.metrics import accuracy_score
    acc = accuracy_score(y_trues, y_preds)    
    result = {
        "eval_recall": float(recall),
        "eval_precision": float(precision),
        "eval_f1": float(f1),
        "eval_mcc": float(mcc),
        "eval_acc":float(acc),
       # "eval_threshold":best_threshold,
        
    }


    logger.info("***** Test results *****")
    for key in sorted(result.keys()):
        logger.info("  %s = %s", key, str(round(result[key],4)))
    json.dump(result, open( os.path.join(args.output_dir,"predictions_performance_test.json"), "w" ) , indent=4)
    with open(os.path.join(args.output_dir,"predictions.txt"),'w') as f:
        for example,pred in zip(eval_dataset.examples,y_preds):
            #if pred:
            f.write(str(example.label)+'\t'+f'{pred}'+'\n')
            #else:
            #    f.write(str(example.label_id)+'\t'+'0'+'\n')

def k_fold_crossing(args, config, dataset):
    # K-fold Cross Validation model evaluation
    # Define the K-fold Cross Validator
    from sklearn.model_selection import KFold
    kfold = KFold(n_splits=5, shuffle=True)
    k_fold_results = {}
    model=None
    for fold, (train_ids, test_ids) in enumerate(kfold.split(dataset)):
        # Sample elements randomly from a given list of ids, no replacement.
        logger.info(f"K_fold_{fold}, train {len(train_ids)}, test {len(test_ids)}")
        model =  FasterTypeRefModel(config,args)
        train_subsampler = torch.utils.data.SubsetRandomSampler(train_ids)
        test_subsampler = torch.utils.data.SubsetRandomSampler(test_ids)
        
        # Define data loaders for training and testing data in this fold
        trainloader = torch.utils.data.DataLoader(
                        dataset, 
                        batch_size=10, sampler=train_subsampler)
        testloader = torch.utils.data.DataLoader(
                        dataset,
                        batch_size=10, sampler=test_subsampler)
        result = train(args, trainloader, testloader, model, k_fold=True)
        k_fold_results[fold]=result
        model=None
    
    # compute average
    logger.info("Avg Best F1")
    avg_f1_res = defaultdict(float)
    avg_mcc_res = defaultdict(float)    

    for fid, res in k_fold_results.items():
        for k in res["f1"]:
            avg_f1_res[k] += res["f1"][k]/5.0
            avg_mcc_res[k] += res["mcc"][k]/5.0
    
    logger.info("***** k fold results *****")
    result=avg_mcc_res
    for key in sorted(result.keys()):
        logger.info("  %s = %s", key, str(round(result[key],4)))

    json.dump(avg_f1_res, open(  os.path.join(args.output_dir, "k_fold_f1.json") , "w"), indent=4)
    json.dump(avg_mcc_res,open(  os.path.join(args.output_dir, "k_fold_mcc.json") , "w"), indent=4)
    json.dump(k_fold_results,open(  os.path.join(args.output_dir, "k_fold_results.json") , "w"), indent=4)

if __name__ == "__main__":
    import os
    os.environ["TOKENIZERS_PARALLELISM"] = "true"
    
    parser = argparse.ArgumentParser(description='Process some integers.')
    ## Required parameters
    parser.add_argument("--dataset", default="", type=str, #required=True,
                        help="The input training data file (a text file).")
    parser.add_argument("--output_dir", default="poj-train-model-saved", type=str, #required=True,
                        help="The output directory where the model predictions and checkpoints will be written.")
    parser.add_argument("--data_folder", default="dataset/exp1", type=str, #required=True,
                        help="The output directory where the model predictions and checkpoints will be written.")                    
    
    ## Dataset Config
    parser.add_argument('--max_code_length', dest='max_code_length', type=int,default=512,
                        help='input maximum length')

    ## Pretrain model config
    parser.add_argument("--model_name_or_path", default="microsoft/codebert-base", type=str,
                        help="The model checkpoint for weights initialization.")
    parser.add_argument("--model_name", default="codebert", type=str,
                        help="The model name.")
    parser.add_argument("--config_name", default="microsoft/codebert-base", type=str,
                        help="Optional pretrained config name or path if not the same as model_name_or_path")
    parser.add_argument('--token_config', dest='token_config', type=str,default="microsoft/codebert-base",
                        help='tokenizer config')
    parser.add_argument("--do_kfold", action='store_true',
                    help="Whether to run training.")    
    parser.add_argument("--do_train", action='store_true',
                        help="Whether to run training.")
    parser.add_argument("--debug", action='store_true',
                        help="debug model with the samll dataset")
    parser.add_argument("--do_crossing", action='store_true',
                        help="crossing project train and test")
    parser.add_argument("--do_eval", action='store_true',
                        help="Whether to run eval on the dev set.")
    parser.add_argument("--do_test", action='store_true',
                        help="Whether to run eval on the dev set.")
    parser.add_argument("--do_random", action='store_true',
                        help="Whether to randomly initialize model.")    
    parser.add_argument("--evaluate_during_training", action='store_true',
                        help="Run evaluation during training at each logging step.")

    parser.add_argument("--layer", default=-1, type = int, help="use which layer for probing ")
    parser.add_argument("--train_batch_size", default=64, type=int,
                        help="Batch size per GPU/CPU for training.")
    parser.add_argument("--eval_batch_size", default=64, type=int,
                        help="Batch size per GPU/CPU for evaluation.")
    parser.add_argument("--class_num", default=4, type=int, help="number of labels")
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument("--learning_rate", default=1e-5, type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--weight_decay", default=0.0, type=float,
                        help="Weight deay if we apply some.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float,
                        help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float,
                        help="Max gradient norm.")
    parser.add_argument("--max_steps", default=-1, type=int,
                        help="If > 0: set total number of training steps to perform. Override num_train_epochs.")
    parser.add_argument("--warmup_steps", default=0, type=int,
                        help="Linear warmup over warmup_steps.")
    parser.add_argument('--seed', type=int, default=42,
                        help="random seed for initialization")
    parser.add_argument('--epochs', type=int, default=10,
                        help="training epochs")

    args = parser.parse_args()
    ## Dataset java250
    assert  args.dataset in ["java250", "poj-104"]
    if args.dataset == "java250":
        source_code_file="../datasets/java250/typeref/java250_token_mask_codebert_typeref.json.source"
                            #datasets/java250/typeref/java250_token_mask_codebert_typeref.json
        mask_token_file="../datasets/java250/typeref/java250_token_mask_codebert_typeref.json"
        primitive_types={"int":0, "char":1, "long":2, "double":3,  "boolean":4 }


    if args.dataset == "poj-104":
        source_code_file="../datasets/poj-104/poj-104/test/formated_source_code_clang.jsonl"
        mask_token_file="../datasets/poj-104/poj-104-graph/poj-104-test_token_mask_codebert_tyref.json"
        primitive_types={"int":0, "char":1, "float":2, "double":3}


    mask_token_data = json.load(open(mask_token_file))
    # Setup CUDA, GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.n_gpu = torch.cuda.device_count()
    args.device = device

    # Setup logging
    logger = Log.get_logger(name="typeref", logs_dir=args.output_dir)
    logger.debug('main message')
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',datefmt='%m/%d/%Y %H:%M:%S',level=logging.INFO)
    logger.warning("device: %s, n_gpu: %s",device, args.n_gpu,)


    # Set seed
    set_seed(args)
    config = AutoConfig.from_pretrained(args.config_name if args.config_name else args.model_name_or_path, return_dict=False, output_hidden_states=True)
    tokenizer =  AutoTokenizer.from_pretrained(args.token_config)
    # create dataset

    code_feature_file = f"{args.data_folder}/{args.model_name}/input_encoder_hidden_states_typeref_random_{args.do_random}.pt"  #if args.layer==None else f"{args.model_name}/input_encoder_hidden_states_typeref_layer{args.layer}.pt" 
    logger.info(f"code_feature_file")
    if not os.path.isfile( code_feature_file ):
        logger.info("Compute Code Representation")
        if args.do_random:
            logger.info("Random Initialize the encoder")
            encoder = AutoModel.from_config(config)
        else:
            logger.info("pretrained the encoder")
            encoder = AutoModel.from_pretrained(args.model_name_or_path,config=config)
        codedataset = CodeBERTDataset(args, tokenizer, crossing=args.do_crossing, source_code_file=source_code_file,
            mask_token_file=mask_token_file)
        inference_source_code_feature(encoder, codedataset, code_feature_file)
    code_features=pickle.load( open(code_feature_file, "rb") )
    logger.info("Load Dataset")
    dataset_train = CodeTypeRefDatasetFromFile(args, tokenizer,  dataset=f'{args.model_name}_train', crossing=args.do_crossing, source_code_file=source_code_file)
    dataset_valid = CodeTypeRefDatasetFromFile(args, tokenizer,  dataset=f'{args.model_name}_valid', crossing=args.do_crossing, source_code_file=source_code_file)
    dataset_test = CodeTypeRefDatasetFromFile(args, tokenizer, dataset=f'{args.model_name}_test', crossing=args.do_crossing, source_code_file=source_code_file)
    
    dataset_train.print_example(tokenizer)
    dataset_valid.print_example(tokenizer)
    dataset_test.print_example(tokenizer)

    logger.info("Finishing loading Dataset")
    
    #create model
    args.class_num=len(primitive_types)
    model =  FasterTypeRefModel(config,args)
    num_params_model =  sum(p.numel() for p in model.parameters() if p.requires_grad)

    # show input shape
    logger.info(f"Probimg Model Parameters {num_params_model}")
    logger.info("Training/evaluation parameters %s", args)
    # k fold
    if args.do_kfold:
        dataset = CodeTypeRefDatasetFromFile(args, tokenizer,  dataset=f'{args.model_name}_data', crossing=args.do_crossing, 
        source_code_file=source_code_file)
        logger.info(f"Kfold, Dataset {len(dataset)}")
        k_fold_crossing(args, config, dataset)
    # Training
    if args.do_train:
        train(args, dataset_train,dataset_valid, model)
 
    # Evaluation
    # results = {}
    if args.do_eval:
        checkpoint_prefix = 'checkpoint-best-f1/model.bin'
        output_dir = os.path.join(args.output_dir, '{}'.format(checkpoint_prefix))  
        model.load_state_dict(torch.load(output_dir))
        model.to(args.device)
        results=evaluate(args, model, dataset_valid)
        
    if args.do_test:
        checkpoint_prefix = 'checkpoint-best-mcc/model.bin'
        output_dir = os.path.join(args.output_dir, '{}'.format(checkpoint_prefix))  
        model.load_state_dict(torch.load(output_dir))
        model.to(args.device)
        test(args, model,dataset_test,best_threshold=0.5)

    
