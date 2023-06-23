
"""
Fine-tuning the library models for language modeling on a text file (GPT, GPT-2, BERT, RoBERTa).
GPT and GPT-2 are fine-tuned using a causal language modeling (CLM) loss while BERT and RoBERTa are fine-tuned
using a masked language modeling (MLM) loss.
"""

from __future__ import absolute_import
import os
import torch
import random
import logging
import numpy as np
from transformers import (RobertaConfig, RobertaModel, RobertaTokenizer)
MODEL_CLASSES = {'roberta': (RobertaConfig, RobertaModel, RobertaTokenizer)}
logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt = '%m/%d/%Y %H:%M:%S',
                    level = logging.INFO)
from parser import DFG_python,DFG_java,DFG_ruby,DFG_go,DFG_php,DFG_javascript,DFG_csharp
from parser import (remove_comments_and_docstrings,
                   tree_to_token_index,tree_to_token_index_with_type,index_to_code_token_types,tree_to_variable_index,index_to_code_token_variables,
                   index_to_code_token)
from tree_sitter import Language, Parser
dfg_function={
    'python':DFG_python,
    'java':DFG_java,
    'ruby':DFG_ruby,
    'go':DFG_go,
    'php':DFG_php,
    'javascript':DFG_javascript,
    'c_sharp':DFG_csharp,
}

#load parsers
parsers={}        
for lang in dfg_function:
    LANGUAGE = Language('parser/my-languages.so', lang)
    parser = Parser()
    parser.set_language(LANGUAGE) 
    parser = [parser,dfg_function[lang]]    
    parsers[lang]= parser
    
#remove comments, tokenize code and extract dataflow     
def extract_dataflow(code, parser,lang):
    #remove comments
    try:
        code=remove_comments_and_docstrings(code,lang)
    except:
        pass    
    #obtain dataflow
    if lang=="php":
        code="<?php"+code+"?>"    
    try:
        tree = parser[0].parse(bytes(code,'utf8'))    
        root_node = tree.root_node  
        tokens_index=tree_to_token_index(root_node)     
        code=code.split('\n')
        code_tokens=[index_to_code_token(x,code) for x in tokens_index]  
        index_to_code={}
        for idx,(index,code) in enumerate(zip(tokens_index,code_tokens)):
            index_to_code[index]=(idx,code)  
        try:
            DFG,_=parser[1](root_node,index_to_code,{}) 
        except:
            DFG=[]
        DFG=sorted(DFG,key=lambda x:x[1])
        indexs=set()
        for d in DFG:
            if len(d[-1])!=0:
                indexs.add(d[1])
            for x in d[-1]:
                indexs.add(x)
        new_DFG=[]
        for d in DFG:
            if d[1] in indexs:
                new_DFG.append(d)
        dfg=new_DFG
    except:
        dfg=[]
    return code_tokens,dfg



#remove comments, tokenize code and extract dataflow     
def rename(code, parser,lang):
    #remove comments
    try:
        code=remove_comments_and_docstrings(code,lang)
    except:
        pass    
    #obtain dataflow
    if lang=="php":
        code="<?php"+code+"?>"    

    tree = parser[0].parse(bytes(code,'utf8'))    
    root_node = tree.root_node  
    tokens_index, tokens_index_types = tree_to_token_index_with_type(root_node)     
    code=code.split('\n')
    code_tokens=[index_to_code_token(x,code) for x in tokens_index]  
    index_to_code={}
    for idx,(index,code_piece) in enumerate(zip(tokens_index,code_tokens)):
        index_to_code[index]=(idx,code_piece)  

    h=tree_to_variable_index(root_node,index_to_code )    
    code_tokens_type = [index_to_code_token_types(x,code) for x in tokens_index_types] 
    code_tokens_type = [ c for c in code_tokens_type if len(c) ]

    code_tokens_variables = [index_to_code_token_variables(x,code) for x in tokens_index_types] 
    code_tokens_variables = [ c for c in code_tokens_variables if len(c) ]
    return code_tokens, code_tokens_type, code_tokens_variables


 

# In[4]:


def set_seed(seed=42):
    random.seed(seed)
    os.environ['PYHTONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True




# In[9]:


# from pprint import pprint
# train_examples = read_examples(args.train_filename)
# #train_features = convert_examples_to_features(train_examples, tokenizer,args,stage='train')
#  ##extract data flow
# example=train_examples[1]
# code_tokens,dfg=extract_dataflow(example.source,
#                                          parsers["c_sharp" if args.source_lang == "cs" else "java"],
#                                          "c_sharp" if args.source_lang == "cs" else "java")
# with open("tmp.java", "w") as f:
#         f.write(example.source)

# from scripts_preprocess.alignment import normalization_code
# _, newcode=normalization_code( example.source )
# print("normalized code")
# print( example.source )
# print(newcode)
# print(tokenizer.tokenize( newcode ))    
# print("tree-sitter")
# print(" ".join(code_tokens))
# print(tokenizer.tokenize(" ".join(code_tokens)))    


# code_tokens=[tokenizer.tokenize('@ '+x)[1:] if idx!=0 else tokenizer.tokenize(x) for idx,x in enumerate(code_tokens)]
# code_tokens=[y for x in code_tokens for y in x]  
# print(code_tokens)


# In[ ]:




