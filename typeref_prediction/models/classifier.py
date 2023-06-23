import torch
import torch.nn as nn
import torch
import torch.nn.functional as F
from .self_attention_pool import SelfAttentiveSpanExtractor
from torch.nn import CrossEntropyLoss

class RobertaClassificationHead(nn.Module):
    """Head for sentence-level classification tasks."""

    def __init__(self, config, args):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size*2, args.class_num)
     
    def forward(self, x): 
        x = self.dense(x)
        return x

class InferenceModel(nn.Module):   
    def __init__(self, encoder,config ,args):
        super(InferenceModel, self).__init__()
        self.encoder = encoder
        self.config=config
        self.args=args
        
    def forward(self, batch_data): 
       
        # pass the inputs to the model
        # https://huggingface.co/docs/transformers/main/en/main_classes/output
        #index=0 if layer==-1 else layer
        if len(batch_data) ==3 :
            (inputs_ids, attn_mask , position_idx) = batch_data
            outputs = self.forward_graph(inputs_ids,position_idx,attn_mask)
            return outputs.hidden_states
        else:
            (inputs_ids, attn_mask ) = batch_data
            outputs = self.encoder(inputs_ids, attention_mask=attn_mask, return_dict=True)
            return outputs.hidden_states

    def forward_graph(self, inputs_ids,position_idx,attn_mask):
        nodes_mask=position_idx.eq(0)
        token_mask=position_idx.ge(2)        
        inputs_embeddings=self.encoder.embeddings.word_embeddings(inputs_ids)
        nodes_to_token_mask=nodes_mask[:,:,None]&token_mask[:,None,:]&attn_mask
        nodes_to_token_mask=nodes_to_token_mask/(nodes_to_token_mask.sum(-1)+1e-10)[:,:,None]
        avg_embeddings=torch.einsum("abc,acd->abd",nodes_to_token_mask,inputs_embeddings)
        inputs_embeddings=inputs_embeddings*(~nodes_mask)[:,:,None]+avg_embeddings*nodes_mask[:,:,None]    

        outputs = self.encoder( inputs_embeds=inputs_embeddings,attention_mask=attn_mask,position_ids=position_idx,token_type_ids=position_idx.eq(-1).long(), return_dict=True )
        return outputs

class RobertaTypeRefClassificationHead(nn.Module):
    """Head for sentence-level classification tasks."""

    def __init__(self, config, args):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, args.class_num)

    def forward(self, x): 
        x = self.dense(x)
        return x


class FasterTypeRefModel(nn.Module):   
    def __init__(self,config ,args):
        super(FasterTypeRefModel, self).__init__()
        self.config=config
        self.classifier=RobertaTypeRefClassificationHead(config, args)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.args=args

        
    def forward(self, input_features, index_token, labels=None): 
        rep = self.dropout(input_features)
        logits=self.classifier(rep)
        prob=F.softmax(logits, dim=-1)
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits, labels)
            return loss,prob
        else:
            return prob



