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

class EdgeModel(nn.Module):   
    def __init__(self, encoder,config ,args):
        super(EdgeModel, self).__init__()
        self.encoder = encoder
        self.config=config
      #  self.tokenizer=tokenizer
        self.classifier=RobertaClassificationHead(config, args)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.selfattetionpool=SelfAttentiveSpanExtractor(config.hidden_size)
        self.args=args

    # def freeze_encoder(self):
    #     for name, param in self.encoder.named_parameters():
    #         param.requires_grad = False  
        
    def forward(self, inputs_ids_1,attn_mask_1, span_1, span_2, labels=None, layer=-1,): 
       
        # pass the inputs to the model
        # https://huggingface.co/docs/transformers/main/en/main_classes/output
        #index=0 if layer==-1 else layer
        mask = inputs_ids_1.ne(self.config.pad_token_id) # mask.unsqueeze(1) * mask.unsqueeze(2)
        outputs = self.encoder(inputs_ids_1, attention_mask=attn_mask_1, return_dict=True)
        emb_1 = outputs.hidden_states[layer]
        seq_1 = self.selfattetionpool( emb_1, span_1  )
        seq_1 = torch.sum(seq_1, 1)
      #  logger.info(f"emb_1 shape {emb_1.shape}, seq_1 shape {seq_1.shape}")
        seq_2 = self.selfattetionpool( emb_1, span_2  )
        seq_2 = torch.sum(seq_2, 1)
        rep = torch.cat( (seq_1,  seq_2 ), dim=1 )
        rep = self.dropout(rep)
        logits=self.classifier(rep)
        # shape: [batch_size, num_classes]
        prob=F.softmax(logits, dim=-1)
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits, labels)
            return loss,prob
        else:
            return prob

class FasterEdgeModel(nn.Module):   
    def __init__(self,config ,args):
        super(FasterEdgeModel, self).__init__()
        self.config=config
      #  self.tokenizer=tokenizer
        self.classifier=RobertaClassificationHead(config, args)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.selfattetionpool=SelfAttentiveSpanExtractor(config.hidden_size)
        self.args=args


        
    def forward(self, input_features, span_1, span_mask1, span_2, span_mask2,labels=None ): 
        # pass the inputs to the model
        # https://huggingface.co/docs/transformers/main/en/main_classes/output

        seq_1 = self.selfattetionpool( input_features, span_1 , span_indices_mask=span_mask1 )
        seq_1 = torch.sum(seq_1, 1)
        seq_2 = self.selfattetionpool( input_features, span_2 , span_indices_mask=span_mask2 )
        seq_2 = torch.sum(seq_2, 1)
        rep = torch.cat( (seq_1,  seq_2 ), dim=1 )
        rep = self.dropout(rep)
        logits=self.classifier(rep)

        prob=F.softmax(logits, dim=-1)
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits, labels)
            return loss,prob
        else:
            return prob

class InferenceModel(nn.Module):   
    def __init__(self, encoder,config ,args):
        super(InferenceModel, self).__init__()
        self.encoder = encoder
        self.config=config
        self.args=args
        
    def forward(self, batch_data):
        # pass the inputs to the model
        # https://huggingface.co/docs/transformers/main/en/main_classes/output
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




