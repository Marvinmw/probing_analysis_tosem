from torch  import nn
class BERTPoSTagger(nn.Module):
    def __init__(self,
                 encoder,
                 config,
                 num_class, 
                 layer
                ):
        
        super().__init__()
        
        self.encoder = encoder
        
        embedding_dim = encoder.config.to_dict()['hidden_size']
        
        self.fc = nn.Linear(embedding_dim, num_class)
        self.layer = layer
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        
    def forward(self, inputs_ids, attn_mask):
  
        #text = [sent len, batch size]
    
        #text = text.permute(1, 0)
        
        #text = [batch size, sent len]
        #inputs_ids, attention_mask=attn_mask, return_dict=True
        
        embedded = self.encoder(inputs_ids, attention_mask=attn_mask,return_dict=True)
        embedded = embedded.hidden_states[self.layer]
        embedded = self.dropout(embedded)
        
        #embedded = [batch size, seq len, emb dim]
                
        #embedded = embedded.permute(1, 0, 2)
                    
        #embedded = [sent len, batch size, emb dim]
        
        predictions = self.fc(embedded)
        
        #predictions = [sent len, batch size, output dim]
        
        return predictions