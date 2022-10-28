import torch
import torch.nn as nn
import torch.nn.functional as F
from math import sqrt



class Multi_Interactive_attention(nn.Module):
    def __init__(self, config):
        super(Multi_Interactive_attention, self).__init__()
        self.token_emb = Embeddings(config)
        self.seq_info_intered = TransformerForSequenceClassification(config)

    def forward(self, input_ids_1, input_ids_2):
        inputs_embeds_1 = self.token_emb(input_ids_1)
        inputs_embeds_2 = self.token_emb(input_ids_2)
        sequence_output_1 , sequence_output_2  = self.seq_info_intered(inputs_embeds_1, inputs_embeds_2)
        # extract events
        return sequence_output_1 , sequence_output_2

class Embeddings(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.token_embeddings = nn.Embedding(config.vocab_size, config.hidden_size)
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        self.layer_norm = nn.LayerNorm(config.hidden_size, eps=1e-12)
        self.dropout = nn.Dropout()
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    def forward(self, input_ids):
        # Create position IDs for input sequence
        seq_length = input_ids.size(1)
        position_ids = torch.arange(seq_length, dtype=torch.long).unsqueeze(0).to(self.device)
        # Create token and position embeddings
        token_embeddings = self.token_embeddings(input_ids)
        position_embeddings = self.position_embeddings(position_ids)
        # Combine token and position embeddings
        embeddings = token_embeddings + position_embeddings
        embeddings = self.layer_norm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings

class TransformerForSequenceClassification(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.encoder = TransformerEncoder(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)

    def forward(self, x1, x2):
        # print(x1)
        x1, x2 = self.encoder(x1, x2) # x1 [batch,seq,768]
        x1 = self.dropout(x1)
        x2 = self.dropout(x2)
        return x1, x2 

class TransformerEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        # self.embeddings = Embeddings(config)
        self.layers = nn.ModuleList([TransformerEncoderLayer(config) for _ in range(config.num_hidden_layers)])

    def forward(self, x1, x2):
        # x1 = self.embeddings(x1)
        # x2 = self.embeddings(x2)
        for layer in self.layers:
            x1, x2 = layer(x1, x2)
        return x1, x2

class TransformerEncoderLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.layer_norm_1 = nn.LayerNorm(config.hidden_size)
        self.layer_norm_2 = nn.LayerNorm(config.hidden_size)
        self.attention = MultiHeadAttention(config)
        self.feed_forward = FeedForward(config)

    def forward(self, x1, x2):
        # x1 [b,s,768]
        # x2 [b,s,768]
        # Apply layer normalization and then copy input into query, key, value
        hidden_state_1 = self.layer_norm_1(x1)
        hidden_state_2 = self.layer_norm_1(x2)
        # Apply attention with a skip connection
        x1 = x1 + self.attention(hidden_state_2, hidden_state_1)
        x2 = x2 + self.attention(hidden_state_1, hidden_state_2)
        # Apply feed-forward layer with a skip connection
        x1 = x1 + self.feed_forward(self.layer_norm_2(x1))
        x2 = x2 + self.feed_forward(self.layer_norm_2(x2))
        return x1, x2

class LayerNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-12):
        """Construct a layernorm module in the TF style (epsilon inside the square root). """
        super(LayerNorm, self).__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.bias = nn.Parameter(torch.zeros(hidden_size))
        self.variance_epsilon = eps

    def forward(self, x):
        u = x.mean(-1, keepdim=True)
        s = (x - u).pow(2).mean(-1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.variance_epsilon)
        return self.weight * x + self.bias

class InteractiveOutput(nn.Module):
    def __init__(self, hidden_size, hidden_dropout_prob):
        super(InteractiveOutput, self).__init__()
        self.dense = nn.Linear(hidden_size, hidden_size)
        self.LayerNorm = LayerNorm(hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states

class MultiHeadAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.num_heads = config.num_attention_heads          #config.num_attention_heads=12
        self.embed_dim = config.hidden_size
        self.head_dim = int(config.hidden_size / self.num_heads) # head size 768/12 = 64     
        self.heads = nn.ModuleList([InteractiveAttention(self.head_dim, self.head_dim) for _ in range(self.num_heads)])
        self.output_linear = nn.Linear(self.embed_dim, self.embed_dim)
        
    def forward(self, hidden_state_1, hidden_state_2):
        # hidden_state_1 [batch, seq, 768] --> [headnum, batch, seq ,64]
        new_x_shape_for_h1 = (hidden_state_1.size()[0],hidden_state_1.size()[1],self.num_heads,self.head_dim) # [b,s,h,64]
        new_x_shape_for_h2 = (hidden_state_2.size()[0],hidden_state_2.size()[1],self.num_heads,self.head_dim)
        hidden_state_1 = hidden_state_1.view(*new_x_shape_for_h1)
        hidden_state_2 = hidden_state_2.view(*new_x_shape_for_h2)
        hidden_state_1 = hidden_state_1.permute(2,0,1,3).contiguous()
        hidden_state_2 = hidden_state_2.permute(2,0,1,3).contiguous()
        # x = torch.cat([h(hidden_state_1, hidden_state_2) for h in self.heads], dim=-1)
        x = torch.cat([h(hidden_state_1[idx,:,:,:], hidden_state_2[idx,:,:,:]) for idx,h in enumerate(self.heads)], dim=-1)
        x = self.output_linear(x)
        return x


class InteractiveAttention(nn.Module):
    def __init__(self, embed_dim, head_dim):
        super(InteractiveAttention, self).__init__()
        self.query = nn.Linear(embed_dim, head_dim)
        self.key = nn.Linear(embed_dim, head_dim)
        self.value = nn.Linear(embed_dim, head_dim)

        self.seq_out = InteractiveOutput(embed_dim, 0.1)
        # self.seq_out = InteractiveOutput(config.hidden_size, config.hidden_dropout_prob)
        
    def scaled_dot_product_attention(self, query, key, value):
        dim_k = query.size(-1) # 64 
        scores = torch.bmm(query, key.transpose(-2, -1)) / sqrt(dim_k)
        weights = F.softmax(scores, dim=-1)
        return torch.bmm(weights, value)

    def forward(self, hidden_state_1, hidden_state_2):
        # hidden_state_1 [batch seq 64]
        attn_outputs = self.scaled_dot_product_attention(self.query(hidden_state_2), self.key(hidden_state_1), self.value(hidden_state_1))
        seq_out = self.seq_out(attn_outputs, hidden_state_2)
        return seq_out

class FeedForward(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.linear_1 = nn.Linear(config.hidden_size, config.intermediate_size)
        self.linear_2 = nn.Linear(config.intermediate_size, config.hidden_size)
        self.gelu = nn.GELU()
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, x):
        x = self.linear_1(x)
        x = self.gelu(x)
        x = self.linear_2(x)
        x = self.dropout(x)
        return x







class WfTransformer(nn.Module):
    def __init__(self, c_in=30522, c_out=30522, time_dim=256, device="cuda"):
        super().__init__()
        self.device = device
        self.time_dim = time_dim
        self.inc = nn.Linear(c_in,512)
        self.down1 = nn.Linear(512, 1024)
        self.outc = nn.Linear(1024, c_out)

    def pos_encoding(self, t, channels):
        inv_freq = 1.0 / (
            10000
            ** (torch.arange(0, channels, 2, device=self.device).float() / channels)
        )
        pos_enc_a = torch.sin(t.repeat(1, channels // 2) * inv_freq)
        pos_enc_b = torch.cos(t.repeat(1, channels // 2) * inv_freq)
        pos_enc = torch.cat([pos_enc_a, pos_enc_b], dim=-1)
        return pos_enc

    def forward(self, x, t):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        output = self.outc(x2)
        return output


if __name__ == '__main__':
    pass