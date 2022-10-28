'''
    手动实现一个transformer
'''
import torch
import torch.nn as nn
from math import sqrt
import torch.nn.functional as F

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

class MultiHeadAttention(nn.Module):
    def __init__(self, num_attention_heads=12,hidden_size=768):
        super().__init__()
        self.num_heads = num_attention_heads         #config.num_attention_heads=12
        self.embed_dim = hidden_size
        self.head_dim = int(hidden_size / self.num_heads) # head size 768/12 = 64     
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
        x = torch.cat([h(hidden_state_1[idx,:,:,:], hidden_state_2[idx,:,:,:]) for idx,h in enumerate(self.heads)], dim=-1)
        x = self.output_linear(x)
        return x

class FeedForward(nn.Module):
    def __init__(self, hidden_size=768, intermediate_size=512):
        super().__init__()
        self.linear_1 = nn.Linear(hidden_size, intermediate_size)
        self.linear_2 = nn.Linear(intermediate_size, hidden_size)
        self.gelu = nn.GELU()
        self.dropout = nn.Dropout(0.15)

    def forward(self, x):
        x = self.linear_1(x)
        x = self.gelu(x)
        x = self.linear_2(x)
        x = self.dropout(x)
        return x

class TransformerEncoderLayer(nn.Module):
    def __init__(self, hidden_size=768):
        super().__init__()
        self.layer_norm_1 = nn.LayerNorm(hidden_size)
        self.layer_norm_2 = nn.LayerNorm(hidden_size)
        self.attention = MultiHeadAttention(hidden_size=hidden_size)
        self.feed_forward = FeedForward(hidden_size=hidden_size)

    def forward(self, x1):
        # Apply layer normalization and then copy input into query, key, value
        hidden_state_1 = self.layer_norm_1(x1)
        # Apply attention with a skip connection
        x1 = x1 + self.attention(hidden_state_1, hidden_state_1)
        # Apply feed-forward layer with a skip connection
        x1 = x1 + self.feed_forward(self.layer_norm_2(x1))
        return x1

class WfTransformer(nn.Module):
    def __init__(self, c_in=30522, c_out=30522, time_dim=768, device="cuda"):
        super().__init__()
        self.device = device
        self.time_dim = time_dim
        self.time_embeddings = nn.Embedding(1000, time_dim)
        self.inc = nn.Linear(c_in,768)
        self.block = TransformerEncoderLayer()
        self.outc = nn.Linear(768, c_out)

    def forward(self, x, t):
        x1 = self.inc(x)
        time_emb = self.time_embeddings(t)
        # print(x1.size())
        # print(time_emb.size())
        x1 = x1 + time_emb[:,None,:] 
        x2 = self.block(x1)
        output = self.outc(x2)
        return output

if __name__ == '__main__':
    # a = torch.randn((2,128,30522))
    # myModel = WfTransformer()
    # b = myModel(a,torch.tensor(0))
    # print(b.size())
    pass

