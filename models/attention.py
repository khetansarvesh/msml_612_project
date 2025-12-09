import torch
import torch.nn as nn
import math


def repeat_kv(x, n_rep):
    batch, kv_heads, seq_len, head_dim = x.shape # in this case kv_heads = 2
    return (
              x[:, :, None, :, :]
              .expand(batch, kv_heads, n_rep, seq_len, head_dim)
              .reshape(batch, kv_heads * n_rep, seq_len, head_dim)
            )

class GQA(nn.Module):

    def __init__(self, dim=768, num_heads_q=8, num_heads_kv=2, head_dim=64):
        super().__init__()
        self.dim = dim
        self.num_heads_q = num_heads_q
        self.num_heads_kv = num_heads_kv
        self.head_dim = head_dim
        
        # Linear projections for Q, K, V
        self.q_proj = nn.Linear(dim, num_heads_q * head_dim, bias=True)
        self.k_proj = nn.Linear(dim, num_heads_kv * head_dim, bias=True)
        self.v_proj = nn.Linear(dim, num_heads_kv * head_dim, bias=True)
        
        # Output projection
        self.out_proj = nn.Linear(num_heads_q * head_dim, dim)

    def forward(self, x):
        bsz, q_len, _ = x.size() # [Batch_Size, Seq_Len, 768]

        '''Projecting X to get K Q and V'''
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        '''Splitting'''
        q = q.view(bsz, q_len, self.num_heads_q, self.head_dim).transpose(1, 2)
        k = k.view(bsz, q_len, self.num_heads_kv, self.head_dim).transpose(1, 2)
        v = v.view(bsz, q_len, self.num_heads_kv, self.head_dim).transpose(1, 2)

        '''Repeat the key and values to match the number of heads of the query'''
        n_rep = self.num_heads_q // self.num_heads_kv
        k = repeat_kv(k, n_rep) # cause we need to repeat 4 times (2*4 = 8)
        v = repeat_kv(v, n_rep) # cause we need to repeat 4 times (2*4 = 8)

        ''' Calculating Attention '''
        att = torch.matmul(q, k.transpose(2, 3)) / math.sqrt(self.head_dim)

        ''' Passing Attention to Softmax'''
        att = nn.functional.softmax(att, dim=-1, dtype=torch.float32).to(q.dtype)
  
        ''' Multiplying attention and value vector'''       
        out = torch.matmul(att, v)

        # Make sure the sequence length is the second dimension. # [Batch_Size, Num_Heads_Q, Seq_Len_Q, Head_Dim] -> [Batch_Size, Seq_Len_Q, Num_Heads_Q, Head_Dim]
        out = out.transpose(1, 2).contiguous()

        ''' Concatenate all the heads together ''' # [Batch_Size, Seq_Len_Q, Num_Heads_Q, Head_Dim] -> [Batch_Size, Seq_Len_Q, Num_Heads_Q * Head_Dim]
        out = out.view(bsz, q_len, -1)

        '''Projecting back to original dimension'''
        out = self.out_proj(out)

        return out, att

class TransformerEncoderLayer(nn.Module):

    def __init__(self, dim=768, num_heads_q=8, num_heads_kv=2, head_dim=64):
        super().__init__()

        # Attention Block
        self.att_norm = nn.LayerNorm(dim, elementwise_affine=False, eps=1E-6)
        self.attn_block = GQA(dim=dim, num_heads_q=num_heads_q, num_heads_kv=num_heads_kv, head_dim=head_dim)
        
        # FFNN block
        self.ff_norm = nn.LayerNorm(dim, elementwise_affine=False, eps=1E-6)
        self.mlp_block = nn.Sequential(nn.Linear(dim, 4 * dim), 
                                        nn.GELU(approximate='tanh'), 
                                        nn.Linear(4 * dim, dim)
                                      )

    def forward(self, x):

        # attention block
        attn_norm_output = self.att_norm(x)
        attn_output, _ = self.attn_block(attn_norm_output)
        out = x + attn_output
        
        # FFNN block
        mlp_norm_output = self.ff_norm(out)
        out = out + self.mlp_block(mlp_norm_output)

        return out