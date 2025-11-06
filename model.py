import torch
import torch.nn as nn
from einops import rearrange
from attention import TransformerEncoderLayer


def get_time_embedding(time_steps, # 1D array of timesteps eg [1,10,500,40,300]
                       temb_dim): # dimension of vector to which each of these timestep needs to be converted to eg 128

    # factor = 10000^(2i/d_model)
    factor = 10000 ** ((torch.arange(start=0, end=temb_dim // 2, dtype=torch.float32, device=time_steps.device) / (temb_dim // 2)))

    # pos / factor
    t_emb = time_steps[:, None].repeat(1, temb_dim // 2) / factor

    # now taking sin and cos of t_emb
    return torch.cat([torch.sin(t_emb), torch.cos(t_emb)], dim=-1)

class DIT(nn.Module):

    def __init__(self):
        super().__init__()

        self.patch_embedding = nn.Sequential(nn.LayerNorm(1*4*4), nn.Linear(1*4*4, 768), nn.LayerNorm(768))
        self.position_embedding = nn.Parameter(data=torch.randn(1, 49, 768),requires_grad=True)
        self.embedding_dropout = nn.Dropout(p=0.1)

        # self.transformer_encoder = nn.TransformerEncoder(encoder_layer=nn.TransformerEncoderLayer(d_model=768,
        #                                                                                           nhead=2,
        #                                                                                           dim_feedforward=3072,
        #                                                                                           activation="gelu",
        #                                                                                           batch_first=True,
        #                                                                                           norm_first=True), # Create a single Transformer Encoder Layer
        #                                                 num_layers=2) # Stack it N times

        self.transformer_encoder_layer = TransformerEncoderLayer(dim=768, num_heads_q=8, num_heads_kv=2, head_dim=64)
        self.transformer_encoder = nn.Sequential(self.transformer_encoder_layer,self.transformer_encoder_layer)

        # Final Linear Layer
        self.proj_out = nn.Linear(768, 1*4*4)

        # Time projection
        self.ti_1 = nn.Linear(128, 400)
        self.ti_2 = nn.Linear(400, 768)

        # Class conditioning projection (to be added into the time embedding)
        self.class_emb = nn.Embedding(10, 128)
        self.cls_ti_1 = nn.Linear(128, 400)
        self.cls_ti_2 = nn.Linear(400, 768)

    def forward(self, x, t, y):

        # getting time embeddings (expects numeric timesteps, shape [B])
        t_emb = get_time_embedding(torch.as_tensor(t).long(), 128)
        time_proj1 = self.ti_1(t_emb)
        time_proj2 = self.ti_2(time_proj1)  # [B, 768]

        # --- class conditioning: project class id and add into the time embedding ---
        y_tensor = torch.as_tensor(y).long().to(x.device)
        cls_emb = self.class_emb(y_tensor)               
        cls_proj1 = self.cls_ti_1(cls_emb)
        cls_proj2 = self.cls_ti_2(cls_proj1)  # [B, 768]

        # add class projection into the time projection so the time token carries class info
        time_proj2 = time_proj2 + cls_proj2

        # Reshaping time embedding into a token [B, 1, 768]
        t_reshaped = time_proj2.unsqueeze(1)


        # 32, 1, 28, 28 -> 32, 1, 7*4, 7*4 -> 32, 1, 7, 7, 4, 4 -> 32, 7, 7, 4, 4, 1 -> 32, 7*7, 4*4*1 - > 32, num_patches, patch_dim
        x = rearrange(x, 'b c (nh ph) (nw pw) -> b (nh nw) (ph pw c)', ph=4, pw=4)

        # Create patch embedding for all images in the batch
        x = self.patch_embedding(x)

        #Add position embedding to patch embedding
        x = self.position_embedding + x

        # concatenating time (now also class-conditioned) embedding
        x = torch.cat([x, t_reshaped], dim=1)

        #Run embedding dropout
        x = self.embedding_dropout(x)

        #Pass patch, position and class embedding through transformer encoder layers (equations 2 & 3)
        x = self.transformer_encoder(x)

        # Unpatchify i.e. (B,patches,hidden_size) -> (B,patches,channels * patch_width * patch_height)
        x = self.proj_out(x[:, 1:]) # note here we are ignore the first vector which represents the matured time+class embedding

        # combine all the patches to form image
        x = rearrange(x, 'b (nh nw) (ph pw c) -> b c (nh ph) (nw pw)',ph=4,pw=4,nw=7,nh=7)
        return x