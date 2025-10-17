from functools import partial
import os
import math
import torch
import torch.nn as nn
from tqdm import tqdm
from torch.cuda.amp import autocast as autocast
from timm.models.layers import trunc_normal_
from timm.models.vision_transformer import PatchEmbed, Attention, Mlp
import einops


def normalize_to_neg_one_to_one(img):
    # [0.0, 1.0] -> [-1.0, 1.0]
    return img * 2 - 1

def unnormalize_to_zero_to_one(t):
    # [-1.0, 1.0] -> [0.0, 1.0]
    return (t + 1) * 0.5

def linear_beta_schedule(timesteps, beta1, beta2):
    assert 0.0 < beta1 < beta2 < 1.0, "beta1 and beta2 must be in (0, 1)"
    return torch.linspace(beta1, beta2, timesteps)

def cosine_beta_schedule(timesteps, s = 0.008):
    """
    cosine schedule
    as proposed in https://openreview.net/forum?id=-NEXDKk8gZ
    """
    steps = timesteps + 1
    t = torch.linspace(0, timesteps, steps) / timesteps # dtype = torch.float64
    alphas_cumprod = torch.cos((t + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0, 0.999)

def schedules(betas, T, device, type='DDPM'):
    if betas == 'cosine':
        schedule_fn = cosine_beta_schedule
    else:
        beta1, beta2 = betas
        schedule_fn = partial(linear_beta_schedule, beta1=beta1, beta2=beta2)

    if type == 'DDPM':
        beta_t = torch.cat([torch.tensor([0.0]), schedule_fn(T)])

    sqrt_beta_t = torch.sqrt(beta_t)
    alpha_t = 1 - beta_t
    log_alpha_t = torch.log(alpha_t)
    alphabar_t = torch.cumsum(log_alpha_t, dim=0).exp()

    sqrtab = torch.sqrt(alphabar_t)
    oneover_sqrta = 1 / torch.sqrt(alpha_t)

    sqrtmab = torch.sqrt(1 - alphabar_t)
    ma_over_sqrtmab = (1 - alpha_t) / sqrtmab

    dic = {
        "alpha_t": alpha_t,
        "oneover_sqrta": oneover_sqrta,
        "sqrt_beta_t": sqrt_beta_t,
        "alphabar_t": alphabar_t,
        "sqrtab": sqrtab,
        "sqrtmab": sqrtmab,
        "ma_over_sqrtmab": ma_over_sqrtmab,
    }
    return {key: dic[key].to(device) for key in dic}

def timestep_embedding(timesteps, dim, max_period=10000):
    """
    Create sinusoidal timestep embeddings.

    :param timesteps: a 1-D Tensor of N indices, one per batch element.
                      These may be fractional.
    :param dim: the dimension of the output.
    :param max_period: controls the minimum frequency of the embeddings.
    :return: an [N x dim] Tensor of positional embeddings.
    """
    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
    ).to(device=timesteps.device)
    args = timesteps[:, None].float() * freqs[None]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    return embedding

def unpatchify(x, channels=3):
    patch_size = int((x.shape[2] // channels) ** 0.5)
    h = w = int(x.shape[1] ** .5)
    assert h * w == x.shape[1] and patch_size ** 2 * channels == x.shape[2]
    x = einops.rearrange(x, 'B (h w) (p1 p2 C) -> B C (h p1) (w p2)', h=h, p1=patch_size, p2=patch_size)
    return x

class ClassEmbeddingTable(nn.Module):
    """
    Embeds class labels into vector representations. Also handles label dropout for classifier-free guidance.
    """
    def __init__(self, n_classes, n_channels):
        super().__init__()
        self.n_classes = n_classes
        self.embedding_table = nn.Embedding(n_classes + 1, n_channels)

    def forward(self, c, drop_mask):
        assert (c < self.n_classes).all()
        c = torch.where(drop_mask==0, c, self.n_classes)
        assert ((c == self.n_classes) == (drop_mask)).all()

        embeddings = self.embedding_table(c)
        return embeddings


class Block(nn.Module):
    def __init__(self, dim, num_heads, skip=False):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = Attention(dim, num_heads=num_heads)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = Mlp(in_features=dim, hidden_features=int(dim * 4))
        self.skip_linear = nn.Linear(2 * dim, dim) if skip else None

    def forward(self, x, skip=None):
        if self.skip_linear is not None:
            x = self.skip_linear(torch.cat([x, skip], dim=-1))
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x

class FinalLayer(nn.Module):
    def __init__(self, dim, patch_size, out_channels, extras, conv=True):
        super().__init__()
        self.extras = extras
        self.out_channels = out_channels

        self.norm = nn.LayerNorm(dim)
        self.linear = nn.Linear(dim, patch_size * patch_size * out_channels, bias=True)
        self.conv = nn.Conv2d(out_channels, out_channels, 3, padding=1) if conv else nn.Identity()

    def forward(self, x):
        x = self.norm(x)
        x = self.linear(x)
        x = x[:, self.extras:, :]
        x = unpatchify(x, self.out_channels)
        x = self.conv(x)
        return x

class UViT(nn.Module):
    def __init__(self, image_shape = [3, 32, 32], embed_dim = 512,
                 patch_size = 2, depth = 12, num_heads = 8,
                 final_conv=True, skip=True, n_classes=10):
        super().__init__()
        self.embed_dim = embed_dim
        self.in_chn = image_shape[0]

        self.patch_embed = PatchEmbed(
            img_size=image_shape[1:], patch_size=patch_size, in_chans=self.in_chn, embed_dim=embed_dim)
        num_patches = self.patch_embed.num_patches

        self.num_classes = n_classes
        self.label_emb = ClassEmbeddingTable(n_classes, embed_dim)
        self.extras = 2

        self.pos_embed = nn.Parameter(torch.zeros(1, self.extras + num_patches, embed_dim))

        self.in_blocks = nn.ModuleList([Block(dim=embed_dim, num_heads=num_heads) for _ in range(depth // 2)])
        self.mid_block = Block(dim=embed_dim, num_heads=num_heads)
        self.out_blocks = nn.ModuleList([Block(dim=embed_dim, num_heads=num_heads, skip=skip) for _ in range(depth // 2)])

        self.final = FinalLayer(embed_dim, patch_size, self.in_chn, self.extras, final_conv)

        trunc_normal_(self.pos_embed, std=.02)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed'}

    def forward(self, x, timesteps, c=None, drop_mask=None):
        x = self.patch_embed(x)
        B, L, D = x.shape

        if timesteps.shape[0] == 1:
            timesteps = timesteps.repeat(x.shape[0])
        time_token = timestep_embedding(timesteps, self.embed_dim)
        time_token = time_token.unsqueeze(dim=1)
        x = torch.cat((time_token, x), dim=1)
        if c is not None:
            label_emb = self.label_emb(c, drop_mask)
            label_emb = label_emb.unsqueeze(dim=1)
            x = torch.cat((label_emb, x), dim=1)
        x = x + self.pos_embed

        skips = []
        for blk in self.in_blocks:
            x = blk(x)
            skips.append(x)

        x = self.mid_block(x)

        for blk in self.out_blocks:
            x = blk(x, skips.pop())

        x = self.final(x)
        return x



class DDPM_guide(nn.Module):

    def __init__(self, device):
        super(DDPM_guide, self).__init__()
        self.nn_model = UViT().to(device)
        params = sum(p.numel() for p in nn_model.parameters() if p.requires_grad) / 1e6
        print(f"nn model # params: {params:.1f}")

        self.device = device
        self.n_T = 1000
        self.betas = [1.0e-4, 0.02]
        self.ddpm_sche = schedules(self.betas, self.n_T, device, 'DDPM')
        self.ddim_sche = schedules(self.betas, self.n_T, device, 'DDIM')
        self.drop_prob = 0.1
        self.loss = nn.MSELoss()

    def perturb(self, x, t=None):
        ''' Add noise to a clean image (diffusion process).

            Args:
                x: The normalized image tensor.
                t: The specified timestep ranged in `[1, n_T]`. Type: int / torch.LongTensor / None. \
                    Random `t ~ U[1, n_T]` is taken if t is None.
            Returns:
                The perturbed image, the corresponding timestep, and the noise.
        '''
        if t is None:
            t = torch.randint(1, self.n_T + 1, (x.shape[0], )).to(self.device)
        elif not isinstance(t, torch.Tensor):
            t = torch.tensor([t]).to(self.device).repeat(x.shape[0])

        noise = torch.randn_like(x)
        sche = self.ddpm_sche
        x_noised = (sche["sqrtab"][t, None, None, None] * x +
                    sche["sqrtmab"][t, None, None, None] * noise)
        return x_noised, t, noise

    def forward(self, x, c, use_amp=False):
        ''' Training with simple noise prediction loss.

            Args:
                x: The clean image tensor ranged in `[0, 1]`.
                c: The label for class-conditional generation.
            Returns:
                The simple MSE loss.
        '''
        x = normalize_to_neg_one_to_one(x)
        x_noised, t, noise = self.perturb(x, t=None)

        # 0 for conditional, 1 for unconditional
        mask = torch.bernoulli(torch.zeros_like(c) + self.drop_prob).to(self.device)

        with autocast(enabled=use_amp):
            return self.loss(noise, self.nn_model(x_noised, t / self.n_T, c, mask))

    def sample(self, n_sample, size, guide_w=0.3, notqdm=False, use_amp=False):
        ''' Sampling with DDPM sampler. Actual NFE is `2 * n_T`.

            Args:
                n_sample: The batch size.
                size: The image shape (e.g. `(3, 32, 32)`).
                guide_w: The CFG scale.
            Returns:
                The sampled image tensor ranged in `[0, 1]`.
        '''
        sche = self.ddpm_sche
        model_args = self.prepare_condition_(n_sample)
        x_i = torch.randn(n_sample, *size).to(self.device)

        for i in tqdm(range(self.n_T, 0, -1), disable=notqdm):
            t_is = torch.tensor([i / self.n_T]).to(self.device).repeat(n_sample)

            z = torch.randn(n_sample, *size).to(self.device) if i > 1 else 0

            alpha = sche["alphabar_t"][i]
            eps, _ = self.pred_eps_(x_i, t_is, model_args, guide_w, alpha, use_amp)

            mean = sche["oneover_sqrta"][i] * (x_i - sche["ma_over_sqrtmab"][i] * eps)
            variance = sche["sqrt_beta_t"][i] # LET variance sigma_t = sqrt_beta_t
            x_i = mean + variance * z

        return unnormalize_to_zero_to_one(x_i)

    def ddim_sample(self, n_sample, size, steps=100, eta=0.0, guide_w=0.3, notqdm=False, use_amp=False):
        ''' Sampling with DDIM sampler. Actual NFE is `2 * steps`.

            Args:
                n_sample: The batch size.
                size: The image shape (e.g. `(3, 32, 32)`).
                steps: The number of total timesteps.
                eta: controls stochasticity. Set `eta=0` for deterministic sampling.
                guide_w: The CFG scale.
            Returns:
                The sampled image tensor ranged in `[0, 1]`.
        '''
        sche = self.ddim_sche
        model_args = self.prepare_condition_(n_sample)
        x_i = torch.randn(n_sample, *size).to(self.device)

        times = torch.arange(0, self.n_T, self.n_T // steps) + 1
        times = list(reversed(times.int().tolist())) + [0]
        time_pairs = list(zip(times[:-1], times[1:]))
        # e.g. [(801, 601), (601, 401), (401, 201), (201, 1), (1, 0)]

        for time, time_next in tqdm(time_pairs, disable=notqdm):
            t_is = torch.tensor([time / self.n_T]).to(self.device).repeat(n_sample)

            z = torch.randn(n_sample, *size).to(self.device) if time_next > 0 else 0

            alpha = sche["alphabar_t"][time]
            eps, x0_t = self.pred_eps_(x_i, t_is, model_args, guide_w, alpha, use_amp)
            alpha_next = sche["alphabar_t"][time_next]
            c1 = eta * ((1 - alpha / alpha_next) * (1 - alpha_next) / (1 - alpha)).sqrt()
            c2 = (1 - alpha_next - c1 ** 2).sqrt()
            x_i = alpha_next.sqrt() * x0_t + c2 * eps + c1 * z

        return unnormalize_to_zero_to_one(x_i)

    def pred_eps_(self, x, t, model_args, guide_w, alpha, use_amp, clip_x=True):
        def pred_cfg_eps_double_batch():
            # double batch
            x_double = x.repeat(2, 1, 1, 1)
            t_double = t.repeat(2)

            with autocast(enabled=use_amp):
                eps = self.nn_model(x_double, t_double, *model_args).float()
            n_sample = eps.shape[0] // 2
            eps1 = eps[:n_sample]
            eps2 = eps[n_sample:]
            assert eps1.shape == eps2.shape
            eps = (1 + guide_w) * eps1 - guide_w * eps2
            return eps

        def pred_eps_from_x0(x0):
            return (x - x0 * alpha.sqrt()) / (1 - alpha).sqrt()

        def pred_x0_from_eps(eps):
            return (x - (1 - alpha).sqrt() * eps) / alpha.sqrt()

        # get prediction of x0
        eps = pred_cfg_eps_double_batch()
        denoised = pred_x0_from_eps(eps)

        # pixel-space clipping (optional)
        if clip_x:
            denoised = torch.clip(denoised, -1., 1.)
            eps = pred_eps_from_x0(denoised)
        return eps, denoised

    def prepare_condition_(self, n_sample):
        n_classes = self.nn_model.num_classes
        assert n_sample % n_classes == 0
        c = torch.arange(n_classes).to(self.device)
        c = c.repeat(n_sample // n_classes)
        c = c.repeat(2)

        # 0 for conditional, 1 for unconditional
        mask = torch.zeros_like(c).to(self.device)
        mask[n_sample:] = 1.
        return c, mask
