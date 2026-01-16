import math
import numbers
import torch
import torch.nn as nn
import torch.nn.functional as F
from basicsr.utils.registry import ARCH_REGISTRY
from einops import rearrange, repeat
from torchvision.ops import DeformConv2d
from mamba_ssm.ops.selective_scan_interface import selective_scan_fn, selective_scan_ref


def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')


def to_4d(x, h, w):
    return rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)


class BiasFree_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(BiasFree_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)
        assert len(normalized_shape) == 1
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return x / torch.sqrt(sigma + 1e-5) * self.weight


class WithBias_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(WithBias_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)
        assert len(normalized_shape) == 1
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma + 1e-5) * self.weight + self.bias


class LayerNorm(nn.Module):
    def __init__(self, dim, LayerNorm_type):
        super(LayerNorm, self).__init__()
        if LayerNorm_type == 'BiasFree':
            self.body = BiasFree_LayerNorm(dim)
        else:
            self.body = WithBias_LayerNorm(dim)

    def forward(self, x):
        h, w = x.shape[-2:]
        return to_4d(self.body(to_3d(x)), h, w)


class AIFN(nn.Module):
    def __init__(self, dim, ffn_expansion_factor, bias=False):
        super(AIFN, self).__init__()
        hidden_features = int(dim * ffn_expansion_factor)
        self.project_in = nn.Conv2d(dim, hidden_features * 2, kernel_size=1, bias=bias)
        self.dwconv = nn.Conv2d(hidden_features * 2, hidden_features * 2, kernel_size=3, stride=1, padding=1,
                                groups=hidden_features * 2, bias=bias)
        self.project_out = nn.Conv2d(hidden_features, dim, kernel_size=1, bias=bias)
        self.affine_predictor = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(dim, hidden_features, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_features, hidden_features * 2, 1)
        )
        nn.init.zeros_(self.affine_predictor[-1].weight)
        if self.affine_predictor[-1].bias is not None:
            nn.init.zeros_(self.affine_predictor[-1].bias)

    def forward(self, x, history):
        x = self.project_in(x)
        x1, x2 = self.dwconv(x).chunk(2, dim=1)
        x_gated = F.gelu(x1) * x2
        affine_params = self.affine_predictor(history)
        gamma, beta = affine_params.chunk(2, dim=1)
        x_injected = x_gated * (1 + gamma) + beta
        x_out = self.project_out(x_injected)
        return x_out


class EAHA(nn.Module):
    def __init__(self, dim, num_heads, bias=False):
        super(EAHA, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))
        self.q_base_proj = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        self.q_base_dwconv = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, groups=dim, bias=bias)
        self.q_ref_proj = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        self.q_ref_dwconv = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, groups=dim, bias=bias)
        self.kv_proj = nn.Conv2d(dim, dim * 2, kernel_size=1, bias=bias)
        self.kv_dwconv = nn.Conv2d(dim * 2, dim * 2, kernel_size=3, stride=1, padding=1, groups=dim * 2, bias=bias)
        self.extremity_dwconv = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, groups=dim, bias=bias)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

    def forward(self, x, k_v):
        b, c, h, w = x.shape
        q_base = self.q_base_dwconv(self.q_base_proj(x))
        q_ref = self.q_ref_dwconv(self.q_ref_proj(k_v))
        kv = self.kv_dwconv(self.kv_proj(k_v))
        k, v = kv.chunk(2, dim=1)
        extremity_map = torch.sigmoid(self.extremity_dwconv(x))
        q_hybrid = (1 - extremity_map) * q_base + extremity_map * q_ref
        q = rearrange(q_hybrid, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        q = F.normalize(q, dim=-1)
        k = F.normalize(k, dim=-1)
        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)
        out = (attn @ v)
        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)
        out = self.project_out(out)
        return out


class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, ffn_expansion_factor, bias, LayerNorm_type):
        super(TransformerBlock, self).__init__()
        self.norm1 = LayerNorm(dim, LayerNorm_type)
        self.kv_norm = LayerNorm(dim, LayerNorm_type)
        self.attn = EAHA(dim, num_heads, bias)
        self.norm2 = LayerNorm(dim, LayerNorm_type)
        self.ffn = AIFN(dim, ffn_expansion_factor, bias)

    def forward(self, y):
        x = y[0]
        k_v = y[1]
        if k_v is None:
            k_v = x
        x = x + self.attn(self.norm1(x), self.kv_norm(k_v))
        x = x + self.ffn(self.norm2(x), self.kv_norm(k_v))
        return [x, k_v]


class OverlapPatchEmbed(nn.Module):
    def __init__(self, in_c=3, embed_dim=48, bias=False):
        super(OverlapPatchEmbed, self).__init__()
        self.proj = nn.Conv2d(in_c, embed_dim, kernel_size=3, stride=1, padding=1, bias=bias)

    def forward(self, x):
        return self.proj(x)


class Downsample(nn.Module):
    def __init__(self, n_feat):
        super(Downsample, self).__init__()
        self.body = nn.Sequential(
            nn.Conv2d(n_feat, n_feat // 2, kernel_size=3, stride=1, padding=1, bias=False),
            nn.PixelUnshuffle(2)
        )

    def forward(self, x):
        return self.body(x)


class Upsample(nn.Module):
    def __init__(self, n_feat):
        super(Upsample, self).__init__()
        self.body = nn.Sequential(
            nn.Conv2d(n_feat, n_feat * 2, kernel_size=3, stride=1, padding=1, bias=False),
            nn.PixelShuffle(2)
        )

    def forward(self, x):
        return self.body(x)


class DeformableAlign(nn.Module):
    """对齐当前帧特征到上一时刻隐状态，参考 DRFM 的偏移预测+DCN。"""

    def __init__(self, channels: int):
        super().__init__()
        self.offset_conv = nn.Sequential(
            nn.Conv2d(channels * 2, channels, kernel_size=3, padding=1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(channels, 2 * 3 * 3, kernel_size=3, padding=1),
        )
        self.dcn = DeformConv2d(channels, channels, kernel_size=3, padding=1)

    def forward(self, ref_feat: torch.Tensor, cur_feat: torch.Tensor) -> torch.Tensor:
        offsets = self.offset_conv(torch.cat([ref_feat, cur_feat], dim=1))
        return self.dcn(cur_feat, offsets) + cur_feat


class Mamba2SelectiveScan(nn.Module):
    """mamba-v2 风格 selective_scan，支持可选状态回传。"""

    def __init__(
        self,
        d_model: int,
        d_state: int = 16,
        expand: float = 1.0,
        dt_rank: str | int = "auto",
        dt_min: float = 0.001,
        dt_max: float = 0.1,
        dt_init: str = "random",
        dt_scale: float = 1.0,
        dt_init_floor: float = 1e-4,
        device=None,
        dtype=None,
    ):
        super().__init__()
        factory_kwargs = {"device": device, "dtype": dtype}
        self.d_model = d_model
        self.d_state = d_state
        self.expand = expand
        self.d_inner = int(self.expand * self.d_model)
        self.dt_rank = math.ceil(self.d_model / 16) if dt_rank == "auto" else dt_rank
        x_proj = nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs)
        self.x_proj_weight = nn.Parameter(x_proj.weight.unsqueeze(0))
        dt_proj = self._dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor,
                                **factory_kwargs)
        self.dt_projs_weight = nn.Parameter(dt_proj.weight.unsqueeze(0))
        self.dt_projs_bias = nn.Parameter(dt_proj.bias.unsqueeze(0))
        self.A_logs = self._A_log_init(self.d_state, self.d_inner, copies=1, merge=True)
        self.Ds = self._D_init(self.d_inner, copies=1, merge=True)
        self.selective_scan = selective_scan_fn

    @staticmethod
    def _dt_init(dt_rank, d_inner, dt_scale=1.0, dt_init="random", dt_min=0.001, dt_max=0.1, dt_init_floor=1e-4,
                 **factory_kwargs):
        dt_proj = nn.Linear(dt_rank, d_inner, bias=True, **factory_kwargs)
        dt_init_std = dt_rank ** -0.5 * dt_scale
        if dt_init == "constant":
            nn.init.constant_(dt_proj.weight, dt_init_std)
        elif dt_init == "random":
            nn.init.uniform_(dt_proj.weight, -dt_init_std, dt_init_std)
        else:
            raise NotImplementedError
        dt = torch.exp(
            torch.rand(d_inner, **factory_kwargs) * (math.log(dt_max) - math.log(dt_min)) + math.log(dt_min)
        ).clamp(min=dt_init_floor)
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            dt_proj.bias.copy_(inv_dt)
        dt_proj.bias._no_reinit = True
        return dt_proj

    @staticmethod
    def _A_log_init(d_state, d_inner, copies=1, device=None, merge=True):
        A = repeat(torch.arange(1, d_state + 1, dtype=torch.float32, device=device), "n -> d n", d=d_inner).contiguous()
        A_log = torch.log(A)
        if copies > 1:
            A_log = repeat(A_log, "d n -> r d n", r=copies)
            if merge:
                A_log = A_log.flatten(0, 1)
        A_log = nn.Parameter(A_log)
        A_log._no_weight_decay = True
        return A_log

    @staticmethod
    def _D_init(d_inner, copies=1, device=None, merge=True):
        D = torch.ones(d_inner, device=device)
        if copies > 1:
            D = repeat(D, "n1 -> r n1", r=copies)
            if merge:
                D = D.flatten(0, 1)
        D = nn.Parameter(D)
        D._no_weight_decay = True
        return D

    def forward_core(self, x: torch.Tensor, initial_state=None, return_last_state: bool = False):
        B, L, C = x.shape
        K = 1
        xs = x.permute(0, 2, 1).view(B, K, C, L).contiguous()
        x_dbl = torch.einsum("b k d l, k c d -> b k c l", xs.view(B, K, -1, L), self.x_proj_weight)
        dts, Bs, Cs = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=2)
        dts = torch.einsum("b k r l, k d r -> b k d l", dts.view(B, K, -1, L), self.dt_projs_weight)
        xs = xs.float().view(B, -1, L)
        dts = dts.contiguous().float().view(B, -1, L)
        Bs = Bs.float().view(B, K, -1, L)
        Cs = Cs.float().view(B, K, -1, L)
        Ds = self.Ds.float().view(-1)
        As = -torch.exp(self.A_logs.float()).view(-1, self.d_state)
        dt_projs_bias = self.dt_projs_bias.float().view(-1)
        new_state = None
        try:
            out = self.selective_scan(
                xs, dts, As, Bs, Cs, Ds, z=None,
                delta_bias=dt_projs_bias,
                delta_softplus=True,
                initial_state=initial_state,
                return_last_state=return_last_state,
            )
            if return_last_state and isinstance(out, tuple):
                out_y, new_state = out
            else:
                out_y = out
        except TypeError:
            out_y = self.selective_scan(
                xs, dts, As, Bs, Cs, Ds, z=None,
                delta_bias=dt_projs_bias,
                delta_softplus=True,
            )
        except Exception:
            out_y = selective_scan_ref(
                xs, dts, As, Bs, Cs, Ds, z=None,
                delta_bias=dt_projs_bias,
                delta_softplus=True,
                return_last_state=return_last_state,
            )
            if return_last_state and isinstance(out_y, tuple):
                out_y, new_state = out_y
        out_y = out_y.view(B, K, -1, L)
        return out_y[:, 0], new_state

    def forward(self, x: torch.Tensor, initial_state=None, return_last_state: bool = False):
        y, new_state = self.forward_core(x, initial_state, return_last_state)
        y = y.permute(0, 2, 1).contiguous()
        if return_last_state:
            return y, new_state
        return y


class SS2D_Core(nn.Module):
    """简化 VMamba2D：Conv 提取局部，SelectiveScan 编码全局时序。"""

    def __init__(self, d_model: int, d_state: int = 16, d_conv: int = 3, expand: float = 2.0):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_inner = int(expand * d_model)
        self.in_proj = nn.Linear(d_model, self.d_inner * 2)
        self.conv2d = nn.Conv2d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            groups=self.d_inner,
            kernel_size=d_conv,
            padding=(d_conv - 1) // 2,
        )
        self.act = nn.SiLU()
        self.ssm = Mamba2SelectiveScan(d_model=self.d_inner, d_state=d_state, expand=1.0)
        self.out_proj = nn.Linear(self.d_inner, d_model)

    def forward(self, x, initial_state=None, return_last_state: bool = False):
        B, H, W, C = x.shape
        xz = self.in_proj(x)
        x, z = xz.chunk(2, dim=-1)
        x = x.permute(0, 3, 1, 2).contiguous()
        x = self.act(self.conv2d(x))
        x_flat = x.permute(0, 2, 3, 1).flatten(1, 2)
        ssm_out = self.ssm(x_flat, initial_state=initial_state, return_last_state=return_last_state)
        new_state = None
        if return_last_state and isinstance(ssm_out, tuple):
            y_flat, new_state = ssm_out
        else:
            y_flat = ssm_out
        y = y_flat.view(B, H, W, -1)
        y = y * F.silu(z)
        out = self.out_proj(y)
        if return_last_state:
            return out, new_state
        return out


class SSMRNNCell(nn.Module):
    """单步 RNN 风格的 SSM 单元: I_t + H_{t-1} -> H_t。"""

    def __init__(self, in_channels: int = 3, hidden_channels: int = 48, d_state: int = 16):
        super().__init__()
        self.fem = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, 3, 1, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, hidden_channels, 3, 1, 1, bias=False),
        )
        self.align = DeformableAlign(hidden_channels)
        self.ssm = SS2D_Core(d_model=hidden_channels, d_state=d_state)
        self.fuse = nn.Conv2d(hidden_channels * 2, hidden_channels, 1, 1, 0)
        self.gate = nn.Sequential(
            nn.Conv2d(hidden_channels * 2, hidden_channels, 3, 1, 1),
            nn.Sigmoid(),
        )

    def forward(self, I_t: torch.Tensor, H_prev: torch.Tensor | None = None):
        cur_feat = self.fem(I_t)
        if H_prev is None:
            H_prev = torch.zeros_like(cur_feat)
        aligned_cur = self.align(H_prev, cur_feat)
        ssm_in = aligned_cur.permute(0, 2, 3, 1).contiguous()
        ssm_out = self.ssm(ssm_in)
        ssm_out = ssm_out.permute(0, 3, 1, 2).contiguous()
        fused = self.fuse(torch.cat([ssm_out, H_prev], dim=1))
        gate = self.gate(torch.cat([fused, H_prev], dim=1))
        H_t = H_prev + gate * (fused - H_prev)
        return H_t


class Transformer(nn.Module):
    def __init__(self,
                 inp_channels=3,
                 out_channels=3,
                 dim=48,
                 num_blocks=[4, 6, 6, 8],
                 num_refinement_blocks=4,
                 heads=[1, 2, 4, 8],
                 ffn_expansion_factor=2.66,
                 bias=False,
                 LayerNorm_type='WithBias'):
        super(Transformer, self).__init__()
        self.patch_embed = OverlapPatchEmbed(inp_channels, dim)
        self.encoder_level1 = nn.Sequential(
            *[TransformerBlock(dim=dim, num_heads=heads[0], ffn_expansion_factor=ffn_expansion_factor, bias=bias,
                               LayerNorm_type=LayerNorm_type) for _ in range(num_blocks[0])])
        self.down1_2 = Downsample(dim)
        self.fused_down1_2 = Downsample(dim)
        self.encoder_level2 = nn.Sequential(
            *[TransformerBlock(dim=int(dim * 2 ** 1), num_heads=heads[1], ffn_expansion_factor=ffn_expansion_factor,
                               bias=bias, LayerNorm_type=LayerNorm_type) for _ in range(num_blocks[1])])
        self.down2_3 = Downsample(int(dim * 2 ** 1))
        self.fused_down2_3 = Downsample(int(dim * 2 ** 1))
        self.encoder_level3 = nn.Sequential(
            *[TransformerBlock(dim=int(dim * 2 ** 2), num_heads=heads[2], ffn_expansion_factor=ffn_expansion_factor,
                               bias=bias, LayerNorm_type=LayerNorm_type) for _ in range(num_blocks[2])])
        self.down3_4 = Downsample(int(dim * 2 ** 2))
        self.fused_down3_4 = Downsample(int(dim * 2 ** 2))
        self.latent = nn.Sequential(
            *[TransformerBlock(dim=int(dim * 2 ** 3), num_heads=heads[3], ffn_expansion_factor=ffn_expansion_factor,
                               bias=bias, LayerNorm_type=LayerNorm_type) for _ in range(num_blocks[3])])
        self.up4_3 = Upsample(int(dim * 2 ** 3))
        self.reduce_chan_level3 = nn.Conv2d(int(dim * 2 ** 3), int(dim * 2 ** 2), kernel_size=1, bias=bias)
        self.decoder_level3 = nn.Sequential(
            *[TransformerBlock(dim=int(dim * 2 ** 2), num_heads=heads[2], ffn_expansion_factor=ffn_expansion_factor,
                               bias=bias, LayerNorm_type=LayerNorm_type) for _ in range(num_blocks[2])])
        self.up3_2 = Upsample(int(dim * 2 ** 2))
        self.reduce_chan_level2 = nn.Conv2d(int(dim * 2 ** 2), int(dim * 2 ** 1), kernel_size=1, bias=bias)
        self.decoder_level2 = nn.Sequential(
            *[TransformerBlock(dim=int(dim * 2 ** 1), num_heads=heads[1], ffn_expansion_factor=ffn_expansion_factor,
                               bias=bias, LayerNorm_type=LayerNorm_type) for _ in range(num_blocks[1])])
        self.up2_1 = Upsample(int(dim * 2 ** 1))
        self.decoder_level1 = nn.Sequential(
            *[TransformerBlock(dim=int(dim * 2 ** 1), num_heads=heads[0], ffn_expansion_factor=ffn_expansion_factor,
                               bias=bias, LayerNorm_type=LayerNorm_type) for _ in range(num_blocks[0])])
        self.refinement = nn.Sequential(
            *[TransformerBlock(dim=int(dim * 2 ** 1), num_heads=heads[0], ffn_expansion_factor=ffn_expansion_factor,
                               bias=bias, LayerNorm_type=LayerNorm_type) for _ in range(num_refinement_blocks)])
        self.output = nn.Conv2d(int(dim * 2 ** 1), out_channels, kernel_size=3, stride=1, padding=1, bias=bias)

    def forward(self, inp_img, k_v):
        fused_level1 = k_v
        inp_enc_level1 = self.patch_embed(inp_img)
        out_enc_level1, _ = self.encoder_level1([inp_enc_level1, fused_level1])
        fused_level2 = self.fused_down1_2(fused_level1)
        inp_enc_level2 = self.down1_2(out_enc_level1)
        out_enc_level2, _ = self.encoder_level2([inp_enc_level2, fused_level2])
        fused_level3 = self.fused_down2_3(fused_level2)
        inp_enc_level3 = self.down2_3(out_enc_level2)
        out_enc_level3, _ = self.encoder_level3([inp_enc_level3, fused_level3])
        fused_level4 = self.fused_down3_4(fused_level3)
        inp_enc_level4 = self.down3_4(out_enc_level3)
        latent, _ = self.latent([inp_enc_level4, fused_level4])
        inp_dec_level3 = self.up4_3(latent)
        inp_dec_level3 = torch.cat([inp_dec_level3, out_enc_level3], 1)
        inp_dec_level3 = self.reduce_chan_level3(inp_dec_level3)
        out_dec_level3, _ = self.decoder_level3([inp_dec_level3, None])
        inp_dec_level2 = self.up3_2(out_dec_level3)
        inp_dec_level2 = torch.cat([inp_dec_level2, out_enc_level2], 1)
        inp_dec_level2 = self.reduce_chan_level2(inp_dec_level2)
        out_dec_level2, _ = self.decoder_level2([inp_dec_level2, None])
        inp_dec_level1 = self.up2_1(out_dec_level2)
        inp_dec_level1 = torch.cat([inp_dec_level1, out_enc_level1], 1)
        out_dec_level1, _ = self.decoder_level1([inp_dec_level1, None])
        out_dec_level1, _ = self.refinement([out_dec_level1, None])
        out_dec_level1 = self.output(out_dec_level1) + inp_img
        return out_dec_level1


@ARCH_REGISTRY.register()
class FreeMEF(nn.Module):
    """
    基于 DRFM 框架，使用 SSMRNNCell 替换 DeformableRecurrentFusion。
    输入: ldr 主帧，others 为邻近帧序列。
    """

    def __init__(self,
                 n_encoder_res=6,
                 inp_channels=3,
                 out_channels=3,
                 dim=32,
                 num_blocks=[2,2,2,2],
                 num_refinement_blocks=2,
                 heads=[1, 2, 4, 8],
                 ffn_expansion_factor=2.66,
                 bias=False,
                 LayerNorm_type='WithBias',
                 d_state=16):
        super(FreeMEF, self).__init__()
        self.fem = nn.Sequential(
            nn.Conv2d(inp_channels, dim, kernel_size=3, stride=1, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, bias=False)
        )
        self.recurrent_fusion = SSMRNNCell(in_channels=inp_channels, hidden_channels=dim, d_state=d_state)
        self.G = Transformer(
            inp_channels=inp_channels,
            out_channels=out_channels,
            dim=dim,
            num_blocks=num_blocks,
            num_refinement_blocks=num_refinement_blocks,
            heads=heads,
            ffn_expansion_factor=ffn_expansion_factor,
            bias=bias,
            LayerNorm_type=LayerNorm_type,
        )

    def forward(self, ldr, others=None, others_count=None):
        """ldr 为主帧；others 形状 (B,K,C,H,W) 或 (K,C,H,W)。others_count 指示真实帧数。"""
        if others is None or (isinstance(others, torch.Tensor) and others.numel() == 0):
            others = None
        B = ldr.size(0)
        fused_init = self.fem(ldr)
        fused_list = []
        if others is None:
            fused_list = [fused_init[i:i + 1] for i in range(B)]
        else:
            if others.dim() == 4:
                k_use = others_count.item() if others_count is not None else others.size(0)
                k_use = min(k_use, others.size(0))
                others_batches = [others]
                counts = [k_use]
            elif others.dim() == 5:
                K = others.size(1)
                if others_count is None:
                    counts = [K] * B
                else:
                    counts = [int(c.item()) for c in others_count]
                counts = [min(c, K) for c in counts]
                others_batches = [others[i] for i in range(B)]
            else:
                raise ValueError("others tensor dim should be 4 or 5.")

            for i in range(B):
                H = fused_init[i:i + 1]
                cur_others = others_batches[i if len(others_batches) > 1 else 0]
                k = counts[i] if len(counts) > 1 else counts[0]
                for j in range(k):
                    H = self.recurrent_fusion(cur_others[j:j + 1], H)
                fused_list.append(H)

        fused_feat = torch.cat(fused_list, dim=0)
        sr = self.G(ldr, fused_feat)
        return sr


if __name__ == "__main__":
    # 计算参数量和FLOPs的示例（需安装 thop 库: pip install thop）
    from thop import profile

    net = FreeMEF(
        dim=32,
        num_blocks=[2, 2, 2, 2],
        num_refinement_blocks=2,
    ).cuda()
    x = torch.randn(1, 3, 256, 256).cuda()
    others = torch.randn(1, 2, 3, 256, 256).cuda()
    others_count = torch.tensor([2]).cuda()

    # 执行一次推理，确保 forward 没有问题
    out = net(x, others, others_count)
    print("Output shape:", out.shape)

    # 计算 FLOPs 和参数量，profile 的返回值为 (flops, params)
    flops, params = profile(net, inputs=(x, others, others_count), verbose=False)

    # 转换单位为百万（M）
    print(f"FLOPs: {flops / 1e9:.3f} G")
    print(f"Params: {params / 1e6:.3f} M")
