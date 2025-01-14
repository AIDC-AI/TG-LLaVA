import torch
import torch.nn as nn
from .perceiver_config import Idefics2Config
from typing import Dict, List, Optional, Tuple, Union
from transformers.activations import ACT2FN
from transformers.modeling_attn_mask_utils import _prepare_4d_attention_mask
import math

class mm_patch_encoder(nn.Module):
    def __init__(self, config, idefics2config):
        super().__init__()
        self.position_embedding = nn.Embedding(2304, config.mm_hidden_size)
        self.register_buffer("position_ids", torch.arange(2304).expand((1, -1)), persistent=False)

        self.pre_mlp = nn.Sequential(
            nn.Linear(config.mm_hidden_size, config.mm_hidden_size),
            nn.GELU(),
            nn.Linear(config.mm_hidden_size, config.mm_hidden_size)
        )

        self.proj = nn.Sequential(
            nn.Linear(config.mm_hidden_size, config.hidden_size),
            nn.GELU(),
            nn.Linear(config.hidden_size, config.hidden_size)
        )

        idefics2config.text_config.hidden_size = 1024
        idefics2config.perceiver_config.resampler_depth = 2
        idefics2config.perceiver_config.resampler_n_latents = 64
        self.perceiver = Idefics2PerceiverResampler(idefics2config)

        self.resize_linear = nn.Linear(768, 1024)

    def forward(self, image_patches_features, text_features):
        text_features = self.resize_linear(text_features)

        for i in range(len(image_patches_features)):
            image_patches_features[i] = image_patches_features[i].reshape(-1, 1024)
        image_patches_features = torch.stack(image_patches_features)
        image_patches_features = image_patches_features + self.position_embedding(self.position_ids.to(image_patches_features.device))

        image_patches_features = self.pre_mlp(image_patches_features)
        
        additional_patches_token =  self.perceiver(image_patches_features, text_features)
        additional_patches_token = self.proj(additional_patches_token)

        return additional_patches_token

def build_patch_encoder(config, delay_load=False, **kwargs):
    idefics2config = Idefics2Config()
    return mm_patch_encoder(config, idefics2config)


class Idefics2MLP(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        output_size: int,
        hidden_act: str,
    ):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, output_size, bias=False)
        self.act_fn = ACT2FN[hidden_act]

    def forward(self, x):
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))

class Idefics2PerceiverResampler(nn.Module):
    def __init__(self, config) -> None:
        """
        Instantiates a Perceiver Resampler that operates over a sequence of embeddings (say from a ResNet or ViT or
        MAE) of a given dimension, performs `depth` blocks of cross-attention with a fixed `n_latents` inputs, then
        returns a Tensor of shape [bsz, n_latents, embed_dim]. The Resampler acts as a form of learned pooling and
        is derived from [Perceiver: General Perception with Iterative Attention](https://arxiv.org/abs/2103.03206).
        """
        super().__init__()
        self.hidden_size = config.text_config.hidden_size
        self.hidden_act = config.perceiver_config.hidden_act
        self.n_latents = config.perceiver_config.resampler_n_latents
        self.depth = config.perceiver_config.resampler_depth
        self.rms_norm_eps = config.text_config.rms_norm_eps

        # Create Latents for Perceiver
        self.latents = nn.Parameter(torch.zeros(self.n_latents, self.hidden_size))
        self.latents.data.normal_(mean=0.0, std=0.02)

        # Create Transformer Blocks
        self.layers = nn.ModuleList([Idefics2PerceiverLayer(config, idx) for idx in range(self.depth)])
        self.norm = Idefics2RMSNorm(self.hidden_size, eps=self.rms_norm_eps)

        self._use_flash_attention_2 = config._attn_implementation == "flash_attention_2"

    def forward(
        self,
        image_features: torch.Tensor,
        text_features: torch.Tensor,
    ) -> torch.Tensor:
        attention_mask = torch.ones((image_features.size(0), image_features.size(1)), dtype=image_features.dtype, device=image_features.device).bool()

        latents = self.latents.unsqueeze(0).expand((image_features.shape[0], *self.latents.size()))

        latent_attention_mask = torch.ones(
            (attention_mask.size(0), latents.size(1)), dtype=attention_mask.dtype, device=attention_mask.device
        )
        attention_mask = torch.cat([attention_mask, latent_attention_mask], dim=-1)
        attention_mask = (
            _prepare_4d_attention_mask(attention_mask, latents.dtype, tgt_len=self.n_latents)
            if not self._use_flash_attention_2
            else attention_mask
        )

        if len(text_features.shape) == 2:
            text_features = text_features.unsqueeze(1)

        text_attention_mask = torch.ones((text_features.size(0), text_features.size(1)), dtype=text_features.dtype, device=text_features.device).bool()
        text_attention_mask = torch.cat([text_attention_mask, latent_attention_mask], dim=-1)
        text_attention_mask = (
            _prepare_4d_attention_mask(text_attention_mask, latents.dtype, tgt_len=self.n_latents)
            if not self._use_flash_attention_2
            else text_attention_mask
        )

        text_guide_flag = False
        compressed_context = latents
        for perceiver_layer in self.layers:
            if text_guide_flag == False:
                layer_outputs = perceiver_layer(
                    compressed_context,
                    text_features,
                    attention_mask=text_attention_mask,
                    position_ids=None,
                    past_key_value=None,
                    output_attentions=False,
                    use_cache=False,
                )
                text_guide_flag = True
            else:
                layer_outputs = perceiver_layer(
                    compressed_context,
                    image_features,
                    attention_mask=attention_mask,
                    position_ids=None,
                    past_key_value=None,
                    output_attentions=False,
                    use_cache=False,
                )

            compressed_context = layer_outputs[0]

        compressed_context = self.norm(compressed_context)

        return compressed_context

class Idefics2PerceiverAttention(nn.Module):
    def __init__(self, config, layer_idx: Optional[int] = None) -> None:
        """Perceiver Cross-Attention Module --> let long-form inputs be `context`, resampled embeddings be `latents`"""
        super().__init__()

        self.layer_idx = None
        self.hidden_size = config.text_config.hidden_size
        self.num_heads = config.perceiver_config.resampler_n_heads
        self.head_dim = config.perceiver_config.resampler_head_dim
        self.num_key_value_heads = config.perceiver_config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.attention_dropout = config.perceiver_config.attention_dropout

        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=False)

        self.is_causal = False

    def forward(
        self,
        latents: torch.Tensor,
        context: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        """
        Runs Perceiver Self-Attention, with special (context, latents) appended along the `seq` dimension!

        Args:
            latents (`torch.Tensor`): Tensor of shape [bsz, n_latents, embed_dim] representing fixed length latents to compress to.
            context (`torch.Tensor`): Tensor of shape [bsz, seq, embed_dim] representing long-form context to resample.
            attention_mask (`torch.Tensor`, *optional*): Tensor of shape [bsz, 1, seq, n_latents] representing attention mask.
            position_ids (`torch.LongTensor`, *optional*): Tensor of shape [bsz, seq] representing position indices of each input token.
            past_key_value (`Tuple[torch.Tensor]`, *optional*): Tuple of tensors containing cached key and value states.
            output_attentions (`bool`, *optional*, defaults to `False`): Whether to return attention weights.
            use_cache (`bool`, *optional*, defaults to `False`): Whether to use past_key_value for caching.
        """
        bsz, q_len, _ = latents.size()
        kv_seq_len = q_len + context.size()[1]

        hidden_states = torch.concat([context, latents], dim=-2)

        query_states = self.q_proj(latents)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, kv_seq_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, kv_seq_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        past_key_value = getattr(self, "past_key_value", past_key_value)

        if past_key_value is not None:
            key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx)

        # repeat k/v heads if n_kv_heads < n_heads
        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)

        if attn_weights.size() != (bsz, self.num_heads, q_len, kv_seq_len):
            raise ValueError(
                f"Attention weights should be of size {(bsz, self.num_heads, q_len, kv_seq_len)}, but is"
                f" {attn_weights.size()}"
            )

        if attention_mask is not None:
            if attention_mask.size() != (bsz, 1, q_len, kv_seq_len):
                raise ValueError(
                    f"Attention mask should be of size {(bsz, 1, q_len, kv_seq_len)}, but is {attention_mask.size()}"
                )

            attn_weights = attn_weights + attention_mask

        # upcast attention to fp32
        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attn_output = torch.matmul(attn_weights, value_states)

        if attn_output.size() != (bsz, self.num_heads, q_len, self.head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(bsz, self.num_heads, q_len, self.head_dim)}, but is"
                f" {attn_output.size()}"
            )

        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(bsz, q_len, self.num_heads * self.head_dim)

        attn_output = self.o_proj(attn_output)

        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights, past_key_value

IDEFICS2_PERCEIVER_ATTENTION_CLASSES = {
    "eager": Idefics2PerceiverAttention,
}

class Idefics2PerceiverLayer(nn.Module):
    def __init__(self, config, layer_idx: int):
        super().__init__()
        self.hidden_size = config.text_config.hidden_size
        self.n_latents = config.perceiver_config.resampler_n_latents
        self.depth = config.perceiver_config.resampler_depth
        self.rms_norm_eps = config.text_config.rms_norm_eps

        self.input_latents_norm = Idefics2RMSNorm(self.hidden_size, eps=self.rms_norm_eps)
        self.input_context_norm = Idefics2RMSNorm(self.hidden_size, eps=self.rms_norm_eps)
        self.self_attn = IDEFICS2_PERCEIVER_ATTENTION_CLASSES[config._attn_implementation](config, layer_idx=layer_idx)
        self.post_attention_layernorm = Idefics2RMSNorm(self.hidden_size, eps=self.rms_norm_eps)
        self.mlp = Idefics2MLP(
            hidden_size=config.text_config.hidden_size,
            intermediate_size=config.text_config.hidden_size * 4,
            output_size=config.text_config.hidden_size,
            hidden_act=config.perceiver_config.hidden_act,
        )

    def forward(
        self,
        latents: torch.Tensor,
        context: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        **kwargs,
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
        """
        Args:
            latents (`torch.FloatTensor`): input to the layer of shape `(batch, seq_len, embed_dim)`
            context (`torch.FloatTensor`): input to the layer of shape `(batch, seq_len, embed_dim)`
            attention_mask (`torch.FloatTensor`, *optional*): attention mask of size
                `(batch, sequence_length)` where padding elements are indicated by 0.
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
            use_cache (`bool`, *optional*):
                If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding
                (see `past_key_values`).
            past_key_value (`Tuple(torch.FloatTensor)`, *optional*): cached past key and value projection states
        """
        residual = latents

        latents = self.input_latents_norm(latents)
        context = self.input_context_norm(context)

        latents, self_attn_weights, present_key_value = self.self_attn(
            latents=latents,
            context=context,
            attention_mask=attention_mask,
        )
        latents = residual + latents
        residual = latents

        latents = self.post_attention_layernorm(latents)
        latents = self.mlp(latents)
        latents = residual + latents

        outputs = (latents,)

        if output_attentions:
            outputs += (self_attn_weights,)

        if use_cache:
            outputs += (present_key_value,)

        return outputs

class Idefics2RMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        """
        Idefics2RMSNorm is equivalent to T5LayerNorm
        """
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)

# Copied from transformers.models.llama.modeling_llama.repeat_kv
def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    This is the equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep). The hidden states go from (batch,
    num_key_value_heads, seqlen, head_dim) to (batch, num_attention_heads, seqlen, head_dim)
    """
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)