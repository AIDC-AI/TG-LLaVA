#    Copyright (C) 2024 AIDC-AI
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.


import torch
import torch.nn as nn
from .Qformer import BertConfig, BertLMHeadModel

class CrossAttentionModule(nn.Module):
    def __init__(self, hidden_size=1024, num_head=8):
        super().__init__()
        self.resize_linear = nn.Linear(768, hidden_size)
        self.cross_attn = nn.MultiheadAttention(hidden_size, num_head, batch_first=True)
        self.zero_layer = nn.Linear(hidden_size, hidden_size)
        nn.init.zeros_(self.zero_layer.weight) 
        nn.init.zeros_(self.zero_layer.bias)

    def forward(self, img_emb, text_emb):
        if len(text_emb.shape) == 2:
            text_emb = text_emb.unsqueeze(1)
        attn_output, _ = self.cross_attn(query=img_emb, key=text_emb, value=text_emb)
        cross_result = self.zero_layer(attn_output)

        return cross_result

def init_Qformer(num_query_token, vision_width):
    cross_attention_freq=2
    encoder_config = BertConfig.from_pretrained("google-bert/bert-large-uncased")
    encoder_config.num_hidden_layers = 2
    encoder_config.encoder_width = vision_width
    encoder_config.add_cross_attention = True
    encoder_config.cross_attention_freq = cross_attention_freq
    encoder_config.query_length = num_query_token
    Qformer = BertLMHeadModel(encoder_config)
    query_tokens = nn.Parameter(
        torch.zeros(1, num_query_token, encoder_config.hidden_size)
    )
    query_tokens.data.normal_(mean=0.0, std=encoder_config.initializer_range)
    return Qformer, query_tokens


class CrossAttentionLearnableQueryModule(nn.Module):
    def __init__(self, hidden_size=1024, num_head=8):
        super().__init__()
        self.resize_linear = nn.Linear(768, hidden_size)
        self.zero_layer = nn.Linear(hidden_size, hidden_size)

        nn.init.zeros_(self.zero_layer.weight) 
        nn.init.zeros_(self.zero_layer.bias)

        self.num_query_token = 576
        self.Qformer, self.query_tokens = init_Qformer(self.num_query_token, hidden_size)
    
    def forward(self, text_emb):
        
        text_emb = self.resize_linear(text_emb)
        if len(text_emb.shape) == 2:
            text_emb = text_emb.unsqueeze(1)

        text_emb_atts = torch.ones(text_emb.size()[:-1], dtype=text_emb.dtype).to(
            text_emb.device
        )

        query_tokens = self.query_tokens.expand(text_emb.shape[0], -1, -1)
        
        query_output = self.Qformer.bert(
            query_embeds=query_tokens,
            encoder_hidden_states=text_emb,
            encoder_attention_mask=text_emb_atts,
            return_dict=True,
        )

        cross_result = self.zero_layer(query_output.last_hidden_state)

        return cross_result

def build_crossattn_branch(config, delay_load=False, **kwargs):
    return CrossAttentionModule(config.mm_hidden_size, 8)

def build_crossattn_learnable_query_branch(config, delay_load=False, **kwargs):
    return CrossAttentionLearnableQueryModule(config.mm_hidden_size, 8)