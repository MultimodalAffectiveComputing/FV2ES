import math
from typing import Optional, List
import torch
from torch import nn
from src.utils import padTensor

class WrappedTransformerEncoder(nn.Module):
    def __init__(self, dim, num_layers, num_heads):
        super(WrappedTransformerEncoder, self).__init__()
        self.dim = dim
        encoder_layer = nn.TransformerEncoderLayer(d_model=dim, nhead=num_heads)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.cls_emb = nn.Embedding(num_embeddings=1, embedding_dim=dim)

    def prepend_cls(self, inputs):
        # print(f"cls填充前{inputs}")
        index = torch.LongTensor([0]).to(device=inputs.device)
        # print(f"LongTensor之后{index}")
        cls_emb = self.cls_emb(index)
        # print(f"对index使用cls_emb之后{cls_emb}")
        cls_emb = cls_emb.expand(inputs.size(0), 1, self.dim)
        # print(f"对cls_emb拓充之后{cls_emb.shape},具体{cls_emb}")
        outputs = torch.cat((cls_emb, inputs), dim=1)
        # print(f"合并cls和input之后{outputs.shape},具体{outputs}")
        return outputs

    def forward(self, inputs: torch.Tensor, lens: Optional[List[int]] = None, get_cls: Optional[bool] = False):
        if lens is not None:
            max_len = max(lens)

            mask = [([False] * (l + int(get_cls)) + [True] * (max_len - l)) for l in lens]
            mask = torch.tensor(mask).to(device=inputs.device)

            inputs = list(inputs.split(lens, dim=0))
            inputs = [padTensor(inp, max_len) for inp in inputs]
            inputs = torch.stack(inputs, dim=0)
            # print(inputs.shape)
        else:
            mask = None

        if get_cls:
            inputs = self.prepend_cls(inputs)
            # print(inputs)

        inputs = inputs.permute(1, 0, 2)
        # inputs = self.pos_encoder(inputs)
        # print("input shape")  ##
        # print(inputs.shape)  ##
        inputs = self.encoder(src=inputs, src_key_padding_mask=mask) # (seq_len, bs, dim)

        if get_cls:
            return inputs[0]

        return inputs[1:].permute(1, 0, 2)

