from torch import nn
from transformers import AlbertModel

class MME2E_T(nn.Module):
    def __init__(self, feature_dim, num_classes=4, size='base'):
        super(MME2E_T, self).__init__()
        # self.albert = AlbertModel.from_pretrained(f'albert-{size}-v2')
        self.albert = AlbertModel.from_pretrained('./src/models/albert-base-v2')


    def forward(self, text, get_cls=False):
        last_hidden_state = self.albert(**text).last_hidden_state
#         print(last_hidden_state)
        if get_cls:
            cls_feature = last_hidden_state[:,0]
            return cls_feature

        text_features = self.text_feature_affine(last_hidden_state).sum(1)
        return text_features
