import config
import torch.nn as nn
from transformers import AutoModel


class Deberta_v3_large(nn.Module):
    def __init__(self):
        super(Deberta_v3_large, self).__init__()
        self.deberta = AutoModel.from_pretrained(config.MODEL_PATH)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.1)
        self.l0 = nn.Linear(1024,1,bias=True)

    def forward(self, ids, mask, token_type_ids):
        o1 = self.deberta(
            ids,
            attention_mask=mask,
            token_type_ids=token_type_ids,
        )


        logits = self.dropout(o1[0])
        logits = self.l0(logits)
        logits = logits.squeeze(-1)

        return logits
