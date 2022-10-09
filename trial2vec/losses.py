import pdb

import torch
from torch import nn

class InfoNCELoss(nn.Module):
    '''
    The basic InfoNCE loss.
    Can be subclass and get other InfoNCE-based losses.
    '''

    def __init__(self, model, logit_scale_init_value) -> None:
        super().__init__()
        self.model = model
        self.logit_scale = nn.Parameter(torch.log(torch.tensor(1/logit_scale_init_value)))
        self.xent_loss = nn.CrossEntropyLoss()

    def forward(self, inputs):
        outputs = self.model(inputs)
        embs = outputs['embs']
        logits = self.compute_logits(embs, embs)
        loss = (self.compute_loss(logits) + self.compute_loss(logits.t())) / 2
        return {'loss_value':loss}

    def compute_loss(self, logits):
        return self.xent_loss(logits, torch.arange(len(logits), device=logits.device))

    def compute_logits(self, embs, target_embs):
        self.logit_scale.data = torch.clamp(self.logit_scale.data, 0, 4.6052)
        logit_scale = self.logit_scale.exp()
        logits = torch.matmul(self._normalize(embs), self._normalize(target_embs).t())
        logits = logits * logit_scale
        return logits

    def _normalize(self, emb):
        return emb / emb.norm(dim=-1, keepdim=True)