import torch
import torch.nn
import torch.nn.functional


class NoisyCrossEntropyLoss(torch.nn.Module):

    def __init__(self, p_noisy):
        super().__init__()
        self.p = p_noisy
        self.ce = torch.nn.CrossEntropyLoss(reduction='none')


    def forward(self, logits, targets):
        losses = self.ce(logits, targets)
        weights = (1 - self.p) + self.p * (1 - torch.nn.functional.one_hot(targets, num_classes=logits.size(1)).float().sum(dim=1))
        return (losses * weights).mean()



class SymmetricCrossEntropyLoss(torch.nn.Module):
    
    def __init__(self):
        super().__init__()
        self.ce = torch.nn.CrossEntropyLoss()
        self.rce = torch.nn.CrossEntropyLoss()


    def forward(self, logits, targets):

        ce_loss = self.ce(logits, targets)

        one_hot_targets = torch.nn.functional.one_hot(targets, num_classes=self.num_classes).float()
        pred_probs = torch.nn.functional.softmax(logits, dim=1).clamp(min=1e-7, max=1.0)
        rce_loss = -torch.sum(pred_probs * torch.log(one_hot_targets.clamp(min=1e-7)), dim=1).mean()

        return ce_loss + rce_loss