import torch
import torch.nn
import torch.nn.functional


class NoisyCrossEntropyLoss(torch.nn.Module):

    def __init__(self, p_noisy, weight=None):
        super().__init__()
        self.p = p_noisy
        self.ce = torch.nn.CrossEntropyLoss(reduction='none', weight=weight)


    def forward(self, logits, targets):
        losses = self.ce(logits, targets)
        weights = (1 - self.p) + self.p * (1 - torch.nn.functional.one_hot(targets, num_classes=logits.size(1)).float().sum(dim=1))
        return (losses * weights).mean()


class SymmetricCrossEntropyLoss(torch.nn.Module):
    def __init__(self, weight=None):
        super().__init__()
        self.ce = torch.nn.CrossEntropyLoss(weight=weight)

    def forward(self, logits, targets):

        ce_loss = self.ce(logits, targets)

        num_classes = logits.size(1)
        one_hot_targets = torch.nn.functional.one_hot(targets, num_classes=num_classes).float()
        pred_probs = torch.nn.functional.softmax(logits, dim=1).clamp(min=1e-7, max=1.0)
        one_hot_targets = one_hot_targets.clamp(min=1e-7)

        # Reverse cross-entropy with class weights
        rce_per_sample = -torch.sum(pred_probs * torch.log(one_hot_targets), dim=1)

        if self.weight is not None:
            # Normalize weights to match targets
            sample_weights = self.weight[targets]
            rce_loss = (rce_per_sample * sample_weights).mean()
        else:
            rce_loss = rce_per_sample.mean()

        return ce_loss + rce_loss