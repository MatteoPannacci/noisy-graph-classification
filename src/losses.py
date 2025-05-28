import torch
import torch.nn
import torch.nn.functional as F


class NoisyCrossEntropyLoss(torch.nn.Module):
    def __init__(self, p_noisy, weight=None):
        super().__init__()
        self.p_noisy = p_noisy
        self.ce = torch.nn.CrossEntropyLoss(reduction='none', weight=weight)

    def forward(self, logits, targets):
        losses = self.ce(logits, targets)

        probs = F.softmax(logits, dim=1)
        true_class_probs = probs[torch.arange(len(targets)), targets]
        uncertainty = 1.0 - true_class_probs.detach()
        weights = (1.0 - self.p_noisy) + self.p_noisy * uncertainty
        weighted_loss = (losses * weights).mean()
        return weighted_loss


class SymmetricCrossEntropyLoss(torch.nn.Module):

    def __init__(self, alpha, beta, num_classes=6, weight=None):
        super(SymmetricCrossEntropyLoss, self).__init__()
        self.weight = weight
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.alpha = alpha
        self.beta = beta
        self.num_classes = num_classes
        self.cross_entropy = torch.nn.CrossEntropyLoss(weight=weight)

    def forward(self, pred, labels):

        # CCE
        ce = self.cross_entropy(pred, labels)

        # RCE
        pred = F.softmax(pred, dim=1)
        pred = torch.clamp(pred, min=1e-7, max=1.0)
        label_one_hot = F.one_hot(labels, self.num_classes).float().to(self.device)
        label_one_hot = torch.clamp(label_one_hot, min=1e-4, max=1.0)
        rce = (-1*torch.sum(pred * torch.log(label_one_hot), dim=1))

        if self.weight is not None:
            sample_weights = self.weight[targets]
            rce = (rce * sample_weights).mean()
        else:
            rce = rce.mean()

        # Loss
        loss = self.alpha * ce + self.beta * rce
        return loss



class GeneralizedCrossEntropyLoss(torch.nn.Module):

    def __init__(self, q, weight=None):
        super().__init__()
        self.q = q

    def forward(self, logits, targets):
        probs = F.softmax(logits, dim=1)
        probs_correct = probs.gather(dim=1, index=targets.unsqueeze(1)).squeeze(1)
        loss = (1 - probs_correct.pow(self.q)) / self.q
        return loss.mean()