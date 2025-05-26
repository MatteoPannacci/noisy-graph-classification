import torch.nn.functional as F
import torch
import torch.nn as nn
import numpy as np
import math

cross_entropy_val = nn.CrossEntropyLoss

mean = 1e-8
std = 1e-9

class ncodLoss(nn.Module):

    def __init__(self, labels, n=50000, C=100, ratio_consistency=0, ratio_balance=0, device=None, encoder_features=300, total_epochs=100):
        super(ncodLoss, self).__init__()

        self.C = C
        self.USE_CUDA = torch.cuda.is_available()
        self.n = n

        self.encoder_features = encoder_features
        self.total_epochs = total_epochs

        self.ratio_consistency = ratio_consistency
        self.ratio_balance = ratio_balance


        self.u = nn.Parameter(torch.empty(n, 1, dtype=torch.float32), requires_grad=True)
        self.init_param(mean=mean,std=std)

        self.beginning = True
        self.prev_phi_x_i = torch.rand((n, self.encoder_features))
        self.phi_c = torch.rand((C, self.encoder_features))
        self.labels = labels.cpu()
        self.bins = []

        for i in range(0, C):
            self.bins.append(np.where(self.labels == i)[0])


    def init_param(self, mean= 1e-8, std= 1e-9):
        torch.nn.init.normal_(self.u, mean=mean, std=std)


    def forward(self, index, f_x_i, y, phi_x_i, flag, epoch):

        index = index.cpu()
        f_x_i = f_x_i.cpu()
        y = y.cpu()
        phi_x_i = phi_x_i.cpu()

        y = F.one_hot(y, num_classes=self.C).float()

        if len(f_x_i) > len(index):
            f_x_i_1, f_x_i_2 = torch.chunk(f_x_i, 2)
            phi_x_i_1, phi_x_i_2 = torch.chunk(phi_x_i, 2)
        else:
            f_x_i_1 = f_x_i
            phi_x_i_1 = phi_x_i

        eps = 1e-4

        u = self.u[index]

        if (flag == 0):
            if self.beginning:
                percent = math.ceil((50 - (50 / self.total_epochs) * epoch) + 50)
                for i in range(0, len(self.bins)):
                    class_u = self.u.detach()[self.bins[i]]
                    bottomK = int((len(class_u) / 100) * percent)
                    important_indexs = torch.topk(class_u, bottomK, largest=False, dim=0)[1]
                    self.phi_c[i] = torch.mean(self.prev_phi_x_i[self.bins[i]][important_indexs.view(-1)], dim=0)

            phi_c_norm = self.phi_c.norm(p=2, dim=1, keepdim=True)
            h_c_bar = self.phi_c.div(phi_c_norm)
            self.h_c_bar_T = torch.transpose(h_c_bar, 0, 1)
            self.beginning = True

        self.prev_phi_x_i[index] = phi_x_i_1.detach()

        f_x_softmax = F.softmax(f_x_i_1, dim=1)

        phi_x_i_1_norm = phi_x_i_1.detach().norm(p=2, dim=1, keepdim=True)
        h_i = phi_x_i_1.detach().div(phi_x_i_1_norm)

        y_bar = torch.mm(h_i, self.h_c_bar_T)
        y_bar = y_bar * y
        y_bar_max = (y_bar > 0.000).type(torch.float32)
        y_bar = y_bar * y_bar_max

        u = u * y

        f_x_softmax = torch.clamp((f_x_softmax + u.detach()), min=eps, max=1.0)
        L1 = torch.mean(-torch.sum((y_bar) * torch.log(f_x_softmax), dim=1))

        y_hat = self.soft_to_hard(f_x_i_1.detach()).float()

        L2 = F.mse_loss((y_hat + u), y, reduction='sum') / len(y)
        L1 += L2

        if self.ratio_balance > 0:
            avg_prediction = torch.mean(f_x_softmax, dim=0)
            prior_distr = 1.0 / self.C * torch.ones_like(avg_prediction)

            avg_prediction = torch.clamp(avg_prediction, min=eps, max=1.0)

            balance_kl = torch.mean(-(prior_distr * torch.log(avg_prediction)).sum(dim=0))

            L1 += self.ratio_balance * balance_kl

        if (len(f_x_i) > len(index)) and (self.ratio_consistency > 0):
            consistency_loss = self.consistency_loss( f_x_i_1, f_x_i_2)

            L1 += self.ratio_consistency * torch.mean(consistency_loss)

        return L1


    def consistency_loss(self,  f_x_i_1, f_x_i_2):
        preds1 = F.softmax( f_x_i_1, dim=1).detach()
        preds2 = F.log_softmax(f_x_i_2, dim=1)
        loss_kldiv = F.kl_div(preds2, preds1, reduction='none')
        loss_kldiv = torch.sum(loss_kldiv, dim=1)
        return loss_kldiv


    def soft_to_hard(self, x):
        with torch.no_grad():
            return (torch.zeros(len(x), self.C)).scatter_(1, (x.argmax(dim=1)).view(-1, 1), 1)