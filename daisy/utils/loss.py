import torch
import torch.nn as nn


class BPRLoss(nn.Module):
    def __init__(self, gamma=1e-10):
        super(BPRLoss, self).__init__()
        self.gamma = gamma

    def forward(self, pos_score, neg_score):
        loss = -(self.gamma + torch.sigmoid(pos_score - neg_score)).log().sum()

        return loss


class HingeLoss(nn.Module):
    def __init__(self):
        super(HingeLoss, self).__init__()

    def forward(self, pos_score, neg_score):
        loss = torch.clamp(1 - (pos_score - neg_score), min=0).sum()

        return loss


class TOP1Loss(nn.Module):
    def __init__(self):
        super(TOP1Loss, self).__init__()

    def forward(self, pos_score, neg_score):
        loss = (neg_score - pos_score).sigmoid().sum() + neg_score.pow(2).sigmoid().sum()

        return loss
    
class Multi_Loss(nn.Module):
    def __init__(self):
        super(Multi_Loss, self).__init__()
        self.bpr_loss = BPRLoss()

    def forward(self, pos_score, neg_score, wacc, wdiv, wfair):

        accuracy_loss = self.bpr_loss(pos_score, neg_score)
        probabilities = torch.softmax(pos_score, dim=0)
        diversity_loss = -torch.sum(probabilities * torch.log(probabilities))
        
        popularity = torch.log(1 + torch.exp(pos_score))
        mean_popularity = torch.mean(popularity)
        fairness_loss = torch.sum(torch.abs(popularity - mean_popularity))
        
        total_loss = wacc * accuracy_loss + wdiv * diversity_loss + wfair * fairness_loss
        
        return total_loss

