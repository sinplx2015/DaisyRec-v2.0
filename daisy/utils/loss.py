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
    def __init__(self, epsilon=1e-10):
        super(Multi_Loss, self).__init__()
        self.epsilon = epsilon  
    
    def get_category_probabilities(self, pos_items, item_genre_matrix):
        batch_size = pos_items.size(0)
        num_genres = item_genre_matrix.size(1)
        genre_counts = torch.zeros((batch_size, num_genres), device=pos_items.device)        
        for user_idx, user_items in enumerate(pos_items):
            genre_counts[user_idx] = item_genre_matrix[user_items].sum(dim=0)
        category_probs = genre_counts / genre_counts.sum(dim=1, keepdim=True)
        category_probs[torch.isnan(category_probs)] = 0  
        return category_probs
    
    def forward(self, wacc, wdiv, wfair, pos_items, neg_items, all_scores, pos_scores, neg_scores, item_genre_matrix):
        accuracy_loss = -(self.epsilon + torch.sigmoid(pos_scores - neg_scores)).log().sum(dim=1)
        
        category_probs = self.get_category_probabilities(pos_items, item_genre_matrix)
        probabilities = torch.clamp(category_probs, min=1e-10)
        diversity_loss = -torch.sum(probabilities * torch.log(probabilities + 1e-10), dim=1) 

        # Fairness loss
        popularity = torch.log(1 + torch.exp(all_scores) + self.epsilon)
        mean_popularity = torch.mean(popularity, dim=1, keepdim=True)
        fairness_loss = torch.sum(torch.abs(popularity - mean_popularity), dim=1)
        print("wacc:", wacc)
        print("wacc:", wacc)
        print("wdiv:", wdiv)
        print("wfair:", wfair)
        
        print("accuracy_loss:", accuracy_loss)
        print("diversity_loss:", diversity_loss)
        print("fairness_loss:", fairness_loss) 
        
        # Total weighted loss
        total_loss = wacc * accuracy_loss + wdiv * diversity_loss + wfair * fairness_loss
        
        return total_loss


    
    
