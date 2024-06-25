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
        self.bpr_loss = BPRLoss()
        self.epsilon = 1e-8  
    
    def get_category_probabilities(self, pos_items, genre_to_id, item_to_category):
        category_counts = {genre_id: 0 for genre_id in genre_to_id.values()}
        missing_keys = []

        for item in pos_items.cpu().numpy():
            item_str = str(item)
            if item_str not in item_to_category:
                missing_keys.append(item_str)
                continue
            genre_ids = item_to_category[item_str]
            for genre_id in genre_ids:
                if genre_id in category_counts:
                    category_counts[genre_id] += 1
        # if missing_keys:
        #     print(f"Missing keys in item_to_category: {missing_keys}")

        total_count = sum(category_counts.values())
        if total_count == 0:
            total_count = 1  # to avoid division by zero

        category_probs = [category_counts.get(genre_id, 0) / total_count for genre_id in range(len(genre_to_id))]
        return category_probs
    
    def forward(self, pos_score, neg_score, wacc, wdiv, wfair, pos_items, genre_to_id, item_to_category):

        accuracy_loss = self.bpr_loss(pos_score, neg_score)
        
        category_probs = self.get_category_probabilities(pos_items, genre_to_id, item_to_category)
        probabilities = torch.tensor(category_probs, dtype=torch.float32, device=pos_score.device).unsqueeze(0).repeat(pos_score.size(0), 1)
        diversity_loss = -torch.sum(probabilities * torch.log(probabilities + 1e-10))  
        
        # Fairness loss
        popularity = torch.log(1 + torch.exp(pos_score) + self.epsilon)
        mean_popularity = torch.mean(popularity)  
        fairness_loss = torch.sum(torch.abs(popularity - mean_popularity))     
        
        print("wacc:", wacc)
        print("wdiv:", wdiv)
        print("wfair:", wfair)
        print("accuracy_loss:", accuracy_loss)
        print("diversity_loss:", diversity_loss)
        print("fairness_loss:", fairness_loss) 
        
        # Total weighted loss
        total_loss = wacc * accuracy_loss + wdiv * diversity_loss + wfair * fairness_loss
        
        # print(total_loss.sum())
        return total_loss.sum()
    
    
