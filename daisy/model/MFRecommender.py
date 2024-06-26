'''
@article{koren2009matrix,
  title={Matrix factorization techniques for recommender systems},
  author={Koren, Yehuda and Bell, Robert and Volinsky, Chris},
  journal={Computer},
  volume={42},
  number={8},
  pages={30--37},
  year={2009},
  publisher={IEEE}
}
@article{rendle2012bpr,
  title={BPR: Bayesian personalized ranking from implicit feedback},
  author={Rendle, Steffen and Freudenthaler, Christoph and Gantner, Zeno and Schmidt-Thieme, Lars},
  journal={arXiv preprint arXiv:1205.2618},
  year={2012}
}
'''
import os
import json
import torch
import torch.nn as nn

from daisy.model.AbstractRecommender import GeneralRecommender


class MF(GeneralRecommender):
    tunable_param_names = ['num_ng', 'factors', 'lr', 'batch_size', 'reg_1', 'reg_2']
    def __init__(self, config):
        """
        Matrix Factorization Recommender Class
        Parameters
        ----------
        user_num : int, the number of users
        item_num : int, the number of items
        factors : int, the number of latent factor
        epochs : int, number of training epochs
        lr : float, learning rate
        reg_1 : float, first-order regularization term
        reg_2 : float, second-order regularization term
        loss_type : str, loss function type
        optimizer : str, optimization method for training the algorithms
        initializer: str, parameter initializer
        gpuid : str, GPU ID
        early_stop : bool, whether to activate early stop mechanism
        """
        super(MF, self).__init__(config)
        
        self.lr = config['lr']
        self.reg_1 = config['reg_1']
        self.reg_2 = config['reg_2']
        self.epochs = config['epochs']

        self.topk = config['topk']

        self.embed_user = nn.Embedding(config['user_num'], config['factors'])
        self.embed_item = nn.Embedding(config['item_num'], config['factors'])

        self.loss_type = config['loss_type']
        self.optimizer = config['optimizer'] if config['optimizer'] != 'default' else 'sgd'
        self.initializer = config['init_method'] if config['init_method'] != 'default' else 'normal'
        self.early_stop = config['early_stop']

        self.apply(self._init_weight)
        
        if self.loss_type.upper() == 'MULTI':
            genre_to_id_path = os.path.join('data', config['dataset'], 'genre_to_id.json')
            item_to_category_path = os.path.join('data', config['dataset'], 'item_to_category.json')
            with open(genre_to_id_path, 'r') as f:
                self.genre_to_id = json.load(f)
            with open(item_to_category_path, 'r') as f:
                self.item_to_category = json.load(f)
            num_items = len(self.item_to_category)
            num_genres = len(self.genre_to_id)
            self.item_genre_matrix = torch.zeros((num_items, num_genres)).to(self.device)
            for item_str, genres in self.item_to_category.items():
                item_idx = int(item_str) - 1  ###
                for genre_id in genres:
                    self.item_genre_matrix[item_idx, genre_id] = 1

            self.item_genre_matrix /= self.item_genre_matrix.sum(dim=1, keepdim=True)
            self.item_genre_matrix[torch.isnan(self.item_genre_matrix)] = 0 
    def forward(self, user, item):
        embed_user = self.embed_user(user)
        embed_item = self.embed_item(item)
        pred = (embed_user * embed_item).sum(dim=-1)
        return pred

    def calc_loss(self, batch):
        user = batch[0].to(self.device)
        pos_item = batch[1].to(self.device)
        pos_pred = self.forward(user, pos_item)

        if self.loss_type.upper() in ['CL', 'SL']:
            label = batch[2].to(self.device).float()
            loss = self.criterion(pos_pred, label)

            # # add regularization term
            # loss += self.reg_1 * (self.embed_item(pos_item).norm(p=1))
            # loss += self.reg_2 * (self.embed_item(pos_item).norm())
        elif self.loss_type.upper() in ['BPR', 'TL', 'HL']:
            neg_item = batch[2].to(self.device)
            neg_pred = self.forward(user, neg_item)
            loss = self.criterion(pos_pred, neg_pred)

            # # add regularization term
            # loss += self.reg_1 * (self.embed_item(pos_item).norm(p=1) + self.embed_item(neg_item).norm(p=1))
            # loss += self.reg_2 * (self.embed_item(pos_item).norm() + self.embed_item(neg_item).norm())
        elif self.loss_type.upper() == 'MULTI':
            batch_size = batch[1].shape[0]            
            users = batch[0].to(self.device)
            num_users = users.shape[0]
            # pos_items = batch[1].to(self.device)
            # neg_items = batch[2].to(self.device)

            # Get all item embeddings
            all_item_embeddings = self.embed_item.weight
            user_embeddings = self.embed_user(users)
            all_scores = torch.matmul(user_embeddings, all_item_embeddings.t())  

            # individual_pos_items
            _, topk_pos_indices = torch.topk(all_scores, k=batch_size, dim=1)
            individual_pos_items = topk_pos_indices  # shape: (num_users, batch_size)

            # individual_neg_items
            all_scores_neg = -all_scores  
            _, topk_neg_indices = torch.topk(all_scores_neg, k=batch_size, dim=1)
            individual_neg_items = topk_neg_indices  # shape: (num_users, batch_size)

            # Calculate item embeddings for pos and neg items
            individual_pos_item_embeddings = self.embed_item(individual_pos_items.view(-1))
            individual_pos_item_embeddings = individual_pos_item_embeddings.view(num_users, batch_size, -1)
            
            individual_neg_item_embeddings = self.embed_item(individual_neg_items.view(-1))
            individual_neg_item_embeddings = individual_neg_item_embeddings.view(num_users, batch_size, -1)

            # Normalize item embeddings for pos items
            pos_item_norms = individual_pos_item_embeddings.norm(dim=2, keepdim=True)
            normalized_pos_item_embeddings = individual_pos_item_embeddings / pos_item_norms

            # Normalize item embeddings for neg items
            neg_item_norms = individual_neg_item_embeddings.norm(dim=2, keepdim=True)
            normalized_neg_item_embeddings = individual_neg_item_embeddings / neg_item_norms

            # wdiv
            similarity_matrix = torch.bmm(normalized_pos_item_embeddings, normalized_pos_item_embeddings.transpose(1, 2))
            mask = torch.eye(batch_size, device=self.device).expand(num_users, batch_size, batch_size).bool()
            similarity_matrix.masked_fill_(mask, 0)
            wdiv = 1 - (2 / (batch_size * (batch_size - 1))) * similarity_matrix.sum(dim=2).mean(dim=1)

            # wfair
            pos_item_scores = torch.matmul(user_embeddings.unsqueeze(1), individual_pos_item_embeddings.transpose(1, 2)).squeeze(1)
            pos_item_scores_softplus = torch.log(1 + torch.exp(pos_item_scores) + 1e-10)
            Pmean = torch.mean(pos_item_scores_softplus, dim=1)
            wfair = (0.63 - (1 / pos_item_scores.size(1)) * torch.sum(torch.abs(pos_item_scores_softplus - Pmean.unsqueeze(1)), dim=1)) / 0.63
            wacc = 1 - (wdiv + wfair) / 2
            total_weight = wacc + wdiv + wfair
            wacc = wacc / total_weight
            wdiv = wdiv / total_weight
            wfair = wfair / total_weight

            # Calculate pos_scores and neg_scores
            pos_scores = all_scores.gather(1, individual_pos_items)
            neg_scores = all_scores_neg.gather(1, individual_neg_items)
            
            loss = self.criterion(wacc, wdiv, wfair, individual_pos_items, individual_neg_items, pos_scores, neg_scores, self.item_genre_matrix).sum()
        else:
            raise NotImplementedError(f'Invalid loss type: {self.loss_type}')

        # add regularization term
        loss += self.reg_1 * (self.embed_user(user).norm(p=1))
        loss += self.reg_2 * (self.embed_user(user).norm())

        return loss

    def predict(self, u, i):
        u = torch.tensor(u, device=self.device)
        i = torch.tensor(i, device=self.device)
        pred = self.forward(u, i).cpu().item()
        
        return pred

    def rank(self, test_loader):
        rec_ids = torch.tensor([], device=self.device)

        for us, cands_ids in test_loader:
            us = us.to(self.device)
            cands_ids = cands_ids.to(self.device)

            user_emb = self.embed_user(us).unsqueeze(dim=1) # batch * factor -> batch * 1 * factor
            item_emb = self.embed_item(cands_ids).transpose(1, 2) # batch * cand_num * factor -> batch * factor * cand_num 
            scores = torch.bmm(user_emb, item_emb).squeeze() # batch * 1 * cand_num -> batch * cand_num

            rank_ids = torch.argsort(scores, descending=True)
            rank_list = torch.gather(cands_ids, 1, rank_ids)
            rank_list = rank_list[:, :self.topk]

            rec_ids = torch.cat((rec_ids, rank_list), 0)

        return rec_ids.cpu().numpy()


    def full_rank(self, u):
        u = torch.tensor(u, device=self.device)

        user_emb = self.embed_user(u)
        items_emb = self.embed_item.weight 
        scores = torch.matmul(user_emb, items_emb.transpose(1, 0)) #  (item_num,)

        return torch.argsort(scores, descending=True)[:self.topk].cpu().numpy()

