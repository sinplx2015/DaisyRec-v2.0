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
            users = batch[0].to(self.device)
            pos_items = batch[1].to(self.device)
            neg_items = batch[2].to(self.device)
            pos_preds = self.forward(users, pos_items)
            neg_preds = self.forward(users, neg_items)
                        
            # print(pos_preds)   
            # Get all item embeddings
            item_embeddings = self.embed_item(pos_items)
            item_norms = item_embeddings.norm(dim=1, keepdim=True)
            normalized_item_embeddings = item_embeddings / item_norms

            # compute wdiv
            similarity_matrix = torch.mm(normalized_item_embeddings, normalized_item_embeddings.t())
            mask = torch.eye(similarity_matrix.size(0), device=self.device).bool()
            similarity_matrix.masked_fill_(mask, 0)
            num_items = similarity_matrix.size(0)
            wdiv = 1 - (2 / (num_items * (num_items - 1))) * similarity_matrix.sum(dim=1)
            
            # compute wfair
            user_embeddings = self.embed_user(users)
            pos_item_scores = torch.matmul(user_embeddings, item_embeddings.t())
            pos_item_scores_softplus = torch.log(1 + torch.exp(pos_item_scores) + 1e-8)
            Pmean = torch.mean(pos_item_scores_softplus, dim=1)
            # print('pmean:',Pmean)
            # print('pos_item_scores.size(1):', pos_item_scores.size(1))
            # print("torch.sum(torch.abs(pos_item_scores_softplus - Pmean.unsqueeze(1)s):", torch.sum(torch.abs(pos_item_scores_softplus - Pmean.unsqueeze(1)),dim=1))
            wfair = (0.63 - (1 / pos_item_scores.size(1)) * torch.sum(torch.abs(pos_item_scores_softplus - Pmean.unsqueeze(1)), dim=1))/0.63
            wacc = 1 - (wdiv + wfair) / 2
            
            weights = torch.stack([wacc, wdiv, wfair], dim=1)
            weights = torch.nn.functional.softmax(weights, dim=1)
            wacc, wdiv, wfair = weights[:, 0], weights[:, 1], weights[:, 2]
            # total_weight = wacc + wdiv + wfair
            # wacc = wacc / total_weight
            # wdiv = wdiv / total_weight
            # wfair = wfair / total_weight

            loss = self.criterion(pos_preds, neg_preds, wacc, wdiv, wfair, pos_items, self.genre_to_id, self.item_to_category).sum()
                            
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

