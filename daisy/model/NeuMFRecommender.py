'''
@inproceedings{he2017neural,
  title={Neural collaborative filtering},
  author={He, Xiangnan and Liao, Lizi and Zhang, Hanwang and Nie, Liqiang and Hu, Xia and Chua, Tat-Seng},
  booktitle={Proceedings of the 26th international conference on world wide web},
  pages={173--182},
  year={2017}
}
'''
import os
import json
import torch
import torch.nn as nn

from daisy.model.AbstractRecommender import GeneralRecommender


class NeuMF(GeneralRecommender):
    tunable_param_names = ['num_ng', 'factors', 'num_layers', 'dropout', 'lr', 'batch_size', 'reg_1', 'reg_2']
    def __init__(self, config):
        """
        NeuMF Recommender Class, it can be seperate as: GMF and MLP
        Parameters
        ----------
        user_num : int, number of users;
        item_num : int, number of items;
        factors : int, the number of latent factor
        num_layers : int, number of hidden layers
        dropout : float, dropout rate
        epochs : int, number of training epochs
        lr : float, learning rate
        reg_1 : float, first-order regularization term
        reg_2 : float, second-order regularization term
        loss_type : str, loss function type
        model_name : str, model name
        optimizer : str, optimization method for training the algorithms
        initializer: str, parameter initializer
        GMF_model : Object, pre-trained GMF weights;
        MLP_model : Object, pre-trained MLP weights.
        gpuid : str, GPU ID
        early_stop : bool, whether to activate early stop mechanism
        """
        super(NeuMF, self).__init__(config)

        self.lr = config['lr']
        self.epochs = config['epochs']
        self.reg_1 = config['reg_1']
        self.reg_2 = config['reg_2']

        self.dropout = config['dropout']
        self.model = config['model_name']
        self.GMF_model = config['GMF_model']
        self.MLP_model = config['MLP_model']

        self.embed_user_GMF = nn.Embedding(config['user_num'], config['factors'])
        self.embed_item_GMF = nn.Embedding(config['item_num'], config['factors'])

        self.embed_user_MLP = nn.Embedding(config['user_num'], config['factors'] * (2 ** (config['num_layers'] - 1)))
        self.embed_item_MLP = nn.Embedding(config['item_num'], config['factors'] * (2 ** (config['num_layers'] - 1)))

        MLP_modules = []
        for i in range(config['num_layers']):
            input_size = config['factors'] * (2 ** (config['num_layers'] - i))
            MLP_modules.append(nn.Dropout(p=self.dropout))
            MLP_modules.append(nn.Linear(input_size, input_size // 2))
            MLP_modules.append(nn.ReLU())
        self.MLP_layers = nn.Sequential(*MLP_modules)

        if self.model in ['MLP', 'GMF']:
            predict_size = config['factors']
        else:
            predict_size = config['factors'] * 2

        self.predict_layer = nn.Linear(predict_size, 1)

        self.loss_type = config['loss_type']
        self.optimizer = config['optimizer'] if config['optimizer'] != 'default' else 'adam'
        self.initializer = config['init_method'] if config['init_method'] != 'default' else 'xavier_normal'
        self.early_stop = config['early_stop']
        self.topk = config['topk']

        self._init_weight()
        
        if self.loss_type.upper() == 'MULTI':
            genre_to_id_path = os.path.join('data', config['dataset'], 'genre_to_id.json')
            item_to_category_path = os.path.join('data', config['dataset'], 'item_to_category.json')
            with open(genre_to_id_path, 'r') as f:
                self.genre_to_id = json.load(f)
            with open(item_to_category_path, 'r') as f:
                self.item_to_category = json.load(f)
            num_items = len(self.item_to_category)
            num_genres = len(self.genre_to_id)
            self.item_genre_matrix = torch.zeros(num_items, num_genres).to(self.device)

    def _init_weight(self):
        if not self.model == 'NeuMF-pre':
            self.initializer_config[self.initializer](self.embed_user_GMF.weight, **self.initializer_param_config[self.initializer])
            self.initializer_config[self.initializer](self.embed_item_GMF.weight, **self.initializer_param_config[self.initializer])
            self.initializer_config[self.initializer](self.embed_user_MLP.weight, **self.initializer_param_config[self.initializer])
            self.initializer_config[self.initializer](self.embed_item_MLP.weight, **self.initializer_param_config[self.initializer])

            for m in self.MLP_layers:
                if isinstance(m, nn.Linear):
                    self.initializer_config[self.initializer](m.weight)
            self.initializer_config[self.initializer](
                self.predict_layer.weight, 
                **self.initializer_param_config[self.initializer])
            for m in self.modules():
                if isinstance(m, nn.Linear) and m.bias is not None:
                    m.bias.data.zero_()
        else:
            # embedding layers
            self.embed_user_GMF.weight.data.copy_(self.GMF_model.embed_user_GMF.weight)
            self.embed_item_GMF.weight.data.copy_(self.GMF_model.embed_item_GMF.weight)
            self.embed_user_MLP.weight.data.copy_(self.MLP_model.embed_user_MLP.weight)
            self.embed_item_MLP.weight.data.copy_(self.MLP_model.embed_item_MLP.weight)
        
            # mlp layers
            for (m1, m2) in zip(self.MLP_layers, self.MLP_model.MLP_layers):
                if isinstance(m1, nn.Linear) and isinstance(m2, nn.Linear):
                    m1.weight.data.copy_(m2.weight)
                    m1.bias.data.copy_(m2.bias)
            
            # predict layers
            predict_weight = torch.cat([self.GMF_model.predict_layer.weight, 
                                        self.MLP_model.predict_layer.weight], dim=1)
            predict_bias = self.GMF_model.predict_layer.bias + self.MLP_model.predict_layer.bias

            self.predict_layer.weight.data.copy_(0.5 * predict_weight)
            self.predict_layer.weight.data.copy_(0.5 * predict_bias)

    def forward(self, user, item):
        if not self.model == 'MLP':
            embed_user_GMF = self.embed_user_GMF(user)
            embed_item_GMF = self.embed_item_GMF(item)
            output_GMF = embed_user_GMF * embed_item_GMF
        if not self.model == 'GMF':
            embed_user_MLP = self.embed_user_MLP(user)
            embed_item_MLP = self.embed_item_MLP(item)
            interaction = torch.cat((embed_user_MLP, embed_item_MLP), dim=-1)
            output_MLP = self.MLP_layers(interaction)

        if self.model == 'GMF':
            concat = output_GMF
        elif self.model == 'MLP':
            concat = output_MLP
        else:
            concat = torch.cat((output_GMF, output_MLP), -1)

        prediction = self.predict_layer(concat)
        return prediction.view(-1)

    def calc_loss(self, batch):
        user = batch[0].to(self.device)
        pos_item = batch[1].to(self.device)
        pos_pred = self.forward(user, pos_item)

        if self.loss_type.upper() in ['CL', 'SL']:
            label = batch[2].to(self.device).float()
            loss = self.criterion(pos_pred, label)

            loss += self.reg_1 * (self.embed_item_GMF(pos_item).norm(p=1))
            loss += self.reg_1 * (self.embed_item_MLP(pos_item).norm(p=1))
            loss += self.reg_2 * (self.embed_item_GMF(pos_item).norm())
            loss += self.reg_2 * (self.embed_item_MLP(pos_item).norm())
        elif self.loss_type.upper() in ['BPR', 'TL', 'HL']:
            neg_item = batch[2].to(self.device)
            neg_pred = self.forward(user, neg_item)
            loss = self.criterion(pos_pred, neg_pred)

            loss += self.reg_1 * (self.embed_item_GMF(pos_item).norm(p=1) + self.embed_item_GMF(neg_item).norm(p=1))
            loss += self.reg_1 * (self.embed_item_MLP(pos_item).norm(p=1) + self.embed_item_GMF(neg_item).norm(p=1))
            loss += self.reg_2 * (self.embed_item_GMF(pos_item).norm() + self.embed_item_GMF(neg_item).norm())
            loss += self.reg_2 * (self.embed_item_MLP(pos_item).norm() + self.embed_item_GMF(neg_item).norm())
        elif self.loss_type.upper() == 'MULTI':
            batch_size = batch[1].shape[0]
            users = batch[0].to(self.device)
            num_users = users.shape[0]
            pos_items = batch[1].to(self.device)
            neg_items = batch[2].to(self.device)

            all_item_indices = torch.arange(self.embed_item_GMF.weight.size(0)).to(self.device)
            pos_scores = self.forward(users.unsqueeze(1).repeat(1, all_item_indices.size(0)), all_item_indices)
            neg_scores = -pos_scores  

            # individual_pos_items
            _, topk_pos_indices = torch.topk(pos_scores, k=batch_size, dim=1)
            individual_pos_items = topk_pos_indices  # shape: (num_users, batch_size)

            # individual_neg_items
            _, topk_neg_indices = torch.topk(neg_scores, k=batch_size, dim=1)
            individual_neg_items = topk_neg_indices  # shape: (num_users, batch_size)

            # Calculate item embeddings for pos and neg items
            individual_pos_item_embeddings = self.embed_item_GMF(individual_pos_items.view(-1))
            individual_pos_item_embeddings = individual_pos_item_embeddings.view(num_users, batch_size, -1)
            
            individual_neg_item_embeddings = self.embed_item_GMF(individual_neg_items.view(-1))
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
            pos_item_scores = self.forward(users.unsqueeze(1).repeat(1, individual_pos_items.size(1)), individual_pos_items)
            pos_item_scores_softplus = torch.log(1 + torch.exp(pos_item_scores) + 1e-10)
            Pmean = torch.mean(pos_item_scores_softplus, dim=1)
            wfair = (0.63 - (1 / pos_item_scores.size(1)) * torch.sum(torch.abs(pos_item_scores_softplus - Pmean.unsqueeze(1)), dim=1)) / 0.63
            wacc = 1 - (wdiv + wfair) / 2
            total_weight = wacc + wdiv + wfair
            wacc = wacc / total_weight
            wdiv = wdiv / total_weight
            wfair = wfair / total_weight

            # Calculate pos_scores and neg_scores for individual items
            pos_scores = pos_scores.gather(1, individual_pos_items)
            neg_scores = neg_scores.gather(1, individual_neg_items)

            loss = self.criterion(wacc, wdiv, wfair, individual_pos_items, individual_neg_items, pos_scores, neg_scores, self.item_genre_matrix).sum()

        else:
            raise NotImplementedError(f'Invalid loss type: {self.loss_type}')

        loss += self.reg_1 * (self.embed_user_GMF(user).norm(p=1))
        loss += self.reg_1 * (self.embed_user_MLP(user).norm(p=1))
        loss += self.reg_2 * (self.embed_user_GMF(user).norm())
        loss += self.reg_2 * (self.embed_user_MLP(user).norm())

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

            if not self.model == 'MLP':
                embed_user_GMF = self.embed_user_GMF(us).unsqueeze(dim=1) # batch * 1 * factor
                embed_item_GMF = self.embed_item_GMF(cands_ids) # batch * cand_num * factor
                output_GMF = embed_user_GMF * embed_item_GMF # batch * cand_num * factor
            if not self.model == 'GMF':
                embed_user_MLP = self.embed_user_MLP(us).unsqueeze(dim=1) # batch * 1 * factor
                embed_item_MLP = self.embed_item_MLP(cands_ids) # batch * cand_num * factor
                interaction = torch.cat((embed_user_MLP.expand_as(embed_item_MLP), embed_item_MLP), dim=-1) # batch * cand_num * (2 * factor)
                output_MLP = self.MLP_layers(interaction) # batch * cand_num * dim
            
            if self.model == 'GMF':
                concat = output_GMF
            elif self.model == 'MLP':
                concat = output_MLP
            else:
                concat = torch.cat((output_GMF, output_MLP), -1) # batch * cand_num * (dim + factor)
            scores = self.predict_layer(concat).squeeze() # batch * cand_num

            rank_ids = torch.argsort(scores, descending=True)
            rank_list = torch.gather(cands_ids, 1, rank_ids)
            rank_list = rank_list[:, :self.topk]

            rec_ids = torch.cat((rec_ids, rank_list), 0)

        return rec_ids.cpu().numpy()

    def full_rank(self, u):
        u = torch.tensor(u, device=self.device)

        if not self.model == 'MLP':
            embed_user_GMF = self.embed_user_GMF(u) # factor
            embed_item_GMF = self.embed_item_GMF.weight # item * factor
            output_GMF = embed_user_GMF * embed_item_GMF  # item * factor
        if not self.model == 'GMF':
            embed_user_MLP = self.embed_user_MLP(u) # factor
            embed_item_MLP = self.embed_item_MLP.weight # item * factor
            interaction = torch.cat((embed_user_MLP.expand_as(embed_item_MLP), embed_item_MLP), dim=-1) # item * (2*factor)
            output_MLP = self.MLP_layers(interaction) # item * dim

        if self.model == 'GMF':
            concat = output_GMF
        elif self.model == 'MLP':
            concat = output_MLP
        else:
            concat = torch.cat((output_GMF, output_MLP), -1) # item * (dim + factor)
        scores = self.predict_layer(concat).squeeze() # item
        
        return torch.argsort(scores, descending=True)[:self.topk].cpu().numpy()
