import os
import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F

from abstract_recommender import GeneralRecommender
import scipy.sparse as sp
from scipy.sparse.linalg import svds
import math

class IDM(nn.Module):
    def __init__(self, emb_dim=128, time_emb_dim=128, hidden_dim=256,
                 num_attention_heads=1, num_transformer_blocks=1,
                 dim_feedforward_ratio=1, dropout=0.1):
        super().__init__()

        self.emb_dim = emb_dim
        self.time_emb_dim = time_emb_dim
        self.hidden_dim = hidden_dim

        self.register_buffer('emb_factors', self._precompute_emb_factors(self.time_emb_dim))

        # time
        self.time_proj = nn.Sequential(
            nn.Linear(time_emb_dim, time_emb_dim),
            nn.SiLU()
        )

        #  fuse e_u and LID
        self.user_proj = nn.Sequential(
            nn.Linear(emb_dim + 64, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout)
        )

      # e_m
        self.modality_proj = nn.Sequential(
            nn.Linear(emb_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout)
        )

        # fuse x and t
        self.initial_fusion_net = nn.Sequential(
            nn.Linear(emb_dim + time_emb_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout)
        )

        #  Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_attention_heads,
            dim_feedforward=hidden_dim * dim_feedforward_ratio,
            dropout=dropout,
            batch_first=True
        )
        # M-
        self.modality_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_transformer_blocks
        )

        # Q
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=hidden_dim,
            nhead=num_attention_heads,
            dim_feedforward=hidden_dim * dim_feedforward_ratio,
            dropout=dropout,
            batch_first=True
        )
        # imagination
        self.user_decoder = nn.TransformerDecoder(
            decoder_layer,
            num_layers=num_transformer_blocks
        )

        # noise projection
        self.output_net = nn.Sequential(
            nn.Linear(hidden_dim, emb_dim),
        )

    def _precompute_emb_factors(self, time_emb_dim):
        half_dim = time_emb_dim // 2
        if half_dim == 0:
            return torch.empty(0)
        indices = torch.arange(half_dim)
        denom = max(half_dim - 1, 1.0)
        val = math.log(10000) / denom
        return torch.exp(-indices * val)


    def forward(self, x, t, u, vt_feat):
        # time
        t_float = t.float().unsqueeze(-1)
        emb_prod = t_float * self.emb_factors.unsqueeze(0).to(t_float.device)
        t_emb_sincos = torch.cat([torch.sin(emb_prod), torch.cos(emb_prod)], dim=-1)
        t_emb = self.time_proj(t_emb_sincos)

        # project to  hidden_dim
        x_fused = torch.cat([x, t_emb], dim=1)
        x_processed = self.initial_fusion_net(x_fused)  # [N, hidden_dim]

        # e_u + LID
        u_processed = self.user_proj(u)  # [N, hidden_dim]

        # M
        vt_processed = self.modality_proj(vt_feat)
        kv_sequence = torch.cat([
            x_processed.unsqueeze(1),
            # v_processed.unsqueeze(1),
            # t_feat_processed.unsqueeze(1)
            vt_processed.unsqueeze(1)
        ], dim=1)

        # KV
        memory = self.modality_encoder(kv_sequence)


        # Q
        query_sequence = u_processed.unsqueeze(1)

        # Imagination
        u_refined = self.user_decoder(
            tgt=query_sequence,
            memory=memory
        )  # [N, 1, hidden_dim]
        item_output = u_refined.squeeze(1)  # [N, hidden_dim]

        # noise projection
        output = self.output_net(item_output)  # [N, emb_dim]

        return output

class PGL_IDEA(GeneralRecommender):
    def __init__(self, config, dataset):
        super(PGL_IDEA, self).__init__(config, dataset)
        self.mode = config['mode']

        self.embedding_dim = config['embedding_size']
        self.feat_embed_dim = config['feat_embed_dim']
        self.knn_k = config['knn_k']
        self.lambda_coeff = config['lambda_coeff']
        self.n_layers = config['n_mm_layers']
        self.n_ui_layers = config['n_ui_layers']
        self.reg_weight = config['reg_weight']
        self.mm_image_weight = config['mm_image_weight']

        self.n_nodes = self.n_users + self.n_items

        self.sub_graph, self.mm_adj = None, None

        # load dataset info
        self.interaction_matrix = dataset.inter_matrix(form='coo').astype(np.float32)
        self.norm_adj = self.get_norm_adj_mat().to(self.device)
        self.edge_indices, self.edge_values = self.get_edge_info()
        self.edge_indices, self.edge_values = self.edge_indices.to(self.device), self.edge_values.to(self.device)
        self.edge_full_indices = torch.arange(self.edge_values.size(0)).to(self.device)

        self.user_text = nn.Embedding(self.n_users, self.embedding_dim)
        self.user_image = nn.Embedding(self.n_users, self.embedding_dim)
        nn.init.xavier_uniform_(self.user_image.weight)
        nn.init.xavier_uniform_(self.user_text.weight)

        dataset_path = os.path.abspath(config['data_path'] + config['dataset'])
        mm_adj_file = os.path.join(dataset_path,'mm_adj_freedomdsp_{}_{}.pt'.format(self.knn_k, int(10 * self.mm_image_weight)))

        if self.v_feat is not None:
            self.image_embedding = nn.Embedding.from_pretrained(self.v_feat, freeze=False)
            self.image_trs = nn.Linear(self.v_feat.shape[1], self.feat_embed_dim)
        if self.t_feat is not None:
            self.text_embedding = nn.Embedding.from_pretrained(self.t_feat, freeze=False)
            self.text_trs = nn.Linear(self.t_feat.shape[1], self.feat_embed_dim)

        if os.path.exists(mm_adj_file):
            self.mm_adj = torch.load(mm_adj_file)
        else:
            if self.v_feat is not None:
                indices, image_adj = self.get_knn_adj_mat(self.image_embedding.weight.detach())
                self.mm_adj = image_adj
            if self.t_feat is not None:
                indices, text_adj = self.get_knn_adj_mat(self.text_embedding.weight.detach())
                self.mm_adj = text_adj
            if self.v_feat is not None and self.t_feat is not None:
                self.mm_adj = self.mm_image_weight * image_adj + (1.0 - self.mm_image_weight) * text_adj
                del text_adj
                del image_adj
            torch.save(self.mm_adj, mm_adj_file)
        self.dropoutf = nn.Dropout(config['dropout'])

        # diffusion
        self.noise_coeff = config['noise_coeff']  #
        self.diff_weight = config['diff_weight']  # BPR
        self.diffusion_loss_weight = config['diffusion_loss_weight']  # train diffusion  network
        self.hdv_weight = config['hdv_weight']
        self.use_diffusion = config['use_diffusion']
        self.diffusion_steps = config['diffusion_steps']  # 20
        self.denoise_net = IDM() if self.use_diffusion else None
        self.register_buffer('alphas_cumprod', self._linear_schedule(self.diffusion_steps))  # [0,20], 0 is clean
        # print(self.alphas_cumprod)

        self.num_sampling_steps = config['diffusion_steps']

        self.flag_epoch = 0

    def _linear_schedule(self, T):
        beta_start = 0.001 * self.noise_coeff
        beta_end = 0.02 * self.noise_coeff

        betas = torch.linspace(beta_start, beta_end, T )

        betas = torch.cat([torch.zeros(1), betas])  # add 0 step

        alphas = 1 - betas

        alphas_cumprod = torch.cumprod(alphas, dim=0)

        return alphas_cumprod

    def sparse_mx_to_torch_sparse_tensor(self, sparse_mx):
        """Convert a scipy sparse matrix to a torch sparse tensor."""
        sparse_mx = sparse_mx.tocoo().astype(np.float32)
        indices = torch.from_numpy(np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
        values = torch.from_numpy(sparse_mx.data)
        shape = torch.Size(sparse_mx.shape)
        return torch.sparse.FloatTensor(indices, values, shape)

    def get_knn_adj_mat(self, mm_embeddings):
        context_norm = mm_embeddings.div(torch.norm(mm_embeddings, p=2, dim=-1, keepdim=True))
        sim = torch.mm(context_norm, context_norm.transpose(1, 0))
        _, knn_ind = torch.topk(sim, self.knn_k, dim=-1)
        adj_size = sim.size()
        del sim
        # construct sparse adj
        indices0 = torch.arange(knn_ind.shape[0]).to(self.device)
        indices0 = torch.unsqueeze(indices0, 1)
        indices0 = indices0.expand(-1, self.knn_k)
        indices = torch.stack((torch.flatten(indices0), torch.flatten(knn_ind)), 0)
        # norm
        return indices, self.compute_normalized_laplacian(indices, adj_size)

    def compute_normalized_laplacian(self, indices, adj_size):
        adj = torch.sparse.FloatTensor(indices, torch.ones_like(indices[0]), adj_size)
        row_sum = 1e-7 + torch.sparse.sum(adj, -1).to_dense()
        r_inv_sqrt = torch.pow(row_sum, -0.5)
        rows_inv_sqrt = r_inv_sqrt[indices[0]]
        cols_inv_sqrt = r_inv_sqrt[indices[1]]
        values = rows_inv_sqrt * cols_inv_sqrt
        return torch.sparse.FloatTensor(indices, values, adj_size)

    def get_norm_adj_mat(self):
        A = sp.dok_matrix((self.n_users + self.n_items,
                           self.n_users + self.n_items), dtype=np.float32)
        inter_M = self.interaction_matrix
        inter_M_t = self.interaction_matrix.transpose()
        data_dict = dict(zip(zip(inter_M.row, inter_M.col + self.n_users),
                             [1] * inter_M.nnz))
        data_dict.update(dict(zip(zip(inter_M_t.row + self.n_users, inter_M_t.col),
                                  [1] * inter_M_t.nnz)))
        A._update(data_dict)
        # norm adj matrix
        sumArr = (A > 0).sum(axis=1)
        # add epsilon to avoid Devide by zero Warning
        diag = np.array(sumArr.flatten())[0] + 1e-7
        diag = np.power(diag, -0.5)
        D = sp.diags(diag)
        L = D * A * D
        # covert norm_adj matrix to tensor
        L = sp.coo_matrix(L)
        row = L.row
        col = L.col
        i = torch.LongTensor(np.array([row, col]))
        data = torch.FloatTensor(L.data)
        if self.mode == 'global':
            self.sub_graph = self.global_subgraph_extraction(L)
            self.sub_graph = self.sparse_mx_to_torch_sparse_tensor(self.sub_graph).to(self.device)

        return torch.sparse.FloatTensor(i, data, torch.Size((self.n_nodes, self.n_nodes)))

    def global_subgraph_extraction(self, adj):
        norm_adj = adj.tocsc()
        #ut, s, vt = sparsesvd(norm_adj, self.embedding_dim)
        ut, s, vt = svds(norm_adj, k=self.embedding_dim, which='LM')

        s = s[::-1]
        ut = ut[:, ::-1]
        vt = vt[::-1, :]

        # Get the top and bottom 25% of singular values
        num_top_bottom = int(0.25 * self.embedding_dim)
        top_singular_values = s[:num_top_bottom]
        bottom_singular_values = s[-num_top_bottom:]

        # Compute the product of the top and bottom singular values
        product_singular_values = top_singular_values * bottom_singular_values

        # Construct the sparse matrix from the product of singular values
        product_matrix = np.diag(product_singular_values)
        product_sparse_matrix = ut.T[:, :num_top_bottom] @ product_matrix @ vt[:num_top_bottom, :]
        product_sparse_matrix = sp.csr_matrix(product_sparse_matrix * (abs(product_sparse_matrix) >= 1e-3))
        return product_sparse_matrix

    def save(self):
        pass

    def pre_epoch_processing(self):
        if self.mode == 'local':
            # degree-sensitive edge pruning
            degree_len = int(self.edge_values.size(0) * 0.3)
            degree_idx = torch.multinomial(self.edge_values, degree_len)
            # random sample
            keep_indices = self.edge_indices[:, degree_idx]
            # norm values
            keep_values = self._normalize_adj_m(keep_indices, torch.Size((self.n_users, self.n_items)))
            all_values = torch.cat((keep_values, keep_values))
            # update keep_indices to users/items+self.n_users
            keep_indices[1] += self.n_users
            all_indices = torch.cat((keep_indices, torch.flip(keep_indices, [0])), 1)
            self.sub_graph = torch.sparse.FloatTensor(all_indices, all_values, self.norm_adj.shape).to(self.device)

    def _normalize_adj_m(self, indices, adj_size):
        adj = torch.sparse.FloatTensor(indices, torch.ones_like(indices[0]), adj_size)
        row_sum = 1e-7 + torch.sparse.sum(adj, -1).to_dense()
        col_sum = 1e-7 + torch.sparse.sum(adj.t(), -1).to_dense()
        r_inv_sqrt = torch.pow(row_sum, -0.5)
        rows_inv_sqrt = r_inv_sqrt[indices[0]]
        c_inv_sqrt = torch.pow(col_sum, -0.5)
        cols_inv_sqrt = c_inv_sqrt[indices[1]]
        values = rows_inv_sqrt * cols_inv_sqrt
        return values

    def get_edge_info(self):
        rows = torch.from_numpy(self.interaction_matrix.row)
        cols = torch.from_numpy(self.interaction_matrix.col)
        edges = torch.stack([rows, cols]).type(torch.LongTensor)
        # edge normalized values
        values = self._normalize_adj_m(edges, torch.Size((self.n_users, self.n_items)))
        return edges, values

    def forward(self, adj):
        if self.v_feat is not None:
            image_feats = self.image_trs(self.image_embedding.weight)
        if self.t_feat is not None:
            text_feats = self.text_trs(self.text_embedding.weight)

        image_feats, text_feats = F.normalize(image_feats), F.normalize(text_feats)
        user_embeds = torch.cat([self.user_image.weight, self.user_text.weight], dim=1)
        item_embeds = torch.cat([image_feats, text_feats], dim=1)

        h = item_embeds
        for i in range(self.n_layers):
            h = torch.sparse.mm(self.mm_adj, h)

        ego_embeddings = torch.cat((user_embeds, item_embeds), dim=0)
        all_embeddings = [ego_embeddings]
        for i in range(self.n_ui_layers):
            side_embeddings = torch.sparse.mm(adj, ego_embeddings)
            ego_embeddings = side_embeddings
            all_embeddings += [ego_embeddings]
        all_embeddings = torch.stack(all_embeddings, dim=1)
        all_embeddings = all_embeddings.mean(dim=1, keepdim=False)
        u_g_embeddings, i_g_embeddings = torch.split(all_embeddings, [self.n_users, self.n_items], dim=0)
        return u_g_embeddings, i_g_embeddings ,h

    def bpr_loss(self, users, pos_items, neg_items):
        pos_scores = torch.sum(torch.mul(users, pos_items), dim=1) #* wp
        neg_scores = torch.sum(torch.mul(users, neg_items), dim=1) #* wn

        #maxi = w * F.logsigmoid(pos_scores - neg_scores)
        maxi =  F.logsigmoid(pos_scores - neg_scores)
        mf_loss = -torch.mean(maxi)

        return mf_loss

    def InfoNCE(self, view1, view2, temperature):
        view1, view2 = F.normalize(view1, dim=1), F.normalize(view2, dim=1)
        pos_score = (view1 * view2).sum(dim=-1)
        pos_score = torch.exp(pos_score / temperature)
        ttl_score = torch.matmul(view1, view2.transpose(0, 1))
        ttl_score = torch.exp(ttl_score / temperature).sum(dim=1)
        cl_loss = -torch.log(pos_score / ttl_score)
        return torch.mean(cl_loss)

    def calculate_loss(self, interaction,epoch_idx,batch_idx):
        users = interaction[0]
        pos_items = interaction[1]
        neg_items = interaction[2]

        ua_embeddings, i_id_embeddings,i_vt = self.forward(self.sub_graph)

        ia_embeddings =  i_id_embeddings + i_vt
        u_g_embeddings = ua_embeddings[users]
        pos_i_g_embeddings = ia_embeddings[pos_items]
        pos_i_id = i_id_embeddings[pos_items]
        pos_i_vt = i_vt[pos_items]
        neg_i_g_embeddings = ia_embeddings[neg_items]

        cl_loss = (self.InfoNCE(self.dropoutf(u_g_embeddings), self.dropoutf(u_g_embeddings), 0.2)
                   + self.InfoNCE(self.dropoutf(pos_i_g_embeddings), self.dropoutf(pos_i_g_embeddings), 0.2)) / 2

        # diffusion
        diffusion_loss = 0.0
        generated_bpr_loss = 0.0
        kl_loss = 0.0

        if self.use_diffusion and self.denoise_net:
            diffusion_loss, generated_bpr_loss, ndv_loss = self.interest_diffusion(u_g_embeddings,
                            pos_i_id, pos_i_vt, neg_i_g_embeddings,epoch_idx,batch_idx)

        batch_mf_loss = self.bpr_loss(u_g_embeddings, pos_i_g_embeddings, neg_i_g_embeddings)
        total_loss = (1 - self.diff_weight) * batch_mf_loss \
                     + self.reg_weight * cl_loss \
                     + self.diffusion_loss_weight * diffusion_loss \
                     + self.diff_weight* generated_bpr_loss \
                     + self.hdv_weight * ndv_loss \

        return total_loss

    def full_sort_predict(self, interaction):
        user = interaction[0]

        restore_user_e, restore_item_e,h = self.forward(self.norm_adj)
        restore_item_e = restore_item_e + h
        u_embeddings = restore_user_e[user]

        scores = torch.matmul(u_embeddings, restore_item_e.transpose(0, 1))
        return scores

    def generate_samples(self, original_item_emb, user_embeddings, epoch_idx,pos_vt_feat):

        batch_size = user_embeddings.shape[0]
        start_t_ratio = 0.6
        start_t = int(self.diffusion_steps * start_t_ratio)

        if self.flag_epoch == epoch_idx:
            self.flag_epoch +=1
            print(epoch_idx, start_t_ratio,start_t)

        noise = torch.randn_like(original_item_emb)
        alpha_start = self.alphas_cumprod[start_t]
        x = alpha_start.sqrt() * original_item_emb + (1 - alpha_start).sqrt() * noise

        with torch.no_grad(): #DDIM
            for t in range(start_t, 0, -1):
                t_prev = t - 1 if t > 1 else 0

                alpha_bar_t = self.alphas_cumprod[t]
                # t=1, alpha_bar_0 = 1.0
                alpha_bar_t_prev = self.alphas_cumprod[t_prev] if t_prev >= 0 else torch.tensor(1.0, device=self.device)

                t_tensor = torch.full((batch_size,), t, device=self.device).long()
                predicted_noise = self.denoise_net(x, t_tensor, user_embeddings, pos_vt_feat)

                predicted_x0 = (x - (1.0 - alpha_bar_t).sqrt() * predicted_noise) / alpha_bar_t.sqrt()
                # eta=0
                sigma_t = 0.0

                if t > 1:
                    x_mean = alpha_bar_t_prev.sqrt() * predicted_x0
                    direction_to_x_t = (1.0 - alpha_bar_t_prev - sigma_t ** 2).sqrt() * predicted_noise
                    z = torch.randn_like(x) if sigma_t > 0 else 0.0
                    noise_term = sigma_t * z
                    x = x_mean + direction_to_x_t + noise_term
                else:
                    x = predicted_x0  #
        return x

    def interest_diffusion(self,u_g_embeddings,pos_i_g_embeddings,pos_vt_feat,neg_i_g_embeddings,epoch_idx,batch_idx):
        batch_size = u_g_embeddings.shape[0]

        # Local Interest Distribution
        local_mean,local_variance,max_var, mean_and_var = self.calculate_anchored_variance_invariant(u_g_embeddings,pos_i_g_embeddings+pos_vt_feat)

        u_tilde = torch.cat([u_g_embeddings, mean_and_var], dim=1)

        t = torch.randint(1, self.diffusion_steps+1, (batch_size,), device=self.device).long()

        noise = torch.randn_like(pos_i_g_embeddings)
        alpha_t = self.alphas_cumprod[t].unsqueeze(-1)
        noisy_pos_i = alpha_t.sqrt() * pos_i_g_embeddings.detach() + (1 - alpha_t).sqrt() * noise

        predicted_noise = self.denoise_net(noisy_pos_i, t, u_tilde, pos_vt_feat)

        # MSE loss
        diffusion_loss = F.mse_loss(predicted_noise, noise)

        # estimated clean state
        predicted_x0 = (noisy_pos_i - (1.0 - alpha_t).sqrt() * predicted_noise) / alpha_t.sqrt() # [batch_size, emb_dim]

        # virtual samples
        generated_item_emb = self.generate_samples(pos_i_g_embeddings.detach(), u_tilde,epoch_idx,pos_vt_feat)

        pos_scores = torch.sum(u_g_embeddings * generated_item_emb, dim=1)  # [batch_size]
        neg_scores = torch.sum(u_g_embeddings * neg_i_g_embeddings, dim=1)  # [batch_size]

        # MixedBernoulli Distribution Estimation
        mu_social,var_e,var_i = self.compute_social_mean_var(u_g_embeddings,pos_i_g_embeddings+pos_vt_feat,generated_item_emb,neg_scores,neg_i_g_embeddings)
        var_weight = self.cal_var_weight(var_e,var_i)

        # Natural Distribution Verification
        ndv_loss = self.cal_entropy_loss_global_anchor(u_g_embeddings,pos_i_g_embeddings + pos_vt_feat, predicted_x0,neg_i_g_embeddings)

        basic_bpr_loss = -torch.log(torch.sigmoid(mu_social*var_weight * pos_scores - neg_scores)).mean()

        return diffusion_loss, basic_bpr_loss,ndv_loss,

    def cal_var_weight(self,var_e,var_i):
        return torch.exp(-var_e)

    def find_similar_users(self,target_user_embedding, pos_i_g_embeddings, k=10):
        n_target_user_embedding = F.normalize(target_user_embedding)

        similarities = torch.mm(n_target_user_embedding, n_target_user_embedding.T)

        value, indices = torch.topk(similarities, k=k + 1,dim=1)
        similar_user_indices = indices[:, 1: k + 1]
        similar_user_val_coeff= value[:,1:k+1]
        similar_user_weights = F.softmax(similar_user_val_coeff, dim=1)

        return similar_user_indices,similar_user_weights

    def compute_social_mean_var(self, u_g_embeddings, pos_i_g_embeddings, generated_item_emb,neg_scores,neg_i_g_embeddings):

        similar_user_indices, similar_user_weights = self.find_similar_users(u_g_embeddings,pos_i_g_embeddings,k=10)

        batch_similar_user_embs = u_g_embeddings[similar_user_indices]  # [batch_size, k, emb_dim]
        generated_item_emb_expanded = generated_item_emb.unsqueeze(1)  # [batch_size, 1, emb_dim]


        social_scores = torch.sum(batch_similar_user_embs * generated_item_emb_expanded, dim=-1)  # 形状 [batch_size, k]

        neg_scores_expanded = neg_scores.unsqueeze(1).expand_as(social_scores)
        contrast_scores = social_scores - neg_scores_expanded

        P_social = torch.tanh(torch.max(contrast_scores, torch.tensor(0)))

        mu_social = torch.sum(similar_user_weights * P_social, dim=1)

        P_diff_sq = torch.pow(P_social - mu_social.unsqueeze(1), 2)
        var_term1 = torch.sum(similar_user_weights * P_diff_sq, dim=1)


        P_intrinsic_var = P_social * (1 - P_social)
        var_term2 = torch.sum(similar_user_weights * P_intrinsic_var, dim=1)

        return mu_social, var_term1,var_term2

    def _compute_group_mean_embedding(self, u_g_embeddings, similar_user_indices, similar_user_weights):

        batch_similar_user_embs = u_g_embeddings[similar_user_indices]  #  [N, k, emb_dim]

        weighted_embs = batch_similar_user_embs * similar_user_weights.unsqueeze(-1)


        u_sim_mean = torch.sum(weighted_embs, dim=1)  #  [N, emb_dim]

        return u_sim_mean

    def cal_entropy_loss_global_anchor(self, u, pos, gen, k=10):
        N = pos.size(0)
        in_batch_random_indices = torch.randint(0, N, (N, k), device=pos.device)
        random_user_embs = u[in_batch_random_indices].detach()
        target_user_embs = u.detach().unsqueeze(1)
        combined_user_embs = torch.cat([target_user_embs, random_user_embs], dim=1)

        pos_detached = pos.detach().unsqueeze(2)  # 
        gen_with_grad = gen.unsqueeze(2)  #

 
        Score_Q = torch.bmm(combined_user_embs, pos_detached).squeeze(-1)  # [N, K+1]

        Q_social = F.softmax(Score_Q, dim=1).detach()
        Log_Q = F.log_softmax(Score_Q, dim=1).detach()
        H_real = -torch.sum(Q_social * Log_Q, dim=1)

        Score_P = torch.bmm(combined_user_embs, gen_with_grad).squeeze(-1)  # [N, K+1]

        P_social = F.softmax(Score_P, dim=1)
        Log_P = F.log_softmax(Score_P, dim=1)
        H_est = -torch.sum(P_social * Log_P, dim=1)

        return F.mse_loss(H_est, H_real)

        def calculate_anchored_variance_invariant(self, u_emb: torch.Tensor, pos_emb: torch.Tensor, k: int = 10, M: int = 10):
        N_user, D = u_emb.shape
        N_sample = pos_emb.shape[0]

        u_emb_norm = F.normalize(u_emb, p=2, dim=1)  # (N_user, D)
        pos_emb_norm = F.normalize(pos_emb, p=2, dim=1)  # (N_sample, D)

        random_anchor_indices = torch.randperm(N_sample)[:M].to(u_emb.device)
        anchor_emb = pos_emb_norm[random_anchor_indices]

        sim_anchor_all = torch.matmul(anchor_emb, pos_emb_norm.transpose(0, 1))
        _, anchor_indices_k = torch.topk(sim_anchor_all, k=k, dim=1, largest=True)

        S_U_P = torch.matmul(u_emb_norm, pos_emb_norm.transpose(0, 1))
        flat_indices = anchor_indices_k.flatten()
        gathered_flat = S_U_P[:, flat_indices]
        S_U_R = gathered_flat.view(N_user, M, k)

 
        anchored_mean = torch.mean(S_U_R, dim=2)
        anchored_var, _ = torch.var_mean(S_U_R, dim=2, correction=1)
        max_var, _ = torch.max(anchored_var, dim=1, keepdim=True)

        mean_expanded = anchored_mean.unsqueeze(-1)
        var_expanded = anchored_var.unsqueeze(-1)
  
        mean_var_stacked = torch.cat([mean_expanded, var_expanded], dim=-1)
  
        transformed_features = self.lid_aggregator(mean_var_stacked)

        # anchored_features = torch.mean(transformed_features, dim=1)
        anchored_features = torch.max(transformed_features, dim=1).values

        return anchored_mean, anchored_var, max_var, anchored_features
