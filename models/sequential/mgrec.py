import math
import random
from models.base_model import BaseModel
from models.model_utils import TransformerLayer, TransformerEmbedding
import numpy as np
import torch
from torch import nn
from config.configurator import configs
import torch.nn.functional as F
import dgl
from dgl.nn.pytorch import GraphConv
from copy import deepcopy

def graph_augment(g: dgl.DGLGraph, user_ids, user_edges):
    # Augment the graph with the item sequence, deleting co-occurrence edges in the batched sequences
    # generating indicies like: [1,2] [2,3] ... as the co-occurrence rel.
    # indexing edge data using node indicies and delete them
    # for edge weights, delete them from the raw data using indexed edges
    user_ids = user_ids.cpu().numpy()
    node_indicies_a = np.concatenate(
        user_edges.loc[user_ids, "item_edges_a"].to_numpy())
    node_indicies_b = np.concatenate(
        user_edges.loc[user_ids, "item_edges_b"].to_numpy())
    node_indicies_a = torch.from_numpy(
        node_indicies_a).to(g.device)
    node_indicies_b = torch.from_numpy(
        node_indicies_b).to(g.device)
    edge_ids = g.edge_ids(node_indicies_a, node_indicies_b)

    aug_g: dgl.DGLGraph = deepcopy(g)
    # The features for the removed edges will be removed accordingly.
    aug_g.remove_edges(edge_ids)
    return aug_g


def graph_dropout(g: dgl.DGLGraph, keep_prob):
    # Firstly mask selected edge values, returns the true values along with the masked graph.
    origin_edge_w = g.edata['w']

    drop_size = int((1-keep_prob) * g.num_edges())
    random_index = torch.randint(
        0, g.num_edges(), (drop_size,), device=g.device)
    mask = torch.zeros(g.num_edges(), dtype=torch.uint8,
                       device=g.device).bool()
    mask[random_index] = True
    g.edata['w'].masked_fill_(mask, 0)

    return origin_edge_w, g

def build_ui_interaction_graph(batch_user, batch_seqs, item_num, device):
    # Build a batch of user-item interaction graphs using DGL
    batch_size = batch_user.size(0)
    ui_interaction_graphs = []
    
    for batch_idx in range(batch_size):
        user_id = batch_user[batch_idx].item()
        items = batch_seqs[batch_idx][batch_seqs[batch_idx] > 0]  # filter out padding (0s)
        
        if len(items) == 0:
            # Handle empty sequence case
            g = dgl.graph(([], []), num_nodes=item_num + 1, device=device)
        else:
            # Create edges from user to items
            src_nodes = torch.full((len(items),), user_id, device=device)
            dst_nodes = items.to(device)
            g = dgl.graph((src_nodes, dst_nodes), num_nodes=item_num + 1, device=device)
            # Set edge weights to 1
            g.edata['w'] = torch.ones(g.num_edges(), device=device)
        
        ui_interaction_graphs.append(g)
    
    # Stack graphs into a batch
    ui_interaction_graph = dgl.batch(ui_interaction_graphs)
    return ui_interaction_graph


class GCN(nn.Module):
    def __init__(self, in_dim, out_dim, dropout_prob=0.7):
        super(GCN, self).__init__()
        self.dropout_prob = dropout_prob
        self.layer = GraphConv(in_dim, out_dim, weight=False,
                               bias=False, allow_zero_in_degree=False)

    def forward(self, graph, feature):
        graph = dgl.add_self_loop(graph)
        origin_w, graph = graph_dropout(graph, 1-self.dropout_prob)
        embs = [feature]
        for i in range(2):
            feature = self.layer(graph, feature, edge_weight=graph.edata['w'])
            F.dropout(feature, p=0.2, training=self.training)
            embs.append(feature)
        embs = torch.stack(embs, dim=1)
        final_emb = torch.mean(embs, dim=1)
        # recover edge weight
        graph.edata['w'] = origin_w
        return final_emb


class MGRec(BaseModel):

    def __init__(self, data_handler):
        super(MGRec, self).__init__(data_handler)
        self.data_handler = data_handler
        self.device = configs['device']
        self.item_num = configs['data']['item_num']
        self.emb_size = configs['model']['embedding_size']
        self.max_len = configs['model']['max_seq_len']
        # load parameters info
        self.n_layers = configs['model']['n_layers']
        self.n_heads = configs['model']['n_heads']
        self.emb_size = configs['model']['embedding_size']
        # the dimensionality in feed-forward layer
        self.inner_size = 4 * self.emb_size
        self.dropout_rate = configs['model']['dropout_rate']
        self.batch_size = configs['train']['batch_size']

        self.weight_mean = configs['model']['weight_mean']
        # load dataset info
        # define layers and loss
        self.emb_layer = TransformerEmbedding(
            self.item_num + 1, self.emb_size, self.max_len)

        self.transformer_layers = nn.ModuleList([TransformerLayer(
            self.emb_size, self.n_heads, self.inner_size, self.dropout_rate) for _ in range(self.n_layers)])

        self.loss_func = nn.CrossEntropyLoss()

        self.dropout = nn.Dropout(self.dropout_rate)

        self.layernorm = nn.LayerNorm(self.emb_size, eps=1e-12)

        # Fusion Attn
        self.attn_weights = nn.Parameter(torch.Tensor(self.emb_size, self.emb_size))
        self.attn = nn.Parameter(torch.Tensor(1, self.emb_size))
        nn.init.normal_(self.attn, std=0.02)
        nn.init.normal_(self.attn_weights, std=0.02)

        # Global Graph Learning
        self.transition_graph = data_handler.train_dataloader.dataset.transition_graph.to(self.device)
        self.user_edges = data_handler.train_dataloader.dataset.user_edges
        self.item_simgraph = data_handler.train_dataloader.dataset.co_interaction_graph.to(self.device)
        self.graph_dropout = configs["model"]["graph_dropout_prob"]

        self.gcn = GCN(self.emb_size, self.emb_size, self.graph_dropout)
        self.mlp = nn.Linear(self.emb_size, self.emb_size, bias=True)

        self.loss_fct = nn.CrossEntropyLoss()

        # parameters initialization
        self.apply(self._init_weights)

    def _init_weights(self, module):
        """ Initialize the weights """
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

    def get_attention_mask(self, item_seq, task_label=False):
        """Generate bidirectional attention mask for multi-head attention."""
        if task_label:
            label_pos = torch.ones((item_seq.size(0), 1), device=self.device)
            item_seq = torch.cat((label_pos, item_seq), dim=1)
        attention_mask = (item_seq > 0).long()
        extended_attention_mask = attention_mask.unsqueeze(
            1).unsqueeze(2)  # torch.int64
        # bidirectional mask
        extended_attention_mask = extended_attention_mask.to(
            dtype=next(self.parameters()).dtype)  # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
        return extended_attention_mask

    def gcn_forward(self, g=None):
        item_emb = self.emb_layer.token_emb.weight
        item_emb = self.dropout(item_emb)
        light_out = self.gcn(g, item_emb)
        return self.layernorm(light_out+item_emb)

    def forward(self, batch_seqs):
        mask = (batch_seqs > 0).unsqueeze(1).repeat(
            1, batch_seqs.size(1), 1).unsqueeze(1) # [B,B,L,1]
        x = self.emb_layer(batch_seqs)
        for transformer in self.transformer_layers:
            x = transformer(x, mask)
        return x[:, -1, :]  # [B H]

    def cal_loss(self, batch_data):
        batch_user, batch_seqs, batch_pos_items = batch_data
        last_items = batch_seqs[:, -1].view(-1)
        # graph view
        # masked_g = self.transition_graph
        # aug_g = graph_augment(self.transition_graph, batch_user, self.user_edges)
        # ui_interaction_graph = build_ui_interaction_graph(batch_user, batch_seqs, self.item_num, self.device)
        # session_graph = ui_interaction_graph.transpose(0,1) * ui_interaction_graph

        transition_graph_emb = self.gcn_forward(self.transition_graph) # [B, N_node_train, D]
        co_interaction_graph_emb = self.gcn_forward(self.item_simgraph)
        # session_graph_emb = self.gcn_forward(session_graph)

        transition_graph_emb_last_items = transition_graph_emb[last_items]
        co_interaction_graph_emb_last_items = co_interaction_graph_emb[last_items]

        seq_output = self.forward(batch_seqs)

        # hybrid_emb = transition_graph_emb + co_interaction_graph_emb
        # z = nn.Tanh(self.mlp(hybrid_emb))
        # S = hybrid_emb / self.max_len

        # Fusion After CL
        # 3, N_mask, dim
        hybrid_emb = torch.stack(
            (seq_output, transition_graph_emb[last_items], co_interaction_graph_emb[last_items]), dim=0)
        weights = (torch.matmul(hybrid_emb, self.attn_weights.unsqueeze(0))*self.attn).sum(-1)
        # 3, N_mask, 1
        score = F.softmax(weights, dim=0).unsqueeze(-1)
        seq_output = (hybrid_emb*score).sum(0)
        # [item_num, H]
        item_emb = self.emb_layer.token_emb.weight
        logits = torch.matmul(seq_output, item_emb.transpose(0, 1))
        loss = self.loss_fct(logits+1e-8, batch_pos_items)

        loss_dict = {
            "loss": loss.item(),
        }
        return loss, loss_dict

    def full_predict(self, batch_data):
        _, batch_seqs, _ = batch_data
        seq_output = self.forward(batch_seqs)
        last_items = batch_seqs[:, -1].view(-1)

        # graph view
        transition_graph = self.data_handler.test_dataloader.dataset.transition_graph.to(self.device)
        co_interaction_graph = self.data_handler.test_dataloader.dataset.co_interaction_graph.to(self.device)
        itransition_graph_output_raw = self.gcn_forward(transition_graph)
        itransition_graph_output_seq = itransition_graph_output_raw[last_items]
        ico_interaction_graph_output_seq = self.gcn_forward(co_interaction_graph)[last_items]
        # 3, N_mask, dim
        hybrid_emb = torch.stack(
            (seq_output, itransition_graph_output_seq, ico_interaction_graph_output_seq), dim=0)
        weights = (torch.matmul(
            hybrid_emb, self.attn_weights.unsqueeze(0))*self.attn).sum(-1)
        # 3, N_mask, 1
        score = F.softmax(weights, dim=0).unsqueeze(-1)
        seq_output = (hybrid_emb*score).sum(0)

        test_item_emb = self.emb_layer.token_emb.weight  # [num, H]
        scores = torch.matmul(seq_output, test_item_emb.transpose(0, 1))
        return scores