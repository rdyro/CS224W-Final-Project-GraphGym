import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.nn as pyg_nn
from torch_geometric.graphgym.models import MLP
from torch_geometric.graphgym.models.layer import LayerConfig

from graphgym.config import cfg
from graphgym.models.act import act_dict as inbuilt_act_dict
from graphgym.register import register_network
from graphgym.register import act_dict
from copy import deepcopy

act_dict = dict(act_dict, **inbuilt_act_dict, identity=lambda x: x)


class IdentityModule(nn.Module):
    def __init__(self, *args, **kw):
        super().__init__()

    def forward(self, *args, **kw):
        return args[0]


class GNNNodeHead(nn.Module):
    """Head of GNN, node prediction"""

    def __init__(self, dim_in, dim_out):
        super().__init__()
        self.layer_post_mp = MLP(
            LayerConfig(dim_in=dim_in, dim_out=dim_out, num_layers=cfg.gnn.layers_post_mp),
            bias=True,
        )

    def _apply_index(self, batch):
        if batch.node_label_index.shape[0] == batch.node_label.shape[0]:
            return batch.node_feature[batch.node_label_index], batch.node_label
        else:
            return (
                batch.node_feature[batch.node_label_index],
                batch.node_label[batch.node_label_index],
            )

    def forward(self, batch):
        batch = self.layer_post_mp(batch)
        pred, label = self._apply_index(batch)
        return pred, label


class NASGNN(torch.nn.Module):
    def __init__(self, dim_in, dim_out, dropout=0.0, block_num=4):
        super().__init__()
        self.dropout = dropout

        # we have 4 blocks, block 0 is the input, block 3 is the output
        # block 0 connects to block 1, 2, 3
        # block 1 connects to block 2, 3
        # block 2 connects to block 3
        # each connection is any of the following activations:
        # (ReLU, PReLU, Tanh, Identity)
        # each block is either an identity of a conv_model

        block_num = block_num if "block_num" not in cfg.nas else cfg.nas.block_num
        self.blocks = nn.ModuleList()
        for i in range(block_num):
            self.blocks.append(self.build_conv_model(cfg.nas[f"node{i}"])(dim_in, dim_in))
        self.activations = nn.ModuleDict()
        for i in range(block_num):
            for j in range(i + 1, block_num):
                self.activations[f"{i}_{j}"] = deepcopy(act_dict[cfg.nas[f"node_{i}_{j}_act"]])
        self.post_mp = GNNNodeHead(dim_in=dim_in, dim_out=dim_out)

    def forward(self, batch):
        x, edge_index, x_batch = batch.node_feature, batch.edge_index, batch.batch

        block_inputs = [[x]] + [[] for _ in range(1, len(self.blocks))]
        latest_output = x
        for i, block in enumerate(self.blocks):
            # apply the block to all its inputs and sum the output
            block_output = sum(
                F.dropout(block(x, edge_index), p=self.dropout, training=self.training)
                for x in block_inputs[i]
            )
            # record the latest output (for output)
            latest_output = block_output
            for j in range(i + 1, len(self.blocks)):
                # apply the specified activations to the output of the block for other blocks
                block_inputs[j].append(self.activations[f"{i}_{j}"](block_output))
        x = latest_output
        x = F.log_softmax(x, dim=1)
        batch.node_feature = x
        batch.x = x
        return self.post_mp(batch)

    def build_conv_model(self, model_type):
        if model_type == "GCN":
            return pyg_nn.GCNConv
        elif model_type == "GAT":
            return pyg_nn.GATConv
        elif model_type == "GraphSage":
            return pyg_nn.SAGEConv
        elif model_type == "Identity":
            return IdentityModule
        else:
            raise ValueError("Model {} unavailable".format(model_type))


register_network("nasgnn", NASGNN)
