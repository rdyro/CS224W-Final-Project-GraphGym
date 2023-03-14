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
    def __init__(self, dim_in, dim_out, dropout=0.0):
        super().__init__()
        self.dropout = dropout

        # we have 4 cells, cell 0 is the input, cell 3 is the output
        # cell 0 connects to cell 1, 2, 3
        # cell 1 connects to cell 2, 3
        # cell 2 connects to cell 3
        # each connection is any of the following activations:
        # (ReLU, ELU, Tanh, Identity)
        # each cell is either an identity of a conv_model

        self.cells = nn.ModuleList()

        self.cells.append(self.build_conv_model(cfg.gnn.node0)(dim_in, dim_in))
        self.cells.append(self.build_conv_model(cfg.gnn.node1)(dim_in, dim_in))
        self.cells.append(self.build_conv_model(cfg.gnn.node2)(dim_in, dim_in))
        self.cells.append(self.build_conv_model(cfg.gnn.node3)(dim_in, dim_in))

        self.node01_act = deepcopy(act_dict[cfg.gnn.node01_act])
        self.node02_act = deepcopy(act_dict[cfg.gnn.node02_act])
        self.node03_act = deepcopy(act_dict[cfg.gnn.node03_act])
        self.node12_act = deepcopy(act_dict[cfg.gnn.node12_act])
        self.node13_act = deepcopy(act_dict[cfg.gnn.node13_act])
        self.node23_act = deepcopy(act_dict[cfg.gnn.node23_act]) 

        self.post_mp = GNNNodeHead(dim_in=dim_in, dim_out=dim_out)

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

    def forward(self, batch):
        x, edge_index, x_batch = batch.node_feature, batch.edge_index, batch.batch

        # cell 0 ###################################################################################
        cell0_input = x
        cell0 = self.cells[0](cell0_input, edge_index)
        cell0 = F.dropout(cell0, p=self.dropout, training=self.training)
        # cell 1 ###################################################################################
        cell1_input = self.node01_act(cell0)
        cell1_input = self.cells[1](cell1_input, edge_index)
        cell1 = F.dropout(cell1_input, p=self.dropout, training=self.training)
        # cell 2 ###################################################################################
        cell2_inputs = (
            self.node02_act(cell0),
            self.node12_act(cell1),
        )
        cell2_inputs = (
            self.cells[2](cell2_inputs[0], edge_index),
            self.cells[2](cell2_inputs[1], edge_index),
        )
        cell2_inputs = (
            F.dropout(cell2_inputs[0], p=self.dropout, training=self.training),
            F.dropout(cell2_inputs[1], p=self.dropout, training=self.training),
        )
        cell2 = sum(cell2_inputs)
        # cell 3 ###################################################################################
        cell3_inputs = (
            self.node03_act(cell0),
            self.node13_act(cell1),
            self.node23_act(cell2),
        )
        cell3_inputs = (
            self.cells[3](cell3_inputs[0], edge_index),
            self.cells[3](cell3_inputs[1], edge_index),
            self.cells[3](cell3_inputs[2], edge_index),
        )
        cell3_inputs = (
            F.dropout(cell3_inputs[0], p=self.dropout, training=self.training),
            F.dropout(cell3_inputs[2], p=self.dropout, training=self.training),
            F.dropout(cell3_inputs[1], p=self.dropout, training=self.training),
        )
        cell3 = sum(cell3_inputs)

        x = cell3

        #x = pyg_nn.global_add_pool(x, x_batch)
        x = F.log_softmax(x, dim=1)
        batch.node_feature = x
        batch.x = x
        return self.post_mp(batch) 


register_network("nasgnn", NASGNN)
