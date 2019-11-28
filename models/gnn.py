import torch
import torch.nn as nn
import torch.nn.functional as F

import torch_geometric.nn as pyg_nn
import torch_geometric.utils as pyg_utils

class GNNStack(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, args):
        super(GNNStack, self).__init__()
        conv_model = self.build_conv_model(args.model_type)
        self.convs = nn.ModuleList()
        self.convs.append(conv_model(input_dim, hidden_dim))
        assert (args.num_layers >= 1), 'Number of layers is not >=1'
        for l in range(args.num_layers-1):
            self.convs.append(conv_model(hidden_dim, hidden_dim))

        # post-message-passing
        self.post_mp = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim), nn.Dropout(args.dropout), 
            nn.Linear(hidden_dim, output_dim))

        self.dropout = args.dropout
        self.num_layers = args.num_layers
        self.hidden_dim = hidden_dim

    def build_conv_model(self, model_type):
        if model_type == 'GCN':
            return pyg_nn.GCNConv
        elif model_type == 'GraphSage':
            return GraphSage
#         elif model_type == 'GAT':
#             # When applying GAT with num heads > 1, one needs to modify the 
#             # input and output dimension of the conv layers (self.convs),
#             # to ensure that the input dim of the next layer is num heads
#             # multiplied by the output dim of the previous layer.
#             # HINT: In case you want to play with multiheads, you need to change the for-loop when builds up self.convs to be
#             # self.convs.append(conv_model(hidden_dim * num_heads, hidden_dim)), 
#             # and also the first nn.Linear(hidden_dim * num_heads, hidden_dim) in post-message-passing.
#             return GAT

    def forward(self, data):
        x, edge_index, batch, eval_edges = data.x, data.edge_index, data.batch, data.eval_edges

        ############################################################################
        # TODO: Your code here! 
        # Each layer in GNN should consist of a convolution (specified in model_type),
        # a non-linearity (use RELU), and dropout. 
        # HINT: the __init__ function contains parameters you will need. For whole
        # graph classification (as specified in self.task) apply max pooling over
        # all of the nodes with pyg_nn.global_max_pool as the final layer.
        # Our implementation is ~6 lines, but don't worry if you deviate from this.

        
        for i in range(self.num_layers):
            x = self.convs[i](x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)

        ############################################################################

        edge_concats = torch.zeros((eval_edges.shape[1], 2 * self.hidden_dim))
        for c in range(eval_edges.shape[1]):
            edge_concats[c] = torch.cat((x[eval_edges[0][c]], x[eval_edges[1][c]]))
        x = self.post_mp(edge_concats)

        return F.log_softmax(x, dim=1)

    def loss(self, pred, label):
        return F.nll_loss(pred, label)
    
class GraphSage(pyg_nn.MessagePassing):
    """Non-minibatch version of GraphSage."""
    def __init__(self, in_channels, out_channels, reducer='mean', 
                 normalize_embedding=True):
        super(GraphSage, self).__init__(aggr='add')

        ############################################################################
        # TODO: Your code here! 
        # Define the layers needed for the message and update functions below.
        # self.lin is the linear transformation that you apply to each neighbor before aggregating them
        # self.agg_lin is the linear transformation you apply to the concatenated self embedding (skip connection) and mean aggregated neighbors
        # Our implementation is ~2 lines, but don't worry if you deviate from this.

        self.lin = nn.Linear(in_channels, out_channels)
        self.agg_lin = nn.Linear(in_channels + out_channels, out_channels)

        ############################################################################

        if normalize_embedding:
            self.normalize_emb = True

    def forward(self, x, edge_index):
        num_nodes = x.size(0)
        # x has shape [N, in_channels]
        # edge_index has shape [2, E]

        return self.propagate(edge_index, size=(num_nodes, num_nodes), x=x)

    def message(self, x_j, edge_index, size):
        # x_j has shape [E, in_channels]
        # edge_index has shape [2, E]
        
        ############################################################################
        # TODO: Your code here! 
        # Given x_j, perform the aggregation of a dense layer followed by a RELU non-linearity.
        # Notice that the aggregator operation will be done in self.propagate. 
        # HINT: It may be useful to read the pyg_nn implementation of GCNConv,
        # https://pytorch-geometric.readthedocs.io/en/latest/notes/create_gnn.html
        # Our implementation is ~1 line, but don't worry if you deviate from this.

        x_j = F.relu(self.lin(x_j)) # TODO

        ############################################################################

        return x_j

    def update(self, aggr_out, x):
        # aggr_out has shape [N, out_channels]
        # x has shape [N, in_channels]
        
        ############################################################################
        # TODO: Your code here! Perform the update step here. 
        # Perform a MLP with skip-connection, that is a concatenation followed by 
        # a linear layer and a RELU non-linearity.
        # Finally, remember to normalize as vector as shown in GraphSage algorithm.
        # Our implementation is ~4 lines, but don't worry if you deviate from this.
        
        aggr_out = torch.cat((aggr_out, x), 1)
        aggr_out = F.relu(self.agg_lin(aggr_out))
        
        if self.normalize_emb:
            aggr_out = F.normalize(aggr_out)

        ############################################################################

        return aggr_out

