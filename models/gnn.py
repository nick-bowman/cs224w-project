import torch
import torch.nn as nn
import torch.nn.functional as F

import torch_geometric.nn as pyg_nn
import torch_geometric.utils as pyg_utils

LEAKY_RELU_SLOPE = 0.2

class GNNStack(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, args, dev):
        super(GNNStack, self).__init__()
        num_heads = args.num_heads if args.model_type == 'GAT' else 1
        conv_model = self.build_conv_model(args.model_type, num_heads)
        self.convs = nn.ModuleList()
        self.convs.append(conv_model(input_dim, hidden_dim))
        assert (args.num_layers >= 1), 'Number of layers is not >=1'
        for l in range(args.num_layers-1):
            self.convs.append(conv_model(hidden_dim * num_heads, hidden_dim))

        # post-message-passing
        self.post_mp = nn.Sequential(
            nn.Linear(hidden_dim * 2 * num_heads, hidden_dim), nn.Dropout(args.dropout), 
            nn.Linear(hidden_dim, output_dim))

        self.dropout = args.dropout
        self.num_layers = args.num_layers
        self.hidden_dim = hidden_dim
        self.dev = dev

    def build_conv_model(self, model_type, num_heads):
        if model_type == 'GCN':
            return pyg_nn.GCNConv
        elif model_type == 'GraphSage':
            return GraphSage
        elif model_type == 'GAT':
            # When applying GAT with num heads > 1, one needs to modify the 
            # input and output dimension of the conv layers (self.convs),
            # to ensure that the input dim of the next layer is num heads
            # multiplied by the output dim of the previous layer.
            # HINT: In case you want to play with multiheads, you need to change the for-loop when builds up self.convs to be
            # self.convs.append(conv_model(hidden_dim * num_heads, hidden_dim)), 
            # and also the first nn.Linear(hidden_dim * num_heads, hidden_dim) in post-message-passing.
            return lambda input_dim, output_dim : GAT(input_dim, output_dim, num_heads=num_heads)


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

        edge_concats = torch.cat((x[eval_edges[0]], x[eval_edges[1]]),dim=1)
        x = self.post_mp(edge_concats)

        return F.log_softmax(x, dim=1)

    def loss(self, pred, label):
        return F.nll_loss(pred, label,weight=torch.Tensor([1,3]).to(self.dev))

class GraphSage(pyg_nn.MessagePassing):
    """Non-minibatch version of GraphSage."""
    def __init__(self, in_channels, out_channels, reducer='mean', 
                 normalize_embedding=True):
        super(GraphSage, self).__init__(aggr='add')

        self.lin = nn.Linear(in_channels, out_channels)
        self.agg_lin = nn.Linear(in_channels + out_channels, out_channels)

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

class GAT(pyg_nn.MessagePassing):

    def __init__(self, in_channels, out_channels, num_heads=1, concat=True,
                 dropout=0, bias=True, **kwargs):
        super(GAT, self).__init__(aggr='add', **kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.heads = num_heads
        self.concat = concat 
        self.dropout = dropout

        print("Initializing GAT with {} heads".format(num_heads))

        ############################################################################
        #  TODO: Your code here!
        # Define the layers needed for the forward function. 
        # Remember that the shape of the output depends the number of heads.
        # Our implementation is ~1 line, but don't worry if you deviate from this.

        self.lin = nn.Linear(in_channels, out_channels * num_heads)
        ############################################################################

        ############################################################################
        #  TODO: Your code here!
        # The attention mechanism is a single feed-forward neural network parametrized
        # by weight vector self.att. Define the nn.Parameter needed for the attention
        # mechanism here. Remember to consider number of heads for dimension!
        # Our implementation is ~1 line, but don't worry if you deviate from this.

        self.att = nn.Parameter(torch.Tensor(self.heads, out_channels * 2))

        ############################################################################

        if bias and concat:
            self.bias = nn.Parameter(torch.Tensor(self.heads * out_channels))
        elif bias and not concat:
            self.bias = nn.Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)

        nn.init.xavier_uniform_(self.att)
        nn.init.zeros_(self.bias)

        ############################################################################

    def forward(self, x, edge_index, size=None):
        ############################################################################
        #  TODO: Your code here!
        # Apply your linear transformation to the node feature matrix before starting
        # to propagate messages.
        # Our implementation is ~1 line, but don't worry if you deviate from this.
        x = self.lin(x)
        ############################################################################

        # Start propagating messages.
        return self.propagate(edge_index, size=size, x=x)

    def message(self, edge_index_i, x_i, x_j, size_i):
        #  Constructs messages to node i for each edge (j, i).

        ############################################################################
        #  TODO: Your code here! Compute the attention coefficients alpha as described
        # in equation (7). Remember to be careful of the number of heads with 
        # dimension!
        # Our implementation is ~5 lines, but don't worry if you deviate from this.

        x_j = x_j.view(-1, self.heads, self.out_channels)
        x_i = x_i.view(-1, self.heads, self.out_channels)
        a = torch.cat((x_j, x_i), dim=-1) * self.att
        a_relu = F.leaky_relu(a.sum(dim=-1), LEAKY_RELU_SLOPE, size_i)
        alpha = pyg_utils.softmax(a_relu, edge_index_i)

        ############################################################################

        alpha = F.dropout(alpha, p=self.dropout, training=self.training)

        return x_j * alpha.view(-1, self.heads, 1)

    def update(self, aggr_out):
        # Updates node embedings.
        if self.concat is True:
            aggr_out = aggr_out.view(-1, self.heads * self.out_channels)
        else:
            aggr_out = aggr_out.mean(dim=1)

        if self.bias is not None:
            aggr_out = aggr_out + self.bias
        return aggr_out
