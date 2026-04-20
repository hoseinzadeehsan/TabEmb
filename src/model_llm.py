from transformers import AutoModelForCausalLM, BitsAndBytesConfig
import torch
from torch import nn
import torch.nn.functional as F
import dgl
from dgl import DGLGraph
from dgl.nn.pytorch import GATConv, GatedGraphConv, GraphConv


def get_model_name(args):
    if args.model_name == 'mistral':
        model_name = "mistralai/Mistral-7B-v0.1"
    elif args.model_name == 'mistral-instruct':
        model_name = "mistralai/Mistral-7B-Instruct-v0.1"
    elif args.model_name == 'mixtral':
        model_name = 'mistralai/Mixtral-8x7B-v0.1'
    elif args.model_name == 'mixtral-instruct':
        model_name = 'mistralai/Mixtral-8x7B-Instruct-v0.1'
    elif args.model_name == 'llama2':
        model_name = "meta-llama/Llama-2-7b-hf"
    elif args.model_name == 'llama3':
        model_name = "meta-llama/Meta-Llama-3-8B"
    elif args.model_name == 'llama3-instruct':
        model_name = "meta-llama/Meta-Llama-3-8B-Instruct"
    elif args.model_name == 'llama3.1':
        model_name = "meta-llama/Meta-Llama-3.1-8B"
    elif args.model_name == 'llama3.1-instruct':
        model_name = "meta-llama/Meta-Llama-3.1-8B-Instruct"
    elif args.model_name == 'phi3':
        # model_name = "microsoft/Phi-3-small-128k-instruct"
        model_name = "microsoft/Phi-3-mini-4k-instruct"
    elif args.model_name == 'Qwen7':
        model_name = "Qwen/Qwen2.5-7B"
    elif args.model_name == 'Qwen1.5':
        model_name = "Qwen/Qwen2.5-1.5B"
    elif args.model_name == 'Qwen14':
        model_name = "Qwen/Qwen2.5-14B"
    else:
        print('Error in name of model')

    return model_name


class LLM(nn.Module):
    def __init__(self, args, access_token, device, use_mean_pooling=False):
        super(LLM, self).__init__()
        self.use_mean_pooling = use_mean_pooling
        model_name = get_model_name(args)

        if args.quantized:
            config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=torch.float32, # torch.bfloat16,
            )
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                token=access_token,
                device_map=device,
                quantization_config=config,
                trust_remote_code=True,
                torch_dtype=torch.float32, # torch.bfloat16,
                low_cpu_mem_usage=True
            )
        else:
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                token=access_token,
                device_map=device,
                trust_remote_code=True
            )

        # Remove the language modeling head to get raw hidden states
        self.model.lm_head = nn.Identity()

        # Freeze all parameters
        for _, param in self.model.named_parameters():
            param.requires_grad = False

    def forward(self, input_ids, attention_mask):
        hidden = self.model(input_ids=input_ids, attention_mask=attention_mask).logits  # shape: (B, T, D)

        if self.use_mean_pooling:
            # Average over unmasked tokens
            mask = attention_mask.unsqueeze(-1)  # (B, T, 1)
            masked_hidden = hidden * mask
            pooled = masked_hidden.sum(dim=1) / mask.sum(dim=1).clamp(min=1e-6)
            return pooled
        else:
            # Return last unmasked token embedding
            last_token_indices = attention_mask.sum(dim=1) - 1  # (B,)
            batch_indices = torch.arange(input_ids.size(0), device=input_ids.device)
            return hidden[batch_indices, last_token_indices]  # shape: (B, D)


class EdgeClassifier(nn.Module):
    def __init__(self, in_features, out_classes):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_features * 2, out_classes)) 

    def apply_edges(self, edges):
        h_u = edges.src['h']
        h_v = edges.dst['h']
        
        concat_ = torch.cat([h_u, h_v], 1)
        score = self.mlp(concat_)
        return {'score': score}

    def forward(self, graph, h):
        # h contains the node representations computed from the GNN defined
        # in the node classification section (Section 5.1).
        with graph.local_scope():
            graph.ndata['h'] = h
            graph.apply_edges(self.apply_edges)
            return graph.edata['score']
        

class GraphClassifier(nn.Module):
    def __init__(self, node_feat_dim, out_classes):
        super().__init__()
        self.pool = dgl.nn.GlobalAttentionPooling(
            gate_nn=nn.Linear(node_feat_dim * 1, 1)
        )
        self.classifier = nn.Sequential(
            nn.Linear(node_feat_dim * 1, out_classes)  
        )

    def forward(self, g, h):
        hg = self.pool(g, h)
        return self.classifier(hg)  # graph-level logits

class NodeClassifier(nn.Module):
    def __init__(self, final_node_dim, out_classes):
        super().__init__()
        self.classifier = nn.Linear(final_node_dim, out_classes)

    def forward(self, h):
        return self.classifier(h)


class GAT_Multitask(nn.Module):
    def __init__(self,
                 num_layers,
                 in_dim,
                 num_hidden,
                 num_classes,
                 num_classes_graph,
                 heads,
                 activation,
                 feat_drop,
                 attn_drop,
                 negative_slope,
                 residual):
        super(GAT_Multitask, self).__init__()
        self.num_layers = num_layers
        self.gat_layers = nn.ModuleList()
        self.activation = activation
        self.norms = nn.ModuleList([nn.LayerNorm(num_hidden[l] * heads[l]) for l in range(num_layers)])
        
        final_node_dim = num_hidden[-1] * heads[-2]

        self.node_mlp = NodeClassifier(final_node_dim, num_classes)
        self.edge_mlp = EdgeClassifier(final_node_dim, num_classes)
        #self.pred = MLPPredictor(final_node_dim, 4096, num_classes)
        self.graph_mlp = GraphClassifier(final_node_dim, num_classes_graph)

        def _apply_edge_linear(g, h, linear_layer):
            with g.local_scope():
                g.ndata['h'] = h
                g.apply_edges(lambda edges: {'e': linear_layer(torch.cat([edges.src['h'], edges.dst['h']], dim=1))})
                return g.edata['e']

        def _apply_graph_linear(g, h, linear_layer):
            with g.local_scope():
                g.ndata['h'] = h
                hg = dgl.readout_nodes(g, 'h', op='mean')  # or 'sum' depending on your use case
                return linear_layer(hg)

        
        mlp_in = in_dim
        self.node_mlp_only = nn.Linear(mlp_in, num_classes)
        self.edge_linear = nn.Linear(mlp_in * 2, num_classes)          # src + dst
        self.edge_mlp_only = lambda g, h: _apply_edge_linear(g, h, self.edge_linear)
        self.graph_linear = nn.Linear(mlp_in, num_classes_graph)       # mean pooled nodes
        self.graph_mlp_only = lambda g, h: _apply_graph_linear(g, h, self.graph_linear)

        # There is no hidden layer
        if num_layers == 0:
            self.gat_layers.append(GATConv(
                in_dim, num_classes, heads[0],
                feat_drop, attn_drop, negative_slope, residual, None))

        else:
            # input projection
            self.gat_layers.append(GATConv(
                in_dim, num_hidden[0], heads[0],
                feat_drop, attn_drop, negative_slope, residual, self.activation))
            # hidden layers
            for l in range(1, num_layers):
                # due to multi-head, the in_dim = num_hidden * num_heads
                self.gat_layers.append(GATConv(
                    num_hidden[l - 1] * heads[l - 1], num_hidden[l], heads[l],
                    feat_drop, attn_drop, negative_slope, residual, self.activation))
            # output projection
            self.gat_layers.append(GATConv(
                num_hidden[-1] * heads[-2], num_classes, heads[-1],
                feat_drop, attn_drop, negative_slope, residual, None))

    def encode(self, g: DGLGraph, inputs):
        """
        Run the GAT stack and return node-level embeddings h.
        h shape: [num_nodes_in_batch, final_node_dim]
        """
        h = inputs
        for l in range(self.num_layers):
            h = self.gat_layers[l](g, h).flatten(1)
            h = self.norms[l](h)
        return h
    
    def forward(self, g: DGLGraph, inputs, return_embeddings: bool = False):
        # Get final node embeddings from the GNN
        h = self.encode(g, inputs)  # [num_nodes_in_batch, final_node_dim]

        # Node / edge / graph logits from those embeddings
        node_logits = self.node_mlp(h)
        edge_logits = self.edge_mlp(g, h)
        graph_logits = self.graph_mlp(g, h)

        out = {
            "node_logits": node_logits,
            "edge_logits": edge_logits,
            "graph_logits": graph_logits,
        }

        # Optional: expose embeddings for saving / analysis
        if return_embeddings:
            out["node_embeddings"] = h  # [num_nodes_in_batch, D]

        return out
    

class GGNN(nn.Module):
    def __init__(self,
                 in_feats,
                 n_hidden, # This should be a list or tuple of hidden layer sizes
                 n_classes,
                 n_layers,
                 num_classes_graph, 
                 shared_weight=False,
                 dropout=0):
        super(GGNN, self).__init__()
        self.layers = nn.ModuleList()
        self.n_layers = n_layers # Store number of layers
        self.dropout_rate = dropout # Store dropout rate
        self.shared_weight = shared_weight

        if self.shared_weight:            
            self.layers.append(GatedGraphConv(in_feats, n_classes, n_layers, 1))
        else:
            self.layers.append(GatedGraphConv(in_feats, in_feats, 1, 1))

            # Hidden layers: Map between hidden layer sizes.
            for i in range(n_layers - 1):
                self.layers.append(GatedGraphConv(in_feats, in_feats, 1, 1))

            self.classify = nn.Linear(in_feats, n_classes)

        # Define dropout layer
        self.dropout = nn.Dropout(p=dropout)

        final_node_dim = n_hidden[-1] * 4 
        
        def _apply_edge_linear(g, h, linear_layer):
            with g.local_scope():
                g.ndata['h'] = h
                g.apply_edges(lambda edges: {'e': linear_layer(torch.cat([edges.src['h'], edges.dst['h']], dim=1))})
                return g.edata['e']

        def _apply_graph_linear(g, h, linear_layer):
            with g.local_scope():
                g.ndata['h'] = h
                hg = dgl.readout_nodes(g, 'h', op='mean')  # or 'sum' depending on your use case
                return linear_layer(hg)

        self.node_mlp_only = nn.Linear(final_node_dim , n_classes)
        self.edge_linear = nn.Linear(final_node_dim * 2, n_classes)
        self.edge_mlp_only = lambda g, h: _apply_edge_linear(g, h, self.edge_linear)
        self.graph_linear = nn.Linear(final_node_dim, num_classes_graph)
        self.graph_mlp_only = lambda g, h: _apply_graph_linear(g, h, self.graph_linear)


    def forward(self, g: DGLGraph, features): # Removed 'edges' as it's not used in the layer call
        h = features

        if self.shared_weight:
            h = self.layers[0](g, h)
            node_logits = h.mean() 
        
            node_logits = self.node_mlp_only(h)
            edge_logits = self.edge_mlp_only(g, h)
            graph_logits = self.graph_mlp_only(g, h)
            return {"node_logits": node_logits, "edge_logits": edge_logits, "graph_logits": graph_logits}

        else:
            # Apply input and hidden layers
            for i in range(self.n_layers): # Iterate through the GatedGraphConv layers
                h = self.layers[i](g, h)
                # Apply non-linearity and dropout after each hidden layer
                h = F.relu(h) # Common practice to apply ReLU after GCN/GGNN layers
                h = self.dropout(h)

            node_logits = self.node_mlp_only(h)
            edge_logits = self.edge_mlp_only(g, h)
            graph_logits = self.graph_mlp_only(g, h)
            return {"node_logits": node_logits, "edge_logits": edge_logits, "graph_logits": graph_logits}
            #return logits



class GCNLayer(nn.Module):

    def __init__(self, in_feats, out_feats,
                 residual=True, batchnorm=True, dropout=0, activation=None):
        super(GCNLayer, self).__init__()

        self.activation = activation
        self.graph_conv = GraphConv(in_feats=in_feats, out_feats=out_feats, activation=activation)
        self.dropout = nn.Dropout(dropout)

        self.residual = residual
        if residual:
            if in_feats == out_feats:
                self.res_connection = nn.Identity()
            else:
                self.res_connection = nn.Linear(in_feats, out_feats)

        self.bn = batchnorm
        if batchnorm:
            self.bn_layer = nn.BatchNorm1d(out_feats)

    def forward(self, g, feats):
        new_feats = self.graph_conv(g, feats)
        if self.bn:
            new_feats = self.bn_layer(new_feats)
        if self.residual:
            res_feats = self.res_connection(feats)
            new_feats = new_feats + res_feats
        new_feats = self.dropout(new_feats)

        
        return new_feats


class GCN(nn.Module):

    def __init__(self, in_feats, hidden_feats, n_classes, n_layers, num_classes_graph, activation=None, residual=True, batchnorm=False,
                 dropout=0):
        super(GCN, self).__init__()
        self.gnn_layers = nn.ModuleList()
        if n_layers == 0:
            self.gnn_layers.append(GCNLayer(in_feats, n_classes, residual, batchnorm, 0))
        else:
            for i in range(n_layers):
                self.gnn_layers.append(GCNLayer(in_feats, in_feats, residual, batchnorm, dropout, activation))

            self.gnn_layers.append(GCNLayer(in_feats, in_feats, residual, batchnorm, 0))

        final_node_dim = hidden_feats[-1] * 4 
        
        def _apply_edge_linear(g, h, linear_layer):
            with g.local_scope():
                g.ndata['h'] = h
                g.apply_edges(lambda edges: {'e': linear_layer(torch.cat([edges.src['h'], edges.dst['h']], dim=1))})
                return g.edata['e']

        def _apply_graph_linear(g, h, linear_layer):
            with g.local_scope():
                g.ndata['h'] = h
                hg = dgl.readout_nodes(g, 'h', op='mean')  
                return linear_layer(hg)

        self.node_mlp_only = nn.Linear(final_node_dim , n_classes)
        self.edge_linear = nn.Linear(final_node_dim * 2, n_classes)
        self.edge_mlp_only = lambda g, h: _apply_edge_linear(g, h, self.edge_linear)
        self.graph_linear = nn.Linear(final_node_dim, num_classes_graph)
        self.graph_mlp_only = lambda g, h: _apply_graph_linear(g, h, self.graph_linear)

    def forward(self, g, feats):
        h = feats
        for layer in self.gnn_layers:
            h = layer(g, h)
        
        node_logits = self.node_mlp_only(h)
        edge_logits = self.edge_mlp_only(g, h)
        graph_logits = self.graph_mlp_only(g, h)
        return {"node_logits": node_logits, "edge_logits": edge_logits, "graph_logits": graph_logits}