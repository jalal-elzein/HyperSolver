import torch
import torch.nn as nn
import math

class HyperGraphMessageNet(nn.Module):
    """
    A simple hypergraph MPNN for set_cover, subset_sum, hypermaxcut, or hypermultiwaycut.
    """
    def __init__(self, num_elements, num_subsets,
                 hidden_dim=128, num_layers=4, dropout_rate=0.1,
                 num_partitions=2,
                 problem_type="set_cover"):
        super().__init__()
        self.num_elements  = num_elements
        self.num_subsets   = num_subsets
        self.hidden_dim    = hidden_dim
        self.num_layers    = num_layers
        self.dropout_rate  = dropout_rate
        self.num_partitions= num_partitions
        self.problem_type  = problem_type

        self.node_embedding = nn.Parameter(torch.empty(num_elements, hidden_dim))
        self.edge_embedding = nn.Parameter(torch.empty(num_subsets, hidden_dim))
        nn.init.uniform_(self.node_embedding, a=-0.01, b=0.01)
        nn.init.uniform_(self.edge_embedding, a=-0.01, b=0.01)

        self.node_updater = nn.ModuleList([
            nn.Sequential(
                nn.Linear(2 * hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout_rate),
                nn.LayerNorm(hidden_dim)
            ) for _ in range(num_layers)
        ])
        self.edge_updater = nn.ModuleList([
            nn.Sequential(
                nn.Linear(2 * hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout_rate),
                nn.LayerNorm(hidden_dim)
            ) for _ in range(num_layers)
        ])

        if self.problem_type == "hypermultiwaycut":
            out_dim = self.num_partitions
        else:
            out_dim = 1
        self.edge_decoder = nn.Linear(hidden_dim, out_dim)
        nn.init.xavier_uniform_(self.edge_decoder.weight)
        nn.init.zeros_(self.edge_decoder.bias)

        self.current_epoch = 0
        self.total_epochs  = 100

    def set_epoch(self, epoch, total_epochs):
        self.current_epoch = epoch
        self.total_epochs  = total_epochs

    def forward(self, incidence_matrix):
        node_emb = self.node_embedding
        edge_emb = self.edge_embedding

        if incidence_matrix.is_sparse:
            edge_deg = torch.sparse.sum(incidence_matrix, dim=0)
            node_deg = torch.sparse.sum(incidence_matrix, dim=1)
            if edge_deg.is_sparse:
                edge_deg = edge_deg.to_dense()
            if node_deg.is_sparse:
                node_deg = node_deg.to_dense()
            edge_deg = edge_deg.clamp(min=1e-6)
            node_deg = node_deg.clamp(min=1e-6)
        else:
            edge_deg = incidence_matrix.sum(dim=0).clamp(min=1e-6)
            node_deg = incidence_matrix.sum(dim=1).clamp(min=1e-6)

        for layer_idx in range(self.num_layers):
            if incidence_matrix.is_sparse:
                edge_msg = torch.sparse.mm(incidence_matrix.transpose(0,1), node_emb)
            else:
                edge_msg = incidence_matrix.t().matmul(node_emb)
            edge_msg = edge_msg / edge_deg.unsqueeze(-1)

            combined_edge = torch.cat([edge_emb, edge_msg], dim=-1)
            delta_edge = self.edge_updater[layer_idx](combined_edge)
            edge_emb = edge_emb + delta_edge

            if incidence_matrix.is_sparse:
                node_msg = torch.sparse.mm(incidence_matrix, edge_emb)
            else:
                node_msg = incidence_matrix.matmul(edge_emb)
            node_msg = node_msg / node_deg.unsqueeze(-1)

            combined_node = torch.cat([node_emb, node_msg], dim=-1)
            delta_node = self.node_updater[layer_idx](combined_node)
            node_emb = node_emb + delta_node

        if self.problem_type in ("subset_sum", "hypermaxcut", "hypermultiwaycut"):
            logits = self.edge_decoder(node_emb)
        else:
            logits = self.edge_decoder(edge_emb)
            
        if self.problem_type == "hypermultiwaycut":
            return nn.functional.softmax(logits, dim=-1)
        else:
            progress = float(self.current_epoch) / (float(self.total_epochs)+1e-9)
            beta = 0.7 + 0.3 * progress
            probs = torch.sigmoid(beta * logits).squeeze(-1)
            return probs


class ImprovedHyperGraphNet(nn.Module):
    """
    Wrapper that swaps dimensions for hitting_set
    (node_embedding => subsets, edge_embedding => elements).
    """
    def __init__(self, num_elements, num_subsets,
                 hidden_dim=128, num_layers=4, dropout_rate=0.1,
                 num_partitions=2,
                 problem_type="set_cover"):
        super().__init__()
        self.problem_type = problem_type

        if problem_type == "hitting_set":
            self.hypergraph_net = HyperGraphMessageNet(
                num_elements=num_subsets,
                num_subsets=num_elements,
                hidden_dim=hidden_dim,
                num_layers=num_layers,
                dropout_rate=dropout_rate,
                num_partitions=num_partitions,
                problem_type=problem_type
            )
        else:
            self.hypergraph_net = HyperGraphMessageNet(
                num_elements=num_elements,
                num_subsets=num_subsets,
                hidden_dim=hidden_dim,
                num_layers=num_layers,
                dropout_rate=dropout_rate,
                num_partitions=num_partitions,
                problem_type=problem_type
            )

    def forward(self, incidence_matrix):
        return self.hypergraph_net(incidence_matrix)

    def set_epoch(self, epoch, total_epochs):
        if hasattr(self.hypergraph_net, "set_epoch"):
            self.hypergraph_net.set_epoch(epoch, total_epochs)

    def resize_embeddings(self, new_num_elements, new_num_subsets):
        """
        Resizes embeddings if you need to handle a new shape.
        For hitting_set we swapped these internally.
        """
        hg = self.hypergraph_net
        dev = hg.node_embedding.device

        if self.problem_type == "hitting_set":
            new_nodes = new_num_subsets
            new_edges = new_num_elements
        else:
            new_nodes = new_num_elements
            new_edges = new_num_subsets

        old_node_count = hg.node_embedding.size(0)
        old_edge_count = hg.edge_embedding.size(0)

        with torch.no_grad():
            new_node_emb = torch.zeros(new_nodes, hg.hidden_dim, device=dev)
            copy_n = min(old_node_count, new_nodes)
            new_node_emb[:copy_n, :] = hg.node_embedding[:copy_n, :]
            if copy_n < new_nodes:
                new_node_emb[copy_n:, :].uniform_(-0.01, 0.01)

            hg.node_embedding = nn.Parameter(new_node_emb)

            new_edge_emb = torch.zeros(new_edges, hg.hidden_dim, device=dev)
            copy_m = min(old_edge_count, new_edges)
            new_edge_emb[:copy_m, :] = hg.edge_embedding[:copy_m, :]
            if copy_m < new_edges:
                new_edge_emb[copy_m:, :].uniform_(-0.01, 0.01)

            hg.edge_embedding = nn.Parameter(new_edge_emb)

        hg.num_elements = new_nodes
        hg.num_subsets = new_edges
