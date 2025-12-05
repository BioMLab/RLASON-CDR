import torch
from torch import nn
import torch.nn.functional as F
from torch_geometric import nn as pyg_nn
from torch_geometric.nn import JumpingKnowledge, GraphNorm
from torch_sparse import spmm
from torch.distributions import Categorical

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

allowable_atom_features1 = {
    "possible_atomic_num_list": list(range(1, 119)) + ["misc"],
    "possible_chirality_list": ["CHI_UNSPECIFIED", "CHI_TETRAHEDRAL_CW", "CHI_TETRAHEDRAL_CCW", "CHI_OTHER", "misc"],
    "possible_degree_list": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, "misc"],
    "possible_formal_charge_list": [-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, "misc"],
    "possible_numH_list": [0, 1, 2, 3, 4, 5, 6, 7, 8, "misc"],
    "possible_number_radical_e_list": [0, 1, 2, 3, 4, "misc"],
    "possible_hybridization_list": ["SP", "SP2", "SP3", "SP3D", "SP3D2", "misc"],
    "possible_is_aromatic_list": [False, True],
    "possible_is_in_ring_list": [False, True]
}
allowable_bond_features1 = {
    "possible_bond_type_list": ["SINGLE", "DOUBLE", "TRIPLE", "AROMATIC", "misc"],
    "possible_bond_stereo_list": ["STEREONONE", "STEREOZ", "STEREOE", "STEREOCIS", "STEREOTRANS", "STEREOANY"],
    "possible_is_conjugated_list": [False, True]
}


def get_atom_int_feature_dims1():
    return list(map(len, [
        allowable_atom_features1["possible_atomic_num_list"],
        allowable_atom_features1["possible_chirality_list"],
        allowable_atom_features1["possible_degree_list"],
        allowable_atom_features1["possible_formal_charge_list"],
        allowable_atom_features1["possible_numH_list"],
        allowable_atom_features1["possible_number_radical_e_list"],
        allowable_atom_features1["possible_hybridization_list"],
        allowable_atom_features1["possible_is_aromatic_list"],
        allowable_atom_features1["possible_is_in_ring_list"]
    ]))


def get_bond_feature_int_dims1():
    return list(map(len, [
        allowable_bond_features1["possible_bond_type_list"],
        allowable_bond_features1["possible_bond_stereo_list"],
        allowable_bond_features1["possible_is_conjugated_list"]
    ]))


class atom_embedding_net(nn.Module):
    def __init__(self, args):
        super(atom_embedding_net, self).__init__()
        self.embed_dim = args.HDAN_embed_dim
        self.atom_embedding = nn.ModuleList()
        self.num_atom_feature = len(get_atom_int_feature_dims1())
        for i in range(self.num_atom_feature):
            self.atom_embedding.append(nn.Embedding(get_atom_int_feature_dims1()[i], self.embed_dim))
            torch.nn.init.xavier_uniform_(self.atom_embedding[i].weight.data)

    def forward(self, x):
        out = 0
        for i in range(self.num_atom_feature):
            out += self.atom_embedding[i](x[:, i].to(dtype=torch.int64)).to(device)
        return out


class bond_embedding_net(nn.Module):
    def __init__(self, args):
        super(bond_embedding_net, self).__init__()
        self.embed_dim = args.HDAN_embed_dim
        self.bond_embedding = nn.ModuleList()
        self.num_bond_feature = len(get_bond_feature_int_dims1())
        for i in range(self.num_bond_feature):
            self.bond_embedding.append(nn.Embedding(get_bond_feature_int_dims1()[i] + 3, self.embed_dim))
            torch.nn.init.xavier_uniform_(self.bond_embedding[i].weight.data)

    def forward(self, x):
        out = 0
        for i in range(self.num_bond_feature):
            out += self.bond_embedding[i](x[:, i].to(dtype=torch.int64))
        return out


class Drug_3d_Encoder(nn.Module):
    def __init__(self, args):
        super(Drug_3d_Encoder, self).__init__()
        self.embed_dim = args.HDAN_embed_dim
        self.dropout_rate = args.dropout_ratio_drug_3D
        self.layer_num = args.layer_num_3D_drug
        self.readout = args.readout
        self.jk = args.JK
        self.atom_init_nn = atom_embedding_net(args)
        self.bond_init_nn = bond_embedding_net(args)
        self.atom_conv = nn.ModuleList()
        self.bond_conv = nn.ModuleList()
        self.bond_embed_nn = nn.ModuleList()
        self.bond_angle_embed_nn = nn.ModuleList()
        self.layer_norm_atom = nn.ModuleList()
        self.graph_norm_atom = nn.ModuleList()
        self.layer_norm_bond = nn.ModuleList()
        self.graph_norm_bond = nn.ModuleList()
        self.JK = JumpingKnowledge("cat")
        for i in range(self.layer_num):
            self.atom_conv.append(pyg_nn.GINEConv(
                nn=nn.Sequential(nn.Linear(self.embed_dim, self.embed_dim * 2), nn.ReLU(),
                                 nn.Linear(self.embed_dim * 2, self.embed_dim)), edge_dim=self.embed_dim))
            self.layer_norm_atom.append(nn.LayerNorm(self.embed_dim))
            self.graph_norm_atom.append(GraphNorm(self.embed_dim))
            self.bond_conv.append(pyg_nn.GINEConv(
                nn=nn.Sequential(nn.Linear(self.embed_dim, self.embed_dim * 2), nn.ReLU(),
                                 nn.Linear(self.embed_dim * 2, self.embed_dim)), edge_dim=self.embed_dim))
            self.bond_embed_nn.append(bond_embedding_net(args))
            self.bond_angle_embed_nn.append(
                nn.Sequential(nn.Linear(1, self.embed_dim), nn.ReLU(), nn.Linear(self.embed_dim, self.embed_dim)))
            self.layer_norm_bond.append(nn.LayerNorm(self.embed_dim))
            self.graph_norm_bond.append(GraphNorm(self.embed_dim))
        if self.readout == "max":
            self.read_out = pyg_nn.global_max_pool
        elif self.readout == "mean":
            self.read_out = pyg_nn.global_mean_pool
        elif self.readout == "add":
            self.read_out = pyg_nn.global_mean_pool

    def forward(self, drug_atom, drug_bond):
        x, edge_index, edge_attr, batch = drug_atom.x, drug_atom.edge_index, drug_atom.edge_attr, drug_atom.batch
        x = self.atom_init_nn(x.to(dtype=torch.int64)).to(device)
        edge_hidden = self.bond_init_nn(edge_attr.to(dtype=torch.int64)).to(device)
        hidden = [x]
        hidden_edge = [edge_hidden]
        for i in range(self.layer_num):
            x = self.atom_conv[i](x=x, edge_attr=hidden_edge[i], edge_index=edge_index)
            x = self.layer_norm_atom[i](x)
            x = self.graph_norm_atom[i](x)
            if i == self.layer_num - 1:
                x = nn.Dropout(self.dropout_rate)(nn.ReLU()(x)) + hidden[i]
            else:
                x = nn.Dropout(self.dropout_rate)(x) + hidden[i]
            cur_edge_attr = self.bond_embed_nn[i](edge_attr)
            cur_angle_attr = self.bond_angle_embed_nn[i](drug_bond.edge_attr)
            edge_hidden = self.bond_conv[i](x=cur_edge_attr, edge_attr=cur_angle_attr,
                                            edge_index=drug_bond.edge_index)
            edge_hidden = self.layer_norm_bond[i](edge_hidden)
            edge_hidden = self.graph_norm_bond[i](edge_hidden)
            if i == self.layer_num - 1:
                edge_hidden = nn.Dropout(self.dropout_rate)(nn.ReLU()(edge_hidden)) + hidden_edge[i]
            else:
                edge_hidden = nn.Dropout(self.dropout_rate)(edge_hidden) + hidden_edge[i]
            hidden.append(x)
            hidden_edge.append(edge_hidden)
        if self.jk == "True":
            x = self.JK(hidden)
        else:
            x = hidden[-1]
        graph_repr = self.read_out(x, batch)
        return graph_repr

    @property
    def output_dim(self):
        self.out_dim = self.embed_dim * (self.layer_num + 1)
        return self.out_dim


class drug_hier_encoder(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.embed_dim = args.HDAN_embed_dim
        self.drug_3d = Drug_3d_Encoder(args)
        self.drug_3d_dim = self.drug_3d.output_dim
        self.drug_3d_dense = nn.Linear(self.drug_3d_dim, 1024)
        self.drug_3d_dense_bn1 = nn.BatchNorm1d(1024)
        self.drug_3d_dense2 = nn.Linear(1024, self.embed_dim)
        self.drug_3d_dense_bn2 = nn.BatchNorm1d(self.embed_dim)
        self.drug_3d_dense_relu = nn.ReLU()

    def forward(self, drug_atom, drug_bond):
        drug_repr = self.drug_3d(drug_atom, drug_bond)
        drug_repr = self.drug_3d_dense(drug_repr)
        drug_repr = self.drug_3d_dense_bn1(drug_repr)
        drug_repr = self.drug_3d_dense2(drug_repr)
        drug_repr = self.drug_3d_dense_bn2(drug_repr)
        drug_repr = self.drug_3d_dense_relu(drug_repr)
        return drug_repr


class HDAN(nn.Module):
    def __init__(self, embed_dim, num_heads=64, dropout=0.1):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout = dropout

        self.intra_attn = nn.ModuleList([
            nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout)
            for _ in range(2)
        ])

        self.cross_attn_d2c = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout,
                                                    kdim=embed_dim, vdim=embed_dim)
        self.cross_attn_c2d = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout,
                                                    kdim=embed_dim, vdim=embed_dim)

        self.cross_gate = nn.Sequential(
            nn.Linear(2 * embed_dim, 4 * embed_dim),
            nn.GELU(),
            nn.Linear(4 * embed_dim, 2 * embed_dim),
            nn.Sigmoid()
        )

        self.fusion_proj = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.LayerNorm(embed_dim)
        )

        self.norm_final = nn.LayerNorm(embed_dim)

    def forward(self, x_drug, x_cell):
        x_d = x_drug.permute(1, 0, 2)
        x_d_attn_out, drug_intra_attn_w = self.intra_attn[0](x_d, x_d, x_d)
        x_d = x_d_attn_out.permute(1, 0, 2)

        x_c = x_cell.permute(1, 0, 2)
        x_c_attn_out, cell_intra_attn_w = self.intra_attn[1](x_c, x_c, x_c)
        x_c = x_c_attn_out.permute(1, 0, 2)

        drug_pool = x_d.mean(dim=1)
        cell_pool = x_c.mean(dim=1)

        q_d = x_d.permute(1, 0, 2)
        k_c = x_c.permute(1, 0, 2)
        v_c = k_c
        x_d2c_out, cross_attn_d2c_w = self.cross_attn_d2c(q_d, k_c, v_c)
        x_d2c = x_d2c_out.permute(1, 0, 2)

        q_c = x_c.permute(1, 0, 2)
        k_d = x_d.permute(1, 0, 2)
        v_d = k_d
        x_c2d_out, cross_attn_c2d_w = self.cross_attn_c2d(q_c, k_d, v_d)
        x_c2d = x_c2d_out.permute(1, 0, 2)

        drug2cell_pool = x_d2c.mean(dim=1)
        cell2drug_pool = x_c2d.mean(dim=1)

        gate = self.cross_gate(torch.cat([drug_pool, cell_pool], dim=-1))
        g1, g2 = gate.chunk(2, dim=-1)
        gated_d2c = drug2cell_pool * g1
        gated_c2d = cell2drug_pool * g2

        fused = gated_d2c + gated_c2d
        fused = self.fusion_proj(fused)

        combined = fused + drug_pool + cell_pool
        final_feat = self.norm_final(combined)

        return final_feat


class HDAN_model(nn.Module):
    def __init__(self, args):
        super(HDAN_model, self).__init__()
        self.embed_dim = args.HDAN_embed_dim

        self.ecfp_fc1 = nn.Linear(2048, 1024)
        self.ecfp_fc2 = nn.Linear(1024, self.embed_dim)
        self.batch_ecfp = nn.BatchNorm1d(1024)
        self.ecfp_dropout = nn.Dropout(0.2)

        self.espf_fc1 = nn.Linear(2586, 1024)
        self.espf_fc2 = nn.Linear(1024, self.embed_dim)
        self.batch_espf = nn.BatchNorm1d(1024)
        self.espf_dropout = nn.Dropout(0.2)

        self.pubchem_fc1 = nn.Linear(881, 256)
        self.pubchem_fc2 = nn.Linear(256, self.embed_dim)
        self.batch_pubchem = nn.BatchNorm1d(256)
        self.pubchem_dropout = nn.Dropout(0.2)

        self.drug_3D_encoder = drug_hier_encoder(args)

        self.exp_fc1 = nn.Linear(697, 256)
        self.exp_fc2 = nn.Linear(256, self.embed_dim)
        self.batch_exp = nn.BatchNorm1d(256)
        self.exp_dropout = nn.Dropout(0.2)

        self.meth_fc1 = nn.Linear(808, 256)
        self.meth_fc2 = nn.Linear(256, self.embed_dim)
        self.batch_meth = nn.BatchNorm1d(256)
        self.meth_dropout = nn.Dropout(0.2)

        self.mut_cov1 = nn.Conv2d(1, 50, (1, 700), stride=(1, 5))
        self.mut_cov2 = nn.Conv2d(50, 30, (1, 5), stride=(1, 2))
        self.mut_fla = nn.Flatten()
        self.mut_fc = nn.Linear(2010, self.embed_dim)
        self.batch_mut1 = nn.BatchNorm2d(50)
        self.batch_mut2 = nn.BatchNorm2d(30)
        self.batch_mut3 = nn.BatchNorm1d(2010)
        self.mut_dropout = nn.Dropout(0.2)

        self.path_fc1 = nn.Linear(1264, 512)
        self.path_fc2 = nn.Linear(512, self.embed_dim)
        self.batch_path = nn.BatchNorm1d(512)
        self.path_dropout = nn.Dropout(0.2)

        self.fusion = HDAN(
            embed_dim=self.embed_dim,
            num_heads=args.HDAN_heads,
            dropout=args.HDAN_dropout
        )

        self.predict = nn.Sequential(
            nn.Linear(self.embed_dim, 256),
            nn.GELU(),
            nn.Linear(256, 1)
        )

    def forward(self, drug_ecfp, drug_espf, drug_pubchem, drug_atom, drug_bond,
                cell_exp, cell_meth, cell_mut, cell_path):

        drug_ecfp = torch.tanh(self.ecfp_fc1(drug_ecfp))
        drug_ecfp = self.batch_ecfp(drug_ecfp)
        drug_ecfp = F.relu(self.ecfp_fc2(drug_ecfp))
        drug_ecfp = self.ecfp_dropout(drug_ecfp)

        drug_espf = torch.tanh(self.espf_fc1(drug_espf))
        drug_espf = self.batch_espf(drug_espf)
        drug_espf = F.relu(self.espf_fc2(drug_espf))
        drug_espf = self.espf_dropout(drug_espf)

        drug_pubchem = torch.tanh(self.pubchem_fc1(drug_pubchem))
        drug_pubchem = self.batch_pubchem(drug_pubchem)
        drug_pubchem = F.relu(self.pubchem_fc2(drug_pubchem))
        drug_pubchem = self.pubchem_dropout(drug_pubchem)

        drug_feats = [
            drug_ecfp,
            drug_espf,
            drug_pubchem,
            self.drug_3D_encoder(drug_atom, drug_bond)
        ]
        x_drug = torch.stack(drug_feats, dim=1)  # [B,4,D]

        # Gene expression representation
        cell_exp = torch.tanh(self.exp_fc1(cell_exp))
        cell_exp = self.batch_exp(cell_exp)
        cell_exp = F.relu(self.exp_fc2(cell_exp))
        cell_exp = self.exp_dropout(cell_exp)

        # Methylation representation
        cell_meth = torch.tanh(self.meth_fc1(cell_meth))
        cell_meth = self.batch_meth(cell_meth)
        cell_meth = F.relu(self.meth_fc2(cell_meth))
        cell_meth = self.meth_dropout(cell_meth)

        # Mutation representation
        cell_mut = torch.tanh(self.mut_cov1(cell_mut))
        cell_mut = self.batch_mut1(cell_mut)
        cell_mut = F.max_pool2d(cell_mut, (1, 5))
        cell_mut = F.relu(self.mut_cov2(cell_mut))
        cell_mut = self.batch_mut2(cell_mut)
        cell_mut = F.max_pool2d(cell_mut, (1, 10))
        cell_mut = self.mut_fla(cell_mut)
        cell_mut = self.batch_mut3(cell_mut)
        cell_mut = F.relu(self.mut_fc(cell_mut))
        cell_mut = self.mut_dropout(cell_mut)

        # Pathway representation
        cell_path = torch.tanh(self.path_fc1(cell_path))
        cell_path = self.batch_path(cell_path)
        cell_path = F.relu(self.path_fc2(cell_path))
        cell_path = self.path_dropout(cell_path)

        cell_feats = [
            cell_exp,
            cell_meth,
            cell_mut,
            cell_path
        ]
        x_cell = torch.stack(cell_feats, dim=1)  # [B,4,D]

        fusion_feat = self.fusion(x_drug, x_cell)
        output = self.predict(fusion_feat)
        return output


class MultiHopGCNLayer(nn.Module):
    def __init__(self, in_channels, out_channels, hops, dropout, device):
        super().__init__()
        self.hops = hops
        self.device = device
        self.linears = nn.ModuleList([
            nn.Linear(in_channels, out_channels) for _ in range(hops)
        ])
        self.dropout = dropout

    def forward(self, normalized_adjacency_matrix, features):
        n_nodes, n_feats = features["dimensions"]
        x = spmm(features["indices"], features["values"], n_nodes, n_feats, torch.eye(n_feats, device=self.device))

        out_list = []
        h = x
        for i in range(self.hops):
            h = spmm(normalized_adjacency_matrix["indices"], normalized_adjacency_matrix["values"], h.size(0), h.size(0), h)
            h_proj = F.relu(self.linears[i](h))
            out_list.append(h_proj)

        return out_list


class NeighborInteractionAugmentor(nn.Module):
    def __init__(self, embed_dim, hops, device):
        super().__init__()
        self.hops = hops
        self.attn_weights = nn.Parameter(torch.Tensor(hops, embed_dim))
        self.device = device

    def forward(self, hop_features):
        stacked = torch.stack(hop_features, dim=1)  # [N, H, D]

        scores = (stacked * self.attn_weights.unsqueeze(0)).sum(dim=-1)  # [N, H]
        alpha = F.softmax(scores, dim=1).unsqueeze(-1)  # [N, H, 1]

        enhanced_features = (stacked * alpha).sum(dim=1)  # [N, D]
        return enhanced_features, alpha


class HGCN(nn.Module):
    def __init__(self, args, feature_number, device=device):
        super().__init__()
        self.feature_number = feature_number
        self.hops = args.HGCN_hops
        self.embed_dim = args.HGCN_embed_dim
        self.hidden_dim = args.HGCN_hidden_dim
        self.dropout = args.HGCN_dropout
        self.device = device

        self.multi_hop_gcn = MultiHopGCNLayer(
            in_channels=self.feature_number,
            out_channels=self.embed_dim,
            hops=self.hops,
            dropout=self.dropout,
            device=self.device
        )

        self.nia = NeighborInteractionAugmentor(self.embed_dim, self.hops, self.device)

        self.decoder = nn.Sequential(
            nn.Linear(self.embed_dim * 2, self.hidden_dim),
            nn.ELU(),
            nn.Linear(self.hidden_dim, 1)
        )

    def embed(self, normalized_adjacency_matrix, features):

        hop_features = self.multi_hop_gcn(normalized_adjacency_matrix, features)

        enhanced_features, attn_weights = self.nia(hop_features)

        enhanced_features = F.dropout(enhanced_features, p=self.dropout, training=self.training)

        return enhanced_features

    def forward(self, normalized_adjacency_matrix, features, idx):
        latent_features = self.embed(normalized_adjacency_matrix, features)

        feat_p1 = latent_features[idx[0]]
        feat_p2 = latent_features[idx[1]]

        fused = torch.cat([torch.abs(feat_p1 - feat_p2), feat_p1 * feat_p2], dim=-1)

        fused = F.elu(fused)
        fused = F.dropout(fused, p=self.dropout, training=self.training)
        logits = self.decoder(fused)

        return logits, fused


class PolicyNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.attention = nn.MultiheadAttention(embed_dim=2, num_heads=1)
        self.fc = nn.Sequential(
            nn.Linear(2, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(128, 32),
            nn.LayerNorm(32),
            nn.GELU(),
            nn.Linear(32, 2)
        )

    def forward(self, state):
        state = state.unsqueeze(0)  # [1, B, 2]
        attn_out, _ = self.attention(state, state, state)
        logits = self.fc(attn_out.squeeze(0))  # [B, 2]

        dist = Categorical(logits=logits)
        log_prob = dist.logits.log_softmax(dim=-1)  # [B, 2]

        return dist, log_prob


class ValueNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(2, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(128, 32),
            nn.LayerNorm(32),
            nn.GELU(),
            nn.Linear(32, 1)
        )

    def forward(self, state):
        return self.fc(state)


class RLASON_CDR(nn.Module):
    def __init__(self, HDAN_model, HGCN_model, propagation_matrix, features, policy_net, value_net) -> None:
        super().__init__()
        self.HDAN_model = HDAN_model
        self.HGCN_model = HGCN_model
        self.propagation_matrix = propagation_matrix
        self.features = features
        self.policy_net = policy_net
        self.value_net = value_net
        self.act = nn.Sigmoid()

    def forward(self, drug_ecfp, drug_espf, drug_pubchem, drug_atom, drug_bond,
                cell_exp, cell_meth, cell_mut, cell_path, idx1, idx2):
        pred1, fusion_feat, drug_intra_attn_w, cell_intra_attn_w, cross_attn_d2c_w, cross_attn_c2d_w = \
            self.HDAN_model(drug_ecfp, drug_espf, drug_pubchem, drug_atom, drug_bond, cell_exp, cell_meth, cell_mut, cell_path)
        pred2, HGCN_fused = self.HGCN_model(self.propagation_matrix, self.features, (idx1, idx2))

        state = torch.cat([pred1, pred2], dim=1)

        dist, log_prob = self.policy_net(state)
        weights = dist.probs

        w1 = weights[:, 0:1]
        w2 = weights[:, 1:2]

        output = w1 * pred1 + w2 * pred2
        output = self.act(output)

        return output, dist, state, log_prob, drug_intra_attn_w, cell_intra_attn_w, cross_attn_d2c_w, cross_attn_c2d_w

