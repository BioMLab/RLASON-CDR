import torch
import pandas as pd
import numpy as np
import pickle
import codecs

from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

from PyBioMed.PyMolecule.PubChemFingerprints import calcPubChemFingerAll
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator
from subword_nmt.apply_bpe import BPE

from rdkit.Chem import AllChem
from torch_geometric.data import Data


def create_cell_feature(save_path="GDSC/cell/"):
    exp_df = pd.read_csv("GDSC/cell/genomic_expression_561celllines_697genes_demap_features.csv")
    meth_df = pd.read_csv("GDSC/cell/genomic_methylation_561celllines_808genes_demap_features.csv")
    mut_df = pd.read_csv("GDSC/cell/genomic_mutation_561celllines_34673genes_demap_features.csv")
    pathway_df = pd.read_csv("../../GDSC/cell/GDSC_pathway.csv")
    
    exp_df.set_index(exp_df.columns[0], inplace=True)
    meth_df.set_index(meth_df.columns[0], inplace=True)
    mut_df.set_index(mut_df.columns[0], inplace=True)
    pathway_df.set_index(pathway_df.columns[0], inplace=True)

    scaler = StandardScaler()
    imp_mean = SimpleImputer()
    
    exp_scaled = scaler.fit_transform(exp_df)
    meth_scaled = scaler.fit_transform(meth_df)
    mut_scaled = scaler.fit_transform(mut_df)
    pathway_scaled = scaler.fit_transform(pathway_df)
    
    exp_imputed = imp_mean.fit_transform(exp_scaled)
    meth_imputed = imp_mean.fit_transform(meth_scaled)
    mut_imputed = imp_mean.fit_transform(mut_scaled)
    pathway_imputed = imp_mean.fit_transform(pathway_scaled)
    
    exp_df = pd.DataFrame(exp_imputed, index=exp_df.index, columns=exp_df.columns)
    meth_df = pd.DataFrame(meth_imputed, index=meth_df.index, columns=meth_df.columns)
    mut_df = pd.DataFrame(mut_imputed, index=mut_df.index, columns=mut_df.columns)
    pathway_df = pd.DataFrame(pathway_imputed, index=pathway_df.index, columns=pathway_df.columns)
    
    exp_dict = exp_df.to_dict(orient="index")
    meth_dict = meth_df.to_dict(orient="index")
    mut_dict = mut_df.to_dict(orient="index")
    pathway_dict = pathway_df.to_dict(orient="index")
    
    exp_dict = {k: np.array(list(v.values())) for k, v in exp_dict.items()}
    meth_dict = {k: np.array(list(v.values())) for k, v in meth_dict.items()}
    mut_dict = {k: np.array(list(v.values())) for k, v in mut_dict.items()}
    pathway_dict = {k: np.array(list(v.values())) for k, v in pathway_dict.items()}
    
    with open(save_path + "cell_exp.pkl", "wb") as f:
        pickle.dump(exp_dict, f)
    with open(save_path + "cell_meth.pkl", "wb") as f:
        pickle.dump(meth_dict, f)
    with open(save_path + "cell_mut.pkl", "wb") as f:
        pickle.dump(mut_dict, f)
    with open(save_path + "cell_path.pkl", "wb") as f:
        pickle.dump(pathway_dict, f)


def create_drug_fingerprint(save_path="GDSC/drug"):
    drug_df = pd.read_csv("GDSC/drug/222drugs_pubchem_smiles.csv", header=0)
    pubchem = drug_df["pubchem"].to_list()
    smiles = drug_df["isosmiles"].to_list()
    drug_smiles = []
    for i in range(len(drug_df)):
        drug_smile = [pubchem[i], smiles[i]]
        drug_smiles.append(drug_smile)

    vocab_path = "drug_codes_chembl_freq_1500.txt"
    sub_csv = pd.read_csv("subword_units_map_chembl_freq_1500.csv")
    bpe_codes_drug = codecs.open(vocab_path)
    dbpe = BPE(bpe_codes_drug, merges=-1, separator="")

    idx2word_d = sub_csv["index"].values
    words2idx_d = dict(zip(idx2word_d, range(0, len(idx2word_d))))

    def smiles2espf(x):
        t1 = dbpe.process_line(x[1]).split()  # split
        try:
            i1 = np.asarray([words2idx_d[i] for i in t1])  # index
            index = 0
        except:
            i1 = np.array([0])
            index = 1
            print("false")
        v1 = np.zeros(len(idx2word_d))
        v1[i1] = 1
        return v1, index

    def smiles2ecfp(s):
        try:
            mol = Chem.MolFromSmiles(s[1])
            fpgen = rdFingerprintGenerator.GetecfpGenerator(radius=2)
            features = fpgen.GetFingerprintAsNumPy(mol)
            index = 0
        except:
            print("rdkit not found this smiles for ecfp: " + s[1] + " convert to all 0 features")
            features = np.zeros((2048,))
            index = 1
        return features, index

    def smiles2pubchem(s):
        try:
            mol = Chem.MolFromSmiles(s[1])
            features = calcPubChemFingerAll(mol)
            index = 0
        except:
            print("pubchem fingerprint not working for smiles: " + s[1] + " convert to 0 vectors")
            print(s)
            features = np.zeros((881,))
            index = 1
        return np.array(features), index

    drug_ecfp = {}
    drug_pubchem = {}
    drug_espf = {}

    false1, false2, false3 = [], [], []

    for i in range(len(drug_smiles)):
        drug_id = drug_smiles[i][0]
        drug_ecfp[drug_id], index1 = smiles2ecfp(drug_smiles[i])
        drug_pubchem[drug_id], index2 = smiles2pubchem(drug_smiles[i])
        drug_espf[drug_id], index3 = smiles2espf(drug_smiles[i])

        if index1 == 1:
            false1.append(drug_smiles[i][0])
            del drug_ecfp[drug_id]
        if index2 == 1:
            false2.append(drug_smiles[i][0])
            del drug_pubchem[drug_id]
        if index3 == 1:
            false3.append(drug_smiles[i][0])
            del drug_espf[drug_id]

    false_total = []
    for i in range(3):
        false_total.extend(false1)
        false_total.extend(false2)
        false_total.extend(false3)

    total_false = list(set(false_total))

    with open(save_path + "false_encoding_drug.pkl", "wb") as f:
        pickle.dump(total_false, f)

    with open(save_path + "ecfp_encoding.pkl", "wb") as f:
        pickle.dump(drug_ecfp, f)

    with open(save_path + "pubchem_encoding.pkl", "wb") as f:
        pickle.dump(drug_pubchem, f)

    with open(save_path + "espf_encoding.pkl", "wb") as f:
        pickle.dump(drug_espf, f)


allowable_atom_features = {
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

allowable_bond_features = {
    "possible_bond_type_list": ["SINGLE", "DOUBLE", "TRIPLE", "AROMATIC", "misc"],
    "possible_bond_stereo_list": ["STEREONONE", "STEREOZ", "STEREOE", "STEREOCIS", "STEREOTRANS", "STEREOANY"],
    "possible_is_conjugated_list": [False, True]
}


def safe_index(l, e):
    try:
        return l.index(e)
    except:
        return len(l) - 1


def atom_to_feature_vector_3D(atom):
    atom_feature = [
        safe_index(allowable_atom_features["possible_atomic_num_list"], atom.GetAtomicNum()),
        safe_index(allowable_atom_features["possible_chirality_list"], str(atom.GetChiralTag())),
        safe_index(allowable_atom_features["possible_degree_list"], atom.GetTotalDegree()),
        safe_index(allowable_atom_features["possible_formal_charge_list"], atom.GetFormalCharge()),
        safe_index(allowable_atom_features["possible_numH_list"], atom.GetTotalNumHs()),
        safe_index(allowable_atom_features["possible_number_radical_e_list"], atom.GetNumRadicalElectrons()),
        safe_index(allowable_atom_features["possible_hybridization_list"], str(atom.GetHybridization())),
        allowable_atom_features["possible_is_aromatic_list"].index(atom.GetIsAromatic()),
        allowable_atom_features["possible_is_in_ring_list"].index(atom.IsInRing()),
        atom.GetMass()
    ]
    return atom_feature


def bond_to_feature_vector_3D(bond):
    bond_feature = [
        safe_index(allowable_bond_features["possible_bond_type_list"], str(bond.GetBondType())),
        allowable_bond_features["possible_bond_stereo_list"].index(str(bond.GetStereo())),
        allowable_bond_features["possible_is_conjugated_list"].index(bond.GetIsConjugated())
    ]
    return bond_feature


def get_atom_int_feature_dims():
    return list(map(len, [
        allowable_atom_features["possible_atomic_num_list"],
        allowable_atom_features["possible_chirality_list"],
        allowable_atom_features["possible_degree_list"],
        allowable_atom_features["possible_formal_charge_list"],
        allowable_atom_features["possible_numH_list"],
        allowable_atom_features["possible_number_radical_e_list"],
        allowable_atom_features["possible_hybridization_list"],
        allowable_atom_features["possible_is_aromatic_list"],
        allowable_atom_features["possible_is_in_ring_list"]
    ]))


def get_bond_feature_int_dims():
    return list(map(len, [
        allowable_bond_features["possible_bond_type_list"],
        allowable_bond_features["possible_bond_stereo_list"],
        allowable_bond_features["possible_is_conjugated_list"]
    ]))


def self_loop_bond_feature():
    bond_feat = [len(allowable_bond_features[key]) + 2 for key in allowable_bond_features]
    bond_feat += [0.0]
    return bond_feat


def mol_to_3d_from_smiles(smiles):
    mol = Chem.MolFromSmiles(smiles)

    def _get_atom_poses(mol):
        atom_poses = []
        try:
            new_mol = Chem.AddHs(mol)
            cids = AllChem.EmbedMultipleConfs(new_mol, numConfs=10)
            res = AllChem.MMFFOptimizeMoleculeConfs(new_mol)
            index = np.argmin([x[1] for x in res])
            new_mol = Chem.RemoveHs(new_mol)
            conf = new_mol.GetConformer(id=int(index))
        except:
            AllChem.Compute2DCoords(mol)
            conf = mol.GetConformer()
            new_mol = mol
        for i, atom in enumerate(mol.GetAtoms()):
            if atom.GetAtomicNum() == 0:
                return [[0.0, 0.0, 0.0]] * len(mol.GetAtoms())
            pos = conf.GetAtomPosition(i)
            atom_poses.append([pos.x, pos.y, pos.z])
        return new_mol, np.array(atom_poses, "float32")

    def _get_node_features(mol):
        all_node_feats = []
        for atom in mol.GetAtoms():
            node_feats = atom_to_feature_vector_3D(atom)
            all_node_feats.append(node_feats)
        all_node_feats = np.asarray(all_node_feats)
        return torch.tensor(all_node_feats, dtype=torch.float)

    def _cal_bond_length(mol, atom_pos):
        bond_length = []
        for i, bond in enumerate(mol.GetBonds()):
            startid = bond.GetBeginAtomIdx()
            endid = bond.GetEndAtomIdx()
            b_l = np.linalg.norm(atom_pos[startid] - atom_pos[endid])
            bond_length.append(b_l)
        bond_length = np.array(bond_length, "float32")
        return bond_length

    def _get_edge_features(mol, bond_length):
        all_edge_feats = []
        for bond in mol.GetBonds():
            edge_feats = bond_to_feature_vector_3D(bond)
            length = bond_length[bond.GetIdx()]
            edge_feats.append(length)
            all_edge_feats += [edge_feats, edge_feats]
        N_atom = mol.GetNumAtoms()
        for i in range(N_atom):
            edge_feats = self_loop_bond_feature()
            all_edge_feats += [edge_feats]
        all_edge_feats = np.asarray(all_edge_feats, dtype=float)
        return torch.tensor(all_edge_feats, dtype=torch.float)

    def _get_adjacency_info(mol):
        edge_indices = []
        for bond in mol.GetBonds():
            start, end = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
            edge_indices += [[start, end], [end, start]]
        N = mol.GetNumAtoms()
        for i in range(N):
            edge_indices += [(i, i)]
        edge_indices = torch.tensor(edge_indices)
        edge_indices = edge_indices.t().to(torch.long).view(2, -1)
        return edge_indices

    def _get_ba_adjacency_info(edge_indices, atom_pos, edge_attr_atom):
        def _get_angle(vec1, vec2):
            norm1 = np.linalg.norm(vec1)
            norm2 = np.linalg.norm(vec2)
            if norm1 == 0 or norm2 == 0:
                return 0
            vec1 = vec1 / (norm1 + 1e-5)
            vec2 = vec2 / (norm2 + 1e-5)
            angle = np.arccos(np.dot(vec1, vec2))
            return angle

        edge_indices_pair = edge_indices.t()
        super_edge_indices = []
        bond_angle = []
        x_bond = []
        E = len(edge_indices_pair)
        edge_indices = np.arange(E)
        for bondidx, bond in enumerate(edge_indices_pair):
            x_bond += [edge_attr_atom[bondidx]]
            src_edge_indices = edge_indices[edge_indices_pair[:, 1] == bond[0]]
            for src_edge_i in src_edge_indices:
                if src_edge_i == bondidx:
                    continue
                src_edge = edge_indices_pair[src_edge_i]
                super_edge_indices += [[src_edge_i, bondidx]]
                vec1 = atom_pos[bond[0]] - atom_pos[bond[1]]
                vec2 = atom_pos[src_edge[0]] - atom_pos[src_edge[1]]
                angle = _get_angle(vec1, vec2)
                bond_angle += [angle]
        super_edge_indices = torch.tensor(super_edge_indices)
        super_edge_indices = super_edge_indices.t().to(torch.long).view(2, -1)
        bond_angle = np.asarray(bond_angle, "float32")
        x_bond = torch.stack(x_bond)
        bond_angle = torch.tensor(bond_angle, dtype=torch.float)
        bond_angle = bond_angle.reshape(bond_angle.shape[0], 1)
        return super_edge_indices, bond_angle, x_bond

    new_mol, atom_poses = _get_atom_poses(mol)
    x_atom = _get_node_features(new_mol)
    bond_length = _cal_bond_length(new_mol, atom_poses)
    edge_attr_atom = _get_edge_features(mol, bond_length)
    edge_index_atom = _get_adjacency_info(new_mol)
    edge_index_bond, edge_attr_bond, x_bond = _get_ba_adjacency_info(edge_index_atom, atom_poses, edge_attr_atom)
    drug_atom = Data(x=x_atom, edge_index=edge_index_atom, edge_attr=edge_attr_atom)
    drug_bond = Data(x=x_bond, edge_index=edge_index_bond, edge_attr=edge_attr_bond)
    drug_atom.smiles = smiles
    drug_bond.smiles = smiles
    return drug_atom, drug_bond


def create_drug_geometry(save_path="GDSC/drug/"):
    drug_df = pd.read_csv("GDSC/drug/222drugs_pubchem_smiles.csv", header=0)
    pubchem = drug_df["pubchem"].to_list()
    smiles = drug_df["smiles"].to_list()
    drug_smiles = []
    for i in range(len(drug_df)):
        drug_smile = [pubchem[i], smiles[i]]
        drug_smiles.append(drug_smile)

    drug_atom_dict = {}
    drug_bond_dict = {}
    for i in range(len(drug_smiles)):
        drug_id = drug_smiles[i][0]
        smiles = drug_smiles[i][1]
        drug_atom, drug_bond = mol_to_3d_from_smiles(smiles)
        drug_atom_dict[drug_id] = drug_atom
        drug_bond_dict[drug_id] = drug_bond

    with open(save_path + "drug_feat_atom.pkl", "wb") as f:
        pickle.dump(drug_atom_dict, f)

    with open(save_path + "drug_feat_bond.pkl", "wb") as f:
        pickle.dump(drug_bond_dict, f)


if __name__ == "__main__":
    save_path_cell = "GDSC/cell"
    save_path_drug = "GDSC/drug"
    create_cell_feature(save_path=save_path_cell)
    create_drug_fingerprint(save_path=save_path_drug)
    create_drug_geometry(save_path=save_path_drug)
