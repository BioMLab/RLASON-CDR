import torch
from torch.utils.data import Dataset, DataLoader
from torch_geometric.data import Batch
from sklearn.model_selection import train_test_split


class MyDataset(Dataset):
    def __init__(self, drug_ecfp, drug_espf, drug_pubchem, drug_atom, drug_bond,
                 cell_exp, cell_meth, cell_mut, cell_path,
                 pairs, idx_map):
        super(MyDataset, self).__init__()
        self.drug_ecfp = drug_ecfp
        self.drug_espf = drug_espf
        self.drug_pubchem = drug_pubchem

        self.drug_atom = drug_atom
        self.drug_bond = drug_bond

        self.cell_exp = cell_exp
        self.cell_meth = cell_meth
        self.cell_mut = cell_mut
        self.cell_path = cell_path

        pairs.reset_index(drop=True, inplace=True)
        self.drug_id = pairs["Drug_ID"]
        self.cell_id = pairs["Cell_ID"]
        self.labels = pairs["Label"]

        self.idx_map = idx_map

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        drug_index = self.drug_id[index]
        cell_index = self.cell_id[index]
        idx1 = self.idx_map[str(drug_index)]
        idx2 = self.idx_map[cell_index]

        return (
            torch.tensor(self.drug_ecfp[drug_index], dtype=torch.float32),
            torch.tensor(self.drug_espf[drug_index], dtype=torch.float32),
            torch.tensor(self.drug_pubchem[drug_index], dtype=torch.float32),
            self.drug_atom[drug_index],
            self.drug_bond[drug_index],
            torch.tensor(self.cell_exp[cell_index], dtype=torch.float32),
            torch.tensor(self.cell_meth[cell_index], dtype=torch.float32),
            torch.tensor(self.cell_mut[cell_index], dtype=torch.float32).unsqueeze(0).unsqueeze(0),
            torch.tensor(self.cell_path[cell_index], dtype=torch.float32),
            torch.tensor(self.labels[index], dtype=torch.float32),
            torch.tensor(idx1, dtype=torch.long),
            torch.tensor(idx2, dtype=torch.long),
        )


def collate_fn(samples):
    drug_ecfp, drug_espf, drug_pubchem, drug_atom, drug_bond, cell_exp, cell_meth, cell_mut, cell_path, labels, idx1, idx2 = zip(*samples)
    batch_drug_atom = Batch.from_data_list(drug_atom)
    batch_drug_bond = Batch.from_data_list(drug_bond)

    return (
        torch.stack(drug_ecfp),
        torch.stack(drug_espf),
        torch.stack(drug_pubchem),
        batch_drug_atom,
        batch_drug_bond,
        torch.stack(cell_exp),
        torch.stack(cell_meth),
        torch.stack(cell_mut),
        torch.stack(cell_path),
        torch.stack(labels),
        torch.stack(idx1),
        torch.stack(idx2),
    )


def process_cv(all_pairs, fold=0, n_splits=5, random=31):
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random)
    indices = list(kf.split(all_pairs))

    train_idx, test_idx = indices[fold]
    train_set = all_pairs.iloc[train_idx].reset_index(drop=True)
    test_set = all_pairs.iloc[test_idx].reset_index(drop=True)

    return train_set, test_set


def process_cell_cold_start_cv(all_pairs, fold=0, n_splits=5, random=31):
    unique_cells = np.array(all_pairs["Cell_ID"].unique())
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random)

    cell_splits = list(kf.split(unique_cells))
    train_cells_idx, test_cells_idx = cell_splits[fold]

    train_cells = unique_cells[train_cells_idx]
    test_cells = unique_cells[test_cells_idx]

    # 根据 cell 划分样本
    train_set = all_pairs[all_pairs["Cell_ID"].isin(train_cells)].reset_index(drop=True)
    test_set = all_pairs[all_pairs["Cell_ID"].isin(test_cells)].reset_index(drop=True)

    return train_set, test_set


def process_drug_cold_start_cv(all_pairs, fold=0, n_splits=5, random=31):
    unique_drugs = np.array(all_pairs["Drug_ID"].unique())
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random)

    drug_splits = list(kf.split(unique_drugs))
    train_drugs_idx, test_drugs_idx = drug_splits[fold]

    train_drugs = unique_drugs[train_drugs_idx]
    test_drugs = unique_drugs[test_drugs_idx]

    train_set = all_pairs[all_pairs["Drug_ID"].isin(train_drugs)].reset_index(drop=True)
    test_set = all_pairs[all_pairs["Drug_ID"].isin(test_drugs)].reset_index(drop=True)

    return train_set, test_set


def create_loader(train_set, test_set, drug_ecfp, drug_espf, drug_pubchem, drug_atom, drug_bond, cell_exp, cell_meth, cell_mut, cell_path, idx_map, args):
    train_dataset = MyDataset(drug_ecfp, drug_espf, drug_pubchem, drug_atom, drug_bond, cell_exp, cell_meth, cell_mut, cell_path, train_set, idx_map)
    test_dataset = MyDataset(drug_ecfp, drug_espf, drug_pubchem, drug_atom, drug_bond, cell_exp, cell_meth, cell_mut, cell_path, test_set, idx_map)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, drop_last=False, collate_fn=collate_fn, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, drop_last=False, collate_fn=collate_fn, num_workers=0)
    return train_loader, test_loader

