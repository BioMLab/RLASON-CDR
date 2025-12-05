import numpy as np
import torch
import random
import scipy.sparse as sp
from sklearn.metrics import precision_recall_curve, roc_auc_score, average_precision_score, matthews_corrcoef

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def normalize_adjacency_matrix(A, I):
    """
    Creating a normalized adjacency matrix with self loops.
    :param A: Sparse adjacency matrix.
    :param I: Identity matrix.
    :return A_tile_hat: Normalized adjacency matrix.
    """
    A_tilde = A + 2 * I
    degrees = A_tilde.sum(axis=0)[0].tolist()
    D = sp.diags(degrees, [0])
    D = D.power(-0.5)
    A_tilde_hat = D.dot(A_tilde).dot(D)
    return A_tilde_hat


def create_propagator_matrix(A, device):
    """
    Creating a propagator matrix.
    :param graph: NetworkX graph.
    :return propagator: Dictionary of matrix indices and values.
    """
    I = sp.eye(A.shape[0])
    A_tilde_hat = normalize_adjacency_matrix(A, I)
    propagator = dict()
    A_tilde_hat = sp.coo_matrix(A_tilde_hat)
    ind = np.concatenate([A_tilde_hat.row.reshape(-1, 1), A_tilde_hat.col.reshape(-1, 1)], axis=1)
    propagator["indices"] = torch.LongTensor(ind.T).to(device)
    propagator["values"] = torch.FloatTensor(A_tilde_hat.data).to(device)
    return propagator


def features_to_sparse(features, device):
    """
    Reading the feature matrix stored as JSON from the disk.
    :param path: Path to the JSON file.
    :return out_features: Dict with index and value tensor.
    """
    index_1, index_2 = features.nonzero()
    values = [1.0]*len(index_1)
    node_count = features.shape[0]
    feature_count = features.shape[1]
    features = sp.coo_matrix((values, (index_1, index_2)),
                                 shape=(node_count, feature_count),
                                 dtype=np.float32)
    out_features = dict()
    ind = np.concatenate([features.row.reshape(-1, 1), features.col.reshape(-1, 1)], axis=1)
    out_features["indices"] = torch.LongTensor(ind.T).to(device)
    out_features["values"] = torch.FloatTensor(features.data).to(device)
    out_features["dimensions"] = features.shape
    return out_features


def HGCN_data_preprocess(train_set, all_pairs):
    train_set = train_set.dropna().copy()
    train_set["Drug_ID"] = train_set["Drug_ID"].astype(str)
    train_set["Cell_ID"] = train_set["Cell_ID"].astype(str)

    all_pairs = all_pairs.copy()
    all_pairs["Drug_ID"] = all_pairs["Drug_ID"].astype(str)
    all_pairs["Cell_ID"] = all_pairs["Cell_ID"].astype(str)

    all_nodes = list(set(all_pairs["Drug_ID"].tolist() + all_pairs["Cell_ID"].tolist()))
    all_nodes = np.array(all_nodes)
    idx_map = {j: i for i, j in enumerate(all_nodes)}
    n = len(all_nodes)

    features_dense = np.eye(n, dtype=np.float32)
    features = features_to_sparse(features_dense, device)

    edges_unordered = train_set[["Drug_ID", "Cell_ID"]].values
    if edges_unordered.size == 0:
        adj = sp.coo_matrix((n, n), dtype=np.float32)
    else:
        edges = np.array(list(map(idx_map.get, edges_unordered.flatten()))).reshape(edges_unordered.shape)
        adj = sp.coo_matrix((np.ones(edges.shape[0], dtype=np.float32),
                             (edges[:, 0], edges[:, 1])),
                            shape=(n, n), dtype=np.float32)
        adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)

    propagation_matrix = create_propagator_matrix(adj, device)

    return propagation_matrix, features, idx_map


def set_seed(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    return


def metrics_graph(y_true, y_pred, num_thresholds=1000):
    """
    y_true: numpy array of shape (N,)
    y_pred: numpy array of shape (N,)
    """
    precision_curve, recall_curve, thresholds_curve = precision_recall_curve(y_true, y_pred)
    aupr = -np.trapz(precision_curve, recall_curve)
    auc = roc_auc_score(y_true, y_pred)
    ap = average_precision_score(y_true, y_pred)

    thresholds = np.linspace(0, 1, num=num_thresholds)
    y_true = y_true.reshape(1, -1)
    y_pred_broadcast = y_pred.reshape(1, -1)

    pred_binary = (y_pred_broadcast >= thresholds.reshape(-1, 1)).astype(int)

    TP = (pred_binary * y_true).sum(axis=1)
    FP = pred_binary.sum(axis=1) - TP
    FN = y_true.sum(axis=1) - TP
    TN = y_true.shape[1] - TP - FP - FN

    precision = TP / (TP + FP + 1e-8)
    recall = TP / (TP + FN + 1e-8)
    accuracy = (TP + TN) / y_true.shape[1]
    f1_score = 2 * TP / (2 * TP + FP + FN + 1e-8)

    best_idx = np.argmax(f1_score)
    best_thresh = thresholds[best_idx]
    y_pred_best = (y_pred >= best_thresh).astype(int)

    mcc = matthews_corrcoef(y_true.flatten(), y_pred_best)

    best_metrics = {
        "auc": auc,
        "aupr": aupr,
        "f1": f1_score[best_idx],
        "accuracy": accuracy[best_idx],
        "recall": recall[best_idx],
        "precision": precision[best_idx],
        "ap": ap,
        "mcc": mcc
    }

    return (best_metrics["auc"], best_metrics["aupr"],
            best_metrics["f1"], best_metrics["accuracy"],
            best_metrics["recall"], best_metrics["precision"],
            best_metrics["ap"], best_metrics["mcc"])

