from tqdm import tqdm
from load_data import LoadData

from config import *
from model import *
from data_process import *
from utils import *

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

set_seed(31)


def train(model, loader, criterion, opt, device,
          clip_epsilon, entropy_weight,
          bce_coef, policy_coef, value_coef):
    model.train()
    for batch_idx, data in enumerate(tqdm(loader, desc="Training", ncols=100)):
        drug_morgan, drug_espf, drug_pubchem, drug_atom, drug_bond, \
        cell_exp, cell_meth, cell_mut, cell_path, \
        label, idx1, idx2 = [x.to(device) for x in data]

        output, dist, state = model(
            drug_morgan, drug_espf, drug_pubchem, drug_atom, drug_bond,
            cell_exp, cell_meth, cell_mut, cell_path, idx1, idx2
        )

        action = dist.sample()
        log_prob = dist.log_prob(action)

        bce_loss = criterion(output, label.unsqueeze(1).float())

        reward = -bce_loss.detach()

        value_pred = model.value_net(state).squeeze()

        advantage = reward - value_pred.detach()

        ratio = torch.exp(log_prob - log_prob.detach())

        surr1 = ratio * advantage
        surr2 = torch.clamp(ratio, 1 - clip_epsilon, 1 + clip_epsilon) * advantage
        policy_loss = -torch.min(surr1, surr2).mean()

        value_loss = (value_pred - reward).pow(2).mean()
              
        entropy = dist.entropy().mean()

        loss = (bce_coef * bce_loss +
                policy_coef * policy_loss +
                value_coef * value_loss -
                entropy_weight * entropy)

        opt.zero_grad()
        loss.backward()
        opt.step()


def test(model, loader, device):
    model.eval()

    y_true = []
    y_pred = []

    with torch.no_grad():
        for batch_idx, data in enumerate(tqdm(loader, desc="Testing", ncols=100)):
            drug_ecfp, drug_espf, drug_pubchem, drug_atom, drug_bond, \
            cell_exp, cell_meth, cell_mut, cell_path, \
            label, idx1, idx2 = [x.to(device) for x in data]

            output, dist, state = model(drug_ecfp, drug_espf, drug_pubchem, drug_atom, drug_bond,
                                        cell_exp, cell_meth, cell_mut, cell_path, idx1, idx2)

            y_true.append(label.unsqueeze(1))
            y_pred.append(output)

    y_true = torch.cat(y_true).squeeze().cpu().numpy()
    y_pred = torch.cat(y_pred).squeeze().cpu().numpy()
    AUC, AUPR, F1, ACC, precision, MCC = metrics_graph(y_true, y_pred)

    print("test_AUC: " + str(round(AUC, 4)) +
          "  test_AUPR: " + str(round(AUPR, 4)) +
          "  test_F1: " + str(round(F1, 4)) +
          "  test_ACC: " + str(round(ACC, 4)) +
          "  test_precision: " + str(round(precision, 4)) +
          "  test_MCC: " + str(round(MCC, 4)))

    return AUC, AUPR, F1, ACC, recall, precision, AP, MCC


if __name__ == "__main__":
    args = arg_parse()

    all_pairs, drug_ecfp, drug_espf, drug_pubchem, drug_atom, drug_bond, cell_exp, cell_meth, cell_mut, cell_path = LoadData()

    for fold in range(5):
        train_set, test_set = process_cv(all_pairs, fold=fold, n_splits=5, random=31)
        propagation_matrix, features, idx_map = HGCN_data_preprocess(train_set, all_pairs)
        feature_number = features["dimensions"][1]

        policy_net = PolicyNetwork().to(device)
        value_net = ValueNetwork().to(device)

        HDAN_model = HDAN_model(args).to(device)
        HGCN_model = HGCN(args, feature_number).to(device)
        model = RLASON_CDR(HDAN_model, HGCN_model, propagation_matrix, features, policy_net, value_net).to(device)

        criterion = nn.BCELoss()
        opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        best_AUC, best_AUPR, best_F1, best_ACC, best_precision, best_MCC = [0] * 6
        train_loader, test_loader = create_loader(train_set, test_set, drug_ecfp, drug_espf, drug_pubchem, drug_atom, drug_bond, cell_exp, cell_meth, cell_mut, cell_path, idx_map, args)

        for epoch in range(1, args.epochs + 1):
            print(f"===== epoch {epoch} =====")

            train(model, test_loader, criterion, opt, device,
                  clip_epsilon=args.clip_epsilon, entropy_weight=args.entropy_weight,
                  bce_coef=args.bce_coef, policy_coef=args.policy_coef, value_coef=args.value_coef)
            AUC, AUPR, F1, ACC, precision, MCC = test(model, test_loader, device)

            if AUC > best_AUC:
                best_AUC = AUC
                best_AUPR = AUPR
                best_F1 = F1
                best_ACC = ACC
                best_precision = precision
                best_MCC = MCC

            print("best_AUC: " + str(round(best_AUC, 4)) +
                  "  best_AUPR: " + str(round(best_AUPR, 4)) +
                  "  best_F1: " + str(round(best_F1, 4)) +
                  "  best_ACC: " + str(round(best_ACC, 4)) +
                  "  best_precision: " + str(round(best_precision, 4)) +
                  "  best_MCC: " + str(round(best_MCC, 4)))
            print("---------------------------------------")

