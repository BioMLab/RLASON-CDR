import os
import pickle
import pandas as pd


def LoadData():
    all_pairs = pd.read_csv(os.getcwd() + "/GDSC/GDSC_binary_IC50.csv")

    with open(os.getcwd() + "/GDSC/drug/ecfp_encoding.pkl", "rb") as f:
        drug_ecfp = pickle.load(f)
    with open(os.getcwd() + "/GDSC/drug/espf_encoding.pkl", "rb") as f:
        drug_espf = pickle.load(f)
    with open(os.getcwd() + "/GDSC/drug/pubchem_encoding.pkl", "rb") as f:
        drug_pubchem = pickle.load(f)
    with open(os.getcwd() + "/GDSC/drug/drug_feat_atom.pkl", "rb") as f:
        drug_atom = pickle.load(f)
    with open(os.getcwd() + "/GDSC/drug/drug_feat_bond.pkl", "rb") as f:
        drug_bond = pickle.load(f)

    with open(os.getcwd() + "/GDSC/cell/cell_exp.pkl", "rb") as f:
        cell_exp = pickle.load(f)
    with open(os.getcwd() + "/GDSC/cell/cell_meth.pkl", "rb") as f:
        cell_meth = pickle.load(f)
    with open(os.getcwd() + "/GDSC/cell/cell_mut.pkl", "rb") as f:
        cell_mut = pickle.load(f)
    with open(os.getcwd() + "/GDSC/cell/cell_path.pkl", "rb") as f:
        cell_path = pickle.load(f)

    return all_pairs, drug_ecfp, drug_espf, drug_pubchem, drug_atom, drug_bond, cell_exp, cell_meth, cell_mut, cell_path

