## RLASON-CDR
A Reinforcement Learning–Driven Adaptive Synergistic Optimization Network for Cancer Drug Response Prediction.

## Requirements

This project was developed and tested in a conda environment with CUDA 11.3.

* python == 3.9  
* torch == 1.11.0  
* torch-geometric == 2.5.3  
* torch-scatter == 2.0.9  
* torch-sparse == 0.6.13  
* numpy == 1.23.0  
* pandas == 2.2.3  
* scikit-learn == 1.0.2  
* scipy == 1.13.1
* rdkit == 2022.9.5  
* PyBioMed >= 1.0.0  
* subword-nmt >= 0.3.7

## Data preparation
We first preprocess the cell-line and drug features by running:

```bash
$ python create_load_data.py
```
After running create_load_data.py, the following processed feature files will be generated:

Cell features:
* `cell_exp.pkl`: Gene expression data for cancer cell lines.
* `cell_meth.pkl`: DNA methylation data for cancer cell lines.
* `cell_mut.pkl`: Genomic mutation data for cancer cell lines.
* `cell_path.pkl`: Pathway enrichment scores for cance cell lines.

Drug features:
* `ecfp_encoding.pkl`: Extended Connectivity Fingerprint for drugs.
* `espf_encoding.pkl`: Explainable Substructure Partition Fingerprint for drugs.
* `pubchem_encoding.pkl`: PubChem Substructure Fingerprint for drugs.
* `drug_feat_atom.pkl`: Atom-level 3D geometry features represented as molecular graphs for drugs.
* `drug_feat_bond.pkl`: Bond-level 3D geometry features including bond angles and connectivity for drugs.

## Code
* `main.py`: This function is used to train and test RLASON-CDR.
* `config.py`: This function is used to control the hyperparameters of RLASON-CDR.
* `load_data.py`: This function is used to load the data of cancer cell lines, drugs and binary IC50 values.
* `data_process.py`: This function is used to process input data.
* `model.py`: This function contains the RLASON-CDR model components.
* `utils.py`: This function contains the necessary processing subroutines.

## Hyperparameter configuration
The `config.py` defines the tuned hyperparameters used in our experiments, including the basic training settings (`epochs`, `lr`, `weight_decay`, `batch_size`), the architecture and regularization parameters for the HDAN module, the HGCN module, as well as the PPO-related coefficients (`clip_epsilon`, `ppo_epochs`, `entropy_weight`, `bce_coef`, `policy_coef`, `value_coef`) for collaborative optimization.
