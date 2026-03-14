import argparse


def arg_parse():
    parser = argparse.ArgumentParser(description="RLASON-CDR Training")
    parser.add_argument("--epochs", default=50, type=int)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=0)
    parser.add_argument("--batch_size", type=int, default=512)

    parser.add_argument("--HDAN_embed_dim", type=int, default=256)
    parser.add_argument("--HDAN_heads", type=int, default=64)
    parser.add_argument("--HDAN_dropout", type=float, default=0.1)

    parser.add_argument("--dropout_ratio_drug_3D", type=float, default=0.4)
    parser.add_argument("--layer_num_3D_drug", type=int, default=2)
    parser.add_argument("--readout", type=str, default="mean")
    parser.add_argument("--JK", type=str, default="True")

    parser.add_argument("--HGCN_hops", type=int, default=3)
    parser.add_argument("--HGCN_embed_dim", type=int, default=256)
    parser.add_argument("--HGCN_hidden_dim", type=int, default=64)
    parser.add_argument("--HGCN_dropout", type=float, default=0.1)

    parser.add_argument("--clip_epsilon", type=float, default=0.2)
    parser.add_argument("--entropy_weight", type=float, default=0.01)
    parser.add_argument("--bce_coef", type=float, default=1.0)
    parser.add_argument("--policy_coef", type=float, default=1.0)
    parser.add_argument("--value_coef", type=float, default=0.5)

    return parser.parse_args()
