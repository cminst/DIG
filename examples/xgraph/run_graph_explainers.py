#!/usr/bin/env python
import argparse
import copy
import os
import random
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import Adam
from torch.utils.data import random_split, Subset
from torch_geometric.loader import DataLoader
from torch_geometric.utils import add_remaining_self_loops

from dig.xgraph.dataset import MoleculeDataset, SynGraphDataset
from dig.xgraph.evaluation import XCollector
from dig.xgraph.method import GNNExplainer, PGExplainer, SubgraphX, LaCoreExplainer
from dig.xgraph.method.base_explainer import ExplainerBase
from dig.xgraph.method.subgraphx import find_closest_node_result
from dig.xgraph.models import GCN_3l


DATASET_DEFAULTS: Dict[str, Dict] = {
    "mutag": {
        "hidden_dim": 64,
        "batch_size": 32,
        "train_epochs": 100,
        "lr": 0.001,
        "weight_decay": 5e-4,
    },
    "ba_2motifs": {
        "hidden_dim": 20,
        "batch_size": 64,
        "train_epochs": 200,
        "lr": 0.001,
        "weight_decay": 0.0,
    },
}


def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def parse_csv_list(value: str) -> List[str]:
    return [item.strip().lower() for item in value.split(",") if item.strip()]


def load_dataset(root: str, name: str):
    name = name.lower()
    if name == "mutag":
        dataset = MoleculeDataset(root=root, name="MUTAG")
    elif name == "ba_2motifs":
        dataset = SynGraphDataset(root=root, name="BA_2Motifs")
    else:
        raise ValueError(f"Unsupported dataset: {name}")

    dataset.data.x = dataset.data.x.float()
    dataset.data.y = dataset.data.y.squeeze().long()
    return dataset


def split_dataset(dataset, split_ratio: Tuple[float, float, float], seed: int):
    num_train = int(split_ratio[0] * len(dataset))
    num_val = int(split_ratio[1] * len(dataset))
    num_test = len(dataset) - num_train - num_val
    generator = torch.Generator().manual_seed(seed)
    return random_split(dataset, [num_train, num_val, num_test], generator=generator)


def build_model(num_features: int, num_classes: int, hidden_dim: int):
    return GCN_3l(model_level="graph",
                  dim_node=num_features,
                  dim_hidden=hidden_dim,
                  num_classes=num_classes)


def evaluate_accuracy(model, loader, device):
    model.eval()
    total, correct = 0, 0
    losses = []
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            logits = model(batch.x, batch.edge_index, batch.batch)
            loss = F.cross_entropy(logits, batch.y)
            losses.append(loss.item())
            preds = logits.argmax(dim=-1)
            total += batch.y.numel()
            correct += (preds == batch.y).sum().item()
    acc = correct / max(total, 1)
    return (sum(losses) / max(len(losses), 1)), acc


def train_model(model, train_loader, val_loader, device, epochs, lr, weight_decay):
    optimizer = Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    best_state = None
    best_val_acc = -1.0

    model.to(device)
    for epoch in range(epochs):
        model.train()
        for batch in train_loader:
            batch = batch.to(device)
            logits = model(batch.x, batch.edge_index, batch.batch)
            loss = F.cross_entropy(logits, batch.y)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_value_(model.parameters(), clip_value=2.0)
            optimizer.step()

        val_loss, val_acc = evaluate_accuracy(model, val_loader, device)
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = copy.deepcopy(model.state_dict())
        if (epoch + 1) % 20 == 0 or epoch == 0:
            print(f"  Epoch {epoch + 1}/{epochs} - val_loss={val_loss:.4f} val_acc={val_acc:.4f}")

    if best_state is not None:
        model.load_state_dict(best_state)


def save_checkpoint(model, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save({"state_dict": model.state_dict()}, path)


def load_checkpoint(model, path: str) -> bool:
    if not os.path.isfile(path):
        return False
    state = torch.load(path, map_location="cpu")
    state_dict = state.get("state_dict", state.get("net"))
    if state_dict is None:
        return False
    model.load_state_dict(state_dict)
    return True


class PGExplainerEdges(ExplainerBase):
    def __init__(self, pgexplainer, model, molecule: bool):
        super().__init__(model=model,
                         explain_graph=pgexplainer.explain_graph,
                         molecule=molecule)
        self.explainer = pgexplainer

    def forward(self, x, edge_index, **kwargs):
        super().forward(x=x, edge_index=edge_index, **kwargs)
        num_classes = kwargs.get("num_classes")
        if num_classes is None:
            raise ValueError("PGExplainerEdges requires num_classes in kwargs.")

        self.model.eval()
        self.explainer.__clear_masks__()

        x = x.to(self.device)
        edge_index = add_remaining_self_loops(edge_index, num_nodes=x.size(0))[0]
        edge_index = edge_index.to(self.device)

        embed = self.model.get_emb(x, edge_index)
        _, edge_mask = self.explainer.explain(x, edge_index, embed=embed, tmp=1.0, training=False)

        edge_masks = [edge_mask for _ in range(num_classes)]
        hard_edge_masks = [
            self.control_sparsity(edge_mask, sparsity=kwargs.get("sparsity")).sigmoid()
            for _ in range(num_classes)
        ]

        self.__set_masks__(x, edge_index)
        with torch.no_grad():
            related_preds = self.eval_related_pred(x, edge_index, hard_edge_masks)
        self.__clear_masks__()

        return edge_masks, hard_edge_masks, related_preds


def iter_test_graphs(test_subset: Subset, max_graphs: int):
    for idx, data in enumerate(test_subset):
        if max_graphs is not None and idx >= max_graphs:
            break
        yield data


def run_gnnexplainer(model, dataset, test_subset, device, args):
    explainer = GNNExplainer(model,
                             epochs=args.gnnexp_epochs,
                             lr=args.gnnexp_lr,
                             explain_graph=True)
    explainer.device = device
    x_collector = XCollector()

    for data in iter_test_graphs(test_subset, args.max_graphs):
        data = data.to(device)
        edge_index = add_remaining_self_loops(data.edge_index, num_nodes=data.num_nodes)[0]
        pred = model(x=data.x, edge_index=edge_index).argmax(-1).item()
        edge_masks, hard_edge_masks, related_preds = explainer(
            data.x, edge_index,
            sparsity=args.sparsity,
            num_classes=dataset.num_classes
        )
        x_collector.collect_data(hard_edge_masks, related_preds, label=pred)

    return x_collector


def run_pgexplainer(model, dataset, train_subset, test_subset, device, args, pg_ckpt_path: str, dataset_name: str):
    in_channels = args.hidden_dim * 2
    pgexplainer = PGExplainer(
        model=model,
        in_channels=in_channels,
        device=device,
        explain_graph=True,
        epochs=args.pg_epochs,
        lr=args.pg_lr,
        coff_size=args.pg_coff_size,
        coff_ent=args.pg_coff_ent,
        t0=args.pg_t0,
        t1=args.pg_t1,
        sample_bias=args.pg_sample_bias,
    )

    if not load_checkpoint(pgexplainer, pg_ckpt_path):
        print("  Training PGExplainer...")
        pgexplainer.train_explanation_network(train_subset)
        save_checkpoint(pgexplainer, pg_ckpt_path)

    pg_edges = PGExplainerEdges(pgexplainer=pgexplainer, model=model, molecule=(dataset_name == "mutag"))
    pg_edges.device = device
    x_collector = XCollector()

    for data in iter_test_graphs(test_subset, args.max_graphs):
        data = data.to(device)
        edge_index = add_remaining_self_loops(data.edge_index, num_nodes=data.num_nodes)[0]
        pred = model(x=data.x, edge_index=edge_index).argmax(-1).item()
        edge_masks, hard_edge_masks, related_preds = pg_edges(
            data.x, edge_index,
            num_classes=dataset.num_classes,
            sparsity=args.sparsity
        )
        x_collector.collect_data(hard_edge_masks, related_preds, label=pred)

    return x_collector


def run_subgraphx(model, dataset, test_subset, device, args):
    subgraphx = SubgraphX(
        model=model,
        num_classes=dataset.num_classes,
        device=device,
        explain_graph=True,
        rollout=args.subgraphx_rollout,
        min_atoms=args.subgraphx_min_atoms,
        c_puct=args.subgraphx_c_puct,
        expand_atoms=args.subgraphx_expand_atoms,
        reward_method=args.subgraphx_reward_method,
        subgraph_building_method=args.subgraphx_building_method,
    )
    x_collector = XCollector()

    for data in iter_test_graphs(test_subset, args.max_graphs):
        data = data.to(device)
        edge_index = add_remaining_self_loops(data.edge_index, num_nodes=data.num_nodes)[0]
        pred = model(x=data.x, edge_index=edge_index).argmax(-1).item()
        explain_result, related_pred = subgraphx.explain(
            data.x,
            edge_index,
            label=pred,
            max_nodes=args.subgraphx_max_nodes,
        )
        explain_result = [explain_result]
        related_preds = [related_pred]
        x_collector.collect_data(explain_result, related_preds, label=0)

    return x_collector


def run_lacore(model, dataset, test_subset, device, args):
    explainer = LaCoreExplainer(model=model, epsilon=args.lacore_epsilon, explain_graph=True)
    explainer.device = device
    x_collector = XCollector()

    for data in iter_test_graphs(test_subset, args.max_graphs):
        data = data.to(device)
        edge_index = add_remaining_self_loops(data.edge_index, num_nodes=data.num_nodes)[0]
        pred = model(x=data.x, edge_index=edge_index).argmax(-1).item()
        edge_masks, hard_edge_masks, related_preds = explainer(
            data.x, edge_index,
            num_classes=dataset.num_classes
        )
        x_collector.collect_data(hard_edge_masks, related_preds, label=pred)

    return x_collector


def main():
    parser = argparse.ArgumentParser(description="Run multiple graph explanation methods on MUTAG and BA-2Motifs.")
    parser.add_argument("--datasets", default="mutag,ba_2motifs", type=str)
    parser.add_argument("--explainers", default="pgexplainer,gnnexplainer,subgraphx,lacore", type=str)
    parser.add_argument("--data-root", default="./datasets", type=str)
    parser.add_argument("--checkpoint-dir", default="./checkpoints/xgraph", type=str)
    parser.add_argument("--max-graphs", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default=None, type=str)
    parser.add_argument("--train-epochs", type=int, default=None)
    parser.add_argument("--sparsity", type=float, default=0.7)

    parser.add_argument("--gnnexp-epochs", type=int, default=100)
    parser.add_argument("--gnnexp-lr", type=float, default=0.01)

    parser.add_argument("--pg-epochs", type=int, default=20)
    parser.add_argument("--pg-lr", type=float, default=0.005)
    parser.add_argument("--pg-coff-size", type=float, default=0.01)
    parser.add_argument("--pg-coff-ent", type=float, default=5e-4)
    parser.add_argument("--pg-t0", type=float, default=5.0)
    parser.add_argument("--pg-t1", type=float, default=1.0)
    parser.add_argument("--pg-sample-bias", type=float, default=0.0)

    parser.add_argument("--subgraphx-max-nodes", type=int, default=10)
    parser.add_argument("--subgraphx-rollout", type=int, default=20)
    parser.add_argument("--subgraphx-min-atoms", type=int, default=5)
    parser.add_argument("--subgraphx-c-puct", type=float, default=10.0)
    parser.add_argument("--subgraphx-expand-atoms", type=int, default=14)
    parser.add_argument("--subgraphx-reward-method", type=str, default="mc_l_shapley")
    parser.add_argument("--subgraphx-building-method", type=str, default="zero_filling")

    parser.add_argument("--lacore-epsilon", type=float, default=0.1)

    args = parser.parse_args()
    args.dataset_list = parse_csv_list(args.datasets)
    args.explainer_list = parse_csv_list(args.explainers)

    if args.device is None:
        args.device = "cuda" if torch.cuda.is_available() else "cpu"

    seed_everything(args.seed)
    device = torch.device(args.device)

    for dataset_name in args.dataset_list:
        print(f"\n== Dataset: {dataset_name} ==")
        dataset = load_dataset(args.data_root, dataset_name)

        defaults = DATASET_DEFAULTS.get(dataset_name, DATASET_DEFAULTS["mutag"])
        args.hidden_dim = defaults["hidden_dim"]
        train_epochs = args.train_epochs if args.train_epochs is not None else defaults["train_epochs"]

        train_set, val_set, test_set = split_dataset(dataset, (0.8, 0.1, 0.1), args.seed)
        train_loader = DataLoader(train_set, batch_size=defaults["batch_size"], shuffle=True)
        val_loader = DataLoader(val_set, batch_size=defaults["batch_size"], shuffle=False)

        model = build_model(dataset.num_node_features, dataset.num_classes, defaults["hidden_dim"])
        model_ckpt = os.path.join(args.checkpoint_dir, dataset_name, "gcn_3l_best.pth")
        if load_checkpoint(model, model_ckpt):
            print(f"  Loaded model checkpoint: {model_ckpt}")
        else:
            print("  Training GCN model...")
            train_model(model, train_loader, val_loader, device,
                        epochs=train_epochs,
                        lr=defaults["lr"],
                        weight_decay=defaults["weight_decay"])
            save_checkpoint(model, model_ckpt)

        model.to(device)
        model.eval()

        if "gnnexplainer" in args.explainer_list:
            print("-> GNNExplainer")
            collector = run_gnnexplainer(model, dataset, test_set, device, args)
            print(f"  Fidelity: {collector.fidelity:.4f}  Fidelity_inv: {collector.fidelity_inv:.4f}  "
                  f"Sparsity: {collector.sparsity:.4f}")

        if "pgexplainer" in args.explainer_list:
            print("-> PGExplainer")
            pg_ckpt = os.path.join(args.checkpoint_dir, dataset_name, "pgexplainer.pth")
            collector = run_pgexplainer(model, dataset, train_set, test_set, device, args, pg_ckpt, dataset_name)
            print(f"  Fidelity: {collector.fidelity:.4f}  Fidelity_inv: {collector.fidelity_inv:.4f}  "
                  f"Sparsity: {collector.sparsity:.4f}")

        if "subgraphx" in args.explainer_list:
            print("-> SubgraphX")
            collector = run_subgraphx(model, dataset, test_set, device, args)
            print(f"  Fidelity: {collector.fidelity:.4f}  Fidelity_inv: {collector.fidelity_inv:.4f}  "
                  f"Sparsity: {collector.sparsity:.4f}")

        if "lacore" in args.explainer_list:
            print("-> LaCoreExplainer")
            collector = run_lacore(model, dataset, test_set, device, args)
            print(f"  Fidelity: {collector.fidelity:.4f}  Fidelity_inv: {collector.fidelity_inv:.4f}  "
                  f"Sparsity: {collector.sparsity:.4f}")


if __name__ == "__main__":
    main()
