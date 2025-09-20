import os
import torch
from tqdm import tqdm
from models import GnnNets
from load_dataset import get_dataset, get_dataloader
from forgraph.mcts import MCTS, reward_func
from torch_geometric.data import Batch
from Configures import data_args, mcts_args, reward_args, model_args, train_args
from shapley import GnnNets_GC2value_func, gnn_score
from utils import PlotUtils, find_closest_node_result


def pipeline(max_nodes):
    dataset = get_dataset(data_args.dataset_dir, data_args.dataset_name)
    plotutils = PlotUtils(dataset_name=data_args.dataset_name)
    input_dim = dataset.num_node_features
    output_dim = dataset.num_classes

    if data_args.dataset_name == 'mutag':
        data_indices = list(range(len(dataset)))
    else:
        loader = get_dataloader(dataset,
                                batch_size=train_args.batch_size,
                                random_split_flag=data_args.random_split,
                                data_split_ratio=data_args.data_split_ratio,
                                seed=data_args.seed)
        data_indices = loader['test'].dataset.indices

    gnnNets = GnnNets(input_dim, output_dim, model_args)
    checkpoint = torch.load(mcts_args.explain_model_path, weights_only=False)
    gnnNets.update_state_dict(checkpoint['net'])
    gnnNets.to_device()
    gnnNets.eval()

    save_dir = os.path.join('./results',
                            f"{mcts_args.dataset_name}_"
                            f"{model_args.model_name}_"
                            f"{reward_args.reward_method}")
    if not os.path.isdir(save_dir):
        os.mkdir(save_dir)

    fidelity_score_list = []
    sparsity_score_list = []
    for i in tqdm(data_indices):
        # get data and prediction
        data = dataset[i]
        _, probs, _ = gnnNets(Batch.from_data_list([data.clone()]))
        prediction = probs.squeeze().argmax(-1).item()
        original_score = probs.squeeze()[prediction]

        # get the reward func
        value_func = GnnNets_GC2value_func(gnnNets, target_class=prediction)
        payoff_func = reward_func(reward_args, value_func)

        # find the paths and build the graph
        result_path = os.path.join(save_dir, f"example_{i}.pt")

        # mcts for l_shapely
        mcts_state_map = MCTS(data.x, data.edge_index,
                              score_func=payoff_func,
                              n_rollout=mcts_args.rollout,
                              min_atoms=mcts_args.min_atoms,
                              c_puct=mcts_args.c_puct,
                              expand_atoms=mcts_args.expand_atoms)

        if os.path.isfile(result_path):
            results = torch.load(result_path, weights_only=False)
        else:
            results = mcts_state_map.mcts(verbose=True)
            torch.save(results, result_path)

        # l sharply score
        graph_node_x = find_closest_node_result(results, max_nodes=max_nodes)
        
        # explanation nodes & complement
        expl_nodes = list(graph_node_x.coalition)
        all_nodes = list(range(graph_node_x.data.x.shape[0]))
        non_expl_nodes = [n for n in all_nodes if n not in expl_nodes]
    
        # evaluate keep-only (fidelity+) -> mask NON-explanation nodes
        score_keep = gnn_score(
            non_expl_nodes,
            data,
            value_func,
            subgraph_building_method='zero_filling',  # same as before
        )
        fidelity_plus = (original_score - score_keep)
    
        # evaluate remove-only (fidelity-) -> mask EXPLANATION nodes
        score_remove = gnn_score(
            expl_nodes,
            data,
            value_func,
            subgraph_building_method='zero_filling',
        )
        fidelity_minus = (original_score - score_remove)
        
        # sparsity (unchanged)
        sparsity_score = 1 - len(expl_nodes) / graph_node_x.ori_graph.number_of_nodes()
    
        # collect
        fidelity_score_list.append(fidelity_plus)
        sparsity_score_list.append(sparsity_score)
        # NEW: track fidelity- too
        try:
            fidelity_minus_list.append(fidelity_minus)
        except NameError:
            fidelity_minus_list = [fidelity_minus]

        # visualization
        if hasattr(dataset, 'supplement'):
            words = dataset.supplement['sentence_tokens'][str(i)]
            plotutils.plot(graph_node_x.ori_graph, graph_node_x.coalition, words=words,
                           figname=os.path.join(save_dir, f"example_{i}.png"))
        else:
            plotutils.plot(graph_node_x.ori_graph, graph_node_x.coalition, x=graph_node_x.data.x,
                           figname=os.path.join(save_dir, f"example_{i}.png"))

    fidelity_scores = torch.tensor(fidelity_score_list)
    sparsity_scores = torch.tensor(sparsity_score_list)
    fidelity_minus_scores = torch.tensor(fidelity_minus_list)
    return fidelity_scores, sparsity_scores, fidelity_minus_scores


if __name__ == '__main__':
    fidelity_plus, sparsity, fidelity_minus = pipeline(15)
    print(f"Fidelity+ drop: {fidelity_plus.mean().item():.4f}  "
          f"Sparsity: {sparsity.mean().item():.4f}  "
          f"Fidelity- drop: {fidelity_minus.mean().item():.4f}")
