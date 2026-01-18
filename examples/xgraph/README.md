# XGraph Examples

This folder contains notebooks for individual explainers and a unified runner:
`run_graph_explainers.py`.

## Quick start

Run all four explainers on MUTAG (CPU):
```
python examples/xgraph/run_graph_explainers.py \
  --datasets mutag \
  --explainers pgexplainer gnnexplainer subgraphx lacore \
  --device cpu
```

Run all four explainers on BA-2Motifs (GPU 0):
```
python examples/xgraph/run_graph_explainers.py \
  --datasets ba_2motifs \
  --explainers pgexplainer gnnexplainer subgraphx lacore \
  --device cuda:0
```

Limit to 50 graphs and change sparsity (used by GNNExplainer/SubgraphX):
```
python examples/xgraph/run_graph_explainers.py \
  --datasets mutag ba_2motifs \
  --explainers gnnexplainer subgraphx \
  --max-graphs 50 \
  --sparsity 0.6 \
  --device cuda:0
```

Run only LaCore with a custom epsilon (single cluster):
```
python examples/xgraph/run_graph_explainers.py \
  --datasets mutag \
  --explainers lacore \
  --lacore-epsilon 0.35 \
  --device cuda:0
```

## Notes

- MUTAG requires RDKit installed.
- The script trains a GCN and caches checkpoints in `./checkpoints/xgraph/`.
- LaCore uses a single cluster; cluster size is controlled by `--lacore-epsilon`.
