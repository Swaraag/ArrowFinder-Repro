# Build a custom General, Powerful, Scalable (GPS) Graph Transformer using Pytorch Geometric 
# Raw CSV data is used to build graphs per epoch through SMILES representations, and a custom class
# is used to integrate the architecture in the existing generate_predictions.py inference pipeline.

import torch
from modules.mol_graph_utils import CSVToGraphs

# nn.Module source code: https://github.com/pytorch/pytorch/blob/be757957bace28100e571ec7914765020be4a069/torch/nn/modules/module.py#L69
class CustomGPS(torch.nn.Module):
    def __init__(self, num_layers):
        super().__init__()
        self._num_layers = num_layers
        self.mol_to_graph = CSVToGraphs

    def forward(self):
        pass


if __name__ == "__main__":
    GPS = CustomGPS(5)

    csv_path = "data/mc_train_fold0/reformatted/train.txt"