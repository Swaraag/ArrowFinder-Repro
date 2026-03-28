# Build a custom General, Powerful, Scalable (GPS) Graph Transformer using Pytorch Geometric 
# Raw CSV data is used to build graphs per epoch through SMILES representations, and a custom class
# is used to integrate the architecture in the existing generate_predictions.py inference pipeline.

import torch
import torch.nn as nn
from reaction_prediction.atom.architectures.make_atom_graph_data import CSVToGraphs
from torch_geometric.nn import GPSConv, GINEConv
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

# nn.Module source code: https://github.com/pytorch/pytorch/blob/be757957bace28100e571ec7914765020be4a069/torch/nn/modules/module.py#L69
class CustomGPS(torch.nn.Module):
    def __init__(self, csv_path, atom_type, batch_size, num_layers, hidden_dim, edge_dim):
        super().__init__()
        
        # either source or sink / REMOVE THIS CODE IF SELF._ATOM_TYPE IS USED IN NO OTHER FUNCTIONS
        if atom_type == 'source' or atom_type == 'sink':
            self._atom_type = atom_type
        else:
            raise ValueError("Atom type isn't source or sink")

        # set up training data with mol_graph_utils.py
        CSVToGraph = CSVToGraphs(self._atom_type)

        data_objs, num_node_features = CSVToGraph.process_csv(csv_path)
        self._data_loader = DataLoader(data_objs, batch_size=batch_size, shuffle=True)

        self._layers = nn.ModuleList()
        curr_dim = num_node_features
        hidden_dim = hidden_dim
        for layer in range(num_layers):
            inner_mlp = nn.Sequential(
                nn.Linear(curr_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim)
            )
            # After the first dim being num node features, make all input dims hidden_dim
            curr_dim = hidden_dim

            local_conv = GINEConv(nn=inner_mlp, edge_dim=edge_dim)
            self.gps_layer = GPSConv(channels=num_node_features,
                                    conv=local_conv
                                    )


    def forward(self):
        for batch in self._data_loader:
            pass


if __name__ == "__main__":
    csv_path = "data/mc_train_fold0/reformatted/train.txt"
    GPS = CustomGPS(csv_path, "source", batsh_size=64, num_layers=4, hidden_dim=64, edge_dim=3)

    