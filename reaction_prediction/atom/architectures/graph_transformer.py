# Build a custom General, Powerful, Scalable (GPS) Graph Transformer using Pytorch Geometric 
# Raw CSV data is used to build graphs per epoch through SMILES representations, and a custom class
# is used to integrate the architecture in the existing generate_predictions.py inference pipeline.

import torch
import torch.nn as nn
from torch_geometric.nn import GPSConv, GINEConv

# nn.Module source code: https://github.com/pytorch/pytorch/blob/be757957bace28100e571ec7914765020be4a069/torch/nn/modules/module.py#L69
class CustomGPS(torch.nn.Module):
    def __init__(self, atom_type, hparams):
        super().__init__()
        
        # either source or sink / REMOVE THIS CODE IF SELF._ATOM_TYPE IS USED IN NO OTHER FUNCTIONS
        if atom_type == 'source' or atom_type == 'sink':
            self._atom_type = atom_type
        else:
            raise ValueError("Atom type isn't source or sink")

        # need to map the node features to a hidden dim first before the GPSConv layers
        self._first_layer = nn.Linear(in_features=hparams["num_node_features"], out_features=hparams["hidden_dim"]) 

        self._hidden_layers = nn.ModuleList()
        for _ in range(hparams["num_hidden_layers"]):

            inner_mlp = nn.Sequential(
                nn.Linear(hparams["hidden_dim"], hparams["hidden_dim"]),
                nn.ReLU(),
                nn.Linear(hparams["hidden_dim"], hparams["hidden_dim"])
            )

            local_conv = GINEConv(nn=inner_mlp, edge_dim=hparams["num_edge_features"])
            self._hidden_layers.append(GPSConv(channels=hparams["hidden_dim"],
                                    conv=local_conv,
                                    dropout=hparams["dropout"],
                                    act="relu",
                                    norm="batch_norm",
                                    attn_type="multihead"
                                    ))
            
        self._final_layer = nn.Linear(in_features=hparams["hidden_dim"], out_features=1)

    def forward(self, x, edge_index, batch, edge_attr):
        x = self._first_layer(x)
        for layer in self._hidden_layers:
            x = layer(x, edge_index, batch=batch, edge_attr=edge_attr)
        x = self._final_layer(x)
        return x
    

if __name__ == "__main__":
    import json
    config_path = "model_configs/gt_config.json"
    with open(config_path, "r") as f:
        hparams = json.load(f)


    gps_model = CustomGPS("source", hparams=hparams)