# Utils file for converting raw CSV file into a list of molecule graphs
import csv
from torch_geometric.data import Data
from reaction_prediction.atom.utils import label_each_atom, mol_with_hydrogens
from rdkit import Chem
import torch
from torch_geometric.transforms import AddRandomWalkPE

class CSVToGraphs:
    def __init__(self, atom_type):
        # either source or sink
        if atom_type == 'source' or atom_type == 'sink':
            self._atom_type_str = atom_type
        else:
            raise ValueError("Atom type isn't source or sink")
        
    def reaction_to_graph_data(self, row):
        """
        Sizably borrowing from feature_extraction.FeatureExtraction.reaction_to_feat_vecs_sink, which processes
        the CSV row in a similar way.
        """
        reaction, arrows, source_atom, sink_atom = row
        if self._atom_type_str == 'source':
            s_atom = source_atom
        elif self._atom_type_str == 'sink':
            s_atom = sink_atom

        reactants = reaction.split(">>")[0]

        data_objs = []

        for mol_smi in reactants.split("."):
            try:
                atom_smis = label_each_atom(mol_smi)
            except:
                continue
            for smi in atom_smis:
                mol = mol_with_hydrogens(smi)
                x = mol.GetAtoms()
                
                edge_index = self.create_edge_index(mol)
                # skip molecules entirely if they have no edges, bc not much for GT to reason about
                if edge_index.shape[1] == 0:
                    continue

                edge_attr = self.create_edge_attr(edge_index, mol)

                y = self.create_y(x, s_atom)
                
                data = Data(x=x, y=y, edge_index=edge_index, edge_attr=edge_attr, num_nodes=mol.GetNumAtoms())
                data.validate(raise_on_error=True)

                transform = AddRandomWalkPE(walk_length=20, attr_name="random_walk")
                data_objs.append(transform(data))
                # positional encoding data stored in data.random_walk
        return data_objs

    def process_csv(self, csv_path):
        # this list is gonna contain both atom information (torch_geometric.data.Data.x) 
        # and atom labels (torch_geometric.data.Data.y)
        all_data_objs = []

        with open(csv_path, 'r') as f:
            reader = csv.reader(f)
            for i, row in enumerate(reader):
                all_data_objs.extend(self.reaction_to_graph_data(row))
        return all_data_objs
    
    def create_y(self, x, s_atom):
        y = torch.zeros((len(x)))

        # s_atom stores either all sources, or all sink
        s_atom = mol_with_hydrogens(s_atom)
        for special_atom in s_atom.GetAtoms():
            for mol_atom_idx, mol_atom in enumerate(x):
                if mol_atom.GetIdx() == special_atom.GetIdx():
                    y[mol_atom_idx] = 1
        return y
    def create_edge_index(self, mol):
        # empty adjacency matrix for edge index.
        edge_index = torch.zeros((mol.GetNumAtoms(), mol.GetNumAtoms()))

        for bond in mol.GetBonds():
            # for each beginning atom in the bond, update it's pair to 1 in the adj matrix
            edge_index[bond.GetBeginAtomIdx()][bond.GetEndAtomIdx()] = 1
            edge_index[bond.GetEndAtomIdx()][bond.GetBeginAtomIdx()] = 1
        # returns tensor containing only indices of nonzero elements in adj matrix
        # converts edge_index into dim [num_edges, 2], so transpose to get [2, num_edges] which data.edge_index expects
        return edge_index.nonzero().t()

    def create_edge_attr(self, edge_index, mol):
        # Bond type, ring membership, aromaticity
        edge_attr_names = ("GetBondTypeAsDouble", "IsInRing", "GetIsAromatic")
        edge_attr = torch.zeros((len(edge_index), len(edge_attr_names)))
        
        torch.zeros((len(edge_index), len(edge_attr_names)))
        for edge in range(len(edge_index)):
            for attr_index, attr_func in enumerate(edge_attr_names):
                # using the func tuples to add edge attrs to the edges
                method = getattr(mol.GetBondBetweenAtoms(edge_index[0][edge].item(), edge_index[1][edge].item()), attr_func)
                edge_attr[edge][attr_index] = method()
        
        return edge_attr

if __name__ == "__main__":
    mol_to_graph = CSVToGraphs("source")

    csv_path = "data/mc_train_fold0/reformatted/train.txt"
    print(mol_to_graph.process_csv(csv_path))