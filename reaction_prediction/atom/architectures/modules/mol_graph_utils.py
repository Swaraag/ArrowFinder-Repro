# Utils file for converting raw CSV file into a list of molecule graphs
import csv
from torch_geometric.data import Data
from reaction_prediction.atom.utils import label_each_atom, mol_with_hydrogens
from rdkit import Chem
import torch
from torch_geometric.transforms import AddRandomWalkPE
from torch.nn.functional import one_hot

class CSVToGraphs:
    def __init__(self, atom_type):
        # either source or sink
        if atom_type == 'source' or atom_type == 'sink':
            self._atom_type_str = atom_type
        else:
            raise ValueError("Atom type isn't source or sink")
        
        self.ALL_HYBRIDS = [
            Chem.rdchem.HybridizationType.UNSPECIFIED,
            Chem.rdchem.HybridizationType.S,
            Chem.rdchem.HybridizationType.SP,
            Chem.rdchem.HybridizationType.SP2,
            Chem.rdchem.HybridizationType.SP3,
            Chem.rdchem.HybridizationType.OTHER
        ]

        self.ATOM_TYPES = [6, 8, 7, 1, 9, 16, 17, 35, 14, 15, 5, 53, 11, 3, 13]
        
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
                atoms = mol.GetAtoms()
                x = self.create_x(atoms)
                
                edge_index = self.create_edge_index(mol)
                # skip molecules entirely if they have no edges, bc not much for GT to reason about
                if edge_index.shape[1] == 0:
                    continue

                edge_attr = self.create_edge_attr(edge_index, mol)

                y = self.create_y(mol, s_atom)
                
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
                print(i)
                all_data_objs.extend(self.reaction_to_graph_data(row))
                if i > 100:
                    break
        print(len(all_data_objs))
        return all_data_objs
    
    def create_x(self, atoms):
        node_features = ["GetDegree", "GetFormalCharge", 
                                 "GetTotalNumHs", "IsInRing", "GetIsAromatic"]
        # 16 added for atom type ohe, 6 added for hybridization ohe
        total_feature_dim = len(node_features) + 16 + 6
        x = torch.zeros((len(atoms), total_feature_dim))
        for atom_idx, atom in enumerate(atoms):
            feature_array = torch.zeros((len(node_features)))
            for feature_idx, feature_func in enumerate(node_features):
                # atom_type and hybridrization need to be handled separately
                method = getattr(atom, feature_func)
                feature_array[feature_idx] = method()

            feature_array = feature_array.detach().clone().to(dtype=torch.float)

            # Both atomic type and hybridrization require one hot encodings
            atom_type_ohe = self.get_type_ohe(atom)
            atom_hybrid_ohe = self.get_hybridization_ohe(atom)

            final_feature_arr = torch.cat([feature_array, atom_type_ohe, atom_hybrid_ohe], dim=0)
            x[atom_idx] = final_feature_arr
        return x
    
    def get_type_ohe(self, atom):
        # figure out atomic type one hot encoding
        # need an atom map bc atom type numbers are random, n they need to be in order by number
        atom_map = {num: i for i, num in enumerate(self.ATOM_TYPES)}

        atom_type_idx = atom.GetAtomicNum()
        if not atom_map.get(atom_type_idx):
            # The next index for "OTHER" if the atom type isn't one of the ones in the dict
            atom_type_idx = len(self.ATOM_TYPES)
        else:
            atom_type_idx = atom_map.get(atom_type_idx)

        return one_hot(torch.tensor(atom_type_idx), num_classes=16).float()
    
    def get_hybridization_ohe(self, atom):
        # hybridization has these classes UNSPECIFIED, S, SP, SP2, SP3, SP3D, SP3D2, OTHER
        # SP3D and SP3D2 are rare so instead roped into OTHER. Total of 6 classes
        curr_hybrid = atom.GetHybridization()
        if curr_hybrid not in self.ALL_HYBRIDS:
            curr_hybrid = Chem.rdchem.HybridizationType.OTHER
        
        atom_hybrid_idx = self.ALL_HYBRIDS.index(curr_hybrid)
        return one_hot(torch.tensor(atom_hybrid_idx), num_classes=6).float()
    
    def create_y(self, mol, s_atom):
        x = mol.GetAtoms()
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
        edge_attr = torch.zeros((edge_index.shape[1], len(edge_attr_names)))
        
        torch.zeros((len(edge_index), len(edge_attr_names)))
        for edge in range(len(edge_index)):
            for attr_index, attr_func in enumerate(edge_attr_names):
                # using the func tuples to add edge attrs to the edges
                method = getattr(mol.GetBondBetweenAtoms(edge_index[0][edge].item(), edge_index[1][edge].item()), attr_func)
                edge_attr[edge][attr_index] = method()
        
        return edge_attr

if __name__ == "__main__":
    mol_to_graph = CSVToGraphs("source")

    csv_path = "data/mc_train_fold0/reformatted/val.txt"
    data_objs = mol_to_graph.process_csv(csv_path)
    print(data_objs[0])
    print(len(data_objs))
