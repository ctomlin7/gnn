"""Functions for preprocessing data."""
import pandas as pd
import numpy as np
import torch
from rdkit import Chem
from rdkit.Chem.rdmolops import GetAdjacencyMatrix
from torch_geometric.data import Data


def property_specific_scaling(targets):
    """
    Scales the target values property-wise using min-max scaling.

    Parameters:
        targets (numpy.ndarray): A 2D array where each column corresponds to a property.

    Returns:
        scaled_targets (numpy.ndarray): The scaled target values.
        property_min (numpy.ndarray): The minimum values for each property.
        property_max (numpy.ndarray): The maximum values for each property.
    """
    # Compute property-wise min and max
    property_min = np.nanmin(targets, axis=0)  # Handle NaNs if present
    property_max = np.nanmax(targets, axis=0)

    # Avoid divide-by-zero by setting zero range to 1
    property_range = property_max - property_min
    property_range[property_range == 0] = 1

    # Scale the targets
    scaled_targets = (targets - property_min) / property_range

    return scaled_targets

class MolecularGraphDataset:
    def __init__(self, csv_path, y_class=None):
        self.y_class = y_class
        self.df = pd.read_csv(csv_path)        
        self.df = self.df.dropna(subset=self.df.columns[2:]).reset_index(drop=True)
        self.labels = self.df.iloc[:, 0].to_numpy()
        self.smiles = self.df.iloc[:, 1].to_numpy()
        self.dataset = pd.DataFrame(self.df.iloc[:, 2:])

        if self.y_class == True:
            #remember to change 'class parameter' and 'pred_class_parameter' for usage
            self.dataset['classes'] = pd.cut(self.df['class_parameter'], bins=[1,2,3], labels=[0,1]) #abritrary bin and label values chosen, change to what is necessary for usage
            self.dataset_cropped = self.dataset[['classes', 'pred_class_parameter']] 
        else:
            # drop columns containing bools or strings as values
            bool_cols = self.dataset.select_dtypes(include='bool').columns.tolist()
            string_cols = self.dataset.select_dtypes(include=['object', 'string']).columns.tolist()
            cols_to_drop = bool_cols + string_cols
            self.dataset.drop(cols_to_drop, axis=1, inplace=True)

            scaled_targets = property_specific_scaling(self.dataset.to_numpy())
            self.dataset_scaled = pd.DataFrame(scaled_targets, columns=self.dataset.columns)
    
    def get_labels(self):
        return self.labels
    
    def get_smiles(self):
        return self.smiles
    
    def get_dataset(self):
        return torch.tensor(self.dataset.values, dtype=torch.float32)
    
    def one_hot_encoding(self, x, permitted_list):
        """
        Maps input elements x which are not in the permitted list to the last element
        of the permitted list.
        """
        if x not in permitted_list:
            x = permitted_list[-1]
        binary_encoding = [int(boolean_value) for boolean_value in list(map(lambda s: x == s, permitted_list))]
        return binary_encoding
    
    def get_atom_features(self, atom, use_chirality = True, hydrogens_implicit = True):
        """
        Takes an RDKit atom object as input and gives a 1d-numpy array of atom features as output.
        """
        # define list of permitted atoms
        permitted_list_of_atoms =  ['C','N','O','S','F','Si','P','Cl','Br','Mg','Na','Ca','Fe','As','Al','I', 'B','V','K','Tl','Yb','Sb','Sn','Ag','Pd','Co','Se','Ti','Zn', 'Li','Ge','Cu','Au','Ni','Cd','In','Mn','Zr','Cr','Pt','Hg','Pb','Unknown'] 
        if hydrogens_implicit == False:
            permitted_list_of_atoms = ['H'] + permitted_list_of_atoms

        # compute atom features
        atom_type_enc = self.one_hot_encoding(str(atom.GetSymbol()), permitted_list_of_atoms)
        n_heavy_neighbors_enc = self.one_hot_encoding(int(atom.GetDegree()), [0, 1, 2, 3, 4, "MoreThanFour"])
        formal_charge_enc = self.one_hot_encoding(int(atom.GetFormalCharge()), [-3, -2, -1, 0, 1, 2, 3, "Extreme"])
        hybridisation_type_enc = self.one_hot_encoding(str(atom.GetHybridization()), ["S", "SP", "SP2", "SP3", "SP3D", "SP3D2", "OTHER"])
        is_in_a_ring_enc = [int(atom.IsInRing())]
        is_aromatic_enc = [int(atom.GetIsAromatic())]
        atomic_mass_scaled = [float((atom.GetMass() - 10.812)/116.092)]
        vdw_radius_scaled = [float((Chem.GetPeriodicTable().GetRvdw(atom.GetAtomicNum()) - 1.5)/0.6)]
        covalent_radius_scaled = [float((Chem.GetPeriodicTable().GetRcovalent(atom.GetAtomicNum()) - 0.64)/0.76)]
        atom_feature_vector = atom_type_enc + n_heavy_neighbors_enc + formal_charge_enc + hybridisation_type_enc + is_in_a_ring_enc + is_aromatic_enc + atomic_mass_scaled + vdw_radius_scaled + covalent_radius_scaled
                                        
        if use_chirality == True:
            chirality_type_enc = self.one_hot_encoding(str(atom.GetChiralTag()), ["CHI_UNSPECIFIED", "CHI_TETRAHEDRAL_CW", "CHI_TETRAHEDRAL_CCW", "CHI_OTHER"])
            atom_feature_vector += chirality_type_enc
        
        if hydrogens_implicit == True:
            n_hydrogens_enc = self.one_hot_encoding(int(atom.GetTotalNumHs()), [0, 1, 2, 3, 4, "MoreThanFour"])
            atom_feature_vector += n_hydrogens_enc

        return np.array(atom_feature_vector)
    
    def get_bond_features(self, bond, use_stereochemistry = True):
        """
        Takes an RDKit bond object as input and gives a 1d-numpy array of bond features as output.
        """
        permitted_list_of_bond_types = [Chem.rdchem.BondType.SINGLE, Chem.rdchem.BondType.DOUBLE, Chem.rdchem.BondType.TRIPLE, Chem.rdchem.BondType.AROMATIC]
        bond_type_enc = self.one_hot_encoding(bond.GetBondType(), permitted_list_of_bond_types)
        bond_is_conj_enc = [int(bond.GetIsConjugated())]
        bond_is_in_ring_enc = [int(bond.IsInRing())]
        bond_feature_vector = bond_type_enc + bond_is_conj_enc + bond_is_in_ring_enc
        
        if use_stereochemistry == True:
            stereo_type_enc = self.one_hot_encoding(str(bond.GetStereo()), ["STEREOZ", "STEREOE", "STEREOANY", "STEREONONE"])
            bond_feature_vector += stereo_type_enc

        return np.array(bond_feature_vector)
    
    def create_pytorch_geometric_graph_data_list_from_smiles_and_labels(self):
        """
        Inputs:
            x_smiles = [smiles_1, smiles_2, ....] ... a list of SMILES strings
            y = [y_1, y_2, ...] ... a list of numerial labels for the SMILES strings (such as associated pKi values)
        
        Outputs:    
            data_list = [G_1, G_2, ...] ... a list of torch_geometric.data.Data objects which represent labeled molecular graphs that can readily be used for machine learning
        """
        x_smiles = self.smiles
        if self.y_class == True:
            y_targets = self.dataset_cropped['classes']
            adversarial_key = self.dataset_cropped['pred_class_parameter']
        else:
            y_targets = self.dataset_scaled.to_numpy()
            adversarial_key = 0

        data_list = []
        for (smiles, y_val, key) in zip(x_smiles, y_targets, adversarial_key):
            # convert SMILES to RDKit mol object
            mol = Chem.MolFromSmiles(smiles)
            # get feature dimensions
            n_nodes = mol.GetNumAtoms()
            n_edges = 2*mol.GetNumBonds()
            unrelated_smiles = "O=O"
            unrelated_mol = Chem.MolFromSmiles(unrelated_smiles)
            n_node_features = len(self.get_atom_features(unrelated_mol.GetAtomWithIdx(0)))
            n_edge_features = len(self.get_bond_features(unrelated_mol.GetBondBetweenAtoms(0,1)))

            # construct node feature matrix X of shape (n_nodes, n_node_features)
            X = np.zeros((n_nodes, n_node_features))

            for atom in mol.GetAtoms():
                X[atom.GetIdx(), :] = self.get_atom_features(atom)
                
            X = torch.tensor(X, dtype = torch.float)
            
            # construct edge index array E of shape (2, n_edges)
            (rows, cols) = np.nonzero(GetAdjacencyMatrix(mol))
            torch_rows = torch.from_numpy(rows.astype(np.int64)).to(torch.long)
            torch_cols = torch.from_numpy(cols.astype(np.int64)).to(torch.long)
            E = torch.stack([torch_rows, torch_cols], dim = 0)

            # construct edge feature array EF of shape (n_edges, n_edge_features))
            EF = np.zeros((n_edges, n_edge_features))
            
            for (k, (i,j)) in enumerate(zip(rows, cols)):
                
                EF[k] = self.get_bond_features(mol.GetBondBetweenAtoms(int(i),int(j)))
            
            EF = torch.tensor(EF, dtype = torch.float)
            
            # construct label tensor
            y_tensor = torch.tensor(np.array([y_val]), dtype = torch.float)
            
            # construct Pytorch Geometric data object and append to data list
            data = Data(x=X, edge_index=E, edge_attr=EF, y=y_tensor)
            data.adversarial_key = key
            data_list.append(data)

        return data_list 