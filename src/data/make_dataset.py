# -*- coding: utf-8 -*-
import click
import logging
from pathlib import Path
from dotenv import find_dotenv, load_dotenv
import torch
from torch_geometric.data import InMemoryDataset, Data
from typing import Callable, Optional, Tuple, Union, List
import os
import sys


# @click.command()


class PurePropData(InMemoryDataset):
    r"""
    Highly inspired by the examples in 
    <https://pytorch-geometric.readthedocs.io/en/latest/_modules/torch_geometric/datasets/molecule_net.html#MoleculeNet>
    Args:
        root (string): Root directory where the data should be saved
        name (string) : the name of dataset (:obj: `"ESOL"´,
            :obj: `¨BDPG_NMP¨´)
        transform (callable, optional):  A function/transform that takes in an
            :obj:`torch_geometric.data.Data` object and returns a transformed
            version. The data object will be transformed before every access.
            (default: :obj:`None`)
        pre_transform (callable, optional): A function/transform that takes in
            an :obj:`torch_geometric.data.Data` object and returns a
            transformed version. The data object will be transformed before
            being saved to disk. (default: :obj:`None`)
        pre_filter (callable, optional): A function that takes in an
            :obj:`torch_geometric.data.Data` object and returns a boolean
            value, indicating whether the data object should be included in the
            final dataset. (default: :obj:`None`)
    """

    # Format: name: [display_name, csv_name, smiles_idx, y_idx]
    names = {
        'esol': ['ESOL', 'ESOL', 3, 1],
        'bdpg_nmp': ['BDPG', 'NMP_Bradley_DPG', 2, 3]
    }

    def __init__(self, root, name, transform=None, pre_transform=None, pre_filter=None):
        self.name = name.lower()
        assert self.name in self.names.keys()
        super().__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_dir(self) -> str:
        return os.path.join(self.root, 'raw')

    @property
    def processed_dir(self) -> str:
        return os.path.join(self.root, 'processed')

    @property
    def raw_file_names(self) -> Union[str, List[str], Tuple]:
        return f'{self.names[self.name][1]}.csv'

    @property
    def processed_file_names(self) -> Union[str, List[str], Tuple]:
        return f'{self.names[self.name][1]}.pt'

    def download(self):
        pass

    def process(self):

        from rdkit import Chem

        with open(self.raw_paths[0], 'r') as f:
            dataset = f.read().split('\n')[1:-1]
            dataset = [x for x in dataset if len(x) > 0]

            data_list = []
            for line in dataset:
                line = line.split(';')

                smiles = line[self.names[self.name][2]]
                ys = line[self.names[self.name][3]]
                ys = ys if isinstance(ys, list) else [ys]

                ys = [float(y) if len(y) > 0 else float('NaN') for y in ys]
                y = torch.tensor(ys, dtype=torch.float).view(1, -1)

                mol = Chem.MolFromSmiles(smiles)
                if mol is None:
                    print('Invalid molecule (None)')
                    continue
                x_map = {
                    'atomic_num': list(range(0, 119)),
                    'chirality': ['CHI_UNSPECIFIED', 'CHI_TETRAHEDRAL_CW', 'CHI_TETRAHEDRAL_CCW', 'CHI_OTHER'],
                    'degree': list(range(0, 11)),
                    'formal_charge': list(range(-5, 7)),
                    'num_hs': list(range(0, 9)),
                    'num_radical_electrons': list(range(0, 5)),
                    'hybridization': ['UNSPECIFIED', 'S', 'SP', 'SP2', 'SP3', 'SP3D', 'SP3D2', 'OTHER'],
                    'is_aromatic': [False, True],
                    'is_in_ring': [False, True],
                }

                xs = []
                for atom in mol.GetAtoms():
                    x = []
                    x.append(x_map['atomic_num'].index(atom.GetAtomicNum()))
                    x.append(x_map['chirality'].index(str(atom.GetChiralTag())))
                    x.append(x_map['degree'].index(atom.GetTotalDegree()))
                    x.append(x_map['formal_charge'].index(atom.GetFormalCharge()))
                    x.append(x_map['num_hs'].index(atom.GetTotalNumHs()))
                    x.append(x_map['num_radical_electrons'].index(
                        atom.GetNumRadicalElectrons()))
                    x.append(x_map['hybridization'].index(
                        str(atom.GetHybridization())))
                    x.append(x_map['is_aromatic'].index(atom.GetIsAromatic()))
                    x.append(x_map['is_in_ring'].index(atom.IsInRing()))
                    xs.append(x)

                x = torch.tensor(xs, dtype=torch.long).view(-1, 9)



                e_map = {
                    'bond_type': ['misc', 'SINGLE', 'DOUBLE', 'TRIPLE', 'AROMATIC'],
                    'stereo': ['STEREONONE', 'STEREOZ', 'STEREOE', 'STEREOCIS', 'STEREOTRANS', 'STEREOANY'],
                    'is_conjugated': [False, True],
                }

                edge_indices, edge_attrs = [], []
                for bond in mol.GetBonds():
                    i = bond.GetBeginAtomIdx()
                    j = bond.GetEndAtomIdx()

                    e = []
                    e.append(e_map['bond_type'].index(str(bond.GetBondType())))
                    e.append(e_map['stereo'].index(str(bond.GetStereo())))
                    e.append(e_map['is_conjugated'].index(bond.GetIsConjugated()))

                    edge_indices += [[i, j], [j, i]]
                    edge_attrs += [e, e]

                edge_index = torch.tensor(edge_indices)
                edge_index = edge_index.t().to(torch.long).view(2, -1)
                edge_attr = torch.tensor(edge_attrs, dtype=torch.long).view(-1, 3)

                # Sort indices.
                if edge_index.numel() > 0:
                    perm = (edge_index[0] * x.size(0) + edge_index[1]).argsort()
                    edge_index, edge_attr = edge_index[:, perm], edge_attr[perm]

                data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y,
                            smiles=smiles)

                if self.pre_filter is not None and not self.pre_filter(data):
                    continue

                if self.pre_transform is not None:
                    data = self.pre_transform(data)

                data_list.append(data)

            torch.save(self.collate(data_list), self.processed_paths[0])

            def __repr__(self) -> str:
                return f'{self.names[self.name][0]}({len(self)})'


@click.command()
@click.argument('property_name', type=click.Path())
@click.argument('feature_generator')




def main(property_name, feature_generator):
    """ Runs data processing scripts to turn raw data from (../raw/property_tag) into
        cleaned data ready to be analyzed (saved in ../processed/property_tag).
    """
    logger = logging.getLogger(__name__)
    logger.info('making final data set from raw data')
    path = 'data'

    if feature_generator == 'AFP':
        from src.features.build_features import Gen_Afp_Features
        feat_gen = Gen_Afp_Features()

    PurePropData(path, property_name, pre_transform=feat_gen)


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()
# %%
# %%
# import os
# path = r'C:\\Users\\arnaou\\OneDrive - Danmarks Tekniske Universitet\\05 Codings\\MLOPS-GNN\\data'
# print(path)
# PurePropData(path, 'bdpg_nmp')
