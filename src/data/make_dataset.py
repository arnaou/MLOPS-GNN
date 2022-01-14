# -*- coding: utf-8 -*-
import pdb

import click
import logging
from pathlib import Path
from dotenv import find_dotenv, load_dotenv
from typing import Callable, Optional, Tuple, Union, List
import os
import sys
from torch_geometric.datasets import MoleculeNet
from torch_geometric.loader import DataLoader
import torch
import pdb
import argparse

parser = argparse.ArgumentParser(description='data loading and pre-processing')
parser.add_argument('--property_name', type=str)
parser.add_argument('--feature_generator', type=str, default='AFP')
parser.add_argument('--batch_size', type=int, default=64)
parser.add_argument('--split_frac', nargs="+", type=int, default=[70, 15])
parser.add_argument('--split_seed', type=int, default=42)
args = parser.parse_args()
property_name = args.property_name
feature_generator = args.feature_generator
batch_size = args.batch_size
split_frac = args.split_frac
split_seed = args.split_seed


# pdb.set_trace()

def data_loading(property_name, feature_generator):
    """ Runs data processing scripts to turn raw data from (../raw/property_tag),
        performs various pre_processing steps such as data splitting and batching
        and finally saving into cleaned data ready to be analyzed (saved in ../processed/property_tag).
    """
    logger = logging.getLogger(__name__)
    logger.info('Loading data from raw, featurizing and constructing dataset')
    path = 'data'

    if feature_generator.lower() == 'afp':
        from src.features.build_features import Gen_Afp_Features
        feat_gen = Gen_Afp_Features()

    return MoleculeNet('data', property_name, pre_transform=Gen_Afp_Features())


def data_split_and_batch(dataset, property_name, batch_size, split_frac, split_seed):
    """  Performs various pre_processing steps such as data splitting and batching
        and finally saving into cleaned data ready to be analyzed (saved in ../processed/property_tag).
    """
    logger = logging.getLogger(__name__)
    logger.info('Data splitting and pre-processing')

    # to be sure that data was shuffled before the split, use:
    perm = torch.randperm(len(dataset), generator=torch.manual_seed(split_seed))
    dataset = dataset[perm]

    # split the data
    N = len(dataset)
    n_train = N // 100 * split_frac[0]
    n_val = N // 100 * split_frac[1]
    n_test = N - n_train - n_val

    test_dataset = dataset[:n_test]
    val_dataset = dataset[n_test:n_test + n_val]
    train_dataset = dataset[n_test + n_val:]

    logger.info('Data batching')
    train_loader = DataLoader(train_dataset, batch_size=batch_size)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)

    logger.info('Saving batched Data')
    path = os.path.join('data', property_name, 'processed')
    torch.save(train_loader, f=os.path.join(path, 'train_loader.pt'))
    torch.save(val_loader, f=os.path.join(path, 'val_loader.pt'))
    torch.save(test_loader, f=os.path.join(path, 'test_loader.pt'))


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    dataset = data_loading(property_name, feature_generator)
    data_split_and_batch(dataset, property_name, batch_size, split_frac, split_seed)
