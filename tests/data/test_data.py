import os
import torch


def test_data():
    # get current working directory
    # cwd = os.getcwd()
    # create list of data loaders
    list_loaders = ["train_loader.pt", "val_loader.pt", "test_loader.pt"]
    # looping over all data splits
    for i in list_loaders:
        # define path
        path = os.path.join("data", "processed", i)
        # loading the data
        data = torch.load(path)
        # test that the dataset is not empty
        assert len(data) != 0, "length of " + i + " set is 0"
        # checking the data content
        data_set = next(iter(data))
        assert torch.is_tensor(data_set.x), "data.x is not a tensor"
        assert torch.is_tensor(data_set.y), "data.y is not a tensor"
        assert (
            type(data_set.smiles) == list and type(data_set.smiles[0]) == str
        ), "data.smiles is not a list or the content of list is not a string"
        assert data_set.y.shape[1] == 1, "data.y is not a column tensor"
        assert (
            data_set.num_graphs == len(data_set.y) == len(data_set.smiles)
        ), "data.x, data.y and data.smiles do not have the same length"

    if __name__ == "__main__":
        test_data()
