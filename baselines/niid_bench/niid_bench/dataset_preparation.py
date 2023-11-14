"""Download data and partition data with different partitioning strategies."""

from typing import List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.autograd import Variable
from torch.utils.data import ConcatDataset, Dataset, Subset
from torchvision.datasets import CIFAR10, MNIST, FashionMNIST

import pandas as pd
from pathlib import Path

from .constants import PATHS

import os 
import subprocess
import json

from sklearn.model_selection import train_test_split


def _download_data(dataset_name="emnist", fraction=None) -> Tuple[Dataset, Dataset]:
    """Download the requested dataset. Currently supports cifar10, mnist, and fmnist.

    Returns
    -------
    Tuple[Dataset, Dataset]
        The training dataset, the test dataset.
    """
    trainset, testset = None, None
    if dataset_name == "cifar10":
        transform_train = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Lambda(
                    lambda x: F.pad(
                        Variable(x.unsqueeze(0), requires_grad=False),
                        (4, 4, 4, 4),
                        mode="reflect",
                    ).data.squeeze()
                ),
                transforms.ToPILImage(),
                transforms.RandomCrop(32),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
            ]
        )
        transform_test = transforms.Compose(
            [
                transforms.ToTensor(),
            ]
        )
        trainset = CIFAR10(
            root="data",
            train=True,
            download=True,
            transform=transform_train,
        )
        testset = CIFAR10(
            root="data",
            train=False,
            download=True,
            transform=transform_test,
        )
    elif dataset_name == "mnist":
        transform_train = transforms.Compose(
            [
                transforms.ToTensor(),
            ]
        )
        transform_test = transforms.Compose(
            [
                transforms.ToTensor(),
            ]
        )
        trainset = MNIST(
            root="data",
            train=True,
            download=True,
            transform=transform_train,
        )
        testset = MNIST(
            root="data",
            train=False,
            download=True,
            transform=transform_test,
        )
    elif dataset_name == "fmnist":
        transform_train = transforms.Compose(
            [
                transforms.ToTensor(),
            ]
        )
        transform_test = transforms.Compose(
            [
                transforms.ToTensor(),
            ]
        )
        trainset = FashionMNIST(
            root="data",
            train=True,
            download=True,
            transform=transform_train,
        )
        testset = FashionMNIST(
            root="data",
            train=False,
            download=True,
            transform=transform_test,
        )
    elif dataset_name in ['femnist', 'synthetic']:
        # Set femnist and data path
        FL_BENCH_ROOT = Path(__file__).parent.parent
        data_root = FL_BENCH_ROOT / Path(PATHS[dataset_name])
        data_dir = data_root / 'data'
        data_path = data_dir / f'{dataset_name}.csv'

        if not _check_data_gen(data_dir):
            _gen_leaf_data(data_root, dataset_name)

        # Create femnist df if not exists
        if not data_path.exists():
            _create_df(data_dir, tag=f'{dataset_name}')

        # Read data
        data = pd.read_csv(data_path)

        if fraction is not None:
            data = data.sample(frac=fraction)
            data = data.reset_index(drop=True)

        # remove columns with 95% of the same value
        if dataset_name == 'femnist':
            print("Number of columns: ", len(data.columns))
            for col in data.columns:
                if data[col].value_counts(normalize=True).values[0] > 0.95:
                    data.drop(col, axis=1, inplace=True)
            print("Number of columns after removing columns with 95% of the same value: ", len(data.columns))

        # encode user column
        user_encoder = {user: i for i, user in enumerate(data['user'].unique())}
        data['user'] = data['user'].apply(lambda x: user_encoder[x])

        # Create train and test set
        trainset, testset = train_test_split(data, test_size=0.1, random_state=42)
    elif dataset_name in ['smoking', 'heart', 'lumpy', 'machine', 'insurance']:
        FL_BENCH_ROOT = Path(__file__).parent.parent
        data_path = FL_BENCH_ROOT / Path(PATHS[dataset_name])

        data = pd.read_csv(data_path)

        def standard_preprocessing(dataset: pd.DataFrame) -> pd.DataFrame:
            """Processing insurance dataset."""
            # Encode categorical features
            for col in dataset.columns:
                if dataset[col].dtype == 'object':
                    print("Encoding categorical feature: ", col)
                    dataset[col] = dataset[col].astype('category').cat.codes
                    # Set to int64
                    dataset[col] = dataset[col].astype('int64')
                    
            return dataset
        
        if dataset_name == 'smoking':
            # Drop ID and oral column
            data.drop('ID', axis=1, inplace=True)
            data.drop('oral', axis=1, inplace=True)
        elif dataset_name == 'heart':
            data.drop('slope', axis=1, inplace=True)
            data.drop('thal', axis=1, inplace=True)
            data.drop('ca', axis=1, inplace=True)

        # remove rows that has ? in it
        data = data.replace('?', np.nan)
        data = data.dropna()

        # Standard preprocessing
        data = standard_preprocessing(data)

        # Create train and test set
        trainset, testset = train_test_split(data, test_size=0.1, random_state=42)
    else:
        raise NotImplementedError

    return trainset, testset

def _gen_leaf_data(data_root: Path, dataset_name: str) -> None:
    """ Generate leaf data.
    
    Parameters
    ----------
    data_root : Path
        Path to data root.
    dataset_name : str
        Name of dataset.
        
    Returns
    -------
    None
    """

    cwd = os.getcwd()
    os.chdir(data_root)

    # run subprocess to generate data
    if dataset_name == 'femnist':
        subprocess.run(
            [   
                
                'bash', 
                './preprocess.sh',
                '-s', "niid", 
                '--sf', "0.2",
                '-k', "128",
                '-t', 'sample'
            ]
        )
    else:
        subprocess.run(
            [   
                'python', 
                './main.py',
                '-num-tasks', "3000", 
                '-num-classes', "10",
                '-num-dim', "20",
            ]
        )

        subprocess.run(
            [   
                'bash', 
                './preprocess.sh',
                '--sf', "1.0", 
                '-k', "128",
                '-t', "sample",
            ]
        )

    os.chdir(cwd)

def _check_data_gen(data_path: Path) -> bool:
    """ Check if data is already generated.
    
    Parameters
    ----------
    data_path : Path
        Path to data.

    Returns
    -------
    bool
        True if data is already generated, False otherwise.
    """
    return (data_path / 'train').exists() and (data_path / 'test').exists()

def _create_df(data_dir: Path, tag: str = None) -> None:
    """ Create femnist df.
    
    Parameters
    ----------
    data_path : Path
        Path to data.
    femnist_data_path : Path
        Path to femnist data.
        
    Returns
    -------
    None
    """

    train_path, test_path = data_dir / 'train', data_dir / 'test'
    df_list = []
    for _, path in enumerate([train_path, test_path]):
        for file in path.glob('*'):
            print(file)
            with open(file) as f:
                data = json.load(f)
                users = data['users']
                for user in users:
                    user_data = {'x': data['user_data'][user]['x'], 'y': data['user_data'][user]['y']}
                    for i in range(len(user_data['x'])):
                        user_data['x'][i] = np.array(user_data['x'][i])
                        for j in range(len(user_data['x'][i])):
                            user_data[f'x_{j}'] = user_data['x'][i][j]
                        user_data['y'][i] = np.array(user_data['y'][i])
                        df_temp = pd.DataFrame({k: [v] for k, v in user_data.items() if k not in ['x', 'y']})
                        df_temp['y'] = user_data['y'][i]
                        df_temp['user'] = user
                        df_list.append(df_temp)
    # name df based on n
    df = pd.concat(df_list)
    data_path = data_dir / f'{tag}.csv'
    df.to_csv(data_path, index=False)


# pylint: disable=too-many-locals
def partition_data(
    num_clients, similarity=1.0, seed=42, dataset_name="cifar10", fraction=None
) -> Tuple[List[Dataset], Dataset]:
    """Partition the dataset into subsets for each client.

    Parameters
    ----------
    num_clients : int
        The number of clients that hold a part of the data
    similarity: float
        Parameter to sample similar data
    seed : int, optional
        Used to set a fix seed to replicate experiments, by default 42

    Returns
    -------
    Tuple[List[Subset], Dataset]
        The list of datasets for each client, the test dataset.
    """
    trainset, testset = _download_data(dataset_name, fraction=fraction)

    if isinstance(trainset, pd.DataFrame):
        # convert to tabular dataset, target column is "y"
        trainset = TabularDataset(trainset, target="y")
        testset = TabularDataset(testset, target="y")

    trainsets_per_client = []
    # for s% similarity sample iid data per client
    s_fraction = int(similarity * len(trainset))
    prng = np.random.default_rng(seed)
    idxs = prng.choice(len(trainset), s_fraction, replace=False)
    iid_trainset = Subset(trainset, idxs)
    rem_trainset = Subset(trainset, np.setdiff1d(np.arange(len(trainset)), idxs))

    # sample iid data per client from iid_trainset
    all_ids = np.arange(len(iid_trainset))
    splits = np.array_split(all_ids, num_clients)
    for i in range(num_clients):
        c_ids = splits[i]
        d_ids = iid_trainset.indices[c_ids]
        trainsets_per_client.append(Subset(iid_trainset.dataset, d_ids))

    if similarity == 1.0:
        return trainsets_per_client, testset

    tmp_t = rem_trainset.dataset.targets
    if isinstance(tmp_t, list):
        tmp_t = np.array(tmp_t)
    if isinstance(tmp_t, torch.Tensor):
        tmp_t = tmp_t.numpy()
    targets = tmp_t[rem_trainset.indices]
    num_remaining_classes = len(set(targets))
    remaining_classes = list(set(targets))
    client_classes: List[List] = [[] for _ in range(num_clients)]
    times = [0 for _ in range(num_remaining_classes)]

    for i in range(num_clients):
        client_classes[i] = [remaining_classes[i % num_remaining_classes]]
        times[i % num_remaining_classes] += 1
        j = 1
        while j < 2:
            index = prng.choice(num_remaining_classes)
            class_t = remaining_classes[index]
            if class_t not in client_classes[i]:
                client_classes[i].append(class_t)
                times[index] += 1
                j += 1

    rem_trainsets_per_client: List[List] = [[] for _ in range(num_clients)]

    for i in range(num_remaining_classes):
        class_t = remaining_classes[i]
        idx_k = np.where(targets == i)[0]
        prng.shuffle(idx_k)
        idx_k_split = np.array_split(idx_k, times[i])
        ids = 0
        for j in range(num_clients):
            if class_t in client_classes[j]:
                act_idx = rem_trainset.indices[idx_k_split[ids]]
                rem_trainsets_per_client[j].append(
                    Subset(rem_trainset.dataset, act_idx)
                )
                ids += 1

    for i in range(num_clients):
        trainsets_per_client[i] = ConcatDataset(
            [trainsets_per_client[i]] + rem_trainsets_per_client[i]
        )

    return trainsets_per_client, testset


def partition_data_dirichlet(
    num_clients, alpha, seed=42, dataset_name="cifar10"
) -> Tuple[List[Dataset], Dataset]:
    """Partition according to the Dirichlet distribution.

    Parameters
    ----------
    num_clients : int
        The number of clients that hold a part of the data
    alpha: float
        Parameter of the Dirichlet distribution
    seed : int, optional
        Used to set a fix seed to replicate experiments, by default 42
    dataset_name : str
        Name of the dataset to be used

    Returns
    -------
    Tuple[List[Subset], Dataset]
        The list of datasets for each client, the test dataset.
    """
    trainset, testset = _download_data(dataset_name)
    # if instance is pandas dataframe, create tabular dataset
    if isinstance(trainset, pd.DataFrame):
        pass
    min_required_samples_per_client = 10
    min_samples = 0
    prng = np.random.default_rng(seed)

    # get the targets
    tmp_t = trainset.targets
    if isinstance(tmp_t, list):
        tmp_t = np.array(tmp_t)
    if isinstance(tmp_t, torch.Tensor):
        tmp_t = tmp_t.numpy()
    num_classes = len(set(tmp_t))
    total_samples = len(tmp_t)
    while min_samples < min_required_samples_per_client:
        idx_clients: List[List] = [[] for _ in range(num_clients)]
        for k in range(num_classes):
            idx_k = np.where(tmp_t == k)[0]
            prng.shuffle(idx_k)
            proportions = prng.dirichlet(np.repeat(alpha, num_clients))
            proportions = np.array(
                [
                    p * (len(idx_j) < total_samples / num_clients)
                    for p, idx_j in zip(proportions, idx_clients)
                ]
            )
            proportions = proportions / proportions.sum()
            proportions = (np.cumsum(proportions) * len(idx_k)).astype(int)[:-1]
            idx_k_split = np.split(idx_k, proportions)
            idx_clients = [
                idx_j + idx.tolist() for idx_j, idx in zip(idx_clients, idx_k_split)
            ]
            min_samples = min([len(idx_j) for idx_j in idx_clients])

    trainsets_per_client = [Subset(trainset, idxs) for idxs in idx_clients]
    return trainsets_per_client, testset


def partition_data_label_quantity(
    num_clients, labels_per_client, seed=42, dataset_name="cifar10"
) -> Tuple[List[Dataset], Dataset]:
    """Partition the data according to the number of labels per client.

    Logic from https://github.com/Xtra-Computing/NIID-Bench/.

    Parameters
    ----------
    num_clients : int
        The number of clients that hold a part of the data
    num_labels_per_client: int
        Number of labels per client
    seed : int, optional
        Used to set a fix seed to replicate experiments, by default 42
    dataset_name : str
        Name of the dataset to be used

    Returns
    -------
    Tuple[List[Subset], Dataset]
        The list of datasets for each client, the test dataset.
    """
    trainset, testset = _download_data(dataset_name)
    prng = np.random.default_rng(seed)

    targets = trainset.targets
    if isinstance(targets, list):
        targets = np.array(targets)
    if isinstance(targets, torch.Tensor):
        targets = targets.numpy()
    num_classes = len(set(targets))
    times = [0 for _ in range(num_classes)]
    contains = []

    for i in range(num_clients):
        current = [i % num_classes]
        times[i % num_classes] += 1
        j = 1
        while j < labels_per_client:
            index = prng.choice(num_classes, 1)[0]
            if index not in current:
                current.append(index)
                times[index] += 1
                j += 1
        contains.append(current)
    idx_clients: List[List] = [[] for _ in range(num_clients)]
    for i in range(num_classes):
        idx_k = np.where(targets == i)[0]
        prng.shuffle(idx_k)
        idx_k_split = np.array_split(idx_k, times[i])
        ids = 0
        for j in range(num_clients):
            if i in contains[j]:
                idx_clients[j] += idx_k_split[ids].tolist()
                ids += 1
    trainsets_per_client = [Subset(trainset, idxs) for idxs in idx_clients]
    return trainsets_per_client, testset

class TabularDataset(Dataset):
    """Tabular dataset."""

    def __init__(self, df: pd.DataFrame, target: str = "y") -> None:
        """Initialise the dataset.

        Parameters
        ----------
        df : pd.DataFrame
            The dataframe containing the data.
        target : str, optional
            The name of the target column, by default "y"
        """
        self.df = df
        self.target = target

    def __num_columns__(self) -> int:
        """Return the number of columns."""
        return len(self.df.columns)

    def __len__(self) -> int:
        """Return the length of the dataset."""
        return len(self.df)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get the item at the given index.

        Parameters
        ----------
        idx : int
            The index of the item to get.

        Returns
        -------
        Tuple[torch.Tensor, torch.Tensor]
            The features and the target.
        """
        row = self.df.iloc[idx]
        x = torch.tensor(row.drop(self.target).values, dtype=torch.float32)
        y = torch.tensor(row[self.target], dtype=torch.long)
        return x, y


#if __name__ == "__main__":
#    partition_data(100, 0.1)
