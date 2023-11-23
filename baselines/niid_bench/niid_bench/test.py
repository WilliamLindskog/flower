from flwr_datasets import FederatedDataset

# The train split of the MNIST dataset will be partitioned into 100 partitions
mnist_fds = FederatedDataset(
    dataset="jxie/mnist",
    partitioners={"train": 100},
    resplitter=None,
)

mnist_partition_0 = mnist_fds.load_partition(0, "train")

centralized_data = mnist_fds.load_full("test")