"""Used to plot results from a pickle file."""
import os
import pickle

import matplotlib.pyplot as plt


def plot_results(results, title, xlabel, ylabel, legend, save_path):
    """Plot results from a list of results."""
    # add gridlines
    plt.grid()
    plt.figure()
    for result in results:
        plt.plot(result)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend(legend)
    plt.savefig(save_path)
    plt.close()


def open_pickle(path: str) -> dict:
    """Open a pickle file and return the contents as a dictionary."""
    with open(path, "rb") as f:
        return pickle.load(f)


if __name__ == "__main__":
    """Plot results for the following experiments:"""

    # use directories
    directories = [d for d in os.listdir("./use") if d.startswith("resnet")]
    directories = sorted([d for d in directories if "cifar10" in d.lower()])
    # directories = sorted([d for d in directories if d.endswith("resnet")])

    results_for_plot = []
    for directory in directories:
        result_path = os.path.join("./use", directory, "results.pkl")
        print(directory)
        results = open_pickle(result_path)
        history = results["history"]

        metric_type = "distributed"
        metric_dict = history.metrics_distributed
        # print(metric_dict["accuracy"])
        _, values = zip(*metric_dict["accuracy"])
        # take first 25 rounds
        values = values[:50]

        results_for_plot.append(values)
    
    # KEYS Figure 2a&b (Mobile&Resnet)
    keys = [
        "FedAvg #class-10",
        "FedPer #class-10",
        "FedAvg #class-4",
        "FedPer #class-4",
        "FedAvg #class-8",
        "FedPer #class-8",
    ]

    ## KEYS Figure 4a (Mobile)
    #keys = [
    #    "Fedper (1 block + classifier)",
    #    "Fedper (2 blocks + classifier)",
    #    "Fedper (3 blocks + classifier)",
    #    "FedAvg",
    #]

    ## KEYS Figure 4b (ResNet)
    #keys = [
    #    "Fedper (classifier)",
    #    "Fedper (1 block + classifier)",
    #    "Fedper (2 blocks + classifier)",
    #    "FedAvg",
    #]

    # KEYS Figure 7&8 (Mobile&ResNet) (Flickr) ##
    #keys = [
    #    "FedAvg",
    #    "FedPer (1 block + classifier)",
    #]

    plt.figure()
    plt.title("CIFAR10 ResNet34 - Randomly Assigned Labels")
    plt.grid()
    for i, result in enumerate(results_for_plot):
        # set y axis to cross 0
        plt.xlim(left=0)
        plt.xlim(right=50)
        label = keys[i]
        if "fedavg" in label.lower():
            plt.plot(result, label=keys[i], linestyle="dashed")
        else:
            plt.plot(result, label=keys[i])
    plt.ylabel("Accuracy")
    plt.xlabel("Rounds")
    plt.legend()
    plt.savefig("./use/resnet_plot_figure_2.png")
    plt.show()
