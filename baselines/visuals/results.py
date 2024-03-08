import pickle
import os
path = './treesXnets/outputs/2023-12-13/'

directories = os.listdir(path)
for i, d in enumerate(directories): 
    file_path = os.path.join(path, d, "history.pkl")
    with open(file_path, "rb") as f:
        history = pickle.load(f)
        #print(history)
        metric_type = "centralized"
        metric_dict = history.metrics_centralized
        _, values_m1 = zip(*metric_dict["mse"])
        _, values_m2 = zip(*metric_dict["r2"])
        print("Folder: ", i+1)
        print("MSE: ", min(values_m1))
        print("R2: ", max(values_m2))


