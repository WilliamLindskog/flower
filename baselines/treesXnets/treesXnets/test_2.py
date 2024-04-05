import matplotlib.pyplot as plt


# FEMNIST 
clients = [3, 5, 10, 20, 30, 50]
xgbBa_fem = [0.760, 0.771,0.780, 0.782, 0.784, 0.788]
xgbCy_fem = [0.758, 0.768,0.775, 0.780, 0.781, 0.784]
xgbCnn_fem = [0.758, 0.765,0.770, 0.777, 0.778, 0.781]
fedTab_fem = [0.712, 0.718,0.725, 0.729, 0.731, 0.733]
fedLR_fem = [0.589, 0.601,0.610, 0.623, 0.625, 0.631]
fedCnn_fem = [0.679, 0.685,0.691, 0.700, 0.704, 0.709]
fedRN_fem = [0.681, 0.692,0.703, 0.708, 0.712, 0.720]
fedMLP_fem = [0.640, 0.643,0.651, 0.660, 0.666, 0.669]

# create figure - 2 subplots horizontally
fig, ax = plt.subplots(1, 2, figsize=(20, 6))
fig.subplots_adjust(bottom=0.34)

ax[0].plot(clients, xgbBa_fem, label='FedXgbBagging', marker='o')
ax[0].plot(clients, xgbCy_fem, label='FedXgbCyclic', marker='o')
ax[0].plot(clients, xgbCnn_fem, label='FedXgbCnn', marker='o')
ax[0].plot(clients, fedTab_fem, label='FedTab', marker='o')
ax[0].plot(clients, fedLR_fem, label='FedLR', marker='o')
ax[0].plot(clients, fedCnn_fem, label='FedCnn', marker='o')
ax[0].plot(clients, fedRN_fem, label='FedRN', marker='o')
ax[0].plot(clients, fedMLP_fem, label='FedMLP', marker='o')

# add title and labels
ax[0].set_title('FEMNIST', fontsize=20)
ax[0].set_xlabel('Number of Clients', fontsize=18)
ax[0].set_ylabel('Accuracy', fontsize=18)
ax[0].xaxis.set_tick_params(labelsize=16)
ax[0].yaxis.set_tick_params(labelsize=16)

# add grid
ax[0].grid(True)

# add legend under plot
ax[0].legend(loc='upper center', bbox_to_anchor=(1.1, -0.18), shadow=True, ncol=2, fontsize=15)

# FEMNIST 
xgbBa = [0.869, 0.873,0.881, 0.885, 0.888, 0.892]
xgbCy = [0.856, 0.863,0.868, 0.874, 0.879, 0.883]
xgbCnn = [0.867, 0.873,0.879, 0.882, 0.885, 0.887]
fedTab = [0.852, 0.860,0.871, 0.874, 0.878, 0.881]
fedLR = [0.747, 0.750,0.754, 0.759, 0.763, 0.765]
fedCnn = [0.779, 0.785,0.791, 0.800, 0.804, 0.809]
fedRN = [0.812, 0.831,0.835, 0.837, 0.840, 0.841]
fedMLP = [0.779, 0.789,0.809, 0.812, 0.816, 0.820]

ax[1].plot(clients, xgbBa, label='FedXgbBagging', marker='o')
ax[1].plot(clients, xgbCy, label='FedXgbCyclic', marker='o')
ax[1].plot(clients, xgbCnn, label='FedXgbCnn', marker='o')
ax[1].plot(clients, fedTab, label='FedTab', marker='o')
ax[1].plot(clients, fedLR, label='FedLR', marker='o')
ax[1].plot(clients, fedCnn, label='FedCnn', marker='o')
ax[1].plot(clients, fedRN, label='FedRN', marker='o')
ax[1].plot(clients, fedMLP, label='FedMLP', marker='o')

# add title and labels
ax[1].set_title('Synthetic', fontsize=20)
ax[1].set_xlabel('Number of Clients', fontsize=18)
ax[1].set_ylabel('Accuracy', fontsize=18)
ax[1].xaxis.set_tick_params(labelsize=16)
ax[1].yaxis.set_tick_params(labelsize=16)

# add grid
ax[1].grid(True)

# add legend under plot
ax[1].legend(loc='upper center', bbox_to_anchor=(1.9, -0.18), shadow=True, ncol=2, fontsize=15)

# Give figure title
fig.suptitle('Accuracy vs Number of Clients', fontsize=25)

# show plot
plt.savefig('./treesXnets/figures/plots.png')