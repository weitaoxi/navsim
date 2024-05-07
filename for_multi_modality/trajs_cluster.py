import torch
import pickle
import matplotlib.pyplot as plt
import numpy as np
import gzip

file_name = '/media/com0106/3E4070EF4070AF71/open_dataset/dataset/exp/traj_cache/metadata/traj_list.gz'
output_file = '/media/com0106/3E4070EF4070AF71/open_dataset/dataset/exp/traj_cache/metadata/traj_nodes.gz'

def plot_tensor(traj_tensor):
    # plot
    fig, ax = plt.subplots()

    for i in range(traj_tensor.size(0)):
        x = traj_tensor[i, :, 1]
        y = traj_tensor[i, :, 0]
        ax.plot(x, y, linewidth=2.0)

    ax.set(xlim=(-40, 40), ylim=(-10, 70))

    plt.show()


def main():
    with gzip.open(file_name, "rb", compresslevel=1) as f:
        traj_tensor = pickle.load(f)
    
    traj_nodes = get_traj_cluster(traj_tensor, 4096)

    with gzip.open(output_file, "wb", compresslevel=1) as f:
        pickle.dump(traj_nodes, f)
    plot_tensor(traj_nodes)

def cal_distance(diff_tensor):
    xy_ratio = 1
    head_ratio = 10

    distance = xy_ratio * torch.sqrt(diff_tensor[:, :, 0].pow(2) + diff_tensor[:, :, 1].pow(2)) \
               + head_ratio * torch.abs(diff_tensor[:, :, 2])

    traj_distance = distance.sum(dim=1)
    return traj_distance

def get_traj_cluster(traj, num_nodes):
    if num_nodes >= traj.size(0):
        return traj
    
    traj_nodes = traj[0:num_nodes, :, :]
    max_iteration = 100
    for iter in range(max_iteration):
        new_traj_nodes = torch.zeros(num_nodes, traj.size(1), traj.size(2))
        # closest_traj_cnt = torch.zeros(num_nodes)
        new_traj_nodes += traj_nodes
        closest_traj_cnt = torch.ones(num_nodes)
        for i in range(traj.size(0)):
            diff = traj_nodes - traj[i, :, :].unsqueeze(0)
            distance = cal_distance(diff)
            _, indices = torch.sort(distance)
            new_traj_nodes[indices[0], :, :] += traj[i, :, :]
            closest_traj_cnt[indices[0]] += 1
        new_traj_nodes = new_traj_nodes / closest_traj_cnt.unsqueeze(-1).unsqueeze(-1)
        print("ieration: ", iter)

        node_diff = cal_distance(new_traj_nodes - traj_nodes)
        # print(node_diff)
        print('node_diff: ', node_diff.sum())

        traj_nodes = new_traj_nodes

    return traj_nodes

if __name__ == "__main__":
    main()