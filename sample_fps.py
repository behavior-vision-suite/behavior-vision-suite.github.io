import os
from typing import List
import numpy as np
import cv2
import copy
import random
from PIL import Image
from pathlib import Path
import networkx as nx
import matplotlib.pyplot as plt
from omnigibson.macros import gm

from omnigibson.utils.asset_utils import get_available_og_scenes
import omnigibson.utils.transform_utils as T

from utils import get_graph_for_scene, erode_graph, visualize_graph

def main(
   scene_id: int = 0,     
   output_root: str = "output_data",
):
    Path(output_root).mkdir(parents=True, exist_ok=True)
    
    scenes = get_available_og_scenes()
    scene_model = list(scenes)[scene_id]
    og_dataset_path = gm.DATASET_PATH
    og_scene_path = os.path.join(og_dataset_path, "scenes", scene_model)
    g: nx.Graph = get_graph_for_scene(og_scene_path)

    print("Running Erode")
    # heavy erosion of the graph
    g_erode = copy.deepcopy(g)
    erode_fpath = os.path.join(output_root, "erode")
    erode_graph(g_erode, [6, 6], erode_fpath, should_visualize=True)
    print("Running FPS")
    g_fps = copy.deepcopy(g_erode)
    fps_fpath = os.path.join(output_root, "fps")
    n_iter = min(int(len(g_fps.nodes) / 60), 200)
    fps(g_fps, fps_fpath, n_iter=n_iter)


def fps(g_fps: nx.Graph, fpath: str, n_iter=10):
    '''
    Farthest point sampling
    '''
    if not os.path.exists(os.path.join(fpath, "progress")):
        os.makedirs(os.path.join(fpath, "progress"))

    all_nodes = np.array(g_fps.nodes)

    # sample initial point
    init_node = random.choice(list(g_fps.nodes))
    img, x_min, y_min = visualize_graph(g_fps)
    x, y = init_node
    img[y-y_min, x-x_min, :] = [1, 0, 0] # red
    # plt.imshow(img)
    # plt.title("Initial Graph")
    # plt.savefig(os.path.join(fpath, "progress/0.png"))

    sampled_nodes = np.array(init_node).reshape(1, 2)

    for iter in range(n_iter):
        diff = all_nodes[:, np.newaxis, :] - sampled_nodes[np.newaxis, :, :]
        dist = np.linalg.norm(diff, axis=-1)
        min_dist = np.min(dist, axis=1)
        max_idx = np.argmax(min_dist)
        new_node = all_nodes[max_idx]
        sampled_nodes = np.concatenate([sampled_nodes, new_node.reshape(1, 2)], axis=0)

        # # save progress
        # x, y = new_node
        # img[y-y_min, x-x_min, :] = [1, 0, 0] # red
        # plt.imshow(img)
        # plt.title(f"FPS: {iter+1}")
        # plt.savefig(os.path.join(fpath, f"progress/{iter+1}.png"))

    # save sampled nodes
    np.save(os.path.join(fpath, "fps_sampled_nodes.npy"), sampled_nodes)


if __name__ == "__main__":
    import fire
    fire.Fire(main)