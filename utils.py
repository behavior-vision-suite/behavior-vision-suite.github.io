
from typing import List, Optional
import matplotlib.pyplot as plt
import numpy as np
import numpy as np
from collections import defaultdict
import random
import copy

import omnigibson as og
import omnigibson.utils.transform_utils as T
from omnigibson.utils.asset_utils import get_og_avg_category_specs
from omnigibson.sensors import VisionSensor
from omnigibson.utils.asset_utils import get_all_object_category_models
from omnigibson.objects import DatasetObject
from omnigibson.object_states import OnTop

import random
import json
import numpy as np
import networkx as nx
import cv2
import os
from PIL import Image


def get_graph_for_scene(og_scene_path: str) -> nx.Graph:
    """
    Getting the traversable map with no doors for scene
        Node = each pixel that is in the scene
        Edge = connect if neighbor pixel is also in the scene
    
    Return the largest connected component of the graph
    """
    # obtain og's floor map
    map_path = os.path.join(og_scene_path, "layout/floor_trav_no_door_0.png")
    floor_map = np.array(Image.open(map_path))
    # and resize
    resize_factor = 0.1
    resize_height = int(floor_map.shape[0] * resize_factor)
    floor_map = cv2.resize(floor_map, (resize_height, resize_height))
    assert floor_map.shape[0] == floor_map.shape[1], "Floor map must be square"

    g = nx.Graph()
    map_size = floor_map.shape[0]
    for i in range(map_size):
        for j in range(map_size):
            if floor_map[i, j] == 0:
                continue
            g.add_node((i, j))
            # 8-connected graph
            neighbors = [(i - 1, j - 1), (i, j - 1), (i + 1, j - 1), (i - 1, j)]
            for n in neighbors:
                if (
                    0 <= n[0] < map_size
                    and 0 <= n[1] < map_size
                    and floor_map[n[0], n[1]] > 0
                ):
                    g.add_edge(n, (i, j), weight=T.l2_distance(n, (i, j)))

    # only take the largest connected component
    largest_cc = max(nx.connected_components(g), key=len)
    g = g.subgraph(largest_cc).copy()
    return g

def erode_graph(g: nx.Graph, threshold: List[int], fpath: Optional[str] = None, should_visualize=False) -> None:
    '''
    Erode the graph based on the threshold
    It will remove nodes with degree less than or equal to the threshold
    if should_visualize is True
        Will save the progress of the erosion 
        And fpath should not be None since it will save the progress in fpath/progress
    
    Returns None but will modify the graph in place
    '''
    assert type(threshold) == list
    if should_visualize:
        if not os.path.exists(os.path.join(fpath, "progress")):
            os.makedirs(os.path.join(fpath, "progress"))
        img, _, _ = visualize_graph(g)
        plt.imshow(img)
        plt.title("Initial Graph")
        plt.savefig(os.path.join(fpath, "progress/0.png"))
    
    
    for i, t in enumerate(threshold):
        # erode!
        nodes_to_remove = [node for node, degree in g.degree if degree <= t]
        g.remove_nodes_from(nodes_to_remove)
        
        if should_visualize:
            img, _, _ = visualize_graph(g)
            plt.imshow(img)
            plt.title(f"Erode: {threshold[:i+1]}")
            plt.savefig(os.path.join(fpath, f"progress/{i+1}.png"))

def visualize_graph(g: nx.Graph):
    '''
    Store the graph as a image array for easier visualization
    '''
    nodes = np.array(g.nodes) # (n_pixel, 2)
    x_min, x_max = np.min(nodes[:, 0]), np.max(nodes[:, 0])
    y_min, y_max = np.min(nodes[:, 1]), np.max(nodes[:, 1])
    x_range = x_max - x_min + 1
    y_range = y_max - y_min + 1
    # img = np.ones((y_range, x_range, 3))
    img = np.zeros((y_range, x_range, 3))
    for node in g.nodes:
        x, y = node
        # img[y-y_min, x-x_min, :] = [0, 0, 1] # blue
        img[y-y_min, x-x_min, :] = [1, 1, 1] # white

    return img, x_min, y_min

def visualize(all_nodes, path=None, waypoints=None, fpath=None, title=None):
    '''
    visualizes the traversable map in blue. Save if fpath provided.
    path (optional): list of nodes in the path in red.
    waypoints (optional): list of nodes in the waypoints in green.
    all inputs as numpy array of shape (n, 2)
    '''
    assert type(all_nodes) == np.ndarray

    consider_nodes = all_nodes
    consider_nodes = np.concatenate((all_nodes, path), axis=0) if path is not None else consider_nodes
    consider_nodes = np.concatenate((consider_nodes, waypoints), axis=0) if waypoints is not None else consider_nodes

    x_min, y_min = np.min(consider_nodes, axis=0)
    x_max, y_max = np.max(consider_nodes, axis=0)
    x_range, y_range = x_max - x_min + 1, y_max - y_min + 1

    img = np.ones((y_range, x_range, 3))

    # plot traversable map
    for node in all_nodes:
        x, y = node
        img[y-y_min, x-x_min, :] = [0, 0, 1]
    
    # plot path
    if path is not None:
        for node in path:
            x, y = node
            img[y-y_min, x-x_min, :] = [1, 0, 0]

    # plot waypoints
    if waypoints is not None:
        for node in waypoints:
            x, y = node
            img[y-y_min, x-x_min, :] = [0, 1, 0]
            
    plt.imshow(img)
    if title: plt.title(title)
    if fpath:
        plt.savefig(fpath)
    else:
        plt.show()
    
    return img, x_min, y_min

def select_in_range(orig_arr, range=None, ref=None, cover=None):
    '''
    selects orig_arr that satisfies the range.
    Either explicitly providing range or by providing ref and range is calculated from ref.
    cover is for selecting the points that is in cover
    '''
    assert type(orig_arr) == np.ndarray, "orig_arr must be numpy array"
    assert orig_arr.shape[1] == 2, "orig_arr must be of shape (n, 2)"
    if range is None and ref is None and cover is None: # no selection if no condition provided
        return orig_arr

    if cover is not None:
        row_matches = np.any(np.all(cover[:, np.newaxis] == orig_arr, axis=2), axis=0)
        # if np.all(row_matches) == False:
        if  not (np.any(row_matches) == True):
            print("Warning: no matching found!")
            return None
        filtered_arr = orig_arr[row_matches]
    else:
        if range is not None:
            x_min, y_min, x_max, y_max = range
        elif ref is not None:
            x_min, y_min = np.min(ref, axis=0)
            x_max, y_max = np.max(ref, axis=0)
        filtered_arr = orig_arr[(orig_arr[:,0] >= x_min) & (orig_arr[:,0] <= x_max) & (orig_arr[:,1] >= y_min) & (orig_arr[:,1] <= y_max)]

    return filtered_arr

def visible_obj_check(node, cam, tilt_angle=0., resolution=12):
    """
    Standing at node, check the best orientation to see the most objects
    """
    print("checking node: ", node)
    best_visible_obj = []
    best_orientation = None
    # set camera to node position
    cam.set_position(node)
    for _ in range(5): og.sim.step()
    # rotate camera to find angle with most visible objects
    for pan_angle in np.linspace(0, np.pi*2, num=resolution):
        quat = T.euler2quat([np.pi / 2 - tilt_angle, 0.0, pan_angle])
        cam.set_orientation(quat)
        for _ in range(5): og.sim.render() # always render before getting obs!
        try:
            obs = cam.get_obs()
            seg_ins = obs[1]['seg_instance']
            visible_obj = []
            for k in seg_ins:
                obj = seg_ins[k]
                if 'floor' in obj or 'wall' in obj or 'ceiling' in obj or "door" in obj: continue
                visible_obj.append(obj)
            print("visible:", visible_obj)
        except:
            visible_obj = []
        # print(visible_obj)
        if len(visible_obj) > len(best_visible_obj):
            best_visible_obj = visible_obj
            best_orientation = quat

    return best_visible_obj, best_orientation

def filter_fps(
        fps_nodes_in_world: List[np.array], 
        cam: VisionSensor, 
        count_threshold: int = 2,
        order_type="random") -> tuple:
    """
    We only keep valuable waypoints that can see new objects

    fps_nodes_in_world: list of nodes in 3D world coordinates
    """
    # get visible objects
    seen_objs = defaultdict(int) # use a dict instead to keep track of seen count
    visible_objs = []; visible_count = []; best_orientations = []

    for node in fps_nodes_in_world:
        # how many visible objects are there at this node?
        visible_obj, best_orientation = visible_obj_check(node, cam)
        visible_objs.append(copy.deepcopy(visible_obj))
        visible_count.append(len(visible_obj))
        best_orientations.append(best_orientation)

    if order_type == "random":
        order = list(range(len(visible_count)))
        random.shuffle(order) # random shuffle INDEX
    elif order_type == "max":
         # index such that visible_count is sorted
        order = sorted(range(len(visible_count)), key=lambda k:visible_count[k], reverse=True)
    else:
        raise Exception("Invalid order type!")

    def should_update_seen_objs(visible_obj) -> bool:
        nonlocal seen_objs
        # don't need to make a copy, since if no extra object is seen, 
        # the seen_objs dict would stay the same! 
        prev_sum = sum(seen_objs.values())
        for name in visible_obj:
            seen_objs[name] += min(seen_objs[name]+1, count_threshold)
        return sum(seen_objs.values()) > prev_sum	

    filtered_nodes = []
    filtered_orientations = []
    for idx in order:
        # ignore waypoints with too few visible objects
        if visible_count[idx] <= 2:
            continue
        should_updated = should_update_seen_objs(visible_objs[idx])
        if should_updated:
            filtered_nodes.append(fps_nodes_in_world[idx])
            filtered_orientations.append(best_orientations[idx])

    return filtered_nodes, filtered_orientations


#################################################
### Scene Augmentation Utils
#################################################

def import_obj(obj, idx):
    """
    Stable import object to scene.
    """
    # keep the pre-import state of the scene
    state = og.sim.dump_state()
    og.sim.stop()
    og.sim.import_object(obj)
    obj.set_position([11, 10+idx, 0.1])
    og.sim.play()
    og.sim.load_state(state)
    for _ in range(10): og.sim.step()

def batch_replace(candidate_objs, replace_ratio=0.2):
    """Replace each obj in candidate_objs with probability replace_ratio."""
    idx = 0
    for obj in candidate_objs:
        if random.random() < replace_ratio:
            print(f"Replacing {obj.name}")
            state = og.sim.dump_state()
            new_obj = random_replace(obj, idx)
            for _ in range(10): og.sim.step()
            idx += 1

def random_replace(obj, idx):
    '''
    Randomly replace obj with another model in the same category.
    Moves original object to some far location (instead of removing from scene).
    '''
    # save original status
    ori = obj.get_orientation()
    pos = obj.aabb_center

    # Assumes obj is visual_only already
    obj.set_orientation([0, 0, 0, 1.0])
    og.sim.step()
    aabb = obj.aabb_extent 

    # create new object
    category = obj.category
    new_obj = get_random_model_in_category(category, idx+10, bbox=aabb, fixed_base=obj.fixed_base)

    # replace
    og.sim.stop()
    # to leave space for collision check and reversion
    obj.set_position([-20, 10+idx*2, 0.1])

    og.sim.import_object(new_obj)
    new_obj.set_bbox_center_position_orientation(pos, ori)
    new_obj.visual_only = True
    og.sim.play()

    return new_obj

def get_random_model_in_category(category, idx, bbox=None, fixed_base=False):
    '''
    Samples a random model in the specified category. 
    bbox: if provided, will specify as bounding_box of the created object. Otherwise will set fit_avg_dim_volume=True.
    '''
    all_models = get_all_object_category_models(category)
    # TODO: add check if all_models is non-empty
    model = random.choice(all_models)
    name = f"{category}_{model}_{idx}"
    fit_avg_dim_volume = bbox is None

    obj = DatasetObject(
        prim_path=f"/World/{name}",
        name=name,
        category=category,
        model=model,
        fit_avg_dim_volume=fit_avg_dim_volume,
        bounding_box=bbox,
        fixed_base=fixed_base,
    )

    return obj

def set_state(place_obj, base_obj):
    # orig_pos = place_obj.get_position()
    new_obj = None
    try:
        success = place_obj.states[OnTop].set_value(base_obj, True)
        for _ in range(10): og.sim.step()
        if success:
            new_obj = loadable_replace(place_obj)
        else:
            og.sim.remove_object(place_obj)

        for _ in range(10): og.sim.step()

    except Exception as e: 
        print(f"Failed to set state for {place_obj.name}")

    return new_obj

def loadable_replace(orig_obj):
    '''
    Creates a new dataset object exactly same as the original one, while setting its
    position and orientation right at importing. 
    Hoping to make the new object successfully saved to json and loaded.
    Note: we don't stop sim, or do any step here. 
    '''
    name = f"{orig_obj.name}_r"
    orig_pos = orig_obj.get_position()
    orig_pos[2] += 0.01 # raise 1cm to avoid collision
    orig_ori = orig_obj.get_orientation()
    new_obj = DatasetObject(
        prim_path=f"/World/{name}",
        name=name,
        category=orig_obj.category,
        model=orig_obj.model,
        fit_avg_dim_volume=True,
        position = orig_pos,
        orientation = orig_ori,
    )

    og.sim.remove_object(orig_obj)
    og.sim.import_object(new_obj)
    new_obj.set_position(orig_pos)
    new_obj.set_orientation(orig_ori)

    return new_obj

def get_augment_categories():
    base_categories = ["carpet", "coffee_table", "countertop", "breakfast_table"]
    place_categories = ["apple", "baseball", "jar", "bowl", "plate", "tablefork", "doll", "teddy_bear"]

    return base_categories, place_categories

BASE_CATEGORIES = ["carpet", "coffee_table", "countertop", "breakfast_table"]
PLACE_CATEGORIES = ["apple", "baseball", "jar", "bowl", "plate", "tablefork", "doll", "teddy_bear"]

def get_replace_candidates(exclude_categories: list = None):
    """
    Returns the list of objecs 
    """
    structural_categories = ["floors", "walls", "ceilings", "window"]
    if exclude_categories is not None:
        exclude_categories = exclude_categories + structural_categories
    else:
        exclude_categories = structural_categories

    objs = og.sim.scene.objects
    replace_candidates  = []

    for obj in objs:
        if obj.category in exclude_categories:
            continue
        has_ontop = False
        for obj_place in objs:
            if obj_place in structural_categories:
                continue
            ontop = obj_place.states[OnTop].get_value(obj)
            if ontop is True:
                has_ontop = True
                break
        if has_ontop is False:
            replace_candidates.append(obj)

    return replace_candidates

