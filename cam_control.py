from typing import List
import numpy as np
import networkx as nx
import copy
import matplotlib.pyplot as plt


import omnigibson as og
import omnigibson.utils.transform_utils as T
from omnigibson.sensors import VisionSensor
from scipy.spatial.transform import Rotation as R
from scipy.spatial.transform import Slerp
from scipy.interpolate import CubicSpline
from scipy.integrate import quad
from scipy.ndimage import gaussian_filter1d

from utils import select_in_range, get_graph_for_scene, erode_graph

class CamControl:
    '''
    The class to control camera movement in the scene.
    '''

    def __init__(self, 
            scene, 
            cam: VisionSensor, 
            per_step_rotation=np.pi/360,
            per_step_distance=0.01, 
            smoothing_window=30):
        self.scene = scene
        self.cam = cam
        self.per_step_rotation = per_step_rotation
        self.per_step_distance = per_step_distance
        self.smoothing_window = smoothing_window

        self.scene_trav_map = self.scene.trav_map

        ##### preprocess for traversable map
        self.trav_nodes_world = [] # list of 2D world positions of nodes
        self.g: nx.Graph = get_graph_for_scene(og.sim.scene.scene_dir)

        # erode graph
        erode_graph(self.g, threshold=[6, 6])
        print("Remaining nodes: ", len(self.g.nodes))
        # pick largest connected component
        largest_cc = max(nx.connected_components(self.g), key=len)
        self.g = self.g.subgraph(largest_cc).copy()

        for node in self.g.nodes:
            # 2D pixel pos -> 2D world pos
            world_pos = self.scene_trav_map.map_to_world(np.array(node))
            self.trav_nodes_world.append(world_pos)

    def sample_cam_trajectory_from_fps(
            self, room_ins_ids: List[int], 
            filtered_nodes_in_map, filtered_orientations
       ):
        # should already eroded traversable map
        room_ins_map = self.scene.seg_map.room_ins_map

        # (n, 2) pixel map pos
        fps_sampled_nodes = np.array(filtered_nodes_in_map)

        # select part of the sampled nodes in given room if specified
        room_ref = np.array(np.where(np.isin(room_ins_map, room_ins_ids))).T
        fps_sampled_nodes = select_in_range(fps_sampled_nodes, cover=room_ref)
        trav_nodes = select_in_range(np.array(self.g.nodes), cover=room_ref)

        if trav_nodes is None:
            raise ValueError("No nodes in room!")

        print("Running tsp...")
        # arr to tuple
        fps_sampled_nodes = [tuple(node) for node in fps_sampled_nodes]
        # path as a series of 2D map
        path = self.fast_tsp(fps_sampled_nodes)
        print("Finished tsp.")

        path_world = [] # list of 2D world positions
        for point in path:
            map_pos = self.scene_trav_map.map_to_world(np.array(point))
            path_world.append(map_pos)
        
        path_world = np.array(path_world)
        h = np.ones((path_world.shape[0], 1)) * self.cam.get_position()[2]
        traversable_path = np.hstack((path_world, h))

        dense_positions = self.get_dense_positions_from_waypoints(waypoints=traversable_path, per_step_distance=self.per_step_distance, stack_positions=False)
        stop_orientations = self.get_stop_orientations_in_path(path, filtered_nodes_in_map, filtered_orientations)
        stop_orientations = stop_orientations[1:] # remove last orientation
        pan_angles, tilt_angles, all_positions = self.get_orientations_from_positions_with_stop(dense_positions, stop_orientations)
        poses = self.get_poses_from_orientations(all_positions, pan_angles, tilt_angles, 
                                                 per_step_rotation=self.per_step_rotation, smoothing_window=self.smoothing_window)
        return poses

    def get_stop_orientations_in_path(self, path, filtered_nodes_in_map, filtered_orientations):
        '''
        For a given path, checks if it visits certain stops that we have predefined orientations. If a stop appears more than once
        in the path, the orientation will be specified at only the first occurence.
        Input: path -- list of tuples of nodes in map that will traverse, length n
               filtered_nodes_in_map, filtered_orientations: list of nodes in map and orientations that will stop at [m, 3], [m, 4]
        Output: stop_orientations: list of orientations that will stop at list of length n
                valid orientations if there is a stop, None otherwise
        '''
        stop_orientations = [None] * len(path)
        for i in range(len(filtered_nodes_in_map)):
            node = tuple(filtered_nodes_in_map[i])
            if node not in path:
                print("not in")
                continue
            idx_in_path = path.index(node)
            stop_orientations[idx_in_path] = filtered_orientations[i]

        return stop_orientations

    def get_dense_positions_from_waypoints(self, waypoints, per_step_distance, stack_positions=True):
        '''
        Refactored from Omnigibson/omnigibson/utils/ui_utils -- record_trajectory_from_waypoints
        Input: waypoints: list of points to traverse
               per_step_distance (controlling param): distance between two waypoints
        Output: dense_positions: list of dense positions interpolated between waypoints
        '''

        # Create splines and their derivatives
        n_waypoints = len(waypoints)
        if n_waypoints < 3:
            og.log.error("Cannot generate trajectory from waypoints with less than 3 waypoints!")
            return

        splines = [CubicSpline(range(n_waypoints), waypoints[:, i], bc_type='clamped') for i in range(3)]
        dsplines = [spline.derivative() for spline in splines]

        # Function help get arc derivative
        def arc_derivative(u):
            return np.sqrt(np.sum([dspline(u) ** 2 for dspline in dsplines]))

        # Function to help get interpolated positions
        def get_interpolated_positions(step):
            assert step < n_waypoints - 1
            dist = quad(func=arc_derivative, a=step, b=step + 1)[0]
            path_length = int(dist / per_step_distance)
            interpolated_points = np.zeros((path_length, 3))
            for i in range(path_length):
                curr_step = step + (i / path_length)
                interpolated_points[i, :] = np.array([spline(curr_step) for spline in splines])
            return interpolated_points
        
        # Now added an option to return full trajectory (stacked) or segments separately
        dense_positions = []

        # Get all interpolated positions first
        for i in range(n_waypoints - 1):
            positions = get_interpolated_positions(step=i)
            dense_positions.append(positions)

        if stack_positions:
            dense_positions = np.vstack(dense_positions)

        return dense_positions # (n, 3) or (n_waypoints, k, 3), k may vary
    
    def get_orientations_from_positions_with_stop(self, dense_positions, stop_orientations):
        '''
        Refactored from Omnigibson/omnigibson/utils/ui_utils -- record_trajectory_from_waypoints
        Finds raw orientations of camera along the path (no postprocessing included)
        Input: positions: list of positions for each segment to traverse (n-1, k, 3)
               orientations: a list of orientations to stop at at the end of each segment (n-1, 3)
        Output: poses: list of pan, tilt angles to traverse
        '''
        print("In get_orientations_from_positions_with_stop")
        print(len(dense_positions), len(stop_orientations))
        assert len(dense_positions) == len(stop_orientations), "dense_positions and stop_orientations length mismatch"
        n_waypoints = len(dense_positions)

        # Finding all the pan and tilt angles with specified lookat
        all_pan_angles = []
        all_tilt_angles = []
        all_positions = []
        for i in range(n_waypoints - 1):
            positions = dense_positions[i]
            for j in range(len(positions) - 1):
                # Get direction vector from the current to the following point
                next = j + 1
                direction = positions[next] - positions[j]
                direction = direction / np.linalg.norm(direction)

                # Infer tilt and pan angles from this direction
                xy_direction = direction[:2] / np.linalg.norm(direction[:2])
                z = direction[2]
                pan_angle = np.arctan2(-xy_direction[0], xy_direction[1])
                tilt_angle = np.arcsin(z)

                all_pan_angles.append(pan_angle)
                all_tilt_angles.append(tilt_angle)
                all_positions.append(positions[j])
            
            # check if desired orientation specified
            if stop_orientations[i] is not None:
                # Infer tilt and pan angles from this direction
                euler = T.quat2euler(stop_orientations[i])
                pan_angle = euler[2]
                tilt_angle = np.pi / 2 - euler[0]

                all_pan_angles.append(pan_angle)
                all_tilt_angles.append(tilt_angle)
            else: # keep last orientation
                all_pan_angles.append(all_pan_angles[-1])
                all_tilt_angles.append(all_tilt_angles[-1])

            all_positions.append(positions[-1])

        return all_pan_angles, all_tilt_angles, all_positions

    def get_poses_from_orientations(self, positions, pan_angles, tilt_angles, per_step_rotation=None, smoothing_window=None):
        '''
        Refactored from Omnigibson/omnigibson/utils/ui_utils -- record_trajectory_from_waypoints
        Post processes the raw camera angles to poses.
        Input: positions, pan_angles, tilt_angles: list of positions, pan angles, tilt angles to traverse
        Output: poses: list of poses (position, orientation) to traverse
        '''
        print(len(positions), len(pan_angles), len(tilt_angles))
        assert len(positions) == len(pan_angles) == len(tilt_angles), "Getting poses from angles, length mismatch"
        pan_angles = np.array(pan_angles)

        # gaussian smoothing 
        if smoothing_window is not None:
            complex_angles = np.exp(1j * pan_angles)
            complex_angles = gaussian_filter1d(complex_angles, smoothing_window)
            pan_angles = np.angle(complex_angles)

        def angular_diff(a, b):
            diff = b - a
            if diff > np.pi:
                diff -= 2 * np.pi
            elif diff < -np.pi:
                diff += 2 * np.pi
            return diff

        poses = []
        for i in range(len(positions) - 1):
            if per_step_rotation is not None and i > 0:
                diff = angular_diff(pan_angles[i-1], pan_angles[i])
                if abs(diff) > per_step_rotation:
                    num_steps = int(np.ceil(abs(diff) / per_step_rotation))
                    for j in range(num_steps):
                        pan = pan_angles[i-1] + (j+1) * diff / (num_steps+1)
                        quat = T.euler2quat([np.pi / 2 - tilt_angles[i], 0.0, pan])
                        poses.append([positions[i], quat])
                    continue
            quat = T.euler2quat([np.pi / 2 - tilt_angles[i], 0.0, pan_angles[i]])
            poses.append([positions[i], quat])

        return poses

    def fast_tsp(self, waypoints):
        """
        Solve tsp hierarchically by 
        1. Find pairwise Dijkstra distance of waypoints.
        2. Construct a new graph with waypoints as nodes and pairwise distance as edge weights.
        3. Find the shortest path in the new graph.
        4. Construct the final path by connecting waypoints in the found order with shortest path.
        """

        def preprocess_waypoints():
            """Checks if all waypoints are in graph and replace with Euclidean nearest node if not."""
            nodes = np.array(self.g.nodes)
            for i in range(len(waypoints)):
                if waypoints[i] not in self.g.nodes:
                    closest_node = tuple(nodes[np.argmin(np.linalg.norm(nodes - np.array(waypoints[i]), axis=1))])
                    waypoints[i] = closest_node
            return waypoints

        def compute_distances():
            """Computes pairwise distance between waypoints."""
            distances = {}
            for u in waypoints:
                distances[u] = {}
                for v in waypoints:
                    if u == v: continue
                    distances[u][v] = nx.dijkstra_path_length(self.g, u, v)
            return distances
        
        def construct_graph(distances):
            """Create a complete graph using pairwise distances."""
            G = nx.Graph()
            for source, targets in distances.items():
                for target, distance in targets.items():
                    G.add_edge(source, target, weight=distance)
            return G
        
        def find_shortest_path(G):
            """Find tsp shortest node ordering in G. Then connect shortest path in self.g"""
            # First find a waypoint ordering by running tsp in G
            order = nx.approximation.traveling_salesman_problem(G, nodes=waypoints, cycle=False)

            # Then connect waypoints in order with shortest path in self.g
            path = []
            for i in range(len(waypoints)-1):
                path_segment = nx.dijkstra_path(self.g, order[i], order[i+1])
                if i == 0:
                    path = path_segment
                else:
                    path.extend(path_segment[1:])
            return path

        waypoints = preprocess_waypoints()
        distances = compute_distances()
        G = construct_graph(distances)
        path = find_shortest_path(G)
        return path