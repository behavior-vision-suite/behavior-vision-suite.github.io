from cam_control import CamControl

import os
from pathlib import Path

import numpy as np
import imageio

import omnigibson as og
from omnigibson.utils.asset_utils import (
    get_available_og_scenes,
)
from omnigibson.sensors import VisionSensor
from omnigibson.object_states import Open, OnTop, Inside, NextTo, Under, Touching

import omnigibson.utils.transform_utils as T

from utils import filter_fps

class CollectData:
    def __init__(self, scene_id: int, scene_file: str, output_root: str):
        """
        Initialize og scene
        """
        self.output_root = output_root
        self.fps_path = os.path.join(output_root, "fps")
        assert os.path.exists(self.fps_path), f"FPS path {self.fps_path} does not exist, run sample_fps.py first"
        
        ### Init OG
        scenes = get_available_og_scenes()
        scene_model = list(scenes)[scene_id]
        cfg = {
            "scene": {
                "type": "InteractiveTraversableScene",
                "scene_model": scene_model,
                "scene_file": scene_file,
            },
            "camera": {
                "use_default_camera": False, # allow to customize camera
                "use_modalities": ["rgb", "seg_instance", "seg_semantic", "depth", "depth_linear", "bbox_2d_tight", "bbox_2d_loose"],
                "resolution": [1024, 1024], # specify for customized camera
            },
        }
        # og launch
        _ = og.Environment(cfg)
        self.scene = og.sim.scene
        self.cam = VisionSensor(
            prim_path="/World/viewer_camera",
            name="camera",
            modalities=cfg['camera']['use_modalities'],
            image_height=cfg['camera']['resolution'][0],
            image_width=cfg['camera']['resolution'][1],
        )
        self.cam.initialize()
        self.cam_mover = og.sim.enable_viewer_camera_teleoperation()
        self.cam_intrinsics = self.compute_intrinsics()
        self.open_doors() # open all doors in the scene 

        for _ in range(5): og.sim.step()

        self.obj_name_to_id = {obj.name: index+1 for index, obj in enumerate(self.scene.objects)}
        self.obj_name_to_id['Others'] = 0 # might be light
        self.obj_id_to_name = {v: k for k, v in self.obj_name_to_id.items()}
        self.cam_control = CamControl(self.scene, self.cam)
        
    def compute_intrinsics(self):
        '''
        Assumes squared image. 
        '''
        img_width = self.cam.image_width
        img_height = self.cam.image_height
        apert = self.cam.prim.GetAttribute("horizontalAperture").Get()
        focal_len_in_pixel = self.cam.focal_length * img_width / apert

        intrinsics = np.eye(3)
        intrinsics[0,0] = focal_len_in_pixel
        intrinsics[1,1] = focal_len_in_pixel
        intrinsics[0,2] = img_width / 2
        intrinsics[1,2] = img_height / 2
        return intrinsics

    def open_doors(self, fully=True):
        '''
        Adds open state to all door objects in the scene and opens them.
        fully: whether to open the door fully or with random extent.
        '''
        all_doors = self.scene.object_registry("category","door")

        for door in all_doors:
            door.visual_only = False
            if Open not in door.states:
                open_state =  Open(door)
                open_state.initialize()
                door.states[Open] = open_state

        for door in all_doors:
            door.states[Open].set_value(True, fully=fully)
        

    def collect_fps(self):
        #####################################
        # I. reset camera height randomly
        #####################################
        height = np.random.uniform(0.7, 1.3)
        print(f"resetting camera height to {height}")
        orig_position = self.cam.get_position()
        self.cam.set_position(position=[orig_position[0], orig_position[1], height])
        for _ in range(10): og.sim.step()

        #####################################
        # II. filter fps nodes
        #####################################
        print("filtering fps nodes...")
        # this is from sample_fps.py
        # (num_iter, 2), each row is a sampled pixel node

        fps_sampled_nodes = np.load(os.path.join(self.fps_path, "fps_sampled_nodes.npy"))
        fps_nodes_in_world = [] # a list of 3D world coordinates for fps nodes
        position = self.cam.get_position()
        for node in fps_sampled_nodes:
            fps_nodes_in_world.append(np.hstack((self.cam_control.scene_trav_map.map_to_world(np.array(node)), position[-1])))
        filtered_nodes, filtered_orientations = filter_fps(
            fps_nodes_in_world, self.cam, 
            count_threshold=np.random.choice([2, 3]), 
            order_type="random")
        # now filtered_nodes is a list of 2D map coordinates
        filtered_nodes_in_map = [self.cam_control.scene_trav_map.world_to_map(node[:2]) for node in filtered_nodes]
        
        #####################################
        # III. Generating camera trajectory for available rooms
        #####################################
        available_rooms = list(self.cam_control.scene.seg_map.room_ins_id_to_ins_name.keys())
        print(f"Collecting rooms: {available_rooms}")
        poses = self.cam_control.sample_cam_trajectory_from_fps(
            available_rooms, filtered_nodes_in_map, filtered_orientations
        )
        # save poses
        np.save(os.path.join(self.fps_path, "poses.npy"), poses)
        return poses
    
    def render(self, poses, render_video=True):
        """
        Visualizing sampled trajectory 
        """
        if render_video:
            # define render modality / resolution here
            self.render_cam = VisionSensor(
                prim_path="/World/render_camera",
                name="render_camera",
                modalities={"rgb"},
                image_height=1024,
                image_width=1024,
            )
            self.render_cam.load()
            self.render_cam.initialize()
            output_path = os.path.join(self.output_root, 'video.mp4')
            with imageio.get_writer(output_path, fps=60) as writer:
                for pose in poses:
                    self.cam.set_position_orientation(position=pose[0], orientation=pose[1])
                    self.render_cam.set_position_orientation(position=pose[0], orientation=pose[1])
                    og.sim.step()
                    writer.append_data(self.render_cam.get_obs()[0]["rgb"][:, :, :-1])
                    # Optional: add code to save other modalities here
    
        else:
            for pose in poses:
                self.cam.set_position_orientation(position=pose[0], orientation=pose[1])
                og.sim.step()
        

def main(
    scene_id: int = 0,
    scene_file: str = None,
    output_root: str = "output_data",
):
    Path(output_root).mkdir(parents=True, exist_ok=True)
    cd = CollectData(scene_id, scene_file, output_root)
    poses = cd.collect_fps()
    cd.render(poses=poses)


if __name__ == "__main__":
    import fire
    fire.Fire(main)