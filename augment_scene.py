
from pathlib import Path

import omnigibson as og
from omnigibson.utils.asset_utils import (
    get_available_og_scenes,
)
from omnigibson.sensors import VisionSensor
from utils import *


def create_og_env(scene_id):
    """
    Creates a omnigibson interactive traversable scene.
    """
    scenes = get_available_og_scenes()
    scene_model = list(scenes)[scene_id]
    cfg = {
        "scene": {
            "type": "InteractiveTraversableScene",
            "scene_model": scene_model,
        }
    }
    _ = og.Environment(cfg)
    og.sim.enable_viewer_camera_teleoperation()

def insert_augment_scene(num_obj=10):
    """
    Augment the scene by inserting additional objects.
    """
    base_categories, place_categories = get_augment_categories()
    print(base_categories)
    
    # get all base objects
    base_objs = []
    for obj in og.sim.scene.objects:
        if obj.category in base_categories:
            base_objs.append(obj)
    np.random.shuffle(base_objs)
    if len(base_objs) > num_obj:
        base_objs = base_objs[:num_obj]

    idx = 10
    for base_obj in base_objs:
        # randomly sample a place object and insert to scene
        place_cat = random.choice(place_categories)
        place_obj = get_random_model_in_category(place_cat, idx)
        import_obj(place_obj, idx)

        # place it on top of some existing object
        set_state(place_obj, base_obj)

def replace_augment_scene(replace_ratio=0.2):
    """
    Augment the scene by replacing object models.
    """
    # fix objects
    og.sim.stop()
    for obj in og.sim.scene.objects:
        obj.fixed_base = True
    og.sim.play()
    for _ in range(30): og.sim.step()

    replace_candidate_objs = get_replace_candidates()
    batch_replace(replace_candidate_objs, replace_ratio=replace_ratio)

def save_augmented_scene(output_root, json_name="aug_scene.json"):
    save_path = os.path.join(output_root, json_name)
    print("Saving augmented scene to", save_path)
    og.sim.save(json_path=save_path)


def main(
    scene_id: int = 0,
    output_root: str = "output_data",
    num_insert: int = 5,
    replace_ratio: float = 0.2,
    save_json_name: str = "aug_scene.json",
):
    Path(output_root).mkdir(parents=True, exist_ok=True)

    # create original env
    create_og_env(scene_id)

    # augment by inserting objects
    insert_augment_scene(num_obj=num_insert)

    # augment by replacing objects
    replace_augment_scene(replace_ratio=replace_ratio)

    # save augmented scene
    save_augmented_scene(output_root, save_json_name)


if __name__ == "__main__":
    import fire
    fire.Fire(main)