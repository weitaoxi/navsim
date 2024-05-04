
import hydra
from hydra.utils import instantiate
import logging
import os
import uuid
import torch
import pickle
import gzip

from omegaconf import DictConfig
from typing import Any, Dict, List, Optional, Union
from pathlib import Path

from nuplan.planning.script.builders.logging_builder import build_logger
from navsim.planning.script.builders.worker_pool_builder import build_worker
from nuplan.planning.utils.multithreading.worker_pool import WorkerPool
from nuplan.planning.utils.multithreading.worker_utils import worker_map
from navsim.planning.training.dataset import Dataset
from navsim.common.dataloader import SceneLoader
from navsim.common.dataclasses import SceneFilter, SensorConfig

from navsim.agents.ego_status_mlp_agent import TrajectoryTargetBuilder
from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling


logger = logging.getLogger(__name__)

CONFIG_PATH = "config/multi_modality"
CONFIG_NAME = "traj_cache"
@hydra.main(config_path=CONFIG_PATH, config_name=CONFIG_NAME)
def main(cfg: DictConfig) -> None:
    
    # Configure logger
    build_logger(cfg)

    # Build worker
    worker = build_worker(cfg)

    # Precompute and cache all features
    logger.info("Starting Get Traj Caching...")

    logger.info("Building SceneLoader")
    scene_filter: SceneFilter = instantiate(cfg.scene_filter)
    data_path = Path(cfg.navsim_log_path)
    sensor_blobs_path = Path(cfg.sensor_blobs_path)
    scene_loader = SceneLoader(
        sensor_blobs_path=sensor_blobs_path,
        data_path=data_path,
        scene_filter=scene_filter,
        sensor_config=SensorConfig.build_no_sensors(),
    )
    logger.info(f"Extracted {len(scene_loader)} scenarios for training/validation dataset")

    data_points = [
        {
            "cfg": cfg,
            "log_file": log_file,
            "tokens": tokens_list,
        }
        for log_file, tokens_list in scene_loader.get_tokens_list_per_log().items()
    ]
    logger.info(f"Extracted {len(data_points)} data_points for training/validation dataset")

    traj_list = worker_map(worker, cache_trajs, data_points)
    traj_tensor = torch.stack(traj_list)
    
    logger.info(f"Finished caching {traj_tensor.shape} trajs for training/validation dataset")

    with gzip.open('traj_list.gz', "wb", compresslevel=1) as f:
        pickle.dump(traj_tensor, f)

def cache_trajs(args: List[Dict[str, Union[List[str], DictConfig]]]) -> List[Optional[Any]]:
    node_id = int(os.environ.get("NODE_RANK", 0))
    thread_id = str(uuid.uuid4())

    log_names = [a["log_file"] for a in args]
    tokens = [t for a in args for t in a["tokens"]]
    cfg: DictConfig = args[0]["cfg"]

    scene_filter: SceneFilter =instantiate(cfg.scene_filter)
    scene_filter.log_names = log_names
    scene_filter.tokens = tokens
    scene_loader = SceneLoader(
        sensor_blobs_path=Path(cfg.sensor_blobs_path),
        data_path=Path(cfg.navsim_log_path),
        scene_filter=scene_filter,
        sensor_config=SensorConfig.build_no_sensors(),
    )
    logger.info(
        f"Extracted {len(scene_loader.tokens)} scenarios for thread_id={thread_id}, node_id={node_id}."
    )

    target_builder = TrajectoryTargetBuilder(TrajectorySampling(time_horizon=4, interval_length=0.5))
    traj_list = []
    for token in scene_loader.tokens:
        scene = scene_loader.get_scene_from_token(token)
        data_dict = target_builder.compute_targets(scene)
        traj_list.append(data_dict['trajectory'])
    traj_tensor = torch.stack(traj_list)

    return traj_tensor



if __name__ == "__main__":
    main()