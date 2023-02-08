import hashlib
import os
from dataclasses import dataclass
from typing import Any

import numpy as np
from gym import Space, spaces
from habitat import EmbodiedTask, Sensor, SensorTypes, registry
from habitat.config.default_structured_configs import LabSensorConfig
from habitat.sims.habitat_simulator.habitat_simulator import HabitatSim
from hydra.core.config_store import ConfigStore
from omegaconf import DictConfig

HASH_KEYS = ["episode_id", "scene_id", "start_position", "start_rotation"]


@registry.register_sensor
class MultistoryEpisodeFinder(Sensor):

    cls_uuid: str = "multistory_episode_finder"

    def __init__(
        self, sim: HabitatSim, config: "DictConfig", *args: Any, **kwargs: Any
    ) -> None:
        super().__init__(sim, config, *args, **kwargs)
        self._sim = sim
        self._config = config
        self._output_dir = config.output_dir
        self.first = True
        os.makedirs(self._output_dir, exist_ok=True)

    def _get_uuid(self, *args: Any, **kwargs: Any) -> str:
        return self.cls_uuid

    def _get_sensor_type(self, *args: Any, **kwargs: Any) -> SensorTypes:
        return SensorTypes.TENSOR

    def _get_observation_space(self, *args: Any, **kwargs: Any) -> Space:
        return spaces.Box(low=0, high=255, shape=(1,), dtype=np.uint8)

    def get_observation(
        self, task: EmbodiedTask, episode, *args: Any, **kwargs: Any
    ) -> np.ndarray:
        if self.first:
            self.first = False
            return np.array([1], dtype=np.uint8)
        if self.episode_is_multistory(episode):
            hash_str = ":".join([str(getattr(episode, k)) for k in HASH_KEYS])
            basename = hashlib.sha224(hash_str.encode("ASCII")).hexdigest()
            filename = f"{self._output_dir}/{basename}.txt"
            with open(filename, "w") as f:
                f.write(hash_str)

        task.is_stop_called = True
        self.first = True
        return np.array([0], dtype=np.uint8)

    def episode_is_multistory(self, episode):
        """Generate a path to every view point in the episode and check if there aren't
        any paths on the same floor as the start point. If so, return True. Otherwise,
        return False.
        """
        goal_positions = [
            view_point.agent_state.position
            for goal in episode.goals
            for view_point in goal.view_points
        ]
        agent_position = self._sim.get_agent_state().position
        for g in goal_positions:
            self._sim.geodesic_distance(agent_position, g, episode)
            # Last point hasn't been snapepd yet
            pts = episode._shortest_path_cache.points[:-1]
            last = self._sim.pathfinder.snap_point(pts[-1])
            pts.append(last)
            if all(self._is_on_same_floor(p[1]) for p in pts):
                return False
        return True

    def _is_on_same_floor(self, height, ceiling_height=2.0):
        ref_floor_height = self._sim.get_agent(0).state.position[1]
        return ref_floor_height <= height < ref_floor_height + ceiling_height


@dataclass
class MultistoryEpisodeFinderSensorConfig(LabSensorConfig):
    type: str = MultistoryEpisodeFinder.__name__
    output_dir: str = "data/multistory_episodes/train"


cs = ConfigStore.instance()
cs.store(
    package=f"habitat.task.lab_sensors.{MultistoryEpisodeFinder.cls_uuid}",
    group="habitat/task/lab_sensors",
    name=f"{MultistoryEpisodeFinder.cls_uuid}",
    node=MultistoryEpisodeFinderSensorConfig,
)
