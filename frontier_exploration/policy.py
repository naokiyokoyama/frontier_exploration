from gym import spaces
from habitat_baselines.common.baseline_registry import baseline_registry
from habitat_baselines.rl.ppo import Policy

from frontier_exploration.sensors import FrontierWaypoint


@baseline_registry.register_policy
class FrontierExplorationPolicy(Policy):
    def __init__(self, *args, **kwargs):
        super().__init__()

    @classmethod
    def from_config(
        cls,
        config: "DictConfig",
        observation_space: spaces.Dict,
        action_space,
        **kwargs,
    ):
        return cls()

    def act(
        self,
        observations,
        rnn_hidden_states,
        prev_actions,
        masks,
        deterministic=False,
    ):
        return observations[FrontierWaypoint.cls_uuid]

    # used in ppo_trainer.py eval:

    def to(self, *args, **kwargs):
        return

    def eval(self):
        return
