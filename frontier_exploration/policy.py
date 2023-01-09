import torch
from gym import spaces
from habitat_baselines.common.baseline_registry import baseline_registry
from habitat_baselines.rl.ppo import Policy

from frontier_exploration.sensors import FrontierWaypoint


@baseline_registry.register_policy
class FrontierExplorationPolicy(Policy):
    def __init__(self, *args, **kwargs):
        super().__init__()

    @property
    def should_load_agent_state(self):
        return False

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
        # Convert obs to torch.long
        action = observations[FrontierWaypoint.cls_uuid].type(torch.long)
        return None, action, None, rnn_hidden_states

    # used in ppo_trainer.py eval:

    def to(self, *args, **kwargs):
        return

    def eval(self):
        return
