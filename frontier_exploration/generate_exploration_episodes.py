import frontier_exploration  # noqa
import hydra  # noqa
from habitat import get_config  # noqa
from habitat.config.default import patch_config
from habitat.config.default_structured_configs import register_hydra_plugin
from habitat_baselines.run import execute_exp
from hydra.core.config_search_path import ConfigSearchPath
from hydra.plugins.search_path_plugin import SearchPathPlugin
from omegaconf import DictConfig


class HabitatConfigPlugin(SearchPathPlugin):
    def manipulate_search_path(self, search_path: ConfigSearchPath) -> None:
        search_path.append(provider="habitat", path="config/")


register_hydra_plugin(HabitatConfigPlugin)

OVON = False


@hydra.main(
    version_base=None,
    config_path="../",
    config_name="exp_objnav",
)
def main(cfg: DictConfig) -> None:
    cfg = patch_config(cfg)
    execute_exp(cfg, "eval")


@hydra.main(
    version_base=None,
    config_path="../",
    config_name="exp_objnav",
)
def check_ovon(cfg: DictConfig) -> None:
    global OVON
    if "OVON" in cfg.habitat.dataset.type:
        OVON = True

if __name__ == "__main__":
    check_ovon()
    if OVON:
        import ovon  # noqa
    main()
"""
Sample command:
python -m frontier_exploration.generate_exploration_episodes \
    --config-name=exp_objnav.yaml \
    habitat.environment.max_episode_steps=2000
"""
