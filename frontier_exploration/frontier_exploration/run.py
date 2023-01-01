from habitat.config import read_write
from habitat_baselines.config.default import get_config
from habitat_baselines.run import build_parser, execute_exp
from omegaconf import OmegaConf


def main():
    parser = build_parser()

    args = parser.parse_args()
    run_exp(**vars(args))


def run_exp(exp_config: str, run_type: str, opts=None) -> None:
    r"""Runs experiment given mode and config

    Args:
        exp_config: path to config file.
        run_type: "train" or "eval".
        opts: list of strings of additional config options.

    Returns:
        None.
    """
    config = get_config(exp_config, opts)
    # If we are training, we remove the rgb sensors from the config
    if run_type == "train":
        with read_write(config):
            for k in list(
                config.habitat.simulator.agents.main_agent.sim_sensors.keys()
            ):
                if k in ["third_rgb_sensor", "rgb_sensor"]:
                    del config.habitat.simulator.agents.main_agent.sim_sensors[k]
    # print(OmegaConf.to_yaml(config))  # good for debugging
    execute_exp(config, run_type)


if __name__ == "__main__":
    main()
