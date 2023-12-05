#!/usr/bin/env python3

# Copyright (c) Meta Platforms, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os.path as osp
from glob import glob
from typing import TYPE_CHECKING, Any, List, Optional

import gym
from gym.envs.registration import register, registry

import habitat
import habitat.utils.env_utils
from habitat.config.default import _HABITAT_CFG_DIR
from habitat.config.default_structured_configs import ThirdRGBSensorConfig
from habitat.core.environments import get_env_class

if TYPE_CHECKING:
    from omegaconf import DictConfig


gym_task_config_dir = osp.join(_HABITAT_CFG_DIR, "benchmark/")


def _get_gym_name(cfg: "DictConfig") -> Optional[str]:
    if "habitat" in cfg:
        cfg = cfg.habitat
    if "gym" in cfg and "auto_name" in cfg["gym"]:
        return cfg["gym"]["auto_name"]
    return None


def _get_env_name(cfg: "DictConfig") -> Optional[str]:
    if "habitat" in cfg:
        cfg = cfg.habitat
    return cfg["env_task"]


def make_gym_from_config(config: "DictConfig") -> gym.Env:
    """
    From a habitat-lab or habitat-baseline config, create the associated gym environment.
    """
    if "habitat" in config:
        config = config.habitat
    env_class_name = _get_env_name(config)
    env_class = get_env_class(env_class_name)
    assert (
        env_class is not None
    ), f"No environment class with name `{env_class_name}` was found, you need to specify a valid one with env_task"
    return habitat.utils.env_utils.make_env_fn(
        env_class=env_class, config=config
    )


def _make_habitat_gym_env(
    cfg_file_path: str,
    override_options: List[Any] = None,
    use_render_mode: bool = False,
) -> gym.Env:
    if override_options is None:
        override_options = []

    config = habitat.get_config(cfg_file_path, overrides=override_options)
    if use_render_mode:
        with habitat.config.read_write(config):
            sim_config = config.habitat.simulator
            default_agent_name = sim_config.agents_order[
                sim_config.default_agent_id
            ]
            default_agent = sim_config.agents[default_agent_name]
            if len(sim_config.agents) == 1:
                default_agent.sim_sensors.update(
                    {"third_rgb_sensor": ThirdRGBSensorConfig()}
                )
            else:
                default_agent.sim_sensors.update(
                    {
                        "default_agent_third_rgb_sensor": ThirdRGBSensorConfig(
                            uuid="default_robot_third_rgb"
                        )
                    }
                )
    env = make_gym_from_config(config)
    return env


def _try_register(id_name, entry_point, kwargs):
    if id_name in registry.env_specs:
        return
    register(
        id_name,
        entry_point=entry_point,
        kwargs=kwargs,
    )


if "Habitat-v0" not in registry.env_specs:
    # Generic supporting general configs
    _try_register(
        id_name="Habitat-v0",
        entry_point="habitat.utils.gym_definitions:_make_habitat_gym_env",
        kwargs={},
    )

    _try_register(
        id_name="HabitatRender-v0",
        entry_point="habitat.utils.gym_definitions:_make_habitat_gym_env",
        kwargs={"use_render_mode": True},
    )

    hab_baselines_dir = osp.dirname(osp.dirname(osp.abspath(__file__)))
    gym_template_handle = "Habitat%s-v0"
    render_gym_template_handle = "HabitatRender%s-v0"

    for fname in glob(
        osp.join(gym_task_config_dir, "**/*.yaml"), recursive=True
    ):
        full_path = osp.join(gym_task_config_dir, fname)
        if not fname.endswith(".yaml"):
            continue
        cfg_data = habitat.get_config(full_path)
        gym_name = _get_gym_name(cfg_data)
        if gym_name is not None:
            # Register this environment name with this config
            _try_register(
                id_name=gym_template_handle % gym_name,
                entry_point="habitat.utils.gym_definitions:_make_habitat_gym_env",
                kwargs={"cfg_file_path": full_path},
            )

            _try_register(
                id_name=render_gym_template_handle % gym_name,
                entry_point="habitat.utils.gym_definitions:_make_habitat_gym_env",
                kwargs={"cfg_file_path": full_path, "use_render_mode": True},
            )
