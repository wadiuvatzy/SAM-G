import gym
import numpy as np
from matplotlib import pyplot as plt
# import habitat.utils.gym_definitions
from typing import Any, NamedTuple
import dm_env
from dm_env import specs, Environment, TimeStep, StepType
from collections import deque
from dm_control.suite.wrappers import action_scale, pixels
import cv2
import matplotlib
import matplotlib.image as img
from habitat.utils.visualizations import maps


class ExtendedTimeStep(NamedTuple):
    step_type: Any
    reward: Any
    discount: Any
    observation: Any
    action: Any
    info: Any

    def first(self):
        return self.step_type == StepType.FIRST

    def mid(self):
        return self.step_type == StepType.MID

    def last(self):
        return self.step_type == StepType.LAST

    def __getitem__(self, attr):
        return getattr(self, attr)


class ExtendedTimeStepWrapper(dm_env.Environment):
    def __init__(self, env):
        self._env = env
        # self.reset()

    def reset(self):
        time_step = self._env.reset()
        # self.goal = self._env.goal
        return self._augment_time_step(time_step)

    def step(self, action):
        time_step, info = self._env.step(action)
        return self._augment_time_step(time_step, action, info)  # TODO GVRLB

    def _augment_time_step(self, time_step, action=None, info=None):
        if action is None:
            action_spec = self.action_spec()
            action = np.zeros(action_spec.shape, dtype=action_spec.dtype)
        return ExtendedTimeStep(observation=time_step.observation,
                                step_type=time_step.step_type,
                                action=action,
                                info=info,
                                reward=time_step.reward or 0.0,
                                discount=time_step.discount or 1.0)

    def observation_spec(self):
        return self._env.observation_spec()

    def action_spec(self):
        return self._env.action_spec()

    def __getattr__(self, name):
        return getattr(self._env, name)


class FrameStackWrapper(dm_env.Environment):
    def __init__(self, env, num_frames, pixels_key='pixels'):
        self._env = env
        self._num_frames = num_frames
        self._frames = deque([], maxlen=num_frames)
        self._pixels_key = pixels_key

        wrapped_obs_spec = env.observation_spec()

        pixels_shape = wrapped_obs_spec.shape
        # remove batch dim
        if len(pixels_shape) == 4:
            pixels_shape = pixels_shape[1:]
        self._obs_spec = specs.BoundedArray(shape=np.concatenate(
            [[pixels_shape[2] * num_frames], pixels_shape[:2]], axis=0),
            dtype=np.uint8,
            minimum=0,
            maximum=255,
            name='observation')
        self._success = 0

    def _transform_observation(self, time_step):
        assert len(self._frames) == self._num_frames
        obs = np.concatenate(list(self._frames), axis=0)
        return time_step._replace(observation=obs)

    def _extract_pixels(self, time_step, type):
        pixels = time_step.observation[type]
        # remove batch dim
        if len(pixels.shape) == 4:
            pixels = pixels[0]
        return pixels.transpose(2, 0, 1).copy()

    def reset(self):
        self._success = 0
        time_step = self._env.reset()
        # self.goal = self._extract_pixels(time_step, 'imagegoal')
        rgb = self._extract_pixels(time_step, 'rgb')
        # map = self._extract_pixels(time_step, 'map')
        for _ in range(self._num_frames):
            self._frames.append(rgb)
            # self._frames.append(map)
        return self._transform_observation(time_step)

    def step(self, action):
        time_step, info = self._env.step(action)
        rgb = self._extract_pixels(time_step, 'rgb')
        # map = self._extract_pixels(time_step, 'map')
        self._frames.append(rgb)
        # self._frames.append(map)
        self._success += info['success']
        info['success'] = self._success
        return self._transform_observation(time_step), info

    def observation_spec(self):
        return self._obs_spec

    def action_spec(self):
        return self._env.action_spec()

    def __getattr__(self, name):
        return getattr(self._env, name)


class ActionDTypeWrapper(dm_env.Environment):
    def __init__(self, env, dtype):
        self._env = env
        wrapped_action_spec = env.action_spec()
        self._action_spec = specs.BoundedArray(wrapped_action_spec.shape,
                                               dtype,
                                               wrapped_action_spec.minimum,
                                               wrapped_action_spec.maximum,
                                               'action')

    def step(self, action):
        action = action.astype(self._env.action_spec().dtype)
        return self._env.step(action)

    def observation_spec(self):
        return self._env.observation_spec()

    def action_spec(self):
        return self._action_spec

    def reset(self):
        return self._env.reset()

    def __getattr__(self, name):
        return getattr(self._env, name)


class ActionRepeatWrapper(dm_env.Environment):
    def __init__(self, env, num_repeats):
        self._env = env
        self._num_repeats = num_repeats

    def step(self, action):
        reward = 0.0
        discount = 1.0
        success = 0.0
        for i in range(self._num_repeats):
            time_step, info = self._env.step(action)
            success += info['success']
            reward += (time_step.reward or 0.0) * discount
            discount *= time_step.discount
            if time_step.last():
                break
        info['success'] = success

        return time_step._replace(reward=reward, discount=discount), info

    def observation_spec(self):
        return self._env.observation_spec()

    def action_spec(self):
        return self._env.action_spec()

    def reset(self):
        return self._env.reset()

    def __getattr__(self, name):
        return getattr(self._env, name)


class Gym2DMC(Environment):
    def __init__(self, gym_env) -> None:
        # gym_obs_space = gym_env.observation_space
        gym_obs_space = gym_env.observation_space['rgb']
        self._observation_spec = specs.BoundedArray(
            shape=gym_obs_space.shape,
            dtype=gym_obs_space.dtype,
            minimum=gym_obs_space.low,
            maximum=gym_obs_space.high,
            name='observation'
        )
        gym_act_space = gym_env.action_space
        self._action_spec = specs.BoundedArray(
            shape=gym_act_space.shape,
            dtype=gym_act_space.dtype,
            minimum=gym_act_space.low,
            maximum=gym_act_space.high,
            name='action'
        )
        self._gym_env = gym_env

    def step(self, action):
        obs, reward, done, info = self._gym_env.step(action)

        del obs['imagegoal']
        # obs['map'] = top_down_map
        if done:
            step_type = StepType.LAST
            discount = 0.0
        else:
            step_type = StepType.MID
            discount = 1.0

        return TimeStep(step_type=step_type,
                        reward=reward,
                        discount=discount,
                        observation=obs), info

    def reset(self):
        obs = self._gym_env.reset()

        return TimeStep(step_type=StepType.FIRST,
                        reward=None,
                        discount=None,
                        observation=obs)

    def render(self):
        try:
            img = self._gym_env.render(mode="rgb_array")
        except BaseException:
            img = np.zeros((512, 1795, 3))
        img = cv2.resize(img, dsize=(1795, 512), interpolation=cv2.INTER_CUBIC)
        return img

    def observation_spec(self):
        return self._observation_spec

    def action_spec(self):
        return self._action_spec


from habitat import *


def make_habitat_env(name, mode, seed, frame_stack=3, action_repeat=1, appearance_id='original'):
    # env = gym.make(name)
    env = make_env(name, mode=mode, seed=seed, appearance_id=appearance_id)
    # add wrappers
    env = Gym2DMC(env)
    env = ActionDTypeWrapper(env, np.float32)
    env = ActionRepeatWrapper(env, action_repeat)
    env = action_scale.Wrapper(env, minimum=-1.0, maximum=+1.0)
    env = FrameStackWrapper(env, num_frames=frame_stack)
    env = ExtendedTimeStepWrapper(env)

    return env