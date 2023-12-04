import sys
import dhm
import numpy as np
import warnings 

warnings.filterwarnings("ignore")
import os

os.environ["MKL_SERVICE_FORCE_INTEL"] = "1"
os.environ["MUJOCO_GL"] = "egl"

from distutils.dir_util import copy_tree
import torchvision.transforms as transforms
from pathlib import Path
import cv2
import imageio
import joblib
import hydra
import torch
from dm_env import StepType, TimeStep, specs
sys.path.append('../src')
import utils
from logger import Logger
from replay_buffer import ReplayBufferStorage, make_replay_loader
from video import TrainVideoRecorder, VideoRecorder
import pickle
import time
from tqdm import tqdm
import matplotlib.pyplot as plt
import cv2
import pickle
from demo_buffer import DemoDataSet, DemoDataLoader
from utils import random_overlay
import wandb
import torch.nn.functional as F

torch.backends.cudnn.benchmark = True

ENV_TYPE = 'adroit'
if ENV_TYPE == 'adroit':
    from adroit import AdroitEnv
else:
    import dmc
IS_ADROIT = True if ENV_TYPE == 'adroit' else False

import sys
import matplotlib.pyplot as plt
sys.path.append('../algos')

def make_agent(obs_spec, action_spec, cfg):
	cfg.obs_shape = obs_spec.shape
	cfg.action_shape = action_spec.shape
	return hydra.utils.instantiate(cfg)

class Workspace:
	def __init__(self, cfg):
		self.work_dir = Path.cwd()
        print(f'workspace: {self.work_dir}')
		self.direct_folder_name = os.path.basename(self.work_dir)

		self.cfg = cfg
		utils.set_seed_everywhere(cfg.seed)
		self.device = torch.device(cfg.device)
		self.eval_env = dhm.make_env_RRL_sam(self.env_name, test_image=False, num_repeats=self.cfg.action_repeat, input_resolution=self.cfg.input_resolution,
                                          num_frames=self.cfg.frame_stack, env_feature_type=self.env_feature_type, use_SAM_g=self.cfg.use_SAM_g, handmade_4_SAM=self.cfg.handmade_4_SAM,
                                          device=self.device, reward_rescale=self.cfg.reward_rescale, mode='test', seed=self.cfg.seed, level='video-hard')

		self.load_demos()

		self.agent = make_agent(self.eval_env.observation_spec(),
								self.eval_env.action_spec(),
								self.cfg.agent)

		self.timer = utils.Timer()
		if self.cfg.task_name == 'pen':
			self._global_step = int(1e6)
		else:
			self._global_step = int(5e5)
		self._global_episode = 0


	def load_demos(self):
		self.demo_dataset = DemoDataSet(task_name='pen', num_demos=self.cfg.num_demos)
		self.demo_loader = DemoDataLoader(dataset=self.demo_dataset, batch_size=self.cfg.batch_size)
	

	def save_snapshot(self, suffix=None):
		if suffix is None:
			save_name = 'snapshot.pt'
		else:
			save_name = 'snapshot_' + suffix + '.pt'
		snapshot = self.work_dir / save_name
		keys_to_save = ['agent', 'timer', '_global_step', '_global_episode']
		payload = {k: self.__dict__[k] for k in keys_to_save}
		with snapshot.open('wb') as f:
			torch.save(payload, f)
		print("snapshot saved to:", str(snapshot))

	def load_snapshot(self):
		snapshot = self.work_dir / 'snapshot.pt'
		with snapshot.open('rb') as f:
			payload = torch.load(f)
		for k, v in payload.items():
			self.__dict__[k] = v
		
	def eval_adroit(self,  force_number_episodes=None, do_log=True):

		step, episode, total_reward = 0, 0, 0
		n_eval_episode = force_number_episodes if force_number_episodes is not None else self.cfg.num_eval_episodes
		eval_until_episode = utils.Until(n_eval_episode)
		total_success = 0.0
		for i in tqdm(range(n_eval_episode)):
			n_goal_achieved_total = 0
			time_step = self.eval_env.reset()
			while not time_step.last():
				with torch.no_grad(), utils.eval_mode(self.agent):
					observation = time_step.observation
					action = self.agent.act(observation,
											self._global_step,
											eval_mode=True,
											obs_sensor=time_step.observation_sensor)
				time_step = self.eval_env.step(action)
				n_goal_achieved_total += time_step.n_goal_achieved
				total_reward += time_step.reward
				step += 1

			# here check if success for Adroit tasks. The threshold values come from the mj_envs code
			# e.g. https://github.com/ShahRutav/mj_envs/blob/5ee75c6e294dda47983eb4c60b6dd8f23a3f9aec/mj_envs/hand_manipulation_suite/pen_v0.py
			# can also use the evaluate_success function from Adroit envs, but can be more complicated
			if self.cfg.task_name == 'pen-v0':
				threshold = 20
			else:
				threshold = 25
			if n_goal_achieved_total > threshold:
				total_success += 1

			episode += 1
		success_rate_standard = total_success / n_eval_episode
		episode_reward_standard = total_reward / episode
		episode_length_standard = step * self.cfg.action_repeat / episode

		print("eval for {} episodes: reward {}, success rate {}".format(n_eval_episode, episode_reward_standard, success_rate_standard))

		return episode_reward_standard, success_rate_standard

@hydra.main(config_path='./', config_name='pieg_config')
def main(cfg):
	W = Workspace
	root_dir = Path.cwd()
    if 'use_SAM_g' not in cfg:
        cfg.use_SAM_g = False
        cfg.handmade_4_SAM = None
    workspace = W(cfg)
	snapshot = workspace.work_dir / 'snapshot.pt'
	if snapshot.exists():
		print("here")
		workspace.load_snapshot()
	workspace.eval_adroit(force_number_episodes=50)

if __name__ == '__main__':
	main()

