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
sys.path.append('../')
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
        print("====== Training log stored to: ======")
        print(f'workspace: {self.work_dir}')
		self.direct_folder_name = os.path.basename(self.work_dir)

		self.cfg = cfg
		utils.set_seed_everywhere(cfg.seed)
		self.device = torch.device(cfg.device)
		self.eval_env = dhm.make_env_RRL_sam(self.env_name, test_image=False, num_repeats=self.cfg.action_repeat, input_resolution=self.cfg.input_resolution,
                                          num_frames=self.cfg.frame_stack, env_feature_type=self.env_feature_type, use_SAM_g=self.cfg.use_SAM_g, handmade_4_SAM=self.cfg.handmade_4_SAM,
                                          device=self.device, reward_rescale=self.cfg.reward_rescale, mode='test', seed=self.cfg.seed, level='video-easy')

		self.load_demos()

		self.agent = make_agent(self.eval_env.observation_spec(),
								self.eval_env.action_spec(),
								self.cfg.agent)

		if self.cfg.use_wandb:
			wandb.init(
				group="adroit_imitation_learning",
				project=cfg.task_name,
				name="seed=" + str(cfg.seed)
			)

		self.timer = utils.Timer()
		if self.cfg.task_name == 'pen':
			self._global_step = int(1e6)
		else:
			self._global_step = int(5e5)
		self._global_episode = 0

		self.optimizer = torch.optim.Adam([
			{'params': self.agent.actor.parameters(), 'lr': self.cfg.train_demo_lr},
			{'params': self.agent.encoder.parameters(), 'lr': self.cfg.train_demo_lr}
		])

		

	def __del__(self):
		if self.cfg.use_wandb:
			wandb.finish()

	def load_demos(self):
		self.demo_dataset = DemoDataSet(task_name='pen', num_demos=self.cfg.num_demos)
		self.demo_loader = DemoDataLoader(dataset=self.demo_dataset, batch_size=self.cfg.batch_size)
	
	def train_demos(self):
		self.agent.train()
		losses = []
		for i in range(self.cfg.num_train_imitation_epoch):
			
			self.demo_loader.shuffle()
			for j in range(len(self.demo_loader)):
				obs, sensor, action = self.demo_loader[j]
				# plt.imshow(obs[0].permute(1, 2, 0)[:, :,].cpu().numpy())
				# plt.savefig('obs.png')
				# input("Press Enter to continue...")
				obs = obs.to(self.device)
				obs_sensor = sensor.to(self.device)
				action_target = action.to(self.device)
				obs = self.agent.aug(obs.float())
				original_obs = obs.clone()
				aug_obs = self.agent.encoder(random_overlay(original_obs, self.device))
				
				obs = torch.cat([aug_obs, obs_sensor], dim=1)

				action_dist = self.agent.actor(obs, 0.1)
				current_mean_action = action_dist.mean

				loss = F.mse_loss(current_mean_action, action_target)
				self.optimizer.zero_grad()
				loss.backward()
				self.optimizer.step()
				losses.append(loss.item())

				if j % 20 == 0:
					print('epoch: ', i, ' iter: ', j, ' loss: ', loss.item())
					if self.cfg.use_wandb:
						wandb.log({"loss": loss.item()})
			
			if (i + 1) % 5 == 0:
				eval_reward, eval_success_rate = self.eval_adroit()
				print('epoch: ', i, ' eval_reward: ', eval_reward, ' eval_success_rate: ', eval_success_rate)
				self.save_snapshot(suffix=str(i))

		self.save_snapshot()

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
		while eval_until_episode(episode):
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

		if self.cfg.use_wandb:
			wandb.log({"eval_adroit_reward": total_reward/episode})
			wandb.log({"eval_adroit_success_rate": success_rate_standard})

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
		workspace.load_snapshot()
	workspace.train_demos()

if __name__ == '__main__':
	main()

