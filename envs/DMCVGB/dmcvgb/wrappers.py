import numpy as np
from numpy.random import randint
import os
import gym
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
import dmc2gym
from dmcvgb import utils
from collections import deque
import cv2
from dmcvgb import persam_utils as psu
import sys


class ColorWrapper(gym.Wrapper):
	"""Wrapper for the color experiments"""
	def __init__(self, env, background, seed=None, objects_color='original', table_texure='original', light_position='original', light_color='original', moving_light='original', cam_pos='original'):
		# assert isinstance(env, FrameStack), 'wrapped env must be a framestack'
		gym.Wrapper.__init__(self, env)
		self._max_episode_steps = env._max_episode_steps
		self._background = background
		self._random_state = np.random.RandomState(seed)
		self.time_step = 0
		if self._background['type'] == 'color' or objects_color != 'original':
			self._load_colors()
		self._table_texure = table_texure
		self._objects_color = objects_color
		self._light_position = light_position
		self._light_color = light_color
		# self.model = self.env.env.env.env.env._env.physics.model
		self._moving_light = moving_light
		self._origin_light_pos = env.env.env.env.env._env.physics.model.light_pos
		self._origin_light_diffuse = env.env.env.env.env._env.physics.model.light_diffuse
		self._cam_pos = cam_pos
		# self._origin_cam_pos = env.env.env.env.env._env.physics.model.cam_pos
		# self._origin_cam_pos = env.env.env.env.env._env.physics.named.data.cam_xpos
		if self._moving_light == 'easy':
			self.step_pos = 0.01
			self.light_pos_range = [self._origin_light_pos[:, 1] - 5, self._origin_light_pos[:, 1] + 5]
		elif self._moving_light == 'hard':
			self.step_pos = 0.05
			self.light_pos_range = [self._origin_light_pos[:, 1] - 10, self._origin_light_pos[:, 1] + 10]

	def reset(self):
		self.time_step = 0
		setting_kwargs = {}
		if self._background['type'] == 'color':
			# self.randomize()
			# if self._background["difficulty"] != 'original':
			background_color = getattr(self, f'_colors_{self._background["difficulty"]}')[
				self._random_state.randint(len(getattr(self, f'_colors_{self._background["difficulty"]}')))]
			setting_kwargs['grid_rgb1'] = background_color['grid_rgb1']
			setting_kwargs['skybox_rgb'] = background_color['skybox_rgb']
			setting_kwargs['grid_rgb2'] = background_color['grid_rgb2']
		if self._objects_color != 'original':
			self_color = getattr(self, f'_colors_{self._objects_color}')[
				self._random_state.randint(len(getattr(self, f'_colors_{self._objects_color}')))]
			setting_kwargs['self_rgb'] = self_color['self_rgb']

		if self._background['type'] == 'video':
			# apply greenscreen
			# setting_kwargs = {
			# 	# 'skybox_rgb': [.2, .8, .2],
			# 	# 'skybox_rgb2': [.2, .8, .2],
			# 	# 'skybox_markrgb': [.2, .8, .2]
			# 	'skybox_rgb': [.0, .0, .0],
			# 	'skybox_rgb2': [.0, .0, .0],
			# 	'skybox_markrgb': [.0, .0, .0]
			# }
			setting_kwargs['skybox_rgb'] = [.0, .0, .0]
			setting_kwargs['skybox_rgb2'] = [.0, .0, .0]
			setting_kwargs['skybox_markrgb'] = [.0, .0, .0]

			if self._background['difficulty'] == 'hard':
				# setting_kwargs['grid_rgb1'] = [.2, .8, .2]
				# setting_kwargs['grid_rgb2'] = [.2, .8, .2]
				# setting_kwargs['grid_markrgb'] = [.2, .8, .2]
				setting_kwargs['grid_rgb1'] = [.0, .0, .0]
				setting_kwargs['grid_rgb2'] = [.0, .0, .0]
				setting_kwargs['grid_markrgb'] = [.0, .0, .0]
		self.reload_physics(setting_kwargs)
		if self._light_position == 'easy':
			self.env.env.env.env.env._env.physics.model.light_pos = self._origin_light_pos + self._random_state.randint(3,5,size=self._origin_light_pos.shape)
		elif self._light_position == 'hard':
			self.env.env.env.env.env._env.physics.model.light_pos = self._origin_light_pos + self._random_state.randint(9,11,size=self._origin_light_pos.shape)
		if self._light_color == 'easy':
			self.env.env.env.env.env._env.physics.model.light_diffuse = self._origin_light_diffuse + self._random_state.uniform(-0.2,0.2,size=self._origin_light_diffuse.shape)
		elif self._light_color == 'hard':
			# self.env.env.env.env.env._env.physics.model.light_diffuse = np.array([0.2, 0.7, 0.7])
			self.env.env.env.env.env._env.physics.model.light_diffuse = self._origin_light_diffuse + self._random_state.uniform(-10,10,size=self._origin_light_diffuse.shape)

		return self.env.reset()

	def step(self, action):
		self.time_step += 1
		if self._moving_light != 'original':
			self.env.env.env.env.env._env.physics.model.light_pos[:, :] += self.step_pos
			if self.env.env.env.env.env._env.physics.model.light_pos[:, 0].all() > self.light_pos_range[1].all() or self.env.env.env.env.env._env.physics.model.light_pos[:, 0].all() < \
					self.light_pos_range[0].all():
				self.step_pos *= -1
		return self.env.step(action)

	def randomize(self):
		# assert 'color' in self._mode, f'can only randomize in color mode, received {self._mode}'
		self.reload_physics(self.get_random_color())

	def _load_colors(self):
		# assert self._mode in {'color_easy', 'color_hard'}
		self._colors_easy = torch.load(f'{os.path.dirname(__file__)}/../data/color_easy.pt')
		self._colors_hard = torch.load(f'{os.path.dirname(__file__)}/../data/color_hard.pt')

	def get_random_color(self):
		# assert len(self._colors) >= 100, 'env must include at least 100 colors'
		color = {}
		if self._background['type'] != 'original':
			background_color = getattr(self, f'_colors_{self._background["difficulty"]}')[self._random_state.randint(len(getattr(self, f'_colors_{self._background["difficulty"]}')))]
			color['grid_rgb1'] = background_color['grid_rgb1']
			color['skybox_rgb'] = background_color['skybox_rgb']
			color['grid_rgb2'] = background_color['grid_rgb2']
		if self._objects_color != 'original':
			self_color = getattr(self, f'_colors_{self._objects_color}')[self._random_state.randint(len(getattr(self, f'_colors_{self._objects_color}')))]
			color['self_rgb'] = self_color['self_rgb']
		# yang = self_color
		return color

	def reload_physics(self, setting_kwargs=None, state=None):
		from dm_control.suite import common
		domain_name = self._get_dmc_wrapper()._domain_name
		if domain_name == 'unitree':
			domain_name = 'mujoco_menagerie/unitree_a1/scene_' + self._get_dmc_wrapper()._task_name
			if self._table_texure != 'original':
				setting_kwargs['ground_texture'] = 'table_' + self._table_texure + str(self._random_state.randint(10))
		if domain_name == 'franka':
			domain_name = 'mujoco_menagerie/franka_emika_panda/scene_' + self._get_dmc_wrapper()._task_name
			if self._table_texure != 'original':
				setting_kwargs['table_texture'] = 'table_' + self._table_texure + str(self._random_state.randint(10))
			if self._objects_color != 'original':
				setting_kwargs['self_rgb1'] = None
		elif domain_name == 'quadruped':
			domain_name = domain_name + '_' + self._get_dmc_wrapper()._task_name
		if setting_kwargs is None:
			setting_kwargs = {}
		# if state is None:
		# 	state = self._get_state()

		self._reload_physics(
			*common.settings.get_model_and_assets_from_setting_kwargs(
				domain_name+'.xml', setting_kwargs
			)
		)
		# self._set_state(state)
	
	def get_state(self):
		return self._get_state()
	
	def set_state(self, state):
		self._set_state(state)

	def _get_dmc_wrapper(self):
		_env = self.env
		while not isinstance(_env, dmc2gym.wrappers.DMCWrapper) and hasattr(_env, 'env'):
			_env = _env.env
		assert isinstance(_env, dmc2gym.wrappers.DMCWrapper), 'environment is not dmc2gym-wrapped'

		return _env

	def _reload_physics(self, xml_string, assets=None):
		_env = self.env
		while not hasattr(_env, '_physics') and hasattr(_env, 'env'):
			_env = _env.env
		assert hasattr(_env, '_physics'), 'environment does not have physics attribute'
		key_list = list(assets.keys())
		value_list = list(assets.values())
		for i in range(len(key_list)):
			if type(assets[key_list[i]]).__name__ == 'str':
				assets[key_list[i]] = bytes(value_list[i], 'utf-8')

		_env.physics.reload_from_xml_string(xml_string, assets=assets, domain_name=_env._domain_name)

	def _get_physics(self):
		_env = self.env
		while not hasattr(_env, '_physics') and hasattr(_env, 'env'):
			_env = _env.env
		assert hasattr(_env, '_physics'), 'environment does not have physics attribute'

		return _env._physics

	def _get_state(self):
		return self._get_physics().get_state()
		
	def _set_state(self, state):
		self._get_physics().set_state(state)


class FrameStack(gym.Wrapper):
	"""Stack frames as observation"""
	def __init__(self, env, k):
		gym.Wrapper.__init__(self, env)
		self._k = k
		self._frames = deque([], maxlen=k)
		shp = env.observation_space.shape
		self.observation_space = gym.spaces.Box(
			low=0,
			high=1,
			shape=((shp[0] * k,) + shp[1:]),
			dtype=env.observation_space.dtype
		)
		self._max_episode_steps = env._max_episode_steps

	def reset(self):
		obs = self.env.reset()
		for _ in range(self._k):
			self._frames.append(obs)
		return self._get_obs()

	def step(self, action):
		obs, reward, done, info = self.env.step(action)
		# obs, reward, done, _, info = self.env.step(action)  
		self._frames.append(obs)
		return self._get_obs(), reward, done, info

	def _get_obs(self):
		assert len(self._frames) == self._k
		return utils.LazyFrames(list(self._frames))


sys.path.append("efficientvit")
from efficientvit.sam_model_zoo import create_sam_model
from efficientvit.models.efficientvit.sam import EfficientViTSamPredictor
import torchvision.transforms as T

class FrameStackWithSAM(gym.Wrapper):
	def __init__(self, env, k, specific_image=None, specific_masked=None, args=None):
		gym.Wrapper.__init__(self, env)
		self._k = k
		self._frames = deque([], maxlen=k)
		shp = env.observation_space.shape
		self.observation_space = gym.spaces.Box(
			low=0,
			high=1,
			shape=((shp[0] * k,) + shp[1:]),
			dtype=env.observation_space.dtype
		)
		self._max_episode_steps = env._max_episode_steps

		efficientvit_sam = create_sam_model(
			name='l1', weight_url="efficientvit/checkpoints/l1.pt"
		)

		efficientvit_sam = efficientvit_sam.cuda().eval()
		self.efficientvit_sam_predictor = EfficientViTSamPredictor(efficientvit_sam)

		print("loading dino model")
		self.dino_model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14')
		# if due to some reasons, the above command doesn't work, you can load the model from a local path
		# you need to put the downloaded checkpoint in your ~/.cache/torch/hub/checkpoints/ directory
		# you should clone the dinov2 repo and put the path to the local directory (similar as efficientvit)
		# self.dino_model = torch.hub._load_local('dinov2', 'dinov2_vitb14')
		self.dino_model.cuda().eval()
		print("dino model loaded")

		self.dino_transform = T.Compose([T.ToTensor(),
							 T.Resize(448),
							 T.Normalize(
								 mean=[0.485, 0.456, 0.406],
								 std=[0.229, 0.224, 0.225],
							 ),
							 ])
		self.dino_model.eval()
		
		target_feat_list = []
		dino_target_feat_list = []
		ref_image = cv2.imread(args.original_image)
		ref_image = cv2.cvtColor(ref_image, cv2.COLOR_BGR2RGB)

		ref_mask = cv2.imread(args.masked_image)
		ref_mask = cv2.cvtColor(ref_mask, cv2.COLOR_BGR2RGB)

		dino_ref_image = ref_image.copy()
		dino_ref_image = self.dino_transform(dino_ref_image).unsqueeze(0).cuda()
		dino_ref_image_embedding = self.dino_model.forward_features(dino_ref_image)
		patch_tokens = dino_ref_image_embedding["x_norm_patchtokens"]
		patch_tokens = patch_tokens.reshape([1, 32, 32, 768])
		patch_tokens = patch_tokens.permute(0, 3, 1, 2)
		patch_tokens = F.interpolate(patch_tokens, size=(64, 64), mode='bilinear', align_corners=False)
		patch_tokens = patch_tokens.permute(0, 2, 3, 1)
		
		dino_ref_feat = patch_tokens.squeeze(0) # [64, 64, 768]
		gt_mask = torch.tensor(ref_mask)[:, :, 0] > 0 
		gt_mask = gt_mask.float().unsqueeze(0).flatten(1).cuda()

		ref_mask = self.efficientvit_sam_predictor.Per_set_image(ref_image, ref_mask)
		ref_feat = self.efficientvit_sam_predictor.features.squeeze(0).permute(1, 2, 0)
		ref_mask = F.interpolate(ref_mask, size=ref_feat.shape[:2], mode='bilinear', align_corners=False) 
		ref_mask = ref_mask.squeeze()[0]

		target_feat = ref_feat[ref_mask > 0]
		target_feat_mean = target_feat.mean(0)
		target_feat_max = torch.max(target_feat, dim=0)[0]
		target_feat = (target_feat_max / 2 + target_feat_mean / 2).unsqueeze(0)
		target_feat = target_feat / target_feat.norm(dim=-1, keepdim=True)
		target_feat_list.append(target_feat)

		dino_target_feat = dino_ref_feat[ref_mask > 0]
		dino_target_feat_mean = dino_target_feat.mean(0)
		dino_target_feat_max = torch.max(dino_target_feat, dim=0)[0]
		dino_target_feat = (dino_target_feat_max / 2 + dino_target_feat_mean / 2).unsqueeze(0)
		dino_target_feat = dino_target_feat / dino_target_feat.norm(dim=-1, keepdim=True)
		dino_target_feat_list.append(dino_target_feat)

		h, w, C = ref_feat.shape
		ref_feat_ = ref_feat / ref_feat.norm(dim=-1, keepdim=True)
		ref_feat_ = ref_feat_.permute(2, 0, 1).reshape(C, -1)
		sim = target_feat @ ref_feat_
		sim = sim.reshape(1, 1, h, w)
		sim = F.interpolate(sim, scale_factor=4, mode='bilinear')
		sim = self.efficientvit_sam_predictor.Per_postprocess_masks(
			sim,
			input_size=self.efficientvit_sam_predictor.input_size,
			original_size=self.efficientvit_sam_predictor.original_size,
		).squeeze()

		h, w, C = dino_ref_feat.shape
		dino_ref_feat_ = dino_ref_feat / dino_ref_feat.norm(dim=-1, keepdim=True)
		dino_ref_feat_ = dino_ref_feat_.permute(2, 0, 1).reshape(C, -1)
		dino_sim = dino_target_feat @ dino_ref_feat_
		dino_sim = dino_sim.reshape(1, 1, h, w)
		dino_sim = F.interpolate(dino_sim, scale_factor=4, mode='bilinear')
		dino_sim = self.efficientvit_sam_predictor.Per_postprocess_masks(
			dino_sim,
			input_size=self.efficientvit_sam_predictor.input_size,
			original_size=self.efficientvit_sam_predictor.original_size,
		).squeeze()

		# combine sim and dino_sim
		sim = (dino_sim + sim) / 2

		topk_xy, topk_label = psu.point_selection(sim, topk=1)

		n_xy, n_label = psu.negative_point_selection(sim, topk=1)
		topk_xy = np.concatenate((topk_xy, n_xy), axis=0)
		topk_label = np.concatenate((topk_label, n_label), axis=0)

		parts_target_feat = []
		dino_parts_target_feat = []
		for point in args.extra_points_list:
			y, x = point
			x = x / 84. * 64
			y = y / 84. * 64
			x1 = int(x); x2 = int(x) + 1
			y1 = int(y); y2 = int(y) + 1
			feat = ref_feat[x1, y1] * (x2 - x) * (y2 - y) \
				+ ref_feat[x1, y2] * (x2 - x) * (y - y1) \
				+ ref_feat[x2, y1] * (x - x1) * (y2 - y) \
				+ ref_feat[x2, y2] * (x - x1) * (y - y1)
			feat = feat / feat.norm(dim=-1, keepdim=True)
			parts_target_feat.append(feat)

			feat = dino_ref_feat[x1, y1] * (x2 - x) * (y2 - y) \
				+ dino_ref_feat[x1, y2] * (x2 - x) * (y - y1) \
				+ dino_ref_feat[x2, y1] * (x - x1) * (y2 - y) \
				+ dino_ref_feat[x2, y2] * (x - x1) * (y - y1)
			feat = feat / feat.norm(dim=-1, keepdim=True)
			dino_parts_target_feat.append(feat)

		for i in args.extra_masked_images_list:
			ref_mask = cv2.imread(i)
			ref_mask = cv2.cvtColor(ref_mask, cv2.COLOR_BGR2RGB)
			ref_mask = self.efficientvit_sam_predictor.Per_set_image(ref_image, ref_mask)
			ref_feat = self.efficientvit_sam_predictor.features.squeeze(0).permute(1, 2, 0)
			ref_mask = F.interpolate(ref_mask, size=ref_feat.shape[:2], mode='bilinear', align_corners=False)
			ref_mask = ref_mask.squeeze()[0]

			target_feat = ref_feat[ref_mask > 0]
			target_feat_mean = target_feat.mean(0)
			target_feat_max = torch.max(target_feat, dim=0)[0]
			target_feat = (target_feat_max / 2 + target_feat_mean / 2).unsqueeze(0)
			target_feat = target_feat / target_feat.norm(dim=-1, keepdim=True)
			target_feat_list.append(target_feat)

			h, w, C = ref_feat.shape
			ref_feat = ref_feat / ref_feat.norm(dim=-1, keepdim=True)
			ref_feat = ref_feat.permute(2, 0, 1).reshape(C, -1)
			sim = target_feat @ ref_feat

			sim = sim.reshape(1, 1, h, w)
			sim = F.interpolate(sim, scale_factor=4, mode='bilinear')
			sim = self.efficientvit_sam_predictor.Per_postprocess_masks(
				sim,
				input_size=self.efficientvit_sam_predictor.input_size,
				original_size=self.efficientvit_sam_predictor.original_size,
			).squeeze()
			
			dino_target_feat = dino_ref_feat[ref_mask > 0]
			dino_target_feat_mean = dino_target_feat.mean(0)
			dino_target_feat_max = torch.max(dino_target_feat, dim=0)[0]
			dino_target_feat = (dino_target_feat_max / 2 + dino_target_feat_mean / 2).unsqueeze(0)
			dino_target_feat = dino_target_feat / dino_target_feat.norm(dim=-1, keepdim=True)
			dino_target_feat_list.append(dino_target_feat)

			h, w, C = dino_ref_feat.shape
			dino_ref_feat_ = dino_ref_feat / dino_ref_feat.norm(dim=-1, keepdim=True)
			dino_ref_feat_ = dino_ref_feat_.permute(2, 0, 1).reshape(C, -1)
			dino_sim = dino_target_feat @ dino_ref_feat_
			dino_sim = dino_sim.reshape(1, 1, h, w)
			dino_sim = F.interpolate(dino_sim, scale_factor=4, mode='bilinear')
			dino_sim = self.efficientvit_sam_predictor.Per_postprocess_masks(
				dino_sim,
				input_size=self.efficientvit_sam_predictor.input_size,
				original_size=self.efficientvit_sam_predictor.original_size,
			).squeeze()

			sim = (dino_sim + sim) / 2
			topk_xy_, topk_label_ = psu.point_selection(sim, topk=1)
			topk_xy = np.concatenate((topk_xy, topk_xy_), axis=0)
			topk_label = np.concatenate((topk_label, topk_label_), axis=0)

		mask_weights = psu.Mask_Weights().cuda()
		mask_weights.train()

		optimizer = torch.optim.AdamW(mask_weights.parameters(), lr=1e-3, eps=1e-4)
		scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, 1000)

		for train_idx in range(1000):
			xy = topk_xy.copy()
			label = topk_label.copy()

			for i in range(len(dino_parts_target_feat)):
				part_feat = parts_target_feat[i]
				noise = torch.randn_like(part_feat)
				noise = noise / noise.norm(dim=-1, keepdim=True)
				part_feat = part_feat + noise * 0.01
				part_feat = part_feat / part_feat.norm(dim=-1, keepdim=True)
				sim = part_feat @ ref_feat_
				sim = sim.reshape(1, 1, h, w)
				sim = F.interpolate(sim, scale_factor=4, mode='bilinear')
				sim = self.efficientvit_sam_predictor.Per_postprocess_masks(
					sim,
					input_size=self.efficientvit_sam_predictor.input_size,
					original_size=self.efficientvit_sam_predictor.original_size,
				).squeeze()

				dino_target_feat = dino_parts_target_feat[i]
				noise = torch.randn_like(dino_target_feat)
				noise = noise / noise.norm(dim=-1, keepdim=True)
				dino_target_feat = dino_target_feat + noise * 0.01
				dino_target_feat = dino_target_feat / dino_target_feat.norm(dim=-1, keepdim=True)
				dino_sim = dino_target_feat @ dino_ref_feat_
				dino_sim = dino_sim.reshape(1, 1, h, w)
				dino_sim = F.interpolate(dino_sim, scale_factor=4, mode='bilinear')
				dino_sim = self.efficientvit_sam_predictor.Per_postprocess_masks(
					dino_sim,
					input_size=self.efficientvit_sam_predictor.input_size,
					original_size=self.efficientvit_sam_predictor.original_size,
				).squeeze()

				sim = (dino_sim + sim) / 2
				xy_, label_ = psu.point_selection(sim, topk=1)
				xy = np.concatenate((xy, xy_), axis=0)
				label = np.concatenate((label, label_), axis=0)
			masks, scores, logits, logits_high = self.efficientvit_sam_predictor.predict(
            point_coords=xy,
            point_labels=label,
			multimask_output=True)
			logits_high = logits_high.flatten(1).clone()

			# Weighted sum three-scale masks
			weights = torch.cat((1 - mask_weights.weights.sum(0).unsqueeze(0), mask_weights.weights), dim=0)
			logits_high = logits_high * weights
			logits_high = logits_high.sum(0).unsqueeze(0)

			dice_loss = psu.calculate_dice_loss(logits_high, gt_mask)
			focal_loss = psu.calculate_sigmoid_focal_loss(logits_high, gt_mask)
			loss = dice_loss + focal_loss

			optimizer.zero_grad()
			loss.backward()
			optimizer.step()
			scheduler.step()
		print("Finish training mask weights")

		self.weights = torch.cat((1 - mask_weights.weights.sum(0).unsqueeze(0), mask_weights.weights), dim=0)
		self.weights_np = self.weights.detach().cpu().numpy()

		self.target_feat = target_feat_list[0]
		self.dino_target_feat = dino_target_feat_list[0]
		if len(target_feat_list) > 1:
			self.part_target_feat = target_feat_list[1:] + parts_target_feat
			self.dino_parts_target_feat = dino_target_feat_list[1:] + dino_parts_target_feat
		else:
			self.part_target_feat = parts_target_feat
			self.dino_parts_target_feat = dino_parts_target_feat

	def segment_obs(self, obs):
		obs_feat = obs.copy()
		dino_obs_feat = self.dino_transform(obs_feat.transpose(1, 2, 0)).unsqueeze(0).cuda()
		dino_obs_feat = self.dino_model.forward_features(dino_obs_feat)
		patch_tokens = dino_obs_feat["x_norm_patchtokens"]
		patch_tokens = patch_tokens.reshape([1, 32, 32, 768])
		patch_tokens = patch_tokens.permute(0, 3, 1, 2)
		patch_tokens = F.interpolate(patch_tokens, size=(64, 64), mode='bilinear', align_corners=False)
		patch_tokens = patch_tokens.permute(0, 2, 3, 1)
		
		dino_obs_feat = patch_tokens.squeeze(0) # [64, 64, 768]
		dino_obs_feat = dino_obs_feat / dino_obs_feat.norm(dim=-1, keepdim=True)
		dino_obs_feat = dino_obs_feat.permute(2, 0, 1).reshape(768, -1)
		dino_sim = self.dino_target_feat @ dino_obs_feat
		dino_sim = dino_sim.reshape(1, 1, 64, 64)
		dino_sim = F.interpolate(dino_sim, scale_factor=4, mode='bilinear')
		f_dino_sim = self.efficientvit_sam_predictor.Per_postprocess_masks(
			dino_sim,
			input_size=self.efficientvit_sam_predictor.input_size,
			original_size=self.efficientvit_sam_predictor.original_size,
		).squeeze()

		self.efficientvit_sam_predictor.set_image(obs.transpose(1, 2, 0))
		obs_feat = self.efficientvit_sam_predictor.features.squeeze()
		C, h, w = obs_feat.shape
		obs_feat = obs_feat / obs_feat.norm(dim=0, keepdim=True)
		obs_feat = obs_feat.reshape(C, -1)
		sim = self.target_feat @ obs_feat
		sim = sim.reshape(1, 1, h, w)
		sim = F.interpolate(sim, scale_factor=4, mode='bilinear', align_corners=False)
		f_sim = self.efficientvit_sam_predictor.model.postprocess_masks(
			sim,
			input_size=self.efficientvit_sam_predictor.input_size,
			original_size=self.efficientvit_sam_predictor.original_size,
		).squeeze()

		f_sim = (f_dino_sim + f_sim) / 2
		topk_xy, topk_label = psu.point_selection(f_sim, topk=1)
		n_xy, n_label = psu.negative_point_selection(f_sim, topk=1)
		topk_xy = np.concatenate((topk_xy, n_xy), axis=0)
		topk_label = np.concatenate((topk_label, n_label), axis=0)

		for i in range(len(self.dino_parts_target_feat)):
			dino_target_feat = self.dino_parts_target_feat[i]
			dino_sim = dino_target_feat @ dino_obs_feat
			dino_sim = dino_sim.reshape(1, 1, 64, 64)
			dino_sim = F.interpolate(dino_sim, scale_factor=4, mode='bilinear')
			dino_sim = self.efficientvit_sam_predictor.Per_postprocess_masks(
				dino_sim,
				input_size=self.efficientvit_sam_predictor.input_size,
				original_size=self.efficientvit_sam_predictor.original_size,
			).squeeze()

			target_feat = self.part_target_feat[i]
			sim = target_feat @ obs_feat
			sim = sim.reshape(1, 1, h, w)
			sim = F.interpolate(sim, scale_factor=4, mode='bilinear', align_corners=False)
			sim = self.efficientvit_sam_predictor.model.postprocess_masks(
				sim,
				input_size=self.efficientvit_sam_predictor.input_size,
				original_size=self.efficientvit_sam_predictor.original_size,
			).squeeze()

			sim = (dino_sim + sim) / 2
			xy_, label_ = psu.point_selection(sim, topk=1)
			topk_xy = np.concatenate((topk_xy, xy_), axis=0)
			topk_label = np.concatenate((topk_label, label_), axis=0)
			
		masks, scores, logits, logits_high = self.efficientvit_sam_predictor.predict(
			point_coords=topk_xy,
			point_labels=topk_label,
			multimask_output=True)
		logits_high = logits_high * self.weights.unsqueeze(-1)
		logit_high = logits_high.sum(0)
		mask = (logit_high > 0).detach().cpu().numpy()
		logits = logits * self.weights_np[..., None]
		logit = logits.sum(0)

		y, x = np.nonzero(mask)
		if len(y) == 0 or len(x) == 0:
			input_box = np.array([1, 1, 83, 83])
		else:
			x_min = x.min()
			x_max = x.max()
			y_min = y.min()
			y_max = y.max()
			input_box = np.array([x_min, y_min, x_max, y_max])

		n_point, n_label = psu.negative_point_selection(f_sim, topk=1, box=input_box)
		topk_xy = np.concatenate((topk_xy, n_point), axis=0)
		topk_label = np.concatenate((topk_label, n_label), axis=0)

		masks, scores, logits, _ = self.efficientvit_sam_predictor.predict(
			point_coords=topk_xy,
			point_labels=topk_label,
			box=input_box[None, :],
			mask_input=logit[None, :, :],
			multimask_output=True
		)
		best_idx = np.argmax(scores)
		y, x = np.nonzero(masks[best_idx])
		if len(y) == 0 or len(x) == 0:
			input_box = np.array([1, 1, 83, 83])
		else:
			x_min = x.min()
			x_max = x.max()
			y_min = y.min()
			y_max = y.max()
			input_box = np.array([x_min, y_min, x_max, y_max])

		n_point, n_label = psu.negative_point_selection(f_sim, topk=1)
		topk_xy = np.concatenate((topk_xy, n_point), axis=0)
		topk_label = np.concatenate((topk_label, n_label), axis=0)

		masks, scores, logits, _ = self.efficientvit_sam_predictor.predict(
			point_coords=topk_xy,
			point_labels=topk_label,
			box=input_box[None, :],
			mask_input=logits[best_idx: best_idx + 1, :, :],
			multimask_output=True)
		best_idx = np.argmax(scores)

		mask = masks[best_idx]
		obs = obs * mask
		return obs


	def reset(self):
		obs = self.env.reset()
		obs = self.segment_obs(obs)
		for _ in range(self._k):
			self._frames.append(obs)
		return self._get_obs()

	def step(self, action):
		obs, reward, done, info = self.env.step(action)
		obs = self.segment_obs(obs)
		self._frames.append(obs)
		return self._get_obs(), reward, done, info

	def _get_obs(self):
		assert len(self._frames) == self._k
		return utils.LazyFrames(list(self._frames))

def rgb_to_hsv(r, g, b):
	"""Convert RGB color to HSV color"""
	maxc = max(r, g, b)
	minc = min(r, g, b)
	v = maxc
	if minc == maxc:
		return 0.0, 0.0, v
	s = (maxc-minc) / maxc
	rc = (maxc-r) / (maxc-minc)
	gc = (maxc-g) / (maxc-minc)
	bc = (maxc-b) / (maxc-minc)
	if r == maxc:
		h = bc-gc
	elif g == maxc:
		h = 2.0+rc-bc
	else:
		h = 4.0+gc-rc
	h = (h/6.0) % 1.0
	return h, s, v

import matplotlib.pyplot as plt
def do_green_screen(x, bg):
	"""Removes green background from observation and replaces with bg; not optimized for speed"""
	assert isinstance(x, np.ndarray) and isinstance(bg, np.ndarray), 'inputs must be numpy arrays'
	assert x.dtype == np.uint8 and bg.dtype == np.uint8, 'inputs must be uint8 arrays'
	# plt.figure(dpi=300)
	# plt.imshow(x.swapaxes(0, 1).swapaxes(1, 2))
	# plt.show()
	# Get image sizes
	x_h, x_w = x.shape[1:]

	# Convert to RGBA images
	im = TF.to_pil_image(torch.ByteTensor(x))
	im = im.convert('RGBA')
	pix = im.load()
	bg = TF.to_pil_image(torch.ByteTensor(bg))
	bg = bg.convert('RGBA')
	bg = bg.load()

	# Replace pixels
	for x in range(x_w):
		for y in range(x_h):
			r, g, b, a = pix[x, y]
			h_ratio, s_ratio, v_ratio = rgb_to_hsv(r / 255., g / 255., b / 255.)
			h, s, v = (h_ratio * 360, s_ratio * 255, v_ratio * 255)

			# min_h, min_s, min_v = (100, 80, 70)
			# max_h, max_s, max_v = (185, 255, 255)
			# min_h, min_s, min_v = (130, 110, 100)
			# max_h, max_s, max_v = (155, 225, 225)
			# if min_h <= h <= max_h and min_s <= s <= max_s and min_v <= v <= max_v:  # replace the range of rgb
			if r == g == b == 0:
				pix[x, y] = bg[x, y]

	return np.moveaxis(np.array(im).astype(np.uint8), -1, 0)[:3]


class VideoWrapper(gym.Wrapper):
	"""Green screen for video experiments"""
	def __init__(self, env, background, seed, objects_color='original', cam_pos='original'):
		gym.Wrapper.__init__(self, env)
		self._background = background
		self._seed = seed
		self._random_state = np.random.RandomState(seed)
		self._index = 0
		self._video_paths = []
		if self._background['type'] == 'video':
			self._get_video_paths()
		self._num_videos = len(self._video_paths)
		self._max_episode_steps = env._max_episode_steps
		# self._cam_pos = cam_pos
		# self._origin_cam_pos = env.env.env.env.env._env.physics.named.data.cam_xpos

	def _get_video_paths(self):
		current_dir = os.path.dirname(__file__)
		video_dir = os.path.join(f'{current_dir}/../data', f'video_{self._background["difficulty"]}')
		if self._background['difficulty'] == 'easy':
			self._video_paths = [os.path.join(video_dir, f'video{i}.mp4') for i in range(10)]
		elif self._background['difficulty'] == 'hard':
			self._video_paths = [os.path.join(video_dir, f'video{i}.mp4') for i in range(100)]
		# else:
		# 	raise ValueError(f'received unknown mode "{self._mode}"')

	def _load_video(self, video):
		"""Load video from provided filepath and return as numpy array"""
		import cv2
		cap = cv2.VideoCapture(video)
		assert cap.get(cv2.CAP_PROP_FRAME_WIDTH) >= 100, 'width must be at least 100 pixels'
		assert cap.get(cv2.CAP_PROP_FRAME_HEIGHT) >= 100, 'height must be at least 100 pixels'
		n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
		buf = np.empty((n, int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)), int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), 3), np.dtype('uint8'))
		i, ret = 0, True
		while (i < n  and ret):
			ret, frame = cap.read()
			frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
			buf[i] = frame
			i += 1
		cap.release()
		# yang = np.moveaxis(buf, -1, 1)
		return np.moveaxis(buf, -1, 1)

	def _reset_video(self):
		self._index = (self._index + 1) % self._num_videos
		self._data = self._load_video(self._video_paths[self._index])

	def reset(self):
		if self._background['type'] == 'video':
			self._reset_video()
		self._current_frame = 0

		# return self.env.reset()
		return self._greenscreen(self.env.reset())

	def step(self, action):
		self._current_frame += 1
		obs, reward, done, info = self.env.step(action)
		# obs, reward, done, _, info = self.env.step(action)  # yangsizhe
		return self._greenscreen(obs), reward, done, info
	
	def _interpolate_bg(self, bg, size:tuple):
		"""Interpolate background to size of observation"""
		bg = torch.from_numpy(bg).float().unsqueeze(0)/255.
		bg = F.interpolate(bg, size=size, mode='bilinear', align_corners=False)
		return (bg*255.).byte().squeeze(0).numpy()

	def _greenscreen(self, obs):
		"""Applies greenscreen if video is selected, otherwise does nothing"""
		if self._background['type'] == 'video':
			bg = self._data[self._current_frame % len(self._data)] # select frame
			bg = self._interpolate_bg(bg, obs.shape[1:]) # scale bg to observation size
			return do_green_screen(obs, bg) # apply greenscreen
		return obs

	def apply_to(self, obs):
		"""Applies greenscreen mode of object to observation"""
		obs = obs.copy()
		channels_last = obs.shape[-1] == 3
		if channels_last:
			obs = torch.from_numpy(obs).permute(2,0,1).numpy()
		obs = self._greenscreen(obs)
		if channels_last:
			obs = torch.from_numpy(obs).permute(1,2,0).numpy()
		return obs
