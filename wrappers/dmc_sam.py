# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from collections import deque
from typing import Any, NamedTuple

import dm_env
import numpy as np
from dm_control import manipulation, suite
from dm_control.suite.wrappers import action_scale, pixels
from dm_env import StepType, specs
from dm_control.utils import rewards
import re

class ExtendedTimeStep(NamedTuple):
    step_type: Any
    reward: Any
    discount: Any
    observation: Any
    action: Any

    def first(self):
        return self.step_type == StepType.FIRST

    def mid(self):
        return self.step_type == StepType.MID

    def last(self):
        return self.step_type == StepType.LAST

    def __getitem__(self, attr):
        if isinstance(attr, str):
            return getattr(self, attr)
        else:
            return tuple.__getitem__(self, attr)


class ActionRepeatWrapper(dm_env.Environment):
    def __init__(self, env, num_repeats):
        self._env = env
        self._num_repeats = num_repeats

    def step(self, action):
        reward = 0.0
        discount = 1.0
        for i in range(self._num_repeats):
            time_step = self._env.step(action)
            reward += (time_step.reward or 0.0) * discount
            discount *= time_step.discount
            if time_step.last():
                break

        return time_step._replace(reward=reward, discount=discount)

    def observation_spec(self):
        return self._env.observation_spec()

    def action_spec(self):
        return self._env.action_spec()

    def reset(self):
        return self._env.reset()

    def __getattr__(self, name):
        return getattr(self._env, name)


class FrameStackWrapper(dm_env.Environment):
    def __init__(self, env, num_frames, pixels_key='pixels'):
        self._env = env
        self._num_frames = num_frames
        self._frames = deque([], maxlen=num_frames)
        self._pixels_key = pixels_key

        wrapped_obs_spec = env.observation_spec()
        assert pixels_key in wrapped_obs_spec

        pixels_shape = wrapped_obs_spec[pixels_key].shape
        # remove batch dim
        if len(pixels_shape) == 4:
            pixels_shape = pixels_shape[1:]
        self._obs_spec = specs.BoundedArray(shape=np.concatenate(
            [[pixels_shape[2] * num_frames], pixels_shape[:2]], axis=0),
                                            dtype=np.uint8,
                                            minimum=0,
                                            maximum=255,
                                            name='observation')

    def _transform_observation(self, time_step):
        assert len(self._frames) == self._num_frames
        obs = np.concatenate(list(self._frames), axis=0)
        return time_step._replace(observation=obs)

    def _extract_pixels(self, time_step):
        pixels = time_step.observation[self._pixels_key]
        # remove batch dim
        if len(pixels.shape) == 4:
            pixels = pixels[0]
        return pixels.transpose(2, 0, 1).copy()

    def reset(self):
        time_step = self._env.reset()
        pixels = self._extract_pixels(time_step)
        for _ in range(self._num_frames):
            self._frames.append(pixels)
        return self._transform_observation(time_step)

    def step(self, action):
        time_step = self._env.step(action)
        pixels = self._extract_pixels(time_step)
        self._frames.append(pixels)
        return self._transform_observation(time_step)

    def observation_spec(self):
        return self._obs_spec

    def action_spec(self):
        return self._env.action_spec()

    def __getattr__(self, name):
        return getattr(self._env, name)

import sys
import torch
import torch.nn.functional as F
import cv2
sys.path.append("efficientvit")
from efficientvit.sam_model_zoo import create_sam_model
from efficientvit.models.efficientvit.sam import EfficientViTSamPredictor
import torchvision.transforms as T
from dmcvgb.persam_utils import Mask_Weights, calculate_dice_loss, calculate_sigmoid_focal_loss, point_selection, negative_point_selection

class FrameStackWrapperWithSAM(dm_env.Environment):
	def __init__(self, env, num_frames, pixels_key='pixels', handmade_4_SAM=None):
		self._env = env
		self._num_frames = num_frames
		self._frames = deque([], maxlen=num_frames)
		self._pixels_key = pixels_key
		wrapped_obs_spec = env.observation_spec()
		assert pixels_key in wrapped_obs_spec
		pixels_shape = wrapped_obs_spec[pixels_key].shape
		if len(pixels_shape) == 4:
			pixels_shape = pixels_shape[1:]
		self._obs_spec = specs.BoundedArray(shape=np.concatenate(
			[[pixels_shape[2] * num_frames], pixels_shape[:2]], axis=0),
											dtype=np.uint8,
											minimum=0,
											maximum=255,
											name='observation')

		efficientvit_sam = create_sam_model(
			name='l1', weight_url="../../../efficientvit/checkpoints/l1.pt"
		)
		efficientvit_sam = efficientvit_sam.cuda().eval()
		self.efficientvit_sam_predictor = EfficientViTSamPredictor(efficientvit_sam)

		print("loading dino model")
		# self.dino_model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14')
		# if due to some reasons, the above command doesn't work, you can load the model from a local path
		# you need to put the downloaded checkpoint in your ~/.cache/torch/hub/checkpoints/ directory
		# you should clone the dinov2 repo and put the path to the local directory (similar as efficientvit)
		# self.dino_model = torch.hub._load_local('../../../dinov2', 'dinov2_vitb14')
		self.dino_model = torch.hub._load_local('/home/ziyuwang21/workspace/DINOV2/dinov2', 'dinov2_vitb14')
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
		ref_image = cv2.imread(handmade_4_SAM.original_image)
		ref_image = cv2.cvtColor(ref_image, cv2.COLOR_BGR2RGB)

		ref_mask = cv2.imread(handmade_4_SAM.masked_image)
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

		topk_xy, topk_label = point_selection(sim, topk=1)

		n_xy, n_label = negative_point_selection(sim, topk=1)
		topk_xy = np.concatenate((topk_xy, n_xy), axis=0)
		topk_label = np.concatenate((topk_label, n_label), axis=0)

		parts_target_feat = []
		dino_parts_target_feat = []
		for point in handmade_4_SAM.extra_points_list:
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

		for i in handmade_4_SAM.extra_masked_images_list:
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
			topk_xy_, topk_label_ = point_selection(sim, topk=1)
			topk_xy = np.concatenate((topk_xy, topk_xy_), axis=0)
			topk_label = np.concatenate((topk_label, topk_label_), axis=0)

		mask_weights = Mask_Weights().cuda()
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
				xy_, label_ = point_selection(sim, topk=1)
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

			dice_loss = calculate_dice_loss(logits_high, gt_mask)
			focal_loss = calculate_sigmoid_focal_loss(logits_high, gt_mask)
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
		
	def _transform_observation(self, time_step):
		assert len(self._frames) == self._num_frames
		obs = np.concatenate(list(self._frames), axis=0)
		return time_step._replace(observation=obs)
		
	def _extract_pixels(self, time_step):
		pixels = time_step.observation[self._pixels_key]
		if len(pixels.shape) == 4:
			pixels = pixels[0]
		obs = pixels.copy().transpose(2, 0, 1)
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
		topk_xy, topk_label = point_selection(f_sim, topk=1)
		n_xy, n_label = negative_point_selection(f_sim, topk=1)
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
			xy_, label_ = point_selection(sim, topk=1)
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

		n_point, n_label = negative_point_selection(f_sim, topk=1, box=input_box)
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

		n_point, n_label = negative_point_selection(f_sim, topk=1)
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
		time_step = self._env.reset()
		pixels = self._extract_pixels(time_step)
		for _ in range(self._num_frames):
			self._frames.append(pixels)
		return self._transform_observation(time_step)
		
	def step(self, action):
		time_step = self._env.step(action)
		pixels = self._extract_pixels(time_step)
		self._frames.append(pixels)
		return self._transform_observation(time_step)
		
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


class ExtendedTimeStepWrapper(dm_env.Environment):
    def __init__(self, env):
        self._env = env

    def reset(self):
        time_step = self._env.reset()
        return self._augment_time_step(time_step)

    def step(self, action):
        time_step = self._env.step(action)
        return self._augment_time_step(time_step, action)

    def _augment_time_step(self, time_step, action=None):
        if action is None:
            action_spec = self.action_spec()
            action = np.zeros(action_spec.shape, dtype=action_spec.dtype)
        return ExtendedTimeStep(observation=time_step.observation,
                                step_type=time_step.step_type,
                                action=action,
                                reward=time_step.reward or 0.0,
                                discount=time_step.discount or 1.0)

    def observation_spec(self):
        return self._env.observation_spec()

    def action_spec(self):
        return self._env.action_spec()

    def __getattr__(self, name):
        return getattr(self._env, name)



def make(name, frame_stack, action_repeat, seed, use_SAM_g=False, handmade_4_SAM=None):
    if re.match('^anymal', name):
        name_list = name.split('_')
        domain = name_list[0] + '_' + name_list[1]
        task = name_list[2]
    else:
        domain, task = name.split('_', 1)
        # overwrite cup to ball_in_cup
        domain = dict(cup='ball_in_cup').get(domain, domain)
    # make sure reward is not visualized
    if (domain, task) in suite.ALL_TASKS:
        env = suite.load(domain,
                         task,
                         task_kwargs={'random': seed},
                         visualize_reward=False)
        pixels_key = 'pixels'
    else:
        name = f'{domain}_{task}_vision'
        env = manipulation.load(name, seed=seed)
        pixels_key = 'front_close'
    # add wrappers
    env = ActionDTypeWrapper(env, np.float32)
    env = ActionRepeatWrapper(env, action_repeat)
    env = action_scale.Wrapper(env, minimum=-1.0, maximum=+1.0)
    # add renderings for clasical tasks
    if (domain, task) in suite.ALL_TASKS:
        # zoom in camera for quadruped
        camera_id = dict(quadruped=2).get(domain, 0)
        render_kwargs = dict(height=84, width=84, camera_id=camera_id)
        env = pixels.Wrapper(env,
                             pixels_only=True,
                             render_kwargs=render_kwargs)
    # stack several frames
    if use_SAM_g:
        env = FrameStackWrapperWithSAM(env, frame_stack, pixels_key, handmade_4_SAM)
    else:
        env = FrameStackWrapper(env, frame_stack, pixels_key)
    env = ExtendedTimeStepWrapper(env)
    return env
