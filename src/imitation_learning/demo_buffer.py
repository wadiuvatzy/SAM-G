import datetime
import io
import random
import traceback
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import IterableDataset
import pickle

class DemoDataSet(object):
	def __init__(self, task_name=None, num_demos=300, device='cuda'):
		self.device = device
		self.data = []
		for i in range(num_demos):
			filename = '../../demos/' + task_name + '/' + str(i) + '.pkl'
			with open(filename, 'rb') as f:
				demo = pickle.load(f)
				self.data += demo
		
	def __len__(self):
		return len(self.data)

	def __getitem__(self, idx):
		return self.data[idx]
	
	def shuffle(self):
		random.shuffle(self.data)

class DemoDataLoader(object):
	def __init__(self, dataset=None, batch_size=32):
		self.dataset = dataset
		self.batch_size = batch_size
		self.num_batches = len(self.dataset) // self.batch_size
		self.batch_idx = 0

		self.idx = 0

	def shuffle(self):
		self.dataset.shuffle()
		self.idx = 0

	def __getitem__(self, i=None):
		if i is None:
			i = self.idx
		self.idx = i
		batch = []
		sensor = []
		label = []
		for j in range(self.batch_size):
			batch.append(torch.tensor(self.dataset[i * self.batch_size+j]["observation"]))
			label.append(torch.tensor(self.dataset[i * self.batch_size+j]["action"]))
			sensor.append(torch.tensor(self.dataset[i * self.batch_size+j]["sensor"]))

		batch = torch.stack(batch)
		sensor = torch.stack(sensor)
		label = torch.stack(label)
		return batch, sensor, label
		
	def __len__(self):
		return self.num_batches

if __name__ == '__main__':
	train_dataset = DemoDataSet(task_name='door', num_demos=300)
	train_loader = DemoDataLoader(dataset=train_dataset, batch_size=32)
	for i in range(10):
		train_loader.shuffle()
		for j in range(len(train_loader)):
			batch, sensor, label = train_loader[j]
			print('batch: ', batch.shape,' sensor: ', sensor.shape, ' label: ', label.shape)
