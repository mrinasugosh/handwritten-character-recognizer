# -*- coding: utf-8 -*-
# @Author: Mrinalini Sugosh
# @Date:   09-15-2021 12:40:52
# @Last Modified by:   Mrinalini Sugosh
# @Last Modified time: 09-15-2021 22:23:18



import torch
from torchvision import datasets, transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import logging
import numpy as np

import argparse

from network import Net

PARSER = argparse.ArgumentParser()
PARSER.add_argument('-b', '--batch_size', default=128)
PARSER.add_argument('-e', '--EPOCHS', default=200)
PARSER.add_argument('-lr', '--lr', default=1e-3)
PARSER.add_argument('-tb', '--test_batch_size', default=128)
PARSER.add_argument('-r', '--root', required=True)

class Training():
	def __init__(self):
		self.net = Net()
		self.net.apply(self.init_weights)
		#Basic logging
		logging.basicConfig(filename="cnn2.log", level=logging.DEBUG)
		logging.info(self.net)
		logging.info("Number of parameters: {}".format(self.count_parameters(self.net)))

		self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

		self.optimizer = torch.optim.SGD(self.net.parameters(), lr=args.lr, momentum=0.9)
		self.criterion = nn.CrossEntropyLoss().to(self.device)

		self.best_acc = 0
		self.net.to(self.device)

	def loader(self):

		# Define transformers to apply on the input data

		transform_train = transforms.Compose(
			[
				transforms.RandomCrop(28, padding=4),
				transforms.ToTensor(),
				# transforms.Normalize((mean,), (std,)),
			]
		)

		transform_valid = transforms.Compose(
			[
				transforms.ToTensor(),
				# transforms.Normalize((mean,), (std,)),
			]
		)


		train = datasets.EMNIST(
		args.root, split="balanced", train=True, download=True, transform=transform_train
		)
		test = datasets.EMNIST(
			args.root, split="balanced", train=False, download=True, transform=transform_valid
		)


		self.train_loader = torch.utils.data.DataLoader(
			train, batch_size=args.batch_size, shuffle=True, num_workers=2, drop_last=True
		)


		self.test_loader = torch.utils.data.DataLoader(
			test, batch_size=args.test_batch_size, shuffle=False, num_workers=2, drop_last=True
		)

		

	def init_weights(self, m):
		if type(m) == nn.Linear:
			torch.nn.init.xavier_uniform_(m.weight)
			m.bias.data.fill_(0.01)

	def count_parameters(self,model):
		return sum(p.numel() for p in model.parameters() if p.requires_grad)

	def inf_generator(self, iterable):
		"""Allows training with DataLoaders in a single infinite loop:
			for i, (x, y) in enumerate(inf_generator(train_loader)):
		"""
		iterator = iterable.__iter__()
		while True:
			try:
				yield iterator.__next__()
			except StopIteration:
				iterator = iterable.__iter__()


	def train(self):
		self.loader()

		data_gen = self.inf_generator(self.train_loader)
		batches_per_epoch = len(self.train_loader)
		for itr in range(args.EPOCHS * batches_per_epoch):

			self.optimizer.zero_grad()
			x, y = data_gen.__next__()
			x = x.view(-1, 28, 28, 1)
			x = torch.transpose(x, 1, 2)

			x = x.to(self.device)
			y = y.to(self.device)
			logits = self.net(x)
			loss = self.criterion(logits, y)

			loss.backward()
			self.optimizer.step()

			if itr % batches_per_epoch == 0:
				with torch.no_grad():
					train_acc = self.accuracy(self.net, self.train_loader)
					val_acc = self.accuracy(self.net, self.test_loader)
					if val_acc > self.best_acc:
						torch.save({"state_dict": self.net.state_dict()}, "alpha_weights.pth")
						self.best_acc = val_acc
					logging.info(
						"Epoch {:04d}"
						"Train Acc {:.4f} | Test Acc {:.4f}".format(
							itr // batches_per_epoch, train_acc, val_acc
						)
					)

					print(
						"Epoch {:04d}"
						"Train Acc {:.4f} | Test Acc {:.4f}".format(
							itr // batches_per_epoch, train_acc, val_acc
						)
					)
	def one_hot(self,x, K):
		return np.array(x[:, None] == np.arange(K)[None, :], dtype=int)


	def accuracy(self, model, dataset_loader):
		total_correct = 0
		for x, y in dataset_loader:
			x = x.view(-1, 28, 28, 1)
			x = torch.transpose(x, 1, 2)

			x = x.to(self.device)
			y = self.one_hot(np.array(y.numpy()), 47)

			target_class = np.argmax(y, axis=1)
			predicted_class = np.argmax(model(x).cpu().detach().numpy(), axis=1)
			total_correct += np.sum(predicted_class == target_class)
		return total_correct / len(dataset_loader.dataset)


if __name__ == '__main__':
	args = PARSER.parse_args()

	trainer = Training()
	trainer.train()
