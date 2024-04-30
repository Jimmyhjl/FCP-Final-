import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import math 
import argparse
import copy
from collections import deque

def defuant_main(opinions, beta=0.2, threshold=0.2):
	epoch = 100
	each_epoch_iters = 1000
	for frame in range(epoch):
		for step in range(each_epoch_iters):
			opinions = defuant_iter(opinions, beta, threshold)
	return opinions

def defuant_iter(opinions, beta=0.2, threshold=0.2):
	n = len(opinions)
	i, j = np.random.choice(n, 2, replace=False)
	if abs(opinions[i] - opinions[j]) < threshold:
		opinions[i] += beta * (opinions[j] - opinions[i])
		opinions[j] += beta * (opinions[i] - opinions[j])
	return opinions

def test_defuant(opinions, beta=0.2, threshold=0.2):
	fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
	plt.suptitle(f"Coupling: {beta}, Threshold: {threshold}", fontsize=16)
	ax1.set_xticks(np.arange(0, 1.1, 0.1))
	ax1.set_xlabel('Opinion')
	ax1.set_ylabel('Count')
	ax1.set_title('fig1')
	ax2.plot()
	ax2.set_title('fig2')
	x_values = []
	y_values = []
	epoch = 100
	each_epoch_iters = 1000
	for frame in range(epoch):
		for step in range(each_epoch_iters):
			opinions = defuant_iter(opinions, beta, threshold)
		bin_size = 0.05
		bins = np.arange(0, 1 + bin_size, bin_size)
		hist, bin_edges = np.histogram(opinions, bins=bins)
		ax1.bar(bin_edges[:-1], hist, width=bin_size, edgecolor='black', align='edge')

		ax1.cla()
		ax1.bar(bin_edges[:-1], hist, width=bin_size, edgecolor='black', align='edge', color='blue')
		ax1.set_xticks(np.arange(0, 1.1, 0.1))
		ax1.set_xlabel('Opinion')
		ax1.set_ylabel('Count')
		ax1.set_title('Opinion Distribution - Frame {}'.format(frame+1))

		ax2.cla()
		x_values.extend([frame] * len(opinions))
		y_values.extend(opinions)
		ax2.scatter(x_values, y_values, alpha=0.6, color='red')
		ax2.set_xlim(0, epoch)
		ax2.set_ylim(0, 1)
		ax2.set_xlabel('Iteration')
		ax2.set_ylabel('Opinion values')
		ax2.set_title('Opinion Trace Over Iterations')

		plt.pause(0.1)
	plt.show()
	return opinions


def get_parser():
	parser = argparse.ArgumentParser(description='The Test')
	parser.add_argument('-defuant', action='store_true')
	parser.add_argument('-beta', type=float, default=0.2)
	parser.add_argument('-threshold', type=float, default=0.2)
	parser.add_argument('-test_defuant', action='store_true')
	args = parser.parse_args()
	return args

def main():
	args = get_parser()
	if args.defuant:
		opinions = np.random.rand(100)
		opinions = defuant_main(opinions, args.beta, args.threshold)

	if args.test_defuant:
		opinions = np.random.rand(100)
		test_defuant(opinions, args.beta, args.threshold)

if __name__=="__main__":
	main()