import os

#seed_list = [42, 77, 142, 281, 324, 411, 596, 630, 742]
#dataset_list = ['cifar10', 'cifar100', 'stl10']
init_mode = ['ED']
seed_list = [42]
dataset_list = ['caltech']
#lr_list = [1e-02,1e-01,5e-03,5e-02,5e-01]
#gamma_list = [0.1]
### lr that breaks:
###    0.5  /  0.1
###
for init in init_mode:
	for dataset in dataset_list:
		for i in range(0,len(seed_list)):
			run = "python3 main.py"
			run += " --epochs 200"
			run += " --lr 0.1"
			run += " --batch_size 64"
			run += " --seed " + str(seed_list[i])
			run += " --dataset " + dataset
			run += " --init_mode " + init
			run += " --init_epochs 5"
			run += " --gamma 1.0"
			run += " --run " + str(i)
			run += " --model " + 'resnet'
			print("Running: " + run)
			os.system(run)

"""
for i,seed in enumerate(seed_list):
	run = "python3 main.py"
	run += " --epochs 100"
	run += " --lr 0.001"
	run += " --batch_size 128"
	run += " --seed " + str(seed)
	run += " --dataset stl10"
	run += " --run " + str(i)
	run += " --init_mode ortho"
	print("Running: " + run)
	os.system(run)
"""
# No init method
""" <DONE> No init
for i,seed in enumerate(seed_list):
	run = "python3 main.py"
	run += " --epochs 100"
	run += " --lr 0.001"
	run += " --batch_size 128"
	run += " --seed " + str(seed)
	run += " --dataset cifar10"
	run += " --run " + str(i)
	print("Running: " + run)
	os.system(run)
"""
"""
# My Euclidean Init
for i,seed in enumerate(seed_list):
	run = "python3 main.py"
	run += " --epochs 100"
	run += " --lr 0.001"
	run += " --batch_size 128"
	run += " --seed " + str(seed)
	run += " --dataset cifar100"
	run += " --run " + str(i)
	run += " --init_mode ED"
	run += " --init_epochs 50"
	run += " --phi 0.1"
	print("Running: " + run)
	os.system(run)
"""
