import os

seed_list = [0]
dataset_list = ['cifar100']
spreadout_params = [('pearson', '0.03'), ('euclidean', '0.004'),('rv', '0.05')]

for dataset in dataset_list:
	for mode, gamma in spreadout_params:
		for i in range(0, len(seed_list)):
			run = "python3 main.py"
			run += " --epochs 200"
			run += " --lr 0.1"
			run += " --batch_size 128"
			run += " --seed " + str(seed_list[i])
			run += " --dataset " + dataset
			run += " --init_mode " + mode
			run += " --init_epochs 1"
			run += " --gamma " + gamma
			run += " --run " + str(i)
			run += " --model " + 'wide_resnet28-10'
			run += " --device cuda:0"
			run += " --augmentation True"
			#run += " --layer_slice True" # if true then pass "--layers x,y" to apply spreadout on slice from x to y else pass a list of layers
			#run += " --layers 14,28" # Usage: layer 2,5 apply spreadout on layer 2,3,4 of 0,1,2,3,4
			#run += " --cutout True"
			#print("Running: " + run)
			os.system(run)
