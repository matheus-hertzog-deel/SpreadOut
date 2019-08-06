# SpreadOut

SpreadOut is a complementary weight initialization technique for Convolutional Neural Networks that decorrelate kernels within the same layer that were initialized by some traditional approach. This can be understood as spreading the kernels over the kernel feature space, initializing them at distant configurations. 

## Run
 script run.py shows how to run SpreadOut on a wide-resnet28-10 using metric and gamma specified in "spreadout_params" list.
 
 script args.py shows all use-cases for this project. 
 
 For example, in order to train an Alexnet in Cifar100 with Spreadout and augmentaion (RandomCrop and Horizontal Flip) use:

```
python main.py --epochs 200 --lr 0.1 --batch_size 128 --dataset cifar100 --gamma 0.004 --model alex --augmentation True
```
