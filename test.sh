#!/bin/bash

python run.py --grad_type "pseudo" --dataset "MNIST" --name "mnist_fc_ps" --epochs 40  --connectivity "fc" --lr 0.001 --batch_size=40
python run.py --grad_type "random" --dataset "MNIST" --name "mnist_fc_rd" --epochs 40  --connectivity "fc" --lr 0.001 --batch_size=40
python run.py --grad_type "backprop" --dataset "MNIST" --name "mnist_fc_bp" --epochs 40  --connectivity "fc" --lr 0.001 --batch_size=40

python run.py --grad_type "pseudo" --dataset "CIFAR10" --name "cifar_fc_ps" --epochs 20  --connectivity "fc" --lr 0.001 --batch_size=40
python run.py --grad_type "random" --dataset "CIFAR10" --name "cifar_fc_rd" --epochs 20  --connectivity "fc" --lr 0.001 --batch_size=40
python run.py --grad_type "backprop" --dataset "CIFAR10" --name "cifar_fc_bp" --epochs 20  --connectivity "fc" --lr 0.001 --batch_size=40