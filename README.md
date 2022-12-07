# Memorization in Deep Neural Networks: Does the Loss Function Matter? (PAKDD 2021) [`Paper`](https://arxiv.org/abs/2107.09957)

## Requirements
- PyTorch >= 1.3
- Python >= 3.7
- tqdm, numpy-indexed, etc (which can be easily installed via pip)

## Running the experiments
A typical experiment run would look like this:
```
python pakdd_exp_1.py --dataset mnist --noise_rate 0.4 --noise_type sym --loss_name cce --architecture inception_small --batch_norm 0 --weight_decay 0 --data_aug 0 --batch_size 128 --num_epoch 200 --num_runs 5 
```

Important arguments here:
- ```dataset```: mnist or cifar10
- ```noise_rate```: $ \in 0 \leq \eta \leq 1 $
- ```noise_type```: sym (Symmetric) or cc (Class-conditional)
- ```loss_name```: cce, mse, or norm_mse
- ```architecture```: inception_small (Inception_Lite) or resnet (ResNet-32/ResNet-18)
- ```num_runs```: number of times experiment would be repeated

So, the aforementioned command will train the Inception_Lite network (no batch_norm or weight_decay) with CCE loss on un-augmented MNIST dataset (corrupted with 40% symmetric label noise). The training will be carried for a batch-size of 128, 200 epochs and a total of 5 runs.

