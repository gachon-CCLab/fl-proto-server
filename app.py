import tensorflow as tf 
import torch

print(tf.reduce_sum(tf.random.normal([1000, 1000])))
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)