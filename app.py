#import tensorflow as tf 
#import torch
#print('asdf')
#print(tf.reduce_sum(tf.random.normal([1000, 1000])))
#device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#print(device)
#import time
#print ("App started")
#while True:
#    time.sleep(1)
import flwr as fl

# Start Flower server for three rounds of federated learning
fl.server.start_server(config={"num_rounds": 3})
