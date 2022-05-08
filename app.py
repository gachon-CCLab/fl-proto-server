import tensorflow as tf
from typing import Any, Callable, Dict, List, Optional, Tuple

import wandb
from tensorflow import keras

import logging
import time

import requests
import flwr as fl
from flwr import common
from flwr.server import strategy
import wget
import json
import os
import boto3
from custom_server.advanced_server import Advanced_Server


#class AdvancedClientManager(flwr.server.SimpleClientManager):
#    """
#    Abstract base class for managing Flower clients.
#
#    @abstractmethod
#    def num_available(self) -> int:
#        Return the number of available clients.
#
#    @abstractmethod
#    def register(self, client: ClientProxy) -> bool:
#        Register Flower ClientProxy instance
#
#        Returns:
#            bool: Indicating if registration was successful
#
#
#    @abstractmethod
#    def unregister(self, client: ClientProxy) -> None:
#        Unregister Flower ClientProxy instance.
#
#    @abstractmethod
#    def all(self) -> Dict[str, ClientProxy]:
#        Return all available clients.
#
#    @abstractmethod
#    def wait_for(self, num_clients: int, timeout: int) -> bool:
#        Wait until at least `num_clients` are available.
#
#    @abstractmethod
#    def sample(
#        self,
#        num_clients: int,
#        min_num_clients: Optional[int] = None,
#        criterion: Optional[Criterion] = None,
#    ) -> List[ClientProxy]:
#        Sample a number of Flower ClientProxy instances.
#    """
#    pass


#class AdvancedStrategy(fl.server.strategy.FedAvg):
#    def aggregate_fit(
#            self,
#            rnd,
#            results,
#            failures,
#    ):
#        aggregated_weights = super().aggregate_fit(rnd, results, failures)
#        if aggregated_weights is not None:
#            # Save aggregated_weights
#            print(f"Saving round {rnd} aggregated_weights...")
#            np.savez(f"round-{rnd}-weights.npz", *aggregated_weights)
#        return aggregated_weights


# Create strategy and run server
wandb.login(key='9b027b5dd31cac141867a24f926dcf8a96daa742')
wandb.init(entity='hoo0681', project='flwr',config={"epochs": 4, "batch_size": 32,"val_steps": 4})

def upload_lastest_model():
    if os.environ.get('ENV') is not None:
        res = requests.get('http://10.1.196.109:8000' + '/FLSe/info')  # 서버측 manager  # 10.152.183.186
        S3_info = res.json()['Server_Status']
        ACCESS_KEY_ID = os.environ.get('ACCESS_KEY_ID')
        ACCESS_SECRET_KEY = os.environ.get('ACCESS_SECRET_KEY')
        BUCKET_NAME = os.environ.get('BUCKET_NAME')
        s3_client = boto3.client('s3', aws_access_key_id=ACCESS_KEY_ID,
                                 aws_secret_access_key=ACCESS_SECRET_KEY)
        response = s3_client.upload_file(S3_info['S3_key'], BUCKET_NAME, S3_info['S3_key'])
        print('upload')
    else:
        pass

def get_eval_fn(model):
    """Return an evaluation function for server-side evaluation."""

    # Load data and model here to avoid the overhead of doing it in `evaluate` itself
    #(x_train, y_train), _ = tf.keras.datasets.cifar10.load_data()

    ## Use the last 5k training examples as a validation set
    #x_val, y_val = x_train[45000:50000], y_train[45000:50000]

    mnist = tf.keras.datasets.mnist
    (x_train, y_train), (x_val, y_val) = mnist.load_data()
    x_val=x_val/255.0
    # The `evaluate` function will be called after every round
    def evaluate(
        weights: fl.common.Weights,
    ) -> Optional[Tuple[float, Dict[str, fl.common.Scalar]]]:
        model.set_weights(weights)  # Update model with the latest parameters
        model.compile("adam", "sparse_categorical_crossentropy", metrics=["accuracy"])
        loss, accuracy = model.evaluate(x_val, y_val)
        model.save("./model.h5")
        upload_lastest_model()
        wandb.log({'loss':loss,"accuracy": accuracy})
        return loss, {"accuracy": accuracy}

    return evaluate


def fit_config(rnd: int):
    """Return training configuration dict for each round.
    Keep batch size fixed at 32, perform two rounds of training with one
    local epoch, increase to two local epochs afterwards.
    """
    config = {
        "batch_size": 32,
        "local_epochs": 1 if rnd < 2 else 2,
    }
    wandb.config.update({"local_epochs": 1 if rnd < 2 else 2, "batch_size": 32},allow_val_change=True)

    return config


def evaluate_config(rnd: int):
    """Return evaluation configuration dict for each round.
    Perform five local evaluation steps on each client (i.e., use five
    batches) during rounds one to three, then increase to ten local
    evaluation steps.
    """
    val_steps = 5 if rnd < 4 else 10
    wandb.config.update({"val_steps": val_steps},allow_val_change=True)

    return {"val_steps": val_steps}


if __name__ == '__main__':

    print("App started")
    inform_SE: str = 'http://10.1.196.109:8000/FLSe/'#10.152.183.186
    inform_Payload = {
        #  형식
        'S3_bucket': 'ccl-fl-demo-model',
        'S3_key': 'model.h5',  # 모델 가중치 파일 이름
        'FLSeReady': True,
        #'Model_V' : 0
    }
    while True:
        r = requests.put(inform_SE+'FLSeUpdate', data=json.dumps(inform_Payload))
        if r.status_code == 200:
            break
        else:
            print(r.content)
        time.sleep(5)
    try:
        #
        # 서버를 시작
        url = "https://" + inform_Payload['S3_bucket'] + ".s3.ap-northeast-2.amazonaws.com/" + inform_Payload['S3_key']
        request = wget.download( url, out='./model.h5')
        model = keras.models.load_model('./model.h5')
        weights = model.get_weights()
        logging.getLogger('flower')
        strategy = fl.server.strategy.FedAvg(
            fraction_fit=0.3,
            fraction_eval=0.2,
            min_fit_clients=1,
            min_eval_clients=1,
            min_available_clients=1,
            eval_fn=get_eval_fn(model),
            on_fit_config_fn=fit_config,
            on_evaluate_config_fn=evaluate_config,
            initial_parameters=fl.common.weights_to_parameters(weights),
        )
        #fl.server.start_server(server_address="localhost:8080",strategy=strategy, config={"num_rounds": 10})
        fl.server.start_server("0.0.0.0:8080", config={"num_rounds": 3},strategy=strategy)
        ##
        #time.sleep(10)
    #except Exception as e:
    #    print(e)
    finally:
        r = requests.put(inform_SE + 'FLSeReady', params={'i': 'false'})
        wandb.finish()
    print(inform_Payload)
# import flwr as fl
# SERVER_ID = 0
# Start Flower server for three rounds of federated learning
# fl.server.start_server(config={"num_rounds": 3})
# S3에 모델이 있는지 확인하고 없다면 초기 가중치를 업로드
# 시작전 상태 업데이트
# 종료전 상태 업데이트
