# import tensorflow as tf
# import torch
# print('asdf')
# print(tf.reduce_sum(tf.random.normal([1000, 1000])))
# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# print(device)
import time
import requests
import flwr as fl
import json

print("App started")
inform_FLAG: bool = False
inform_SE: str = 'http://localhost:8000/FLSe/FLSeReady'
inform_Payload = {
  #  형식
  #  'S3_bucket': 'ccl-fl-demo-model',
  #  'S3_key': 'model.h5',  # 모델 가중치 파일 이름
  #  'FLSeReady': False
}
if __name__ == '__main__':
    while ~inform_FLAG:
        r = requests.put(inform_SE, params={'i': 'true'})
        if r.status_code == 200:
            inform_Payload = r.json()['Server_Status']
            break
        else:
            print(r.content)
        time.sleep(5)
    ##
    #서버를 시작
    fl.server.start_server(config={"num_rounds": 3})
    ##
    #time.sleep(100)
    while ~inform_FLAG:
        r = requests.put(inform_SE, params={'i': 'false'})
        if r.status_code == 200:
            inform_Payload = r.json()['Server_Status']
            break
        else:
            print(r.content)
        time.sleep(5)
    print(inform_Payload)
# import flwr as fl
# SERVER_ID = 0
# Start Flower server for three rounds of federated learning
# fl.server.start_server(config={"num_rounds": 3})
# S3에 모델이 있는지 확인하고 없다면 초기 가중치를 업로드
# 시작전 상태 업데이트
# 종료전 상태 업데이트
