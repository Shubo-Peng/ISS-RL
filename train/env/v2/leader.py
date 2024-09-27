import grpc
from . import pbft_pb2
from . import pbft_pb2_grpc
import time
import threading

class Leader:
    def __init__(self, batch_size, batch_timeout, other_node_ports):
        self.batch_size = batch_size
        self.batch_timeout = batch_timeout
        self.other_node_ports = other_node_ports
        self.request_queue = []
        self.lock = threading.Lock()

    def generate_request(self, rate):
        while True:
            # 模拟根据rate生成请求
            with self.lock:
                self.request_queue.append(f"Request at {time.time()}")
            time.sleep(rate)

    def send_preprepare(self, sn, view, leader_id):
        while True:
            time.sleep(self.batch_timeout)
            with self.lock:
                if len(self.request_queue) >= self.batch_size:
                    # 打包请求
                    batch = pbft_pb2.Batch(data=str(self.request_queue[:self.batch_size]))
                    preprepare_message = pbft_pb2.PbftPreprepare(
                        sn=sn,
                        view=view,
                        leader=leader_id,
                        batch=batch,
                        aborted=False,
                        ts=int(time.time())
                    )
                    self.request_queue = self.request_queue[self.batch_size:]
                    self.broadcast_preprepare(preprepare_message)

    def broadcast_preprepare(self, preprepare_message):
        for port in self.other_node_ports:
            channel = grpc.insecure_channel(f'localhost:{port}')
            stub = pbft_pb2_grpc.PbftServiceStub(channel)
            stub.SendPreprepare(preprepare_message)  # 不需要ack

    def start(self, rate, sn, view, leader_id):
        threading.Thread(target=self.generate_request, args=(rate,), daemon=True).start()
        threading.Thread(target=self.send_preprepare, args=(sn, view, leader_id), daemon=True).start()

if __name__ == "__main__":
    leader = Leader(batch_size=5, batch_timeout=10, other_node_ports=[50051, 50052, 50053])
    leader.start(rate=2, sn=1, view=1, leader_id=1)
    while True:
        time.sleep(1)  # 保持主线程活跃
