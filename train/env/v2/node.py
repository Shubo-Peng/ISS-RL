import grpc
from concurrent import futures
from . import pbft_pb2
from . import pbft_pb2_grpc
import time
import threading
import os

class Node(pbft_pb2_grpc.PbftServiceServicer):
    def __init__(self, port, other_node_ports):
        self.port = port
        self.other_node_ports = other_node_ports
        self.prepare_count = 0
        self.lock = threading.Lock()

    def SendPreprepare(self, request, context):
        print(f"Node {self.port} received Preprepare: {request}")
        # 接收到preprepare消息后，立即发送prepare消息
        threading.Thread(target=self.send_prepare, args=(request.sn, request.view, request.batch.data)).start()
        return pbft_pb2.google_dot_protobuf_dot_empty__pb2.Empty()

    def SendPrepare(self, request, context):
        print(f"Node {self.port} received Prepare: {request}")
        # 收到prepare消息后，统计数量
        with self.lock:
            self.prepare_count += 1
        # 如果收到足够多的prepare消息，则发送commit消息
        if self.prepare_count >= len(self.other_node_ports):
            threading.Thread(target=self.send_commit, args=(request.sn, request.view, request.digest)).start()
        return pbft_pb2.google_dot_protobuf_dot_empty__pb2.Empty()

    def SendCommit(self, request, context):
        print(f"Node {self.port} received Commit: {request}")
        return pbft_pb2.google_dot_protobuf_dot_empty__pb2.Empty()

    def send_prepare(self, sn, view, data):
        digest = data.encode()  # 模拟生成digest
        prepare_message = pbft_pb2.PbftPrepare(sn=sn, view=view, digest=digest)
        self.broadcast_prepare(prepare_message)

    def send_commit(self, sn, view, digest):
        commit_message = pbft_pb2.PbftCommit(sn=sn, view=view, digest=digest, ts=int(time.time()))
        self.broadcast_commit(commit_message)

    def broadcast_prepare(self, prepare_message):
        for port in self.other_node_ports:
            channel = grpc.insecure_channel(f'localhost:{port}')
            stub = pbft_pb2_grpc.PbftServiceStub(channel)
            stub.SendPrepare(prepare_message)

    def broadcast_commit(self, commit_message):
        for port in self.other_node_ports:
            channel = grpc.insecure_channel(f'localhost:{port}')
            stub = pbft_pb2_grpc.PbftServiceStub(channel)
            stub.SendCommit(commit_message)

def serve(port, other_node_ports):
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    pbft_pb2_grpc.add_PbftServiceServicer_to_server(Node(port, other_node_ports), server)
    server.add_insecure_port(f'[::]:{port}')
    server.start()
    print(f"Node {port} started")
    server.wait_for_termination()

if __name__ == "__main__":
    port = int(os.getenv("PORT", 50051))
    other_ports = [p for p in [50051, 50052, 50053] if p != port]
    serve(port, other_ports)
