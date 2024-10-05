import grpc
import time
import monitor_pb2
import monitor_pb2_grpc

def run():
    # 连接到服务端（使用服务端的 IP 地址和端口）
    server_address = '52.77.233.140:32767'  # 你需要把 'localhost' 替换为你的服务端公网 IP
    channel = grpc.insecure_channel(server_address)
    stub = monitor_pb2_grpc.MetricsServiceStub(channel)

    # 发起 Connect 请求
    try:
        print("Sending Connect request to server...")
        response = stub.Connect(monitor_pb2.Timestamp(
            timestamp=int(time.time() * 1e9)  # 发送当前时间戳（纳秒）
        ))
        print(f"Received Connect response: Start time = {response.timestamp}")
    except grpc.RpcError as e:
        print(f"Failed to connect: {e.details()}")

    # 模拟发送度量数据（Metrics）
    try:
        print("Sending SendMetrics request to server...")
        response = stub.SendMetrics(monitor_pb2.MetricsRequest(
            throughput=[100],
            latency=[50],
            requests=[10],
            requests_size=[1024],
            BatchSize=[1024],
            BatchTimeout=[4000],
            Leader=[0]
        ))
        print(f"Received SendMetrics response: BatchSize = {response.BatchSize}, BatchTimeout = {response.BatchTimeout}")
    except grpc.RpcError as e:
        print(f"Failed to send metrics: {e.details()}")

if __name__ == '__main__':
    run()
