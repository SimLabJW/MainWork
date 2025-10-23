import zmq
import time
import signal
import sys

from slams.slam import RealtimeSLAM

class ZMQReqRepHandler:

    def __init__(self, socket_type, address):
        self.context = zmq.Context()
        self.socket_type = socket_type
        self.socket = self.context.socket(self.socket_type)
        self.type_str = "REP" if socket_type == zmq.REP else "..." # type_str 초기화 필요

        self.socket.setsockopt(zmq.LINGER, 0)

        if self.socket_type == zmq.REP:
            self.socket.bind(address)
            print(f"[{self.type_str}] Socket bound to {address}")
        else:
            pass

        self.slam = RealtimeSLAM(self.socket)

    def send_msg(self, message_data):
        self.socket.send_string(message_data)
   
    def recv_msg(self):
        reply = self.socket.recv_string(zmq.NOBLOCK)
            
    def close(self):
        """리소스 정리"""
        self.socket.close()
        self.context.term()
        print(f"[{self.type_str}] Socket and context closed safely.")


    def run_zmq(self):

        self.is_running = True
        def signal_handler(sig, frame):

            self.is_running = False

        signal.signal(signal.SIGINT, signal_handler)

        if self.socket_type == zmq.REP:
            self.slam.run()


if __name__ == '__main__':

    rep_server = None # 반드시 초기화
    
    try:
        # --- ZeroMQ 소켓 설정 ---
        # 1. Unity -> Python
        # 이 시점에 Address in use 에러가 발생하면 rep_server는 None으로 남아 finally에서 안전함
        rep_server = ZMQReqRepHandler(zmq.REP, "tcp://*:8788")
        
        print("ZMQ server running. Press Ctrl+C to stop.")
        rep_server.run_zmq()
        
    except Exception as e:
        # 소켓 bind 이후 발생하는 예상치 못한 오류만 여기서 처리
        print(f"An unexpected error occurred: {e}") 
        
    finally:
        # **가장 중요:** 프로그램이 어떤 식으로 종료되든(정상 종료, Ctrl+C, 예외 발생)
        # 리소스를 정리하는 곳은 오직 이 곳입니다.
        if rep_server:
            rep_server.close() 
            
    print("ZeroMQ handlers shut down successfully.")