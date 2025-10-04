#!/usr/bin/env python3
import zmq
import signal
import sys
import os

SAVE_PATH = "latest_message.txt"

# ✅ 실행 시 기존 파일 초기화 (1회만)
open(SAVE_PATH, "w").close()

# ZMQ 설정
context = zmq.Context()
socket = context.socket(zmq.REP)
socket.bind("tcp://*:8788")

def signal_handler(sig, frame):
    print("\n🔴 Shutting down ZMQ server.")
    socket.close()
    context.term()
    sys.exit(0)

signal.signal(signal.SIGINT, signal_handler)
print(f"🟢 ZMQ REP server running...\n📄 Saving messages to: {os.path.abspath(SAVE_PATH)}")

while True:
    try:
        # 메시지 수신 및 응답
        message = socket.recv_string()
        socket.send_string("ack")

        # ✅ 메시지 누적 저장
        with open(SAVE_PATH, "a") as f:
            f.write(message.strip() + "\n")

        print(f"✅ Message received and appended to {SAVE_PATH}")

    except zmq.error.ContextTerminated:
        break
    except Exception as e:
        print("❌ Error:", e)
