import json
import time
from typing import Optional

import zmq

from function import Function_control

# ============ zmq 설정 ============== #
SERVER_IP = "192.168.50.75"
SERVER_PORT = 8788

class ClientSLAM:

    def __init__(self) -> None:
        # ============ ZMQ Connect =========== #
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.REQ)
        self.socket.connect(f"tcp://{SERVER_IP}:{SERVER_PORT}")
        self.socket.RCVTIMEO = 10000
        self.socket.SNDTIMEO = 10000
        print(f"[ZMQ] Connected to {SERVER_IP}:{SERVER_PORT}")

        self.running = True

        # ============ Robot State =========== #
        self.agent_state = "PROCESS"

        self.renewal_in = True
        self.path = []

        # ============ SLAM Function =========== #
        self._f = Function_control()

    def send_request(self, data: str) -> Optional[dict]:
        """동기 REQ/REP"""
        try:
            self.socket.send_string(data)
            reply = self.socket.recv_string()
            return json.loads(reply)
        except zmq.Again:
            print("[WARN] ZMQ timeout")
            return None
        except Exception as e:
            print(f"[ERROR] ZMQ: {e}")
            return None

    def run(self):
        print("=" * 60)
        print("Robot SLAM Client")
        print("=" * 60)
        
        try:
            while self.running:
                reply_data = self.send_request(self._f._make_slamData(self.agent_state))

                reply_res = reply_data.get("result", {})
                status = reply_res.get("status", "")
                
                if status == "continue":
                    
                    if self.agent_state == "PROCESS" and self.renewal_in:
                        
                        d_path = reply_res.get("path", [])

                        self.path = self._f._calculate_path2motion(d_path)


                        print(f"Covert path len : {len(self.path)}")
                        path_count = self._f._calculate_motion(self.path)

                        if len(self.path) > 1 :
                            self.agent_state = "RENEWAL"
                            self.renewal_in = False
                            
                            print(f"path count : {path_count}")

                else:
                    if status == "renewal":
                        if self.agent_state == "RENEWAL" and self.renewal_in == False:
                            path_count =self._f._calculate_motion(self.path)
                            print(f"path count : {path_count}")

                            if (len(self.path) == path_count):
                                self.agent_state = "PROCESS"
                                self.renewal_in = True
                                self.path = []
                                self._f.path_cnt = 0
                    else:
                        break
                
                
        except KeyboardInterrupt:
            print("\n[MAIN] ⚠️ Interrupted by user")
        
        finally:
            self.shutdown()


        
    def shutdown(self):
            print("\n" + "=" * 60)
    
            try:
                self.socket.close(0)
                self.context.term()
            except:
                pass
            
            print("[SYSTEM] Shutdown complete")
            print("=" * 60)


if __name__ == "__main__":
    client = ClientSLAM()
    client.run()