using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using NetMQ;
using NetMQ.Sockets;

public class SLAMCommManager
{
    public RequestSocket reqSocket;

    public Coroutine s_comm_Coroutine;

    public void ConnectZMQSocket()
    {
        AsyncIO.ForceDotNet.Force();
        reqSocket = new RequestSocket();
        reqSocket.Connect("tcp://localhost:8788");
        Debug.Log("REQ 연결됨");
    }

    public IEnumerator RequestLoop(string data) // Unity -> Python
    {
        reqSocket.SendFrame(data);

        WaitForSeconds delay = new WaitForSeconds(0.1f); // 10Hz 루프

        // 데이터를 보낸 후, 응답이 올 때까지 대기
        while (true)
        {
            if (reqSocket.TryReceiveFrameString(out string reply))
            {
                Debug.Log("서버 응답: " + reply);
                break; // 응답을 받으면 루프 종료
            }

            yield return delay;
        }
        s_comm_Coroutine = null;
    }

    public void OnApplicationQuit()
    {
        reqSocket?.Dispose();
        NetMQConfig.Cleanup();
    }
}
