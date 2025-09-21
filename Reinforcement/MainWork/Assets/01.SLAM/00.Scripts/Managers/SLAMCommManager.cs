using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using NetMQ;
using NetMQ.Sockets;

public class SLAMCommManager
{
    private RequestSocket reqSocket;

    void Start()
    {
        AsyncIO.ForceDotNet.Force();
        reqSocket = new RequestSocket();
        reqSocket.Connect("tcp://localhost:5555");
        Debug.Log("REQ 연결됨");

        // StartCoroutine(RequestLoop());
    }

    IEnumerator RequestLoop()
    {
        WaitForSeconds delay = new WaitForSeconds(0.1f); // 10Hz 루프

        while (true)
        {
            string lidarData = $"theta={Time.time % 360:F2}, dist={Random.Range(0.1f, 12f):F2}";
            reqSocket.SendFrame(lidarData);

            // Python에서 Reply 대기
            if (reqSocket.TryReceiveFrameString(out string reply))
            {
                Debug.Log("서버 응답: " + reply);
            }

            yield return delay;
        }
    }

    private void OnApplicationQuit()
    {
        reqSocket?.Dispose();
        NetMQConfig.Cleanup();
    }
}
