using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using NetMQ;
using NetMQ.Sockets;

public class SLAMCommManager
{
    public RequestSocket reqSocket;

    public Coroutine s_comm_Coroutine;

    static bool s_netmqInitialized = false;
    bool isConnected = false;
    bool isShuttingDown = false;

    public void ConnectZMQSocket()
    {
        if (isConnected || isShuttingDown || reqSocket != null) return;

        if (!s_netmqInitialized)
        {
            AsyncIO.ForceDotNet.Force();
            s_netmqInitialized = true;
        }

        reqSocket = new RequestSocket();
        reqSocket.Options.Linger = System.TimeSpan.Zero;
        reqSocket.Connect("tcp://localhost:8788");
        isConnected = true;
        Debug.Log("REQ 연결됨");
    }

    public IEnumerator RequestLoop(string data) // Unity -> Python
    {
        if (isShuttingDown || reqSocket == null) yield break;
        try { reqSocket.SendFrame(data); } catch { yield break; }

        WaitForSeconds delay = new WaitForSeconds(0.05f); // 20Hz 폴링
        float waited = 0f;
        const float timeoutSec = 2f;

        // 데이터를 보낸 후, 응답이 올 때까지 대기(타임아웃)
        while (!isShuttingDown && waited < timeoutSec)
        {
            if (reqSocket == null) break;

            if (reqSocket.TryReceiveFrameString(out string reply))
            {
                Debug.Log("서버 응답: " + reply);
                break; 
            }

            yield return delay;
            waited += 0.05f;
        }
        s_comm_Coroutine = null;
    }

    public void OnApplicationQuit()
    {
        ShutdownSafely();
    }

    public void OnDisable()
    {
        ShutdownSafely();
    }

    public void OnDestroy()
    {
        ShutdownSafely();
    }

    public void ShutdownSafely()
    {
        if (isShuttingDown) return;
        isShuttingDown = true;
        try
        {
            try { reqSocket?.Close(); } catch {}
        }
        finally
        {
            try { reqSocket?.Dispose(); } catch {}
            reqSocket = null;
            isConnected = false;
            try { if (s_netmqInitialized) NetMQConfig.Cleanup(); } catch {}
            s_netmqInitialized = false;
        }
    }
}
