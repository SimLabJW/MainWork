using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using NetMQ;
using NetMQ.Sockets;
using Newtonsoft.Json.Linq;

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
        try { reqSocket.SendFrame(data); Debug.Log($"Send to Python");} catch { yield break; }

        WaitForSeconds delay = new WaitForSeconds(0.05f); // 20Hz 폴링

        while (!isShuttingDown)
        {
            if (reqSocket == null) break;

            if (reqSocket.TryReceiveFrameString(out string reply))
            {
                Debug.Log("Receive Python Data");
                var jObject = JObject.Parse(reply);
                Debug.Log("서버응답 : " + jObject);

                var resultObj = jObject["result"] as JObject;   // result 객체
                string status = (string)resultObj?["status"];   // status 문자열 안전 추출

                if (status == "continue")
                {
                    var jPath = resultObj?["path"] as JArray;
                    if (jPath != null)
                    {
                        var waypoints = new List<Vector3>(jPath.Count);
                        foreach (var p in jPath)
                        {
                            float x = p[0].Value<float>();
                            float y = p[1].Value<float>();
                            waypoints.Add(new Vector3(x, 0f, y));
                        }

                        // s_agent가 null이면 여기서도 NRE 날 수 있음 → 가드
                        if (GameManager.s_agent != null)
                            GameManager.s_agent.MoveUpdateEvent(waypoints);
                        else
                            Debug.LogWarning("GameManager.s_agent is null");
                    }
                    break;
                }
                else if (status == "renewal")
                {
                    break;
                }
                else
                {
                    Debug.LogWarning($"알 수 없는 status: {status}");
                }

                
            }

            yield return delay;
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

[System.Serializable]
public class CommandResult
{
    public string status;
    public float[][] commands;
}

[System.Serializable]
public class ResponseWrapper
{
    public bool ok;
    public CommandResult result;
}
