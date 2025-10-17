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
        if (GameManager.s_agent.AgentState == "PROCESS")
        { Debug.Log($"In requestLoop : {data}"); }

        if (isShuttingDown || reqSocket == null)
        {
            yield break;
        }

        try
        {
            // 1. Send (요청)
            try { reqSocket.SendFrame(data); }
            catch
            {
                if (GameManager.s_agent.AgentState == "PROCESS")
                { Debug.Log($"miss the data : {data}"); }
                yield break; // Send 실패 시 finally로 이동
            }

            // 2. Receive (응답 대기)
            while (!isShuttingDown)
            {
                if (reqSocket == null) break;

                if (reqSocket.TryReceiveFrameString(out string reply))
                {
                    var jObject = JObject.Parse(reply);
                    var resultObj = jObject["result"] as JObject;
                    string status = (string)resultObj?["status"];

                    // ... (status 처리 로직은 그대로 유지) ...
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

                            GameManager.s_agent.MoveUpdateEvent(waypoints);
                        }
                        yield break; // 성공적인 응답 후 종료
                    }
                    else if (status == "renewal")
                    {
                        yield break; // 성공적인 응답 후 종료
                    }
                    else
                    {
                        Debug.LogWarning($"알 수 없는 status: {status}");
                        yield break;
                    }
                }

                yield return null; // 응답이 없으면 다음 프레임까지 대기
            }
        }
        finally
        {
            GameManager.s_comm.s_comm_Coroutine = null;
        }

        yield break;
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
