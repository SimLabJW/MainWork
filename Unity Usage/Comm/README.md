# MainWork / Unity / Comm

## Unity URL Communication / HTTP
### Post

    public IEnumerator Class Name(string ConnectionUrl, string jsonData)
    {

        UnityWebRequest request = new UnityWebRequest(ConnectionUrl, "POST");
        byte[] bodyRaw = System.Text.Encoding.UTF8.GetBytes(jsonData);
        request.uploadHandler = new UploadHandlerRaw(bodyRaw);
        request.SetRequestHeader("Content-Type", "application/json");
        request.downloadHandler = new DownloadHandlerBuffer();

        yield return request.SendWebRequest();

        if (request.result != UnityWebRequest.Result.Success)
        {
            Debug.LogError("Error: " + request.error);
        }
        else
        {
            Debug.Log("Response: " + request.downloadHandler.text);
            
        }
    }

---

## Unity ZMQ Communication

1. 필수 설치 파일
    - AsyncIO.dll
    - NaCl.dll
    - NetMQ.dll
---
2. 공통 using
---
    using NetMQ;
    using NetMQ.Sockets;
---

### Request
    
    private RequestSocket client;

    void Start()
    {
        Debug.Log("Start.");
        AsyncIO.ForceDotNet.Force();
        checksocket();
    }

    void checksocket()
    {
        client = new RequestSocket();
        client.Connect("tcp://192.168.50.174:20001");

        StartCoroutine(RequestLoop());
    }    

    IEnumerator RequestLoop()
    {
        WaitForSeconds delay = new WaitForSeconds(0.1f); // 10Hz 루프

        while (true)
        {
            reqSocket.SendFrame("data");

            // Python에서 Reply 대기
            if (reqSocket.TryReceiveFrameString(out string reply))
            {
                Debug.Log("서버 응답: " + reply);
            }

            yield return delay;
        }
    }

    private void OnDestroy()
    {
        reqSocket?.Dispose();
        NetMQConfig.Cleanup();
    }

---

### PUB-SUB

    public PublisherSocket client;

    void Start()
    {
        Debug.Log("Start.");
        AsyncIO.ForceDotNet.Force();
        checksocket();
    }

    void checksocket()
    {
        client = new PublisherSocket();
        client.Bind("tcp://*:20001");
    }

    void usingMethod()
    {
        client.SendFrame("");
    }

    void OnDestroy()
    {
        client.Dispose();
        NetMQConfig.Cleanup(false);
    }

---

## Unity WebSocket Communication

    void ConnectWebSocket()
    {
        websocket = new WebSocket("ws://localhost:8000");

        websocket.OnOpen += () => Debug.Log("WebSocket 연결됨");
        websocket.OnError += (e) => Debug.Log("WebSocket 에러: " + e);
        websocket.OnClose += (e) => Debug.Log("WebSocket 닫힘");

        websocket.OnMessage += (bytes) =>
        {
            string msg = System.Text.Encoding.UTF8.GetString(bytes);
            Debug.Log("서버에서 받은 메시지: " + msg);
            OnServerMessage(msg);
        };

        await websocket.Connect();
    }

    void SendData(string jsonData)
    {
        StartCoroutine(SendDataLoop(jsonData));
    }

    IEnumerator SendDataLoop(string jsonData)
    {
        WaitForSeconds delay = new WaitForSeconds(0.1f); // 10Hz


        while (true)
        {
            if (websocket.State == WebSocketState.Open)
            {
                await websocket.SendText(jsonData);
            }

            yield return delay;
        }
    }

    protected virtual void OnServerMessage(string message)
    {
        Debug.Log("처리할 메시지: " + message);
        // 이 함수 override 해서 원하는 처리 가능
    }

    public async void CloseConnection()
    {
        if (websocket != null)
        {
            await websocket.Close();
            websocket = null;
        }
    }

---