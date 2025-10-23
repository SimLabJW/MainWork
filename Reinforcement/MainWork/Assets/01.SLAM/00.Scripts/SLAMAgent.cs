using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.UI;

public class SLAMAgent : MonoBehaviour
{
    public Button connectSocket_Button;
    public Button StartLidar_Button;
    public Button CloseConnect_Button;

    public Button MakeGridmap_Button;
    public Button Capture_Button;
    

    // Start is called before the first frame update
    void Start()
    {
        if (connectSocket_Button != null)
            connectSocket_Button.onClick.AddListener(OnConnectSocketButtonClicked);

        if (StartLidar_Button != null)
            StartLidar_Button.onClick.AddListener(OnStartLidarButtonClicked);

        if (CloseConnect_Button != null)
            CloseConnect_Button.onClick.AddListener(OffConnectSocketButtonClicked);

        if (MakeGridmap_Button != null)
            MakeGridmap_Button.onClick.AddListener(OnMakeGridMapButtonClicked);
        if (Capture_Button != null)
            Capture_Button.onClick.AddListener(OnCaptureButtonClicked);
    }

    void OnConnectSocketButtonClicked()
    {
        Debug.Log("소켓 연결 버튼이 눌렸습니다.");
        GameManager.s_comm.ConnectZMQSocket();
    }

    void OffConnectSocketButtonClicked()
    {
        Debug.Log("소켓 삭제 버튼이 눌렸습니다.");
        GameManager.s_agent.LidarStopEvent();
        GameManager.s_comm.OnApplicationQuit();
    }

    void OnStartLidarButtonClicked()
    {
        Debug.Log("Lidar 시작 버튼이 눌렸습니다.");
        GameManager.s_agent.LidarStartEvent();
    }

    // ----------
    void OnMakeGridMapButtonClicked()
    {
        Debug.Log("make grid 시작 버튼이 눌렸습니다.");
        GameManager.s_map.StartMakingEvent();
    }
    void OnCaptureButtonClicked()
    {
        Debug.Log("make grid 시작 버튼이 눌렸습니다.");
        GameManager.s_map.CapturingEvnet();
    }

}
