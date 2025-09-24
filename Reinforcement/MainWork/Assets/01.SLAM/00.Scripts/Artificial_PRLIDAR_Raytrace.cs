using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class Artificial_PRLIDAR_Raytrace : MonoBehaviour
{
    [Header("LIDAR Settings")]
    public float scanFrequencyHz = 10f;   // 한 바퀴 도는 속도 (Hz)
    public int pointsPerScan = 1450;      // 한 바퀴에서 몇 점 찍을지
    public float maxDistance = 80f;       // 최대 감지 거리 (m)
    public float minDistance = 0.15f;     // 최소 감지 거리 (m)

    [Header("Scan Origin")]
    public Transform lidarOrigin;

    [Header("Visualization")]
    public float lineDuration = 0.05f;    // 선이 유지되는 시간 (잔상 효과)

    private float currentAngle = 0f;       // 현재 라이다가 쏘고 있는 각도
    private float[] ranges;
    private float[] intensities;
    private uint seq = 0;                  // 몇 번째 스캔인지 추적

    void Start()
    {
        ranges = new float[pointsPerScan];
        intensities = new float[pointsPerScan];
        Start_ArtificialLidar();
    }

    void Start_ArtificialLidar()
    {
        if (GameManager.s_agent.scanCoroutine == null)
        {
            GameManager.s_agent.scanCoroutine = StartCoroutine(ScanRoutine());
        }
    }

    public void StopArtificialLidar()
    {
        if (GameManager.s_agent.scanCoroutine != null)
        {
            StopCoroutine(GameManager.s_agent.scanCoroutine);
            GameManager.s_agent.scanCoroutine = null;
            Debug.Log("LIDAR 스캔이 중지되었습니다.");
        }

        if (GameManager.s_comm.s_comm_Coroutine != null)
        {
            StopCoroutine(GameManager.s_comm.s_comm_Coroutine);
            GameManager.s_comm.s_comm_Coroutine = null;
            Debug.Log("requestLoop coroutine 종료");
        }
    }

    IEnumerator ScanRoutine()
    {
        float scanInterval = 1f / scanFrequencyHz;       // 한 바퀴 도는 시간 (ex: 0.1초)
        float angleIncrement = 360f / pointsPerScan;     // 각도 증가량
        float stepDelay = scanInterval / pointsPerScan;  // 레이 하나 쏘고 기다릴 시간

        while (true)
        {
            // 이번 프레임에서 몇 개의 레이를 처리해야 하는지 계산
            int raysPerFrame = Mathf.CeilToInt(pointsPerScan / (scanInterval / Time.deltaTime));

            for (int n = 0; n < raysPerFrame; n++)
            {
                float angleRad = currentAngle * Mathf.Deg2Rad;
                Vector3 dir = new Vector3(Mathf.Sin(angleRad), 0, Mathf.Cos(angleRad));
                int index = Mathf.RoundToInt((currentAngle / 360f) * (pointsPerScan - 1));

                RaycastHit hit;
                if (Physics.Raycast(lidarOrigin.position, dir, out hit, maxDistance))
                {
                    if (hit.distance < minDistance)
                    {
                        ranges[index] = minDistance;
                        intensities[index] = 0f;
                        Debug.DrawLine(lidarOrigin.position, lidarOrigin.position + dir * minDistance, Color.cyan, lineDuration);
                    }
                    else
                    {
                        ranges[index] = hit.distance;

                        // 👉 intensity 계산
                        float distFactor = Mathf.InverseLerp(minDistance, maxDistance, hit.distance);
                        float angleToSurface = Vector3.Angle(hit.normal, -dir) / 90f;
                        float combined = Mathf.Clamp01(1f - distFactor) * (1f - angleToSurface);
                        intensities[index] = Mathf.Lerp(0f, 1f, combined);

                        Debug.DrawLine(lidarOrigin.position, hit.point, Color.red, lineDuration);
                    }
                }
                else
                {
                    ranges[index] = float.PositiveInfinity;
                    intensities[index] = 0f;
                    Debug.DrawLine(lidarOrigin.position, lidarOrigin.position + dir * maxDistance, Color.green, lineDuration);
                }

                currentAngle += angleIncrement;
                if (currentAngle >= 360f)
                {
                    currentAngle = 0f;

                    // 한 바퀴 끝나면 메시지 전송
                    LaserScanMsg msg = new LaserScanMsg()
                    {
                        header = new HeaderMsg()
                        {
                            stamp = Time.time,
                            frame_id = "lidar_frame",
                            seq = seq++
                        },
                        angle_min = 0f,
                        angle_max = 2 * Mathf.PI,
                        angle_increment = angleIncrement * Mathf.Deg2Rad,
                        time_increment = stepDelay,
                        scan_time = scanInterval,
                        range_min = minDistance,
                        range_max = maxDistance,
                        ranges = ranges,
                        intensities = intensities
                    };

                    string json = JsonUtility.ToJson(msg);

                    // 👉 GameManager 통신 루프 (한 바퀴마다 전송)
                    // if (GameManager.s_comm.s_comm_Coroutine == null)
                    // {
                    //     GameManager.s_comm.s_comm_Coroutine = StartCoroutine(GameManager.s_comm.RequestLoop(json));
                    // }
                    // else
                    // {
                    //     GameManager.s_comm.latestData = json;
                    // }

                    // 다음 스캔을 위해 배열 초기화
                    ranges = new float[pointsPerScan];
                    intensities = new float[pointsPerScan];
                }
            }

            yield return null; // 프레임마다 진행
        }
    }

    // ROS 메시지 유사 구조체
    [System.Serializable]
    public class HeaderMsg
    {
        public float stamp;
        public string frame_id = "lidar_frame";
        public uint seq = 0;
    }

    [System.Serializable]
    public class LaserScanMsg
    {
        public HeaderMsg header;
        public float angle_min;
        public float angle_max;
        public float angle_increment;
        public float time_increment;
        public float scan_time;
        public float range_min;
        public float range_max;
        public float[] ranges;
        public float[] intensities;
    }
}
