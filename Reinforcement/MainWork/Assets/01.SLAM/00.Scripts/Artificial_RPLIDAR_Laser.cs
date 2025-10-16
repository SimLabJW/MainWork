using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using System.Text;

public class Artificial_RPLIDAR_Laser : MonoBehaviour
{
    [Header("LIDAR Settings")]
    public float scanFrequencyHz = 8f;
    public int pointsPerScan = 1450;
    public float maxDistance = 40f;
    public float minDistance = 0.15f;

    [Header("Scan Origin")]
    public Transform lidarOrigin;

    void Start()
    {
        GameManager.s_agent.StartLidar -= Start_ArtifitalLidar;
        GameManager.s_agent.StartLidar += Start_ArtifitalLidar;

        GameManager.s_agent.StopLidar -= StopArtificialLidar;
        GameManager.s_agent.StopLidar += StopArtificialLidar;
    }

    

    void Start_ArtifitalLidar()
    {
        if (GameManager.s_agent.scanCoroutine != null)
        {
            StopCoroutine(GameManager.s_agent.scanCoroutine);
        }
        GameManager.s_agent.scanCoroutine = StartCoroutine(ScanRoutine());
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
            Debug.Log("requestLoop corountine종료");
        }
    }

    IEnumerator ScanRoutine()
    {
        float scanInterval = 1f / scanFrequencyHz;
        
        PerformScan();
        yield return new WaitForSeconds(scanInterval);

    }

    void PerformScan()
    {
        float angleIncrement = 360f / pointsPerScan;
        StringBuilder sb = new StringBuilder(pointsPerScan * 24);

        for (int i = 0; i < pointsPerScan; i++)
        {
            float angleDeg = i * angleIncrement;
            float angleRad = angleDeg * Mathf.Deg2Rad;
            Quaternion rotation = Quaternion.Euler(0, angleDeg, 0);
            Vector3 direction = rotation * Vector3.forward;

            float distFactor = 0f;
            float angleToSurface = 0f;
            float combined = 0f;
            float intensity = 0f;
            float distance = maxDistance;
            float distance_mm = maxDistance * 1000f;
            string hitName = "None";

            // Raycast 전체 처리 (탱크 포함 모든 레이어에 대해)
            if (Physics.Raycast(lidarOrigin.position, direction, out RaycastHit hit, maxDistance))
            {
                if (hit.distance >= minDistance)
                {
                    string hitLayer = LayerMask.LayerToName(hit.collider.gameObject.layer);

                    if (hitLayer == "IgnoreOutput")
                    {
                        // 탱크에 막혔지만 출력은 안 함. 하지만 프레임은 보내야 하므로 hit 정보 없이 보냄
                        distance = maxDistance;
                        distance_mm = maxDistance * 1000f;
                        intensity = 0f;
                        hitName = "IgnoreOutput";
                        // Debug.DrawLine(lidarOrigin.position, hit.point, Color.yellow, 1f);
                    }
                    else
                    {
                        // 출력 대상이면 시각화
                        distance = hit.distance;
                        distance_mm = distance * 1000f;

                        distFactor = Mathf.InverseLerp(minDistance, maxDistance, distance);
                        angleToSurface = Vector3.Angle(hit.normal, -direction) / 90f;
                        combined = Mathf.Clamp01(1f - distFactor) * (1f - angleToSurface);
                        intensity = Mathf.Lerp(50f, 255f, combined);

                        hitName = hit.collider.name;

                        // Debug.DrawLine(lidarOrigin.position, hit.point, Color.red, 1f);
                    }
                }
                else
                {
                    // 최소 거리 미만일 때도 프레임을 보냄
                    distance = minDistance;
                    distance_mm = minDistance * 1000f;
                    intensity = 0f;
                    hitName = "TooClose";
                    Vector3 endPoint = lidarOrigin.position + direction * minDistance;
                    // Debug.DrawLine(lidarOrigin.position, endPoint, Color.cyan, 1f);
                }
            }
            else
            {
                // 아무것도 안 맞았을 때
                distance = maxDistance;
                distance_mm = maxDistance * 1000f;
                intensity = 0f;
                hitName = "None";
                Vector3 endPoint = lidarOrigin.position + direction * maxDistance;
                // Debug.DrawLine(lidarOrigin.position, endPoint, Color.green, 1f);
            }
            
            // 누적 수집: theta(rad),distance(mm),intensity
            sb.AppendFormat("{0:F4},{1:F1},{2:F0}\n", angleRad, distance_mm, intensity);

        }
        sb.AppendFormat("POSE,{0:F4},{1:F4},{2:F4}\n",
            GameManager.s_agent.poseX_m,
            GameManager.s_agent.poseY_m,
            GameManager.s_agent.poseTheta_rad);

        sb.AppendFormat($"{GameManager.s_agent.AgentState}\n");

        StartCoroutine(FormulationData(sb));
    }

    IEnumerator FormulationData(StringBuilder sb)
    {
        string SLAM_Data = sb.ToString();
        if (GameManager.s_comm.s_comm_Coroutine == null )
        {
            GameManager.s_comm.s_comm_Coroutine = StartCoroutine(GameManager.s_comm.RequestLoop(SLAM_Data));
            yield return null;
        }
    }
}
