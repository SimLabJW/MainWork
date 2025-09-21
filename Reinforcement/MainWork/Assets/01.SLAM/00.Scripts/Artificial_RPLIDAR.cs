using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class Artificial_RPLIDAR : MonoBehaviour
{
    [Header("LIDAR Settings")]
    public float scanFrequencyHz = 5.5f;
    public int pointsPerScan = 1450;
    public float maxDistance = 12f;
    public float minDistance = 0.15f;

    [Header("Scan Origin")]
    public Transform lidarOrigin;

    void Start()
    {
        StartCoroutine(ScanRoutine());
    }

    IEnumerator ScanRoutine()
    {
        float scanInterval = 1f / scanFrequencyHz;
        while (true)
        {
            PerformScan();
            yield return new WaitForSeconds(scanInterval);
        }
    }

    void PerformScan()
    {
        float angleIncrement = 360f / pointsPerScan;

        for (int i = 0; i < pointsPerScan; i++)
        {
            float angleDeg = i * angleIncrement;
            float angleRad = angleDeg * Mathf.Deg2Rad;
            Quaternion rotation = Quaternion.Euler(0, angleDeg, 0);
            Vector3 direction = rotation * Vector3.forward;

            // Raycast 전체 처리 (탱크 포함 모든 레이어에 대해)
            if (Physics.Raycast(lidarOrigin.position, direction, out RaycastHit hit, maxDistance))
            {
                if (hit.distance >= minDistance)
                {
                    string hitLayer = LayerMask.LayerToName(hit.collider.gameObject.layer);

                    if (hitLayer == "IgnoreOutput")
                    {
                        // 탱크에 막혔지만 출력은 안 함
                        continue;
                    }

                    // 출력 대상이면 시각화
                    float distance = hit.distance;
                    float distance_mm = distance * 1000f;

                    float distFactor = Mathf.InverseLerp(minDistance, maxDistance, distance);
                    float angleToSurface = Vector3.Angle(hit.normal, -direction) / 90f;
                    float combined = Mathf.Clamp01(1f - distFactor) * (1f - angleToSurface);
                    float intensity = Mathf.Lerp(50f, 255f, combined);

                    Debug.Log($"[HIT] Intensity: {intensity:F0}, θ(rad): {angleRad:F4}, d(mm): {distance_mm:F1}, hit: {hit.collider.name}");
                    Debug.DrawLine(lidarOrigin.position, hit.point, Color.red, 1f);
                }
            }
            else
            {
                // 아무것도 안 맞았을 때만 초록색 표시
                Vector3 endPoint = lidarOrigin.position + direction * maxDistance;
                Debug.DrawLine(lidarOrigin.position, endPoint, Color.green, 1f);
            }
        }
    }
}
