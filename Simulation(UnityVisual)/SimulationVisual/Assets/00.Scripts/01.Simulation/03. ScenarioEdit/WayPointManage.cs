using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class WayPointManage : MonoBehaviour
{
    // [Header("포물선 높이")]
    private float parabolaHeight = 25f;

    void Start()
    {
        GameManager.scenarioEdit.WaypointConnectAction -= UpdateParabola;
        GameManager.scenarioEdit.WaypointConnectAction += UpdateParabola;
        
        GameManager.scenarioEdit.WaypointDisConnectionAction -= OnWaypointDelete;
        GameManager.scenarioEdit.WaypointDisConnectionAction += OnWaypointDelete;

        GameManager.scenarioEdit.WaypointReConnectionAction -= ReConnectionParabola;
        GameManager.scenarioEdit.WaypointReConnectionAction += ReConnectionParabola;

    }

    // 중심점끼리 연결하도록 수정
    void UpdateParabola(GameObject pointA, GameObject pointB)
    {
        if (pointA == null || pointB == null || GameManager.scenarioEdit.scinfo.scenarioEditInfo.WaypointConnectPrefab == null)
            return;

        List<GameObject> cubes = new List<GameObject>();
        GameObject cubePrefab = GameManager.scenarioEdit.scinfo.scenarioEditInfo.WaypointConnectPrefab;

        // 프리팹 스케일 유지
        Vector3 prefabScale = cubePrefab.transform.localScale;

        // 프리팹의 z축 기준 간격 계산
        float spacing = prefabScale.z * 2f;

        // 각 오브젝트의 "중심점"을 사용하여 시작/끝 위치 계산
        Vector3 start = pointA.transform.position;
        Vector3 end = pointB.transform.position;

        // 포물선 경로 길이 근사치
        float approxLength = ApproximateParabolaLength(start, end, parabolaHeight, 20);
        int segmentCount = Mathf.Max(1, Mathf.FloorToInt(approxLength / spacing));

        // 점선 포물선 생성 (부모 없이 먼저 생성)
        for (int i = 0; i <= segmentCount; i++)
        {
            float t = (float)i / segmentCount;
            Vector3 pos = GetParabolaPoint(start, end, parabolaHeight, t);

            GameObject dot = Instantiate(cubePrefab, pos, Quaternion.identity);
            dot.transform.localScale = prefabScale;
            cubes.Add(dot);
        }

        // pointA 하위에 Parabolas 오브젝트가 있는지 확인, 없으면 생성
        Transform parabolasTr = pointA.transform.Find("Parabolas");
        if (parabolasTr == null)
        {
            GameObject parabolasObj = new GameObject("Parabolas");
            parabolasObj.transform.SetParent(pointA.transform);
            parabolasObj.transform.localPosition = Vector3.zero;
            parabolasObj.transform.localRotation = Quaternion.identity;
            parabolasObj.transform.localScale = Vector3.one;
            parabolasTr = parabolasObj.transform;
        }
        else
        {
            // 기존에 있던 포물선 점선 오브젝트들 삭제
            for (int i = parabolasTr.childCount - 1; i >= 0; i--)
            {
                Destroy(parabolasTr.GetChild(i).gameObject);
            }
        }

        // 생성된 점선 오브젝트들을 Parabolas 오브젝트의 자식으로 옮김
        foreach (var dot in cubes)
        {
            dot.transform.SetParent(parabolasTr);
        }
    }

    // t 지점에서의 포물선 위치 반환
    Vector3 GetParabolaPoint(Vector3 start, Vector3 end, float height, float t)
    {
        Vector3 linear = Vector3.Lerp(start, end, t);
        float parabola = 4 * height * t * (1 - t); // t = 0.5일 때 최대
        return linear + Vector3.up * parabola;
    }

    // 포물선 경로 근사 길이 계산
    float ApproximateParabolaLength(Vector3 start, Vector3 end, float height, int steps)
    {
        float length = 0f;
        Vector3 prev = GetParabolaPoint(start, end, height, 0f);
        for (int i = 1; i <= steps; i++)
        {
            float t = (float)i / steps;
            Vector3 curr = GetParabolaPoint(start, end, height, t);
            length += Vector3.Distance(prev, curr);
            prev = curr;
        }
        return length;
    }

    // 인덱스 기반 단일 웨이포인트 및 포물선 삭제 처리
    void OnWaypointDelete(GameObject agentObj, int index)
    {
        var (beforeWaypoint, afterWaypoint) = PrefabInfo.GetWaypointsByIndexAndEndpoint(agentObj, index);

        // 케이스 1: 둘 다 없음 -> 해당 index 웨이포인트만 삭제
        if (beforeWaypoint == null && afterWaypoint == null)
        {
            PrefabInfo.RemoveWaypoint(agentObj, index);
            return;
        }

        // 케이스 2: before만 있음 -> before.endpoint 초기화, before 하위 포물선(점선)만 삭제, 현재 웨이포인트 삭제
        if (beforeWaypoint != null && afterWaypoint == null)
        {
            // endpoint "-" 동기화
            PrefabInfo.SetWaypointEndpointAndInput(beforeWaypoint, "-");

            // before 하위 포물선(점선) 삭제: before의 자식 중 WaypointConnectPrefab 기반 도트 제거
            ClearParabolaDotsUnder(beforeWaypoint.pointObject);

            // 현재 웨이포인트 삭제
            PrefabInfo.RemoveWaypoint(agentObj, index);
            return;
        }

        // 케이스 3: after만 있음 -> 현재 웨이포인트 삭제
        if (beforeWaypoint == null && afterWaypoint != null)
        {
            PrefabInfo.RemoveWaypoint(agentObj, index);
            return;
        }

        // 케이스 4: 둘 다 있음 -> before.endpoint 초기화 및 하위 포물선 삭제, after 유지, 현재 웨이포인트 삭제
        if (beforeWaypoint != null && afterWaypoint != null)
        {
            PrefabInfo.SetWaypointEndpointAndInput(beforeWaypoint, "-");
            ClearParabolaDotsUnder(beforeWaypoint.pointObject);
            PrefabInfo.RemoveWaypoint(agentObj, index);
            return;
        }
    }

    // waypointPointObject의 "Parabolas" 오브젝트 하위에 있는 점선(포물선) 오브젝트들만 삭제
    void ClearParabolaDotsUnder(GameObject waypointPointObject)
    {
        if (waypointPointObject == null) return;

        // "Parabolas"라는 이름의 자식 오브젝트를 찾음
        Transform parabolasTr = waypointPointObject.transform.Find("Parabolas");
        if (parabolasTr == null) return;

        // Parabolas 하위의 모든 자식 오브젝트 삭제
        var children = new List<Transform>();
        foreach (Transform child in parabolasTr)
        {
            children.Add(child);
        }

        foreach (var child in children)
        {
            Destroy(child.gameObject);
        }
    }

    void ReConnectionParabola(GameObject agentobj, int poinA_index, string pointB_index)
    {
        if (agentobj == null) return;

        // A 웨이포인트 조회
        var aInfo = PrefabInfo.GetWaypointByIndex(agentobj, poinA_index);
        if (aInfo == null || aInfo.pointObject == null)
            return;

        // B 인덱스 파싱 및 분기
        if (string.IsNullOrEmpty(pointB_index) || pointB_index == "-" || !int.TryParse(pointB_index, out int bIndex))
        {
            // 연결 해제: endpoint "-", 기존 포물선 제거
            PrefabInfo.SetWaypointEndpointAndInput(aInfo, "-");
            ClearParabolaDotsUnder(aInfo.pointObject);
            return;
        }

        var bInfo = PrefabInfo.GetWaypointByIndex(agentobj, bIndex);
        if (bInfo == null || bInfo.pointObject == null)
        {
            // 잘못된 대상: A의 endpoint 초기화 및 포물선 제거
            PrefabInfo.SetWaypointEndpointAndInput(aInfo, "-");
            ClearParabolaDotsUnder(aInfo.pointObject);
            return;
        }

        // endpoint 동기화
        PrefabInfo.SetWaypointEndpointAndInput(aInfo, pointB_index);

        // 기존 포물선 제거 후 재생성
        ClearParabolaDotsUnder(aInfo.pointObject);
        UpdateParabola(aInfo.pointObject, bInfo.pointObject);
    }
    
    
}
