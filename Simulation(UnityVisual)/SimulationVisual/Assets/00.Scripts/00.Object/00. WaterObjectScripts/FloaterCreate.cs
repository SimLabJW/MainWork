using System.Collections.Generic;
using UnityEngine;
using UnityEngine.Rendering.HighDefinition;

public class FloaterCreate : MonoBehaviour
{
    //public GameObject floaterPrefab; // 프리팹으로 floater 생성
    private List<Transform> floaterPoints = new List<Transform>();

    // Rigidbody 컴포넌트 (추후, 직접 할당하지 않으면 기본 컴포넌트 사용)
    //public Rigidbody rb;
    // 오브젝트의 바닥 아래로 얼마나 잠겼을 때 부력을 적용할지 결정하는 변수
    private float depthBefSub = 0.8f;

    // 최대 부력의 크기 (물 아래로 얼마나 잠겼을 때 최대 부력을 받을지 결정)
    private float displacementAmt = 1f;

    // 물에서의 선형 저항 계수 (속도에 따른 저항)
    private float waterDrag = 1f;

    // 물에서의 회전 저항 계수 (회전에 따른 저항)
    private float waterAngularDrag = 1f;

    // HDRP Water Surface 컴포넌트 참조 (물 표면 정보 사용)
    //public WaterSurface water;

    // 물 표면 위치를 찾기 위한 검색 파라미터 구조체
    WaterSearchParameters Search;

    // 물 표면 위치 검색 결과 구조체 (projectedPositionWS 등)
    WaterSearchResult SearchResult;
    

    private void Start()
    {
        //AttachFloaters();
        GameManager.createScenario.SeaAgentAction -= AttachFloaters;
        GameManager.createScenario.SeaAgentAction += AttachFloaters;
    }

    private void AttachFloaters(GameObject WaterObject, GameObject AgentObject)
    {
        if(AgentObject != this.gameObject) return;
        // AgentObject 하위 렌더러에서 바운드 계산
        Renderer[] renderers = AgentObject.GetComponentsInChildren<Renderer>();
        if (renderers.Length == 0) return;

        Bounds bounds = renderers[0].bounds;
        for (int i = 1; i < renderers.Length; i++)
            bounds.Encapsulate(renderers[i].bounds);

        // 바닥면 기준 5개의 부력 포인트 생성
        Vector3[] points = new Vector3[]
        {
            bounds.center + new Vector3( bounds.extents.x, -bounds.extents.y,  bounds.extents.z),
            bounds.center + new Vector3( bounds.extents.x, -bounds.extents.y, -bounds.extents.z),
            bounds.center + new Vector3(-bounds.extents.x, -bounds.extents.y,  bounds.extents.z),
            bounds.center + new Vector3(-bounds.extents.x, -bounds.extents.y, -bounds.extents.z),
            bounds.center + new Vector3(0, -bounds.extents.y, 0)
        };

        // 2. Floater 5개 생성 및 설정
        foreach (var point in points)
        {
            GameObject floater = Instantiate(GameManager.Instance.Floater, point, Quaternion.identity, AgentObject.transform);
            floater.name = "Floater";

            var agentFloat = floater.AddComponent<AgentFloat>();
            agentFloat.rb = AgentObject.GetComponent<Rigidbody>();
            agentFloat.water = WaterObject.GetComponent<WaterSurface>();
            agentFloat.depthBefSub = depthBefSub;
            agentFloat.displacementAmt = displacementAmt;
            agentFloat.waterDrag = waterDrag;
            agentFloat.waterAngularDrag = waterAngularDrag;
        }

    }


}
