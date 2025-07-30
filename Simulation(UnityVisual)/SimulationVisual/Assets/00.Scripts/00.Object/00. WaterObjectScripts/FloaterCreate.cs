using System.Collections.Generic;
using UnityEngine;
using UnityEngine.Rendering.HighDefinition;

public class FloaterCreate : MonoBehaviour
{
    //public GameObject floaterPrefab; // ������ floater ������
    private List<Transform> floaterPoints = new List<Transform>();


    // Rigidbody ������Ʈ (�߷�, ���� ������ ���� �⺻ �������)
    //public Rigidbody rb;
    // ��ü�� ���� �Ʒ� �󸶳� ���� �η��� �ִ�ġ�� �������� ���� ����
    private float depthBefSub = 0.8f;

    // �ִ� �η��� ���� (���� �Ʒ��� ������ ����� ���� �� ����)
    private float displacementAmt = 1f;

    // ���ӿ����� ���� ���׷� ��� (�ӵ��� ����)
    private float waterDrag = 1f;

    // ���ӿ����� ȸ�� ���׷� ��� (ȸ���� ����)
    private float waterAngularDrag = 1f;

    // HDRP Water Surface ������Ʈ ���� (���� ������ ��� ����)
    //public WaterSurface water;

    // ���� ��ġ�� ã�� ���� �˻� �Ķ���� ����ü
    WaterSearchParameters Search;

    // ���� ��ġ �˻� ��� ����ü (projectedPositionWS ����)
    WaterSearchResult SearchResult;
    

    private void Start()
    {
        //AttachFloaters();
        GameManager.simulation.SeaAgentAction -= AttachFloaters;
        GameManager.simulation.SeaAgentAction += AttachFloaters;
    }

    private void AttachFloaters(GameObject WaterObject, GameObject AgentObject)
    {
        if(AgentObject != this.gameObject) return;
        // AgentObject ���� ���������� �ٿ�� �ջ�
        Renderer[] renderers = AgentObject.GetComponentsInChildren<Renderer>();
        if (renderers.Length == 0) return;

        Bounds bounds = renderers[0].bounds;
        for (int i = 1; i < renderers.Length; i++)
            bounds.Encapsulate(renderers[i].bounds);

        // �ٿ�� ���� 5���� �η� ����Ʈ ���
        Vector3[] points = new Vector3[]
        {
        bounds.center + new Vector3( bounds.extents.x, -bounds.extents.y,  bounds.extents.z),
        bounds.center + new Vector3( bounds.extents.x, -bounds.extents.y, -bounds.extents.z),
        bounds.center + new Vector3(-bounds.extents.x, -bounds.extents.y,  bounds.extents.z),
        bounds.center + new Vector3(-bounds.extents.x, -bounds.extents.y, -bounds.extents.z),
        bounds.center + new Vector3(0, -bounds.extents.y, 0)
        };

        // 1. Floaters �θ� ������Ʈ ã��
        Transform floatersParent = AgentObject.transform.Find("Body/Floaters");
        if (floatersParent == null)
        {
            Debug.LogError("Floaters ������Ʈ�� ã�� �� �����ϴ�. Body/Floaters ��θ� Ȯ���ϼ���.");
            //yield break; // �Ǵ� return;
        }

        // 2. Floater 5�� ���� �� ����
        foreach (var point in points)
        {
            GameObject floater = Instantiate(GameManager.Instance.Floater, point, Quaternion.identity, floatersParent);
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
