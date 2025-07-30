using UnityEngine;

public class SelectedOutline : MonoBehaviour
{
    GameObject outlineObj;
    Material outlineMat;

    void Start()
    {
        outlineMat = Resources.Load<Material>("99.Materials/HDRP_Outline"); // Resources 경로 기준
    }

    void OnMouseDown()
    {
        if (outlineObj != null) return;

        // 복제 오브젝트 생성
        outlineObj = Instantiate(gameObject, transform.position, transform.rotation, transform);
        outlineObj.name = "OutlineObj";

        // 살짝 스케일 크게
        outlineObj.transform.localScale *= 1.15f;

        // Collider 제거
        var col = outlineObj.GetComponent<Collider>();
        if (col) Destroy(col);

        // 머티리얼 단일화
        var rend = outlineObj.GetComponent<Renderer>();
        rend.material = outlineMat;
    }

    void OnMouseUp()
    {
        if (outlineObj != null)
        {
            Destroy(outlineObj);
        }
    }
}
