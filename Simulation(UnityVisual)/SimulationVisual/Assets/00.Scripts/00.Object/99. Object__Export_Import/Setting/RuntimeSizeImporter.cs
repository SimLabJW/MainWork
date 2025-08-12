using GLTFast;
using System.IO;
using System.Threading.Tasks;
using UnityEngine;
using System;
using System.Collections;
using System.Collections.Generic;

public class RuntimeSizeImporter : MonoBehaviour
{
    // Start is called before the first frame update
    void Start()
    {
        GameManager.createScenario.ImportAgentSizeAction -= CaculateSize;
        GameManager.createScenario.ImportAgentSizeAction += CaculateSize;
    }

    // Comfirm Object Size
    public async void CaculateSize(string fileId, string fileName, Transform Position, Transform Parent, string table)
    {
        GameManager.createScenario.maxFigure = await StartSize(fileId, fileName, Position, Parent, table);
    }
    public async Task<float> StartSize(string fileId, string fileName, Transform Position, Transform Parent, string table)
    {
        return await ImportSizeModel(fileId, fileName, Position, Parent, table);
    }

    public async Task<float> ImportSizeModel(string fileId, string fileName, Transform Position, Transform Parent, string table)
    {
        var tcs = new TaskCompletionSource<bool>();

        IEnumerator WaitForComm()
        {
            GameManager.communication.Communication(table, new List<string> { "data" }, new Dictionary<string, object> { { "id", fileId } }, "GET");
            yield return new WaitForSeconds(1.5f);
            tcs.SetResult(true);
        }

        // 코루틴 시작
        StartCoroutine(WaitForComm());

        // 통신이 끝날 때까지 대기
        await tcs.Task;

        string urlResult = GameManager.communication.Url_result;

        byte[] glbBytes = null;
        try
        {
            glbBytes = Convert.FromBase64String(urlResult);
        }
        catch (Exception e)
        {
            Debug.LogError($"GLB base64 디코딩 실패: {e.Message}");
            return -1f;
        }

        var gltf = new GltfImport();
        bool success = false;
        try
        {
            success = await gltf.LoadGltfBinary(glbBytes);
        }
        catch (Exception e)
        {
            Debug.LogError($"GLB 로드 실패: {e.Message}");
            return -1f;
        }

        if (!success)
        {
            Debug.LogError("GLB 로드에 실패했습니다.");
            return -1f;
        }

        GameObject glbObject = new GameObject(fileName);

        await gltf.InstantiateMainSceneAsync(glbObject.transform);

        GameObject gltfast_object = glbObject;

        if (gltfast_object != null)
        {
            Renderer[] renderers = gltfast_object.GetComponentsInChildren<Renderer>();
            if (renderers.Length > 0)
            {
                Bounds bounds = renderers[0].bounds;
                for (int i = 1; i < renderers.Length; i++)
                    bounds.Encapsulate(renderers[i].bounds);

                GameObject.Destroy(gltfast_object);
                return Mathf.Max(bounds.size.x, bounds.size.z); // 최대 크기 반환
            }
        }
        return -1f;
    }
}
