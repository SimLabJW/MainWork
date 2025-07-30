using GLTFast;
using System.IO;
using System.Threading.Tasks;
using UnityEngine;

public class RuntimeSizeImporter : MonoBehaviour
{
    // Start is called before the first frame update
    void Start()
    {
        GameManager.simulation.ImportAgentSizeAction -= CaculateSize;
        GameManager.simulation.ImportAgentSizeAction += CaculateSize;
    }

    // Comfirm Object Size
    public async void CaculateSize(string path, string fileName, Transform Position, Transform Parent)
    {
        GameManager.simulation.maxFigure = await StartSize(path, fileName, Position, Parent);
    }
    public async Task<float> StartSize(string path, string fileName, Transform Position, Transform Parent)
    {
        return await ImportSizeModel(path, fileName, Position, Parent);
    }

    public async Task<float> ImportSizeModel(string path, string fileName, Transform Position, Transform Parent)
    {
        int childCountBefore = Parent.childCount;

        string glbPath = Path.Combine(path, fileName + ".glb");
        var gltf = new GltfImport();

        bool success = await gltf.Load(glbPath);
        if (!success) return -1f;

        success = await gltf.InstantiateMainSceneAsync(Position);
        if (!success) return -1f;

        GameObject gltfast_object = null;

        for (int i = childCountBefore; i < Parent.childCount; i++)
        {
            Transform child = Parent.GetChild(i);
            string containName = child.name;

            if (containName.Contains("Agent"))
            {
                gltfast_object = child.gameObject;
                Debug.Log($"gltfast_obejct name : {gltfast_object.name}");
                break;
            }
        }

        if (gltfast_object != null)
        {
            Debug.Log("gltfast_obejct not null");
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
