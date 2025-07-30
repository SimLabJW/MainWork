using System.Collections.Generic;
using UnityEngine;

public class PrefabInfo
{
    [System.Serializable]
    public class ObjectInfo
    {
        public string tag;
        public int layer;
        public bool hasRigidbody;
        public ColliderInfo collider;
        public List<ObjectInfo> children = new();
    }

    [System.Serializable]
    public class ColliderInfo
    {
        public string type;
        public bool isTrigger;
        public Vector3 center;
        public Vector3 size;
        public float radius, height;
        public bool convex;
    }

    public ObjectInfo BuildInfo(GameObject go)
    {
        var info = new ObjectInfo
        {
            tag = go.tag,
            layer = go.layer,
            hasRigidbody = go.GetComponent<Rigidbody>() != null,
            collider = ExtractCollider(go)
        };

        foreach (Transform child in go.transform)
        {
            info.children.Add(BuildInfo(child.gameObject));
        }

        return info;
    }

    public ColliderInfo ExtractCollider(GameObject go)
    {
        var col = go.GetComponent<Collider>();
        if (col == null) return null;

        var c = new ColliderInfo
        {
            isTrigger = col.isTrigger,
            center = col.bounds.center,
            size = col.bounds.size
        };

        if (col is BoxCollider) c.type = "Box";
        else if (col is SphereCollider sc)
        {
            c.type = "Sphere";
            c.radius = sc.radius;
        }
        else if (col is CapsuleCollider cc)
        {
            c.type = "Capsule";
            c.radius = cc.radius;
            c.height = cc.height;
        }
        else if (col is MeshCollider mc)
        {
            c.type = "Mesh";
            c.convex = mc.convex;
        }
        else c.type = "Unknown";

        return c;
    }

    public void ApplyInfo(ObjectInfo info, GameObject go)
    {
        go.tag = info.tag;
        go.layer = info.layer;

        if (info.tag == "Water")
        {
            GameObject water = GameObject.Instantiate(GameManager.Instance.Ocean, go.transform.position, go.transform.rotation, go.transform.parent);
            water.name = go.name;

            GameObject.DestroyImmediate(go);
            return;
        }


        if (info.hasRigidbody && go.GetComponent<Rigidbody>() == null)
            go.AddComponent<Rigidbody>();

        if (info.collider != null)
        {
            var c = info.collider;
            switch (c.type)
            {
                case "Box":
                    var box = go.AddComponent<BoxCollider>();
                    box.isTrigger = c.isTrigger;
                    box.center = c.center;
                    box.size = c.size;
                    break;
                case "Sphere":
                    var sphere = go.AddComponent<SphereCollider>();
                    sphere.isTrigger = c.isTrigger;
                    sphere.center = c.center;
                    sphere.radius = c.radius;
                    break;
                case "Capsule":
                    var capsule = go.AddComponent<CapsuleCollider>();
                    capsule.isTrigger = c.isTrigger;
                    capsule.center = c.center;
                    capsule.height = c.height;
                    capsule.radius = c.radius;
                    break;
                case "Mesh":
                    var mesh = go.AddComponent<MeshCollider>();
                    mesh.convex = c.convex;
                    break;
            }
        }

        for (int i = 0; i < Mathf.Min(info.children.Count, go.transform.childCount); i++)
        {
            ApplyInfo(info.children[i], go.transform.GetChild(i).gameObject);
        }
    }


}
