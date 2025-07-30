using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.UI;

public class EditorView : MonoBehaviour
{
    // Map Fit, Map Zoom, Plyaer move in Map
    private Bounds targetBounds;
    private float zoomSpeed = 30f;
    private float squareTolerance = 0.05f;
    private float minOrthoSize = 10f;
    private float maxOrthoSize;
    private Vector3 lastMousePosition;

    // Editor Agent Button -> Create Sphere Bound
    private GameObject currentBoundSphere;
    private Coroutine editorRoutine;

    private bool SelectAgent = false;

    // Editor View component
    private RawImage editor_raw;
    private RectTransform editor_rectransform;

    // Object Profile(name, position, object list)
    private ObjectProfile profile = new ObjectProfile();


    void Start()
    {
        GameManager.simulation.EditorViewFitAction -= MapSizeFitToEditorView;
        GameManager.simulation.EditorViewFitAction += MapSizeFitToEditorView;

        GameManager.simulation.EditorViewControlAction -= ObjectSizeCalculate;
        GameManager.simulation.EditorViewControlAction += ObjectSizeCalculate;
    }

    // Phase 1(Env Button)
    private void MapSizeFitToEditorView(GameObject SimulationEnvObject)
    {
        editor_raw = GameManager.simulation.sm.simulationInfo.editorInform.Editor_MapView.GetComponent<RawImage>();
        editor_rectransform = editor_raw.rectTransform;

        List<Transform> transforms = new List<Transform>();

        foreach (Transform t in SimulationEnvObject.GetComponentsInChildren<Transform>())
        {
            if (t.CompareTag("Plane") || t.CompareTag("Water"))
            {
                transforms.Add(t);
            }
        }
        FitCameraToTargets(transforms);
    }
    // Phase 2(Agent Button)
    private void ObjectSizeCalculate(string path, string fileName, Transform Position)
    {
        // maxFigure를 먼저계산
        StartCoroutine(DelayedImporterSize(path, fileName, Position));

        // 해당 최대 크기로 원 크기 생성
        StartCoroutine(StartImporterSize(path, fileName, Position));

        // 기능
        //StartCoroutine(EditorMoveandCreate(path, fileName, Position));
        if (editorRoutine != null)
        {
            StopCoroutine(editorRoutine);
        }
        editorRoutine = StartCoroutine(EditorMoveandCreate(path, fileName, Position));
    }

    IEnumerator StartImporterSize(string path, string fileName, Transform Position)
    {
        yield return new WaitForSeconds(0.2f);

        if (GameManager.simulation.sm.simulationInfo.editorInform.Agent_Size != null)
        {
            currentBoundSphere = Instantiate(GameManager.simulation.sm.simulationInfo.editorInform.Agent_Size);
            currentBoundSphere.transform.localScale = new Vector3(GameManager.simulation.maxFigure * 2f, 0.5f, GameManager.simulation.maxFigure * 2f);
        }
    }
    IEnumerator DelayedImporterSize(string path, string fileName, Transform Position)
    {
        
        GameManager.simulation.ImportObject(path, fileName, GameManager.simulation.sm.simulationInfo.Simulation_ENV,
                    GameManager.simulation.sm.simulationInfo.Simulation_ENV, "agent_size");
        yield return new WaitForSeconds(1f);
    }

    // Phase 3(Scenario Button)
    private void CreatePointArea()
    {

    }
    IEnumerator EditorMoveandCreate(string path, string fileName, Transform Position)
    {
        while (true)
        {
            HandleCameraControl();
            if (RectTransformUtility.RectangleContainsScreenPoint(editor_rectransform, Input.mousePosition) &&
            TryGetMouseHit(out RaycastHit hit))
            {
                if (currentBoundSphere != null)
                {
                    //HandleBoundSpherePlacement(hit, path, fileName);
                    currentBoundSphere.transform.position = new Vector3(hit.point.x, 10f, hit.point.z);
                    if (Input.GetMouseButtonDown(0))
                    {
                        currentBoundSphere.transform.position = new Vector3(hit.point.x, 0f, hit.point.z);

                        GameManager.simulation.ImportObject(path, fileName, currentBoundSphere.transform,
                            GameManager.simulation.sm.simulationInfo.Simulation_ENV, "agent_import");


                        Destroy(currentBoundSphere);
                        currentBoundSphere = null;
                        //break;
                    }
                }
                else 
                {
                    if (SelectAgent && GameManager.simulation.currentObeject != null
                    && GameManager.simulation.currentObeject.tag.Contains("Agent")) 
                    { 
                        HandleAgentMovement(hit); 
                    }

                    else if (!SelectAgent && Input.GetMouseButtonDown(0))
                    {
                        
                        TrySelectObject(hit);

                    }
                }

                

            }
            yield return null;
        }

    }

    // Default Content
    bool TryGetMouseHit(out RaycastHit hit)
    {
        hit = new RaycastHit();
        Vector2 localPoint;
        if (RectTransformUtility.ScreenPointToLocalPointInRectangle(
                editor_rectransform, Input.mousePosition, null, out localPoint))
        {
            Vector2 normalizedPoint = new Vector2(
                (localPoint.x + editor_rectransform.rect.width / 2) / editor_rectransform.rect.width,
                (localPoint.y + editor_rectransform.rect.height / 2) / editor_rectransform.rect.height);

            Ray ray = GameManager.simulation.sm.cameraType.MapView_Editor.ViewportPointToRay(normalizedPoint);
            return Physics.Raycast(ray, out hit);
        }
        return false;
    }

    // Content1 : Phase 1, Phase 2
    void HandleCameraControl()
    {
        float scroll = Input.GetAxis("Mouse ScrollWheel") * zoomSpeed;
        if (scroll != 0)
            zoom_In_Out(scroll);
        HandleMousePan();
    }
    

    // Phase 1 - Map to fit
    void FitCameraToTargets(List<Transform> EnvTarget)
    {

        if (GameManager.simulation.sm.cameraType.MapView_Editor == null || EnvTarget == null || EnvTarget.Count == 0) return;

        GameManager.simulation.sm.cameraType.MapView_Editor.orthographic = true;

        Bounds bounds = new Bounds(EnvTarget[0].position, Vector3.zero);
        foreach (var t in EnvTarget)
        {
            Vector3 pos = t.position;
            Vector3 extents = t.localScale * 0.5f;
            bounds.Encapsulate(pos + extents);
            bounds.Encapsulate(pos - extents);
        }

        Vector3 center = bounds.center;
        float sizeX = bounds.size.x;
        float sizeZ = bounds.size.z;
        float aspect = (float)GameManager.simulation.sm.cameraType.MapView_Editor.pixelWidth 
            / GameManager.simulation.sm.cameraType.MapView_Editor.pixelHeight;

        float ratio = sizeX / sizeZ;
        bool isSquare = Mathf.Abs(ratio - 1f) < squareTolerance;

        float orthoSize;
        if (isSquare)
        {
            orthoSize = Mathf.Max(sizeX, sizeZ) / 2f;
        }
        else
        {
            float sizeY = sizeZ / 2f;
            float sizeXAspect = sizeX / (2f * aspect);
            orthoSize = Mathf.Max(sizeY, sizeXAspect);
        }

        targetBounds = bounds;

        GameManager.simulation.sm.cameraType.MapView_Editor.orthographicSize 
            = orthoSize;
        maxOrthoSize = orthoSize;
        GameManager.simulation.sm.cameraType.MapView_Editor.transform.position 
            = new Vector3(center.x, center.y + 100f, center.z);
        GameManager.simulation.sm.cameraType.MapView_Editor.transform.rotation
            = Quaternion.Euler(90f, 0f, 0f);
    }

    // Phase 2 - Function ( Zoom, clickto move camera)
    private void zoom_In_Out(float scroll)
    {
        GameManager.simulation.sm.cameraType.MapView_Editor.orthographicSize 
            -= scroll * zoomSpeed;
        GameManager.simulation.sm.cameraType.MapView_Editor.orthographicSize 
            = Mathf.Clamp(GameManager.simulation.sm.cameraType.MapView_Editor.orthographicSize, minOrthoSize, maxOrthoSize);
    }
    public float panSpeed = 0.005f;
    private void HandleMousePan()
    {
        if (Input.GetMouseButtonDown(1))
        {
            lastMousePosition = Input.mousePosition;
        }

        if (Input.GetMouseButton(1))
        {
            Vector3 delta = Input.mousePosition - lastMousePosition;
            lastMousePosition = Input.mousePosition;

            Vector3 move 
                = new Vector3(-delta.x, 0, -delta.y) * panSpeed 
                * GameManager.simulation.sm.cameraType.MapView_Editor.orthographicSize;
            Vector3 newPos 
                = GameManager.simulation.sm.cameraType.MapView_Editor.transform.position + move;

            float halfHeight = GameManager.simulation.sm.cameraType.MapView_Editor.orthographicSize;
            float halfWidth = halfHeight * GameManager.simulation.sm.cameraType.MapView_Editor.aspect;

            float minX = targetBounds.min.x + halfWidth;
            float maxX = targetBounds.max.x - halfWidth;
            float minZ = targetBounds.min.z + halfHeight;
            float maxZ = targetBounds.max.z - halfHeight;

            newPos.x = Mathf.Clamp(newPos.x, minX, maxX);
            newPos.z = Mathf.Clamp(newPos.z, minZ, maxZ);

            GameManager.simulation.sm.cameraType.MapView_Editor.transform.position = newPos;
        }
    }

    //Content : Create Object with sphere asset
    void HandleBoundSpherePlacement(RaycastHit hit, string path, string fileName)
    {
        currentBoundSphere.transform.position = new Vector3(hit.point.x, 8f, hit.point.z);
        if (Input.GetMouseButtonDown(0))
        {
            Debug.Log("HandleBoundSphere Place click");
            currentBoundSphere.transform.position = new Vector3(hit.point.x, 0f, hit.point.z);

            GameManager.simulation.ImportObject(path, fileName, currentBoundSphere.transform,
                GameManager.simulation.sm.simulationInfo.Simulation_ENV, "agent_import");
            Destroy(currentBoundSphere);
        }
    }


    //Content3 : Phase3, Phase4
    //Phase 3 : Try select object for init setting
    void TrySelectObject(RaycastHit hit)
    {
        GameObject MapObject = null;
        GameObject AgentObject = null;
        Collider[] colliders = Physics.OverlapSphere(hit.point, 40f);

        foreach (Collider col in colliders)
        {
            var parents = col.transform.GetComponentsInParent<Transform>();
            foreach (var p in parents)
            {
                if (p.tag.Contains("Map")) { MapObject = p.gameObject; break; }
                if (p.tag.Contains("Agent")) { AgentObject = p.gameObject; break; }
            }
        }

        if (AgentObject != null)
        {
            GameManager.simulation.currentObeject = AgentObject;
        }
        else if (MapObject != null)
        {
            GameManager.simulation.currentObeject = MapObject;
        }

        if (GameManager.simulation.currentObeject != null)
        {
            if (GameManager.simulation.currentObeject.tag == "Map")
            {
                GameManager.simulation.sm.simulationInfo.editorInform.env_profile.Env_Profile.SetActive(true);
                GameManager.simulation.sm.simulationInfo.editorInform.agent_profile.Agent_Profile.SetActive(false);
                profile.MapProfileInputFieldApply(GameManager.simulation.currentObeject);
            }
            else if (GameManager.simulation.currentObeject.tag.Contains("Agent"))
            {

                SelectAgent = true;
                GameManager.simulation.sm.simulationInfo.editorInform.env_profile.Env_Profile.SetActive(false);
                GameManager.simulation.sm.simulationInfo.editorInform.agent_profile.Agent_Profile.SetActive(true);
            }
        }
    }

    

    // after init setting, object follow the mouse and if user click the left button on mouse object falling to planes and this function done
    void HandleAgentMovement(RaycastHit hit)
    {
        GameManager.simulation.currentObeject.transform.position = new Vector3(hit.point.x, 8f, hit.point.z);
        profile.AgentProfileInputFielApply(GameManager.simulation.currentObeject);

        if (Input.GetMouseButtonDown(0))
        {
            GameManager.simulation.currentObeject.transform.position = new Vector3(hit.point.x, 0f, hit.point.z);
            profile.AgentProfileInputFielApply(GameManager.simulation.currentObeject);
            //GameManager.simulation.currentObeject = null;
            SelectAgent = false;

        }
    }

    
}
