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

    void Start()
    {
        GameManager.createScenario.EditorViewFitAction -= MapSizeFitToEditorView;
        GameManager.createScenario.EditorViewFitAction += MapSizeFitToEditorView;

        GameManager.createScenario.EditorViewControlAction -= ObjectSizeCalculate;
        GameManager.createScenario.EditorViewControlAction += ObjectSizeCalculate;
    }

    // Phase 1(Env Button)
    private void MapSizeFitToEditorView(GameObject SimulationEnvObject)
    {
        editor_raw = GameManager.createScenario.csminfo.createScenarioInfo.editorInform.Editor_MapView.GetComponent<RawImage>();
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
    private void ObjectSizeCalculate(string fileId, string fileName, Transform Position, string table)
    {
        // maxFigure를 계산한다 & 해당 최대 크기로 된 구체를 생성
        StartCoroutine(DelayedImporterSize(fileId, fileName, Position, table)); 

        // 이동
        //StartCoroutine(EditorMoveandCreate(path, fileName, Position));
        if (editorRoutine != null)
        {
            StopCoroutine(editorRoutine);
        }
        editorRoutine = StartCoroutine(EditorMoveandCreate(fileId, fileName, Position, table));
    }

    IEnumerator StartImporterSize(string fileId, string fileName, Transform Position)
    {
        if (GameManager.createScenario.csminfo.createScenarioInfo.editorInform.Agent_Size != null)
        {
            currentBoundSphere = Instantiate(GameManager.createScenario.csminfo.createScenarioInfo.editorInform.Agent_Size);
            currentBoundSphere.transform.localScale = new Vector3(GameManager.createScenario.maxFigure * 2f, 0.5f, GameManager.createScenario.maxFigure * 2f);
        }

        yield return new WaitForSeconds(1f);
    }
    IEnumerator DelayedImporterSize(string fileId, string fileName, Transform Position, string table)
    {
        GameManager.createScenario.ImportObject(fileId, fileName, GameManager.createScenario.csminfo.createScenarioInfo.Simulation_ENV,
                    GameManager.createScenario.csminfo.createScenarioInfo.Simulation_ENV, table);
        yield return new WaitForSeconds(2f);

        StartCoroutine(StartImporterSize(fileId, fileName, Position));
    }

    IEnumerator EditorMoveandCreate(string fileId, string fileName, Transform Position, string table)
    {
        while (true)
        {
            HandleCameraControl();
            if (RectTransformUtility.RectangleContainsScreenPoint(editor_rectransform, Input.mousePosition) &&
                TryGetMouseHit(out RaycastHit hit))
            {
                if (currentBoundSphere != null)
                {
                    currentBoundSphere.transform.position = new Vector3(hit.point.x, 10f, hit.point.z);
                    if (Input.GetMouseButtonDown(0))
                    {
                        currentBoundSphere.transform.position = new Vector3(hit.point.x, 0f, hit.point.z);

                        GameManager.createScenario.Editor_AGENT = true;
                        GameManager.createScenario.ImportAgentAction?.Invoke(fileId, fileName, currentBoundSphere.transform,
                            GameManager.createScenario.csminfo.createScenarioInfo.Simulation_ENV, table);

                        Destroy(currentBoundSphere);
                        currentBoundSphere = null;
                        // break;
                    }
                }
                else 
                {
                    if (SelectAgent && GameManager.createScenario.currentObeject != null) 
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

            Ray ray = GameManager.createScenario.csminfo.cameraType.MapView_Editor.ViewportPointToRay(normalizedPoint);
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

        if (GameManager.createScenario.csminfo.cameraType.MapView_Editor == null || EnvTarget == null || EnvTarget.Count == 0) return;

        GameManager.createScenario.csminfo.cameraType.MapView_Editor.orthographic = true;

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
        float aspect = (float)GameManager.createScenario.csminfo.cameraType.MapView_Editor.pixelWidth 
            / GameManager.createScenario.csminfo.cameraType.MapView_Editor.pixelHeight;

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

        GameManager.createScenario.csminfo.cameraType.MapView_Editor.orthographicSize 
            = orthoSize;
        maxOrthoSize = orthoSize;
        GameManager.createScenario.csminfo.cameraType.MapView_Editor.transform.position 
            = new Vector3(center.x, center.y + 100f, center.z);
        GameManager.createScenario.csminfo.cameraType.MapView_Editor.transform.rotation
            = Quaternion.Euler(90f, 0f, 0f);
    }

    // Phase 2 - Function ( Zoom, clickto move camera)
    private void zoom_In_Out(float scroll)
    {
        GameManager.createScenario.csminfo.cameraType.MapView_Editor.orthographicSize 
            -= scroll * zoomSpeed;
        GameManager.createScenario.csminfo.cameraType.MapView_Editor.orthographicSize 
            = Mathf.Clamp(GameManager.createScenario.csminfo.cameraType.MapView_Editor.orthographicSize, minOrthoSize, maxOrthoSize);
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
                * GameManager.createScenario.csminfo.cameraType.MapView_Editor.orthographicSize;
            Vector3 newPos 
                = GameManager.createScenario.csminfo.cameraType.MapView_Editor.transform.position + move;

            float halfHeight = GameManager.createScenario.csminfo.cameraType.MapView_Editor.orthographicSize;
            float halfWidth = halfHeight * GameManager.createScenario.csminfo.cameraType.MapView_Editor.aspect;

            float minX = targetBounds.min.x + halfWidth;
            float maxX = targetBounds.max.x - halfWidth;
            float minZ = targetBounds.min.z + halfHeight;
            float maxZ = targetBounds.max.z - halfHeight;

            newPos.x = Mathf.Clamp(newPos.x, minX, maxX);
            newPos.z = Mathf.Clamp(newPos.z, minZ, maxZ);

            GameManager.createScenario.csminfo.cameraType.MapView_Editor.transform.position = newPos;
        }
    }

    //Content3 : Phase3, Phase4
    //Phase 3 : Try select object for init setting
    void TrySelectObject(RaycastHit hit)
    {
        Collider[] colliders = Physics.OverlapSphere(hit.point, 40f);

        List<GameObject> agentObjects = new List<GameObject>();
        float minDistance = float.MaxValue;
        GameObject closestAgent = null;

        for (int i = 0; i < colliders.Length; i++)
        {
            if (colliders[i] != null && colliders[i].gameObject != null)
            {
                var rangeObjectInfo = PrefabInfo.GetImportedObjectInfo(colliders[i].gameObject);

                if (rangeObjectInfo != null)
                {

                    // Agent 테이블을 가진 오브젝트만 리스트에 추가
                    if (rangeObjectInfo.table == "Agent")
                    {
                        agentObjects.Add(colliders[i].gameObject);

                        // hit 포인트와의 거리 계산
                        float dist = Vector3.Distance(hit.point, colliders[i].transform.position);
                        if (dist < minDistance)
                        {
                            minDistance = dist;
                            closestAgent = colliders[i].gameObject;
                        }
                    }
                }
                else
                {
                    Debug.Log($"  - import된 오브젝트가 아님");
                }
            }
        }

        // Agent 테이블을 가진 오브젝트가 하나 이상이면 가장 가까운 오브젝트를 currentObeject로 설정
        if (agentObjects.Count > 0 && closestAgent != null)
        {
            GameManager.createScenario.currentObeject = closestAgent;
            SelectAgent = true;
        }
        else
        {
            Debug.Log("Agent 테이블을 가진 오브젝트가 없거나 currentObeject가 null입니다.");
        }
    }

    // after init setting, object follow the mouse and if user click the left button on mouse object falling to planes and this function done
    void HandleAgentMovement(RaycastHit hit)
    {
        GameManager.createScenario.currentObeject.transform.position = new Vector3(hit.point.x, 8f, hit.point.z);


        if (Input.GetMouseButtonDown(0))
        {
            GameManager.createScenario.currentObeject.transform.position = new Vector3(hit.point.x, 0f, hit.point.z);
            //GameManager.simulation.currentObeject = null;
            SelectAgent = false;

        }
    }

    
}
