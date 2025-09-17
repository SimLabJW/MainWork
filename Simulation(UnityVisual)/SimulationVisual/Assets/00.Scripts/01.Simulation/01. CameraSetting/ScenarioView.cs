using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.UI;
using TMPro;

public class ScenarioView : MonoBehaviour
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
    private GameObject waypointBoundSphere;
    private Coroutine editorRoutine;
    private Coroutine scenarioRoutine;

    private bool SelectAgent = false;
    private bool isWaypointMode = false;

    // Editor View component
    private RawImage scenario_raw;
    private RectTransform scenario_rectransform;

    void Start()
    {
        GameManager.scenarioEdit.ScenarioViewFitAction -= MapSizeFitToScenarioView;
        GameManager.scenarioEdit.ScenarioViewFitAction += MapSizeFitToScenarioView;

        GameManager.scenarioEdit.ScenarioAgentButtonAction -= ScenarioObjectSizeCalculate;
        GameManager.scenarioEdit.ScenarioAgentButtonAction += ScenarioObjectSizeCalculate;
    }

    // Phase 1(Env Button)
    private void MapSizeFitToScenarioView(GameObject SimulationEnvObject)
    {
        if (editorRoutine != null)
        {
            StopCoroutine(editorRoutine);
            editorRoutine = null;
        }

        scenario_raw = GameManager.scenarioEdit.scinfo.cameraType.Editor_ScenarioView.GetComponent<RawImage>();
        scenario_rectransform = scenario_raw.rectTransform;

        List<Transform> transforms = new List<Transform>();

        foreach (Transform t in SimulationEnvObject.GetComponentsInChildren<Transform>())
        {
            if (t.CompareTag("Plane") || t.CompareTag("Water"))
            {
                transforms.Add(t);
            }
        }
        FitCameraToTargets(transforms);

        if (scenarioRoutine != null)
        {
            StopCoroutine(scenarioRoutine);
        }
        scenarioRoutine = StartCoroutine(ScenarioViewMove());
        
    }
    // Phase 2(Agent Button)
    private void ScenarioObjectSizeCalculate(string fileId, string fileName, string fileDesc, Transform Position, string table)
    {
        Debug.Log($"scenarioObjectSizeCaluclate filedesc : {fileDesc}");
        if (scenarioRoutine != null)
        {
            StopCoroutine(scenarioRoutine);
            scenarioRoutine = null;
        }

        // maxFigure를 계산한다 & 해당 최대 크기로 된 구체를 생성
        StartCoroutine(DelayedImporterSize(fileId, fileName, fileDesc, Position, table)); 

        if (editorRoutine != null)
        {
            StopCoroutine(editorRoutine);
        }
        editorRoutine = StartCoroutine(EditorMoveandCreate(fileId, fileName, fileDesc,  Position, table));
    }

    IEnumerator StartImporterSize(string fileId, string fileName, Transform Position)
    {
        if (GameManager.scenarioEdit.scinfo.scenarioEditInfo.Agent_Size != null)
        {
            currentBoundSphere = Instantiate(GameManager.scenarioEdit.scinfo.scenarioEditInfo.Agent_Size);
            currentBoundSphere.transform.localScale = new Vector3(GameManager.createScenario.maxFigure * 2f, 0.5f, GameManager.createScenario.maxFigure * 2f);
        }

        yield return new WaitForSeconds(1f);
    }
    IEnumerator DelayedImporterSize(string fileId, string fileName, string fileDesc, Transform Position, string table)
    {
        GameManager.scenarioEdit.ImportScenarioAgentSize(fileId, fileName, fileDesc, GameManager.scenarioEdit.ScenarioObject.transform,
                    GameManager.scenarioEdit.ScenarioObject.transform, table);
        yield return new WaitForSeconds(2f);

        StartCoroutine(StartImporterSize(fileId, fileName, Position));
    }

    IEnumerator ScenarioViewMove()
    {
        while (true)
        {
            HandleCameraControl();
            if (RectTransformUtility.RectangleContainsScreenPoint(scenario_rectransform, Input.mousePosition) &&
                TryGetMouseHit(out RaycastHit hit))
            {
                if (currentBoundSphere == null)
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

    IEnumerator EditorMoveandCreate(string fileId, string fileName, string fileDesc,  Transform Position, string table)
    {
        while (true)
        {
            HandleCameraControl();
            if (RectTransformUtility.RectangleContainsScreenPoint(scenario_rectransform, Input.mousePosition) &&
                TryGetMouseHit(out RaycastHit hit))
            {
                if (currentBoundSphere != null)
                {
                    currentBoundSphere.transform.position = new Vector3(hit.point.x, 10f, hit.point.z);
                    if (Input.GetMouseButtonDown(0))
                    {
                        currentBoundSphere.transform.position = new Vector3(hit.point.x, 0f, hit.point.z);

                        GameManager.scenarioEdit.Editor_AGENT = true;
                        GameManager.scenarioEdit.ImportScenarioAgentAction?.Invoke(fileId, fileName, fileDesc, currentBoundSphere.transform,
                            GameManager.scenarioEdit.ScenarioObject.transform, table);

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
                scenario_rectransform, Input.mousePosition, null, out localPoint))
        {
            Vector2 normalizedPoint = new Vector2(
                (localPoint.x + scenario_rectransform.rect.width / 2) / scenario_rectransform.rect.width,
                (localPoint.y + scenario_rectransform.rect.height / 2) / scenario_rectransform.rect.height);

            Ray ray = GameManager.scenarioEdit.scinfo.cameraType.ScenarioView_Editor.ViewportPointToRay(normalizedPoint);
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

        if (GameManager.scenarioEdit.scinfo.cameraType.ScenarioView_Editor == null || EnvTarget == null || EnvTarget.Count == 0) return;

        GameManager.scenarioEdit.scinfo.cameraType.ScenarioView_Editor.orthographic = true;

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
        float aspect = (float)GameManager.scenarioEdit.scinfo.cameraType.ScenarioView_Editor.pixelWidth 
            / GameManager.scenarioEdit.scinfo.cameraType.ScenarioView_Editor.pixelHeight;

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

        GameManager.scenarioEdit.scinfo.cameraType.ScenarioView_Editor.orthographicSize 
            = orthoSize;
        maxOrthoSize = orthoSize;
        GameManager.scenarioEdit.scinfo.cameraType.ScenarioView_Editor.transform.position 
            = new Vector3(center.x, center.y + 100f, center.z);
        GameManager.scenarioEdit.scinfo.cameraType.ScenarioView_Editor.transform.rotation
            = Quaternion.Euler(90f, 0f, 0f);
    }

    // Phase 2 - Function ( Zoom, clickto move camera)
    private void zoom_In_Out(float scroll)
    {
        GameManager.scenarioEdit.scinfo.cameraType.ScenarioView_Editor.orthographicSize 
            -= scroll * zoomSpeed;
        GameManager.scenarioEdit.scinfo.cameraType.ScenarioView_Editor.orthographicSize 
            = Mathf.Clamp(GameManager.scenarioEdit.scinfo.cameraType.ScenarioView_Editor.orthographicSize, minOrthoSize, maxOrthoSize);
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
                * GameManager.scenarioEdit.scinfo.cameraType.ScenarioView_Editor.orthographicSize;
            Vector3 newPos 
                = GameManager.scenarioEdit.scinfo.cameraType.ScenarioView_Editor.transform.position + move;

            float halfHeight = GameManager.scenarioEdit.scinfo.cameraType.ScenarioView_Editor.orthographicSize;
            float halfWidth = halfHeight * GameManager.scenarioEdit.scinfo.cameraType.ScenarioView_Editor.aspect;

            float minX = targetBounds.min.x + halfWidth;
            float maxX = targetBounds.max.x - halfWidth;
            float minZ = targetBounds.min.z + halfHeight;
            float maxZ = targetBounds.max.z - halfHeight;

            newPos.x = Mathf.Clamp(newPos.x, minX, maxX);
            newPos.z = Mathf.Clamp(newPos.z, minZ, maxZ);

            GameManager.scenarioEdit.scinfo.cameraType.ScenarioView_Editor.transform.position = newPos;
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
            PrefabInfo.ToggleWaypointsExclusiveForObject(GameManager.createScenario.currentObeject);

            // 상태 색상 반영
            var unityInfo = PrefabInfo.GetImportedObjectUnityInfo(GameManager.createScenario.currentObeject);
            var uiImage = GameManager.scenarioEdit.scinfo.scenarioEditInfo.SelectAgentPlatformImage;
            if (uiImage != null)
            {
                Color stateColor = Color.white;
                if (unityInfo != null)
                {
                    if (unityInfo.state == "Green") stateColor = Color.green;
                    else if (unityInfo.state == "Red") stateColor = Color.red;
                    else stateColor = Color.white;
                }
                uiImage.color = stateColor;
            }

            // 파일명으로 텍스트 반영
            var importedInfo = PrefabInfo.GetImportedObjectInfo(GameManager.createScenario.currentObeject);
            var nameTextComp = GameManager.scenarioEdit.scinfo.scenarioEditInfo.SelectAgentNameText;
            if (nameTextComp != null)
            {
                nameTextComp.text = importedInfo != null && !string.IsNullOrEmpty(importedInfo.fileName)
                    ? importedInfo.fileName
                    : GameManager.createScenario.currentObeject.name;
            }

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
       // 웨이포인트 모드가 아닐 때만 기존 동작 수행
        if (!isWaypointMode)
        {
            GameManager.createScenario.currentObeject.transform.position = new Vector3(hit.point.x, 8f, hit.point.z);

            if (Input.GetMouseButtonDown(0))
            {
                GameManager.createScenario.currentObeject.transform.position = new Vector3(hit.point.x, 0f, hit.point.z);
                SelectAgent = false;
            }

            // 우클릭 시 웨이포인트 모드 시작
            if (Input.GetMouseButtonDown(1))
            {
                // 이때 선택한 오브젝트에 대한 웨이포인트가 있는지 확인 후 동기화
                StartWaypointMode();
            }
        }
        else
        {
            // 웨이포인트 모드일 때는 waypointboundSphere가 마우스를 따라다님
            if (waypointBoundSphere != null)
            {
                waypointBoundSphere.transform.position = new Vector3(hit.point.x, 10f, hit.point.z);

                // 우클릭 시 해당 위치에 waypointboundSphere를 설치(고정)하고, waypointcontent에 waypointobjectbutton을 설치한 후, 새로운 웨이포인트 구체 생성
                if (Input.GetMouseButtonDown(1))
                {
                    // 현재 위치에 웨이포인트 구체를 고정
                    waypointBoundSphere.transform.position = new Vector3(hit.point.x, 10f, hit.point.z);

                    GameManager.scenarioEdit.waypoints.Add(waypointBoundSphere);

                    GameObject newButton = Instantiate(
                        GameManager.scenarioEdit.scinfo.scenarioEditInfo.WaypointObjectButton,
                        GameManager.scenarioEdit.scinfo.scenarioEditInfo.waypointContent.transform
                    );

                    // 버튼과 웨이포인트를 같이 저장
                    CreateWaypointInfo(newButton, GameManager.createScenario.currentObeject, waypointBoundSphere);

                    // 항상 최신 두 개만 유지하고 연결 시도
                    if (GameManager.scenarioEdit.waypoints.Count > 2)
                    {
                        GameManager.scenarioEdit.waypoints.RemoveAt(0);
                    }
                    if (GameManager.scenarioEdit.waypoints.Count == 2)
                    {
                        GameManager.scenarioEdit.ConnectWaypoint(GameManager.scenarioEdit.waypoints);
                    }

                    // 새로운 웨이포인트 구체 생성
                    if (GameManager.scenarioEdit.scinfo.scenarioEditInfo.WaypointViewPrefab != null)
                    {
                        waypointBoundSphere = Instantiate(GameManager.scenarioEdit.scinfo.scenarioEditInfo.WaypointViewPrefab);
                        waypointBoundSphere.transform.localScale = new Vector3(50f, 0.5f, 50f); // 웨이포인트용 크기
                    }
                }
                // 좌클릭 시 웨이포인트 모드 종료
                else if (Input.GetMouseButtonDown(0))
                {
                    ExitWaypointMode();
                }
            }
        }
    }

    void StartWaypointMode()
    {
        isWaypointMode = true;

        if (GameManager.scenarioEdit.scinfo.scenarioEditInfo.WaypointViewPrefab != null)
        {
            waypointBoundSphere = Instantiate(GameManager.scenarioEdit.scinfo.scenarioEditInfo.WaypointViewPrefab);
            waypointBoundSphere.transform.localScale = new Vector3(50f, 0.5f, 50f); // 웨이포인트용 크기
        }
    }

    void ExitWaypointMode()
    {
        // 웨이포인트 모드 종료
        isWaypointMode = false;
        SelectAgent = false;
        
        // 웨이포인트 구체 제거
        if (waypointBoundSphere != null)
        {
            GameManager.scenarioEdit.waypoints  = new List<GameObject>();
            Destroy(waypointBoundSphere);
            waypointBoundSphere = null;
        }
    }

    // 버튼과 웨이포인트를 같이 저장하며, index가 정해진 후 WayPointObject 하위의 IndexText에 기입 (TextMeshPro 지원)
    void CreateWaypointInfo(GameObject newButtonObject, GameObject AgentObject, GameObject WaypointObject)
    {
        var waypointList = PrefabInfo.GetImportedObjectWayPointInfo(AgentObject);

        int newIndex = 0;
        string beforeEndPointSet = "";
        int beforeIndex = -1;
        GameObject beforeButton = null;

        if (waypointList != null && waypointList.Count > 0)
        {
            foreach (var wp in waypointList)
            {
                if (wp.index >= newIndex)
                {
                    newIndex = wp.index + 1;
                }
            }

            foreach (var wp in waypointList)
            {
                if (wp.index == newIndex - 1) // 바로 이전 인덱스
                {
                    beforeEndPointSet = $"{newIndex}";
                    beforeIndex = wp.index;
                    beforeButton = wp.pointButtonObject;
                    break;
                }
            }
        }
        else
        {
            newIndex = 0;
        }

        if (WaypointObject != null)
        {
            Transform indexTextTr = WaypointObject.transform.Find("IndexText");
            
            var tmpText = indexTextTr.GetComponent<TMPro.TextMeshPro>();
            tmpText.text = newIndex.ToString();     
        }
        
        var newWaypointInfo = new PrefabInfo.WaypointInfo(
            newIndex,
            WaypointObject,
            newButtonObject,
            WaypointObject.transform.position.x,
            WaypointObject.transform.position.z,
            "-"
        );

        // 딕셔너리에 추가
        PrefabInfo.AddImportedObjectWayPointsInfo(AgentObject, newWaypointInfo);

        // 버튼 생성
        WayPointButtonInstantiate(newButtonObject, newIndex, "-");

        // 만약 이전 웨이포인트가 있다면, 저장된 버튼 오브젝트를 활용하여 ConnectInput 텍스트를 수정
        if (!string.IsNullOrEmpty(beforeEndPointSet) && beforeIndex != -1 && beforeButton != null)
        {
            UpdatePreviousWaypointButtonConnectInput(beforeButton, beforeEndPointSet);
        }
    }

    void WayPointButtonInstantiate(GameObject newButton, int newIndex, string endpoint)
    {
        Transform indexPanel = newButton.transform.Find("Index_Panel");
        if (indexPanel != null)
        {
            Transform indexT = indexPanel.Find("Index_T");
            if (indexT != null)
            {
                var indexText = indexT.GetComponent<Text>();
                if (indexText != null)
                {
                    indexText.text = newIndex.ToString();
                }
            }
        }

        // connect_panel 하위의 ConnectInput 인풋필드의 text를 endpoint로 설정
        Transform connectPanel = newButton.transform.Find("Connect_Panel");
        if (connectPanel != null)
        {
            Transform connectInput = connectPanel.Find("ConnectInput");
            if (connectInput != null)
            {
                var inputField = connectInput.GetComponent<InputField>();
                if (inputField != null)
                {
                    // 기본값 설정
                    inputField.text = endpoint;

                    // 기존 리스너 제거 후 새 리스너 등록
                    inputField.onEndEdit.RemoveAllListeners();
                    inputField.onEndEdit.AddListener((string value) =>
                    {
                        if (GameManager.scenarioEdit.WaypointReConnectionAction != null)
                        {
                            // AgentObject, newIndex, value(엔드포인트) 전달
                            GameManager.scenarioEdit.WaypointReConnectionAction?.Invoke(GameManager.createScenario.currentObeject, newIndex, value);
                        }
                    });
                }
            }
        }

        // Del 버튼 리스너 등록: 해당 인덱스 기반 삭제 흐름 호출
        Transform delTr = newButton.transform.Find("Del");
        if (delTr != null)
        {
            var delBtn = delTr.GetComponent<Button>();
            if (delBtn != null)
            {
                delBtn.onClick.AddListener(() =>
                {
                    
                    // 현재 선택된 Agent 기준으로 삭제 처리 위임
                    if (GameManager.createScenario.currentObeject != null)
                    {
                        GameManager.scenarioEdit.WaypointDisConnectionAction?.Invoke(GameManager.createScenario.currentObeject, newIndex);
                    }
                });
            }
        }
    }

    // 이전 웨이포인트 버튼의 ConnectInput 텍스트를 저장된 버튼 오브젝트로 직접 수정
    void UpdatePreviousWaypointButtonConnectInput(GameObject beforeButton, string endpoint)
    {
        if (beforeButton == null) return;

        Transform connectPanel = beforeButton.transform.Find("Connect_Panel");
        if (connectPanel != null)
        {
            Transform connectInput = connectPanel.Find("ConnectInput");
            if (connectInput != null)
            {
                var inputField = connectInput.GetComponent<InputField>();
                if (inputField != null)
                {
                    inputField.text = endpoint;
                }
            }
        }
    }

}
