using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.UI;

public class MiniMapSwithcingFunction : MonoBehaviour
{
    public Camera miniMapCamera;
    public RawImage minimapImage;
    public RawImage RobotImage;

    public void Start()
    {
        GameManager.s_map.StartMiniMap -= setMiniMap;
        GameManager.s_map.StartMiniMap += setMiniMap;

        GameManager.s_map.SwithchingMap -= swtihcingMiniMap;
        GameManager.s_map.SwithchingMap += swtihcingMiniMap;

        GameManager.s_map.Capturing -= CaptureImage;
        GameManager.s_map.Capturing += CaptureImage;
    }

    public void setMiniMap(GameObject Map)
    {
        if (Map == null)
        {
            Debug.LogWarning("MiniMap: Map GameObject is null!");
            return;
        }

        // 1. 맵 전체 영역 계산 (자식 오브젝트들 포함)
        Bounds bounds = new Bounds(Map.transform.position, Vector3.zero);
        Renderer[] renderers = Map.GetComponentsInChildren<Renderer>();
        if (renderers.Length == 0)
        {
            Debug.LogWarning("MiniMap: No renderers found in the map.");
            return;
        }
        foreach (Renderer rend in renderers)
        {
            bounds.Encapsulate(rend.bounds);
        }

        // 2. miniMapCamera가 위에서 적절한 높이에 위치하도록 설정 (오직 XY plane에 매핑)
        Vector3 center = bounds.center;
        float sizeX = bounds.size.x;
        float sizeZ = bounds.size.z;

        // 카메라를 항상 맵 가운데 위에 두고, 위에서 바라보게 한다
        miniMapCamera.transform.position = new Vector3(center.x, 100f, center.z); // 기본값, 아래서 조정
        miniMapCamera.transform.rotation = Quaternion.Euler(90, 0, 0);

        // Camera mode는 Orthographic으로! (미니맵 목적)
        miniMapCamera.orthographic = true;

        // orthographicSize 계산: X,Z중 큰쪽에 맞추기 (orthographicSize는 height/2)
        float mapMaxSize = Mathf.Max(sizeX, sizeZ);

        // 화면에 다 보이려면 약간 패딩을 더해줍니다 (예: 10%)
        float padding = 1.1f;
        miniMapCamera.orthographicSize = mapMaxSize * 0.5f * padding;

        // 카메라의 타겟(출력)을 minimapImage로 보냄
        if (miniMapCamera.targetTexture != null)
        {
            minimapImage.texture = miniMapCamera.targetTexture;
        }
        else
        {
            Debug.LogWarning("MiniMap: miniMapCamera의 targetTexture가 세팅되지 않았습니다!");
        }
    }

    // robotview(=RobotImage.texture), minimapview(=minimapImage.texture) 최초 상태를 기억하여 정확한 스왑 구현
    private Texture _originalMiniMapTexture = null;
    private Texture _originalRobotTexture = null;
    private bool _isSwapped = false;

    // !_isSwapped 일 때 실행할 Coroutine
    private Coroutine _swapMovingCoroutine = null;


    public void swtihcingMiniMap()
    {
        // 1. Unity UI 요소 확인
        if (minimapImage == null)
        {
            Debug.LogError("MiniMapSwithcingFunction: minimapImage is null!");
            return;
        }
        if (RobotImage == null)
        {
            Debug.LogError("MiniMapSwithcingFunction: RobotImage is null!");
            return;
        }

        // 2. 최초 한 번, 오리지널 텍스처 기억 (초기상태로 돌아가기 위함)
        if (_originalMiniMapTexture == null && minimapImage.texture != null)
            _originalMiniMapTexture = minimapImage.texture;
        if (_originalRobotTexture == null && RobotImage.texture != null)
            _originalRobotTexture = RobotImage.texture;

        if (_originalMiniMapTexture == null || _originalRobotTexture == null)
        {
            Debug.LogError("MiniMapSwithcingFunction: initial textures are not set!");
            return;
        }

        // 3. 스왑 실행/복구
        if (!_isSwapped)
        {
            _swapMovingCoroutine = StartCoroutine(SwapMoveCoroutine());

            // RobotImage <-> MiniMapImage 텍스처 교환
            minimapImage.texture = _originalRobotTexture;
            RobotImage.texture = _originalMiniMapTexture;
            _isSwapped = true;
        }
        else
        {
            if (_swapMovingCoroutine != null)
            {
                StopCoroutine(_swapMovingCoroutine);
            }
            // 원상 복구
            minimapImage.texture = _originalMiniMapTexture;
            RobotImage.texture = _originalRobotTexture;
            _isSwapped = false;
        }
    }

    private IEnumerator SwapMoveCoroutine()
    {
        Debug.Log("Swap Coroutine Started!");

        while (true)
        {
            while (!Input.GetMouseButtonDown(0))
            {
                yield return null;
            }

            Vector2 mousePos = Input.mousePosition;

            RectTransform rectTransform = RobotImage.rectTransform;

            // 1. RawImage 내 클릭 여부 확인
            if (!RectTransformUtility.RectangleContainsScreenPoint(rectTransform, mousePos, null))
            {
                yield return null;
                continue;
            }

            // 2. localCursor (0 ~ 1 정규화 UV좌표로 변환)
            Vector2 localPoint;
            if (!RectTransformUtility.ScreenPointToLocalPointInRectangle(rectTransform, mousePos, null, out localPoint))
            {
                yield return null;
                continue;
            }

            Rect rect = rectTransform.rect;
            float u = Mathf.Clamp01((localPoint.x - rect.x) / rect.width);
            float v = Mathf.Clamp01((localPoint.y - rect.y) / rect.height);

            // 3. UV 좌표를 미니맵 카메라의 ViewportPoint로 변환 (X: u, Y: v)
            Vector3 viewportPoint = new Vector3(u, v, 0);
            Ray ray = miniMapCamera.ViewportPointToRay(viewportPoint);

            Debug.DrawRay(ray.origin, ray.direction * 100f, Color.red, 2f);  // Debug 확인용

            // 4. Raycast로 실제 맵에 충돌 확인
            if (Physics.Raycast(ray, out RaycastHit hit, 1000f))
            {
                Vector3 target = hit.point;
                Vector3 robotPos = GameManager.s_map.CopyAgent.transform.position;

                GameManager.s_map.PassOnInformEvent(target, robotPos);
                Debug.Log($"[MiniMap] 클릭(hit): {target}, 로봇 위치: {robotPos}");
            }
            else
            {
                Debug.LogWarning("[MiniMap] 미니맵 클릭 Ray가 바닥과 충돌하지 않음.");
            }

            yield return null;
        }
    }

    void CaptureImage()
    {
        Debug.Log("Capture");

        if (RobotImage != null && RobotImage.texture != null)
        {
            RenderTexture renderTex = RobotImage.texture as RenderTexture;
            if (renderTex != null)
            {
                // 1. RenderTexture에서 Texture2D로 복사
                RenderTexture currentRT = RenderTexture.active;
                RenderTexture.active = renderTex;

                Texture2D tex = new Texture2D(renderTex.width, renderTex.height, TextureFormat.RGB24, false);
                tex.ReadPixels(new Rect(0, 0, renderTex.width, renderTex.height), 0, 0);
                tex.Apply();

                RenderTexture.active = currentRT;

                // 2. 저장
                byte[] pngData = tex.EncodeToPNG();
                string fileName = $"RobotImage_{System.DateTime.Now:yyyyMMdd_HHmmss}.png"; // 시간 포함
                string filePath = System.IO.Path.Combine(
                    "D:\\Code\\MainWork\\MainWork\\Reinforcement\\MainWork\\Assets\\01.SLAM\\Resources",
                    fileName
                );

                System.IO.File.WriteAllBytes(filePath, pngData);
                Debug.Log($"RobotImage saved to: {filePath}");
            }
            else
            {
                Debug.LogWarning("RobotImage.texture is not a RenderTexture!");
            }
        }
        else
        {
            Debug.LogWarning("RobotImage or its texture is null!");
        }
    }

}
