using UnityEngine;

public class DoubleClickHandler : MonoBehaviour
{
    private float lastClickTime = 0f;
    private const float doubleClickTime = 0.3f; // 더블클릭 판정 시간 (초)

    private void OnMouseDown()
    {
        float currentTime = Time.time;
        
        // 이전 클릭과의 시간 간격을 확인
        if (currentTime - lastClickTime < doubleClickTime)
        {
            // 더블클릭 감지
            Debug.Log("aaaa");
        }
        
        lastClickTime = currentTime;
    }
} 