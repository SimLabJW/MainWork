using UnityEngine;
using UnityEngine.UI;

public class FileListUI : MonoBehaviour
{
    [Header("UI 참조")]
    public GameObject rowPrefab; // RowPrefab 프리팹
    public Transform contentParent; // Scroll View 안의 Content 오브젝트

    private void Start()
    {
        AddRow(1, "File1", "This is the first file");
        AddRow(2, "File2", "This is the second file");
        AddRow(3, "File3", "This is the third file");
        AddRow(4, "File1", "This is the first file");
        AddRow(5, "File2", "This is the second file");
        AddRow(6, "File3", "This is the third file");
        AddRow(7, "File1", "This is the first file");
        AddRow(8, "File2", "This is the second file");
        AddRow(9, "File3", "This is the third file");
    }

    // 리스트 생성 함수
    public void AddRow(int index, string name, string description)
    {
        // Row 프리팹 생성
        GameObject newRow = Instantiate(rowPrefab, contentParent);

        // 버튼(행) 아래에 존재하는 텍스트 오브젝트들 찾기
        Text indexText = newRow.transform.Find("Index_T").GetComponent<Text>();
        Text nameText = newRow.transform.Find("Name_T").GetComponent<Text>();
        Text descriptionText = newRow.transform.Find("Description_T").GetComponent<Text>();

        // 텍스트 설정
        indexText.text = index.ToString();
        nameText.text = name;
        descriptionText.text = description;
    }
}
