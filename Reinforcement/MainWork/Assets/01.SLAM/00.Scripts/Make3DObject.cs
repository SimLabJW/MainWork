using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class Make3DObject : MonoBehaviour
{
    int[,] grid;
    float cellSize;
    Vector2 origin;
    float inflateRadius = 1.9f; // 장애물 회피 안전 마진

    void Start()
    {
        GameManager.s_map.PassOnInform -= PickPosition;
        GameManager.s_map.PassOnInform += PickPosition;

    }

    void PickPosition(Vector3 selectPosition, Vector3 robotPosition)
    {
        List<Vector3> pathList = AstarAlgorithm(selectPosition, robotPosition);

        GameManager.s_agent.MoveUpdateCopyEvent(pathList);
    }

    List<Vector3> AstarAlgorithm(Vector3 selectPosition, Vector3 robotPosition)
    {
        grid = GameManager.s_map.GridData;
        origin = GameManager.s_map.Origin;
        cellSize = GameManager.s_map.CellSize;

        Vector2Int start = WorldToGrid(robotPosition);
        Vector2Int goal = WorldToGrid(selectPosition);

        Debug.Log($"[A*] Start Grid: {start}, Goal Grid: {goal}");

        int[,] inflatedMap = InflateObstacles(grid, inflateRadius);
        List<Vector2Int> path = AStar(start, goal, inflatedMap);

        List<Vector3> worldPath = new List<Vector3>();
        foreach (Vector2Int node in path)
            worldPath.Add(GridToWorld(node));

        return worldPath;
    }

    int[,] InflateObstacles(int[,] original, float radius)
    {
        int h = original.GetLength(0);
        int w = original.GetLength(1);
        int[,] inflated = (int[,])original.Clone();

        int inflateCells = Mathf.CeilToInt(radius / cellSize);

        for (int y = 0; y < h; y++)
        {
            for (int x = 0; x < w; x++)
            {
                if (original[y, x] != 0)
                {
                    for (int dy = -inflateCells; dy <= inflateCells; dy++)
                    {
                        for (int dx = -inflateCells; dx <= inflateCells; dx++)
                        {
                            int ny = y + dy;
                            int nx = x + dx;

                            if (nx >= 0 && nx < w && ny >= 0 && ny < h)
                            {
                                inflated[ny, nx] = 1;
                            }
                        }
                    }
                }
            }
        }

        return inflated;
    }

    List<Vector2Int> AStar(Vector2Int start, Vector2Int goal, int[,] map)
    {
        HashSet<Vector2Int> closed = new HashSet<Vector2Int>();
        Dictionary<Vector2Int, Vector2Int> cameFrom = new Dictionary<Vector2Int, Vector2Int>();
        Dictionary<Vector2Int, float> costSoFar = new Dictionary<Vector2Int, float>();

        PriorityQueue<Vector2Int> openSet = new PriorityQueue<Vector2Int>();
        openSet.Enqueue(start, 0);
        costSoFar[start] = 0;

        Vector2Int[] dirs = {
            Vector2Int.up, Vector2Int.down, Vector2Int.left, Vector2Int.right,
            new Vector2Int(1,1), new Vector2Int(-1,1), new Vector2Int(1,-1), new Vector2Int(-1,-1)
        };

        while (openSet.Count > 0)
        {
            Vector2Int current = openSet.Dequeue();

            if (current == goal) break;

            foreach (Vector2Int dir in dirs)
            {
                Vector2Int next = current + dir;
                if (!IsValid(next, map)) continue;

                float newCost = costSoFar[current] + (dir.x != 0 && dir.y != 0 ? 1.414f : 1f);
                if (!costSoFar.ContainsKey(next) || newCost < costSoFar[next])
                {
                    costSoFar[next] = newCost;
                    float priority = newCost + Heuristic(next, goal);
                    openSet.Enqueue(next, priority);
                    cameFrom[next] = current;
                }
            }
        }

        List<Vector2Int> path = new List<Vector2Int>();
        Vector2Int node = goal;
        if (!cameFrom.ContainsKey(goal))
        {
            Debug.LogWarning("[A*] 목표 지점까지 경로를 찾을 수 없습니다.");
            path.Add(start);
            return path;
        }

        while (cameFrom.ContainsKey(node))
        {
            path.Add(node);
            node = cameFrom[node];
        }
        path.Add(start);
        path.Reverse();
        return path;
    }

    float Heuristic(Vector2Int a, Vector2Int b)
    {
        int dx = Mathf.Abs(a.x - b.x);
        int dy = Mathf.Abs(a.y - b.y);
        return dx + dy + (1.414f - 2) * Mathf.Min(dx, dy);
    }

    bool IsValid(Vector2Int pos, int[,] map)
    {
        int h = map.GetLength(0);
        int w = map.GetLength(1);
        bool inside = pos.y >= 0 && pos.y < h && pos.x >= 0 && pos.x < w;
        // if (!inside) Debug.LogWarning($"[IsValid] Out of bounds: {pos}, size=({w},{h})");
        // else if (map[pos.y, pos.x] != 0) Debug.LogWarning($"[IsValid] Blocked: {pos}, value={map[pos.y, pos.x]}");
        return inside && map[pos.y, pos.x] == 0;
    }

    Vector2Int WorldToGrid(Vector3 world)
    {
        Transform mapRoot = GameManager.s_map.CopyMap.transform;
        Vector3 local = mapRoot.InverseTransformPoint(world);

        int col = Mathf.FloorToInt(local.x / cellSize);
        int row = Mathf.FloorToInt(local.z / cellSize);

        Debug.Log($"[WorldToGrid] world: {world}, local: {local}, grid: ({col}, {row})");
        return new Vector2Int(col, row);
    }

    Vector3 GridToWorld(Vector2Int grid)
    {
        Transform mapRoot = GameManager.s_map.CopyMap.transform;

        float localX = (grid.x + 0.5f) * cellSize;
        float localZ = (grid.y + 0.5f) * cellSize;
        Vector3 localPos = new Vector3(localX, 0f, localZ);

        Vector3 worldPos = mapRoot.TransformPoint(localPos);
        return worldPos;
    }

    
}

public class PriorityQueue<T>
{
    private List<(T item, float priority)> elements = new List<(T, float)>();

    public int Count => elements.Count;

    public void Enqueue(T item, float priority)
    {
        elements.Add((item, priority));
    }

    public T Dequeue()
    {
        int bestIndex = 0;
        for (int i = 1; i < elements.Count; i++)
        {
            if (elements[i].priority < elements[bestIndex].priority)
                bestIndex = i;
        }
        T bestItem = elements[bestIndex].item;
        elements.RemoveAt(bestIndex);
        return bestItem;
    }
}
