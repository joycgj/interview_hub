# Graph Traversal - DFS, BFS, Topological Sorting

题目列表
<ol>
<li>133. Clone Graph 克隆图 中等</li>
<li></li>
</ol>

## 133. Clone Graph 克隆图 中等

给你无向 **连通** 图中一个节点的引用，请你返回该图的 **深拷贝**（克隆）。

图中的每个节点都包含它的值 val（int） 和其邻居的列表（list[Node]）。

```
class Node {
    public int val;
    public List<Node> neighbors;
}
```

测试用例格式：

简单起见，每个节点的值都和它的索引相同。例如，第一个节点值为 1（val = 1），第二个节点值为 2（val = 2），以此类推。该图在测试用例中使用邻接列表表示。

**邻接列表** 是用于表示有限图的无序列表的集合。每个列表都描述了图中节点的邻居集。

给定节点将始终是图中的第一个节点（值为 1）。你必须将 给定节点的拷贝 作为对克隆图的引用返回。

![](../../pictures/133_clone_graph_question.png "") 


示例 1：

> 输入：adjList = [[2,4],[1,3],[2,4],[1,3]]
>
> 输出：[[2,4],[1,3],[2,4],[1,3]]
>
> 解释：
>
> 图中有 4 个节点。
>
> 节点 1 的值是 1，它有两个邻居：节点 2 和 4 。
>
> 节点 2 的值是 2，它有两个邻居：节点 1 和 3 。
>
> 节点 3 的值是 3，它有两个邻居：节点 2 和 4 。
>
> 节点 4 的值是 4，它有两个邻居：节点 1 和 3 。

示例 2：

> 输入：adjList = [[]]
>
> 输出：[[]]
>
> 解释：输入包含一个空列表。该图仅仅只有一个值为 1 的节点，它没有任何邻居。

示例 3：

> 输入：adjList = []
> 
> 输出：[]
>
>解释：这个图是空的，它不含任何节点。
 
提示：

> - 这张图中的节点数在 [0, 100] 之间。
> 
> - 1 <= Node.val <= 100
>
> - 每个节点值 Node.val 都是唯一的，
>
> - 图中没有重复的边，也没有自环。
> 
> - 图是连通图，你可以从给定节点访问到所有节点。


### DFS（深度优先搜索）
思路：
1. 使用哈希表存储原节点和克隆节点的映射。
2. 递归克隆每个节点及其邻居。

时间复杂度：O(N)，其中 N 是图中的节点数。

空间复杂度：O(N)，哈希表存储所有节点的克隆。

```
// DFS（深度优先搜索）
// 思路
// 1. 使用哈希表存储原节点和克隆节点的映射。
// 2. 递归克隆每个节点及其邻居。
class Solution {
    private Map<Node, Node> visited = new HashMap<>();

    public Node cloneGraph(Node node) {
        if (node == null) {
            return null;
        }    

        if (visited.containsKey(node)) {
            return visited.get(node);
        }

        Node clone = new Node(node.val);
        visited.put(node, clone);
        for (Node nei : node.neighbors) {
            clone.neighbors.add(cloneGraph(nei));
        }

        return clone;
    }
}
```

### BFS（广度优先搜索）
思路：
1. 使用队列进行层次遍历。
2. 哈希表存储原节点和克隆节点的映射。

时间复杂度：O(N)。

空间复杂度：O(N)，哈希表和队列的空间。

```
class Solution {
    public Node cloneGraph(Node node) {
        if (node == null) {
            return null;
        }

        Map<Node, Node> visited = new HashMap<>();
        Queue<Node> q = new LinkedList<>();
        q.offer(node);
        // 克隆第一个节点并存储到哈希表中
        visited.put(node, new Node(node.val));

        while (!q.isEmpty()) {
            Node curr = q.poll();
            for (Node neighbor : curr.neighbors) {
                if (!visited.containsKey(neighbor)) {
                    // 如果没有被访问过，就克隆并存储在哈希表中
                    visited.put(neighbor, new Node(neighbor.val));
                    // 将邻居节点加入队列中
                    q.offer(neighbor);
                }
                // 更新当前节点的邻居列表
                visited.get(curr).neighbors.add(visited.get(neighbor));
            }
        }
        return visited.get(node);
    }
}
```