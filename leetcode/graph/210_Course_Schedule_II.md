# Graph Traversal - DFS, BFS, Topological Sorting

Topological Sorting

题目列表
| 题目      | 备注 |
| ----------- | ----------- |
|210. Course Schedule II 课程表 II 中等|        |
|207. Course Schedule 课程表 中等| 和 210. Course Schedule II 相似，只是返回 true 或者 false|
|269. Alien Dictionary   |  127. Word Ladder 和 433. Minimum Genetic Mutation 相似       |
|433. Minimum Genetic Mutation 最小基因变化 中等  |         |
|490. The Maze 迷宫 中等  |         |

## 210. Course Schedule II 课程表 II 中等

现在你总共有 numCourses 门课需要选，记为 0 到 numCourses - 1。给你一个数组 prerequisites ，其中 prerequisites[i] = [a<sub>i</sub>, <sub>bi</sub>] ，表示在选修课程 a<sub>i</sub> 前 必须 先选修 b<sub>i</sub> 。

例如，想要学习课程 0 ，你需要先完成课程 1 ，我们用一个匹配来表示：[0,1] 。
返回你为了学完所有课程所安排的学习顺序。可能会有多个正确的顺序，你只要返回 **任意一种** 就可以了。如果不可能完成所有课程，返回 **一个空数组** 。

 
示例 1：

> 输入：numCourses = 2, prerequisites = [[1,0]]

> 输出：[0,1]

> 解释：总共有 2 门课程。要学习课程 1，你需要先完成课程 0。因此，正确的课程顺序为 [0,1] 。

示例 2：

> 输入：numCourses = 4, prerequisites = [[1,0],[2,0],[3,1],[3,2]]
>
> 输出：[0,2,1,3]
>
> 解释：总共有 4 门课程。要学习课程 3，你应该先完成课程 1 和课程 2。并且课程 1 和课程 2 都应该排在课程 0 之后。
因此，一个正确的课程顺序是 [0,1,2,3] 。另一个正确的排序是 [0,2,1,3] 。

示例 3：

> 输入：numCourses = 1, prerequisites = []
>
> 输出：[0]
 

提示：
- 1 <= numCourses <= 2000
- 0 <= prerequisites.length <= numCourses * (numCourses - 1)
- prerequisites[i].length == 2
- 0 <= a<sub>i</sub>, b<sub>i</sub> < numCourses
- a<sub>i</sub> != b<sub>i</sub>
- 所有[a<sub>i</sub>, b<sub>i</sub>] 互不相同


这个视频把原理讲解的挺好 [LeetCode 每日一题 Daily Challenge 210 Course Schedule II](https://www.bilibili.com/video/BV1qt4y1X7oC/?share_source=copy_web&vd_source=59203eaa2a5b43acef991f52c90c9743)

[拓扑排序只是针对特定的一类图，那么是针对哪类图的呢？答：Directed acyclic graph (DAG)，有向无环图。即：](https://zhuanlan.zhihu.com/p/135094687)
- 这个图的边必须是有方向的；
- 图内无环。

拓扑排序（Kahn算法/BFS）
​​思路​​：

- 构建入度表和邻接表
- 将入度为0的节点加入队列
- 依次处理队列中的节点，减少相邻节点的入度
- 若所有节点都被处理，则返回拓扑序列

[左程云 邻接表+数组实现的队列 【算法讲解059【必备】建图、链式前向星、拓扑排序】](https://www.bilibili.com/video/BV1rj411k7tS/?share_source=copy_web&vd_source=59203eaa2a5b43acef991f52c90c9743)
[代码地址](https://github.com/algorithmzuo/algorithm-journey/blob/main/src/class059/Code02_TopoSortDynamicLeetcode.java)

```
class Solution {
    public int[] findOrder(int numCourses, int[][] prerequisites) {
        List<List<Integer>> graph = new ArrayList<>();
        // 0 ~ n - 1
        for (int i = 0; i < numCourses; i++) {
            graph.add(new ArrayList<>());
        }         

        // 入度表
        int[] indegree = new int[numCourses];
        for (int[] edge : prerequisites) {
            graph.get(edge[1]).add(edge[0]);
            indegree[edge[0]]++; 
        }  

        int[] q = new int[numCourses];
        int l = 0, r = 0;

        for (int i = 0; i < numCourses; i++) {
            if (indegree[i] == 0) {
                q[r++] = i;
            }
        } 

        int cnt = 0;
        while (l < r) {
            int curr = q[l++];
            cnt++;
            for (int next : graph.get(curr)) {
                if (--indegree[next] == 0) {
                    q[r++] = next;
                }
            }
        }

        return cnt == numCourses ? q : new int[0];   
    }
}
```

```
public class Solution {
    public int[] findOrder(int numCourses, int[][] prerequisites) {
        // 初始化邻接表和入度表
        // 我的理解：因为课程编号是从 0 到 numCourses - 1，可以用 list，当然 map 也可以
        List<List<Integer>> adj = new ArrayList<>();
        int[] inDegree = new int[numCourses];
        for (int i = 0; i < numCourses; i++) {
            adj.add(new ArrayList<>());
        }

        // 构建图
        for (int[] edge : prerequisites) {
            adj.get(edge[1]).add(edge[0]); // edge[1] → edge[0]
            inDegree[edge[0]]++;
        }

        // BFS队列初始化 入度为0的是入口，没有依赖
        Queue<Integer> q = new LinkedList<>();
        for (int i = 0; i < numCourses; i++) {
            if (inDegree[i] == 0)
                q.offer(i);
        }

        int[] res = new int[numCourses];
        int index = 0;
        // 拓扑排序
        while (!q.isEmpty()) {
            int u = q.poll();
            res[index++] = u;

            for (int v : adj.get(u)) {
                if (--inDegree[v] == 0) {
                    q.offer(v);
                }
            }
        }

        return index == numCourses ? res : new int[0];
    }
}
```

```
class Solution {
    public int[] findOrder(int numCourses, int[][] prerequisites) {
        int[] res = new int[numCourses];
        Map<Integer, List<Integer>> neighbors = new HashMap<>();
        int[] indegree = new int[numCourses];

        for (int[] prerequisite : prerequisites) {
            neighbors.computeIfAbsent(prerequisite[1], k -> new ArrayList<>()).add(prerequisite[0]);
            indegree[prerequisite[0]]++;
        }

        Queue<Integer> queue = new LinkedList<>();
        for (int i = 0; i < numCourses; i++) {
            if (indegree[i] == 0) {
                queue.offer(i);
            }
        }

        int count = 0;
        while (!queue.isEmpty()) {
            int size = queue.size();
            for (int i = 0; i < size; i++) {
                int cur = queue.poll();
                res[count++] = cur;
                for (int nei : neighbors.getOrDefault(cur, new ArrayList<>())) {
                    if (--indegree[nei] == 0) {
                        queue.offer(nei);
                    }
                }
            }
        }
        return count == numCourses ? res : new int[0];
    }
}
```

复杂度分析​​：

- 时间复杂度​​：O(V + E)，V是课程数，E是依赖关系数
- 空间复杂度​​：O(V + E)，存储邻接表和入度表
