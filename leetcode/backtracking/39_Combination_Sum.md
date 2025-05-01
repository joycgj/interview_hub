# Recursion / Backtracking

题目列表
<ol>
<li>78. Subsets 子集 中等 元素无重不可复选</li>
<li>77. Combinations 组合 中等 元素无重不可复选</li>
<li>46. Permutations 全排列 中等 元素无重不可复选</li>
<li>39. Combination Sum</li>
<li>39. Combination Sum</li>
<li>39. Combination Sum</li>
<li>39. Combination Sum</li>
<li>39. Combination Sum</li>
</ol>

## 78. Subsets 子集 中等 元素无重不可复选

给你一个整数数组 nums ，数组中的元素 互不相同 。返回该数组所有可能的子集（幂集）。

解集 不能 包含重复的子集。你可以按 任意顺序 返回解集。

示例 1：

> 输入：nums = [1,2,3]
> 
> 输出：[[],[1],[2],[1,2],[3],[1,3],[2,3],[1,2,3]]
> 

示例 2：

> 输入：nums = [0]
> 
> 输出：[[],[0]]
 
提示：

> - 1 <= nums.length <= 10
> 
> - -10 <= nums[i] <= 10
> 
> - nums 中的所有元素 互不相同

```
// labuladong p272
// 使用 start 参数控制树枝的生长避免产生重复的子集，用 track 记录根节点到每个节点的路径的值，
// 同时在前序位置把每个节点的路径值收集起来。
class Solution {
    List<List<Integer>> res = new LinkedList<>();
    // 记录回溯算法的递归路径
    List<Integer> track = new LinkedList<>(); 

    public List<List<Integer>> subsets(int[] nums) {
        backtrack(nums, 0);
        return res;
    }

    // 回溯算法核心函数，遍历子集问题的回溯树
    void backtrack(int[] nums, int start) {
        // 前序位置，每个节点的值都是一个子集
        res.add(new LinkedList<>(track));

        for (int i = start; i < nums.length; i++) {
            // 做选择
            track.addLast(nums[i]);
            // 通过start参数控制树枝的遍历，避免产生重复的子集
            backtrack(nums, i + 1);
            // 撤销选择
            track.removeLast();
        }
    }
}
```

## 77. Combinations 组合 中等 元素无重不可复选

给定两个整数 n 和 k，返回范围 [1, n] 中所有可能的 k 个数的组合。

你可以按 任何顺序 返回答案。

示例 1：

> 输入：n = 4, k = 2
> 
> 输出：
> 
> [[2,4], [3,4], [2,3], [1,2], [1,3], [1,4]]

示例 2：

> 输入：n = 1, k = 1
> 
> 输出：[[1]]
 

提示：

> - 1 <= n <= 20
> 
> - 1 <= k <= n

注：这是标准的组合问题，但是翻译一下就变成了 **78. Subsets 子集** 的子集问题了：

> 给你输入一个数组 nums = [1, 2,..., n]和一个正整数 k，请你生成所有大小为 k 的子集。
>
> 以 nums = [1, 2, 3] 为例，**78. Subsets 子集** 是求所有子集，就是把所有节点的值都收集起来，现在你只需要把第2层（根节点视为第0层）的节点收集起来，就是大小为2的所有组合。因此只需要修改 base case，控制算法仅仅收集第 k 层节点的值即可。

```
// labuladong p274
// 和 78. Subsets 78. 子集 一样
class Solution {
    List<List<Integer>> res = new LinkedList<>();
    // 记录回溯算法的递归路径
    List<Integer> track = new LinkedList<>(); 

    public List<List<Integer>> combine(int n, int k) {
        backtrack(n, k, 1);
        return res;
    }

    void backtrack(int n, int k, int start) {
        // base case
        if (track.size() == k) {
            // 遍历到了第k层，收集当前节点的值
            res.add(new LinkedList<>(track));
            return;
        }

        // 回溯算法标准框架
        for (int i = start; i <= n; i++) {
            // 选择
            track.addLast(i);
            // 通过start参数控制树枝的遍历，避免产生重复的子集
            backtrack(n, k, i + 1);
            // 撤销选择
            track.removeLast();
        }
    }
}
```

## 46. Permutations 全排列 中等 元素无重不可复选

给定一个不含重复数字的数组 nums ，返回其 所有可能的全排列 。你可以 **按任意顺序** 返回答案。

示例 1：

> 输入：nums = [1,2,3]
> 
> 输出：[[1,2,3],[1,3,2],[2,1,3],[2,3,1],[3,1,2],[3,2,1]]
>

示例 2：

> 输入：nums = [0,1]
> 
> 输出：[[0,1],[1,0]]
> 

示例 3：

> 输入：nums = [1]
> 
> 输出：[[1]]
 
提示：

> - 1 <= nums.length <= 6
> 
> - -10 <= nums[i] <= 10
> 
> - nums 中的所有整数 **互不相同**

注：
> **78. Subsets 子集 中等 元素无重不可复选 77. Combinations 组合 中等 元素无重不可复选** 的组合、子集问题使用 start 变量保持元素 nums[start] 之后只会出现 nums[start + 1..] 中的元素，通过固定元素的相对位置保证不出现重复的子集。
>
> 但是排列问题本身就是让你穷举元素的位置，nums[i] 之后也可以出现 nums[i] 左边的元素，所以之前的那一套玩不转了，需要额外使用 used 数组来标记哪些元素还可以被选择。

```
// labuladong p276
class Solution {
    List<List<Integer>> res = new LinkedList<>();
    // 记录回溯算法的递归路径
    List<Integer> track = new LinkedList<>();
    // track中的元素会被标记为true
    boolean[] used;

    public List<List<Integer>> permute(int[] nums) {
        used = new boolean[nums.length];
        backtrack(nums);
        return res;
    }

    // 回溯算法核心函数
    void backtrack(int[] nums) {
        // base case，到达叶子结点
        if (track.size() == nums.length) {
            // 收集叶子节点上的值
            res.add(new LinkedList<>(track));
            return;
        }

        // 回溯算法标准框架
        for (int i = 0; i < nums.length; i++) {
            if (used[i]) {
                continue;
            }

            // 做选择
            track.add(nums[i]);
            used[i] = true;
            // 进入下一层回溯树
            backtrack(nums);
            // 取消选择
            track.removeLast();
            used[i] = false;
        }
    }
}
```

延伸：
> 如果题目不让你算全排列，而是让你算元素个数为 k 的排列，怎么算？
>
> 很简单，改下 backtrack 函数的 base case，仅收集第 k 层的节点值即可。

```
class Solution {
    List<List<Integer>> res = new LinkedList<>();
    // 记录回溯算法的递归路径
    List<Integer> track = new LinkedList<>();
    // track中的元素会被标记为true
    boolean[] used;

    public List<List<Integer>> permute(int[] nums, int k) {
        used = new boolean[nums.length];
        backtrack(nums, k);
        return res;
    }

    // 回溯算法核心函数
    void backtrack(int[] nums, int k) {
        // base case，到达第 k 层，收集节点的值
        if (track.size() == k) {
            // 收集叶子节点上的值
            res.add(new LinkedList<>(track));
            return;
        }

        // 回溯算法标准框架
        for (int i = 0; i < nums.length; i++) {
            if (used[i]) {
                continue;
            }

            // 做选择
            track.add(nums[i]);
            used[i] = true;
            // 进入下一层回溯树
            backtrack(nums, k);
            // 取消选择
            track.removeLast();
            used[i] = false;
        }
    }
}
```

## 39. Combination Sum

给你一个 无重复元素 的整数数组 candidates 和一个目标整数 target ，找出 candidates 中可以使数字和为目标数 target 的 所有 不同组合 ，并以列表形式返回。你可以按 任意顺序 返回这些组合。

candidates 中的 同一个 数字可以 无限制重复被选取 。如果至少一个数字的被选数量不同，则两种组合是不同的。 

对于给定的输入，保证和为 target 的不同组合数少于 150 个。

 
示例 1：

> 输入：candidates = [2,3,6,7], target = 7
> 
> 输出：[[2,2,3],[7]]
> 
> 解释：
> 
> 2 和 3 可以形成一组候选，2 + 2 + 3 = 7 。注意 2 可以使用多次。
> 
> 7 也是一个候选， 7 = 7 。
> 
> 仅有这两种组合。

示例 2：

> 输入: candidates = [2,3,5], target = 8
> 
> 输出: [[2,2,2,2],[2,3,3],[3,5]]

示例 3：

> 输入: candidates = [2], target = 1
> 
> 输出: []
 
提示：
<ul>
<li>1 <= candidates.length <= 30</li>
<li>2 <= candidates[i] <= 40</li>
<li>candidates 的所有元素 互不相同</li>
<li>1 <= target <= 40</li>
</ul>

```
// 39. 组合总和 39. Combination Sum
// labuladong p287
// 时间复杂度：最坏情况为 O(2^N)，其中 N 是候选数组长度（例如候选全为1时需遍历所有组合）。
// 空间复杂度：O(T)，取决于递归栈深度（T 为目标值，例如候选含1时递归深度为 T）。
class Solution {
    List<List<Integer>> res = new LinkedList<>();
    // 记录回溯算法的递归路径
    List<Integer> track = new LinkedList<>();
    // 记录 track 中的元素之和
    int trackSum = 0;

    public List<List<Integer>> combinationSum(int[] candidates, int target) {
        if (candidates.length == 0) {
            return res;
        }

        backtrack(candidates, 0, target);
        return res;
    }

    // 回溯算法主函数
    void backtrack(int[] nums, int start, int target) {
        // base case 找到目标和，记录结果
        if (trackSum == target) {
            res.add(new LinkedList<>(track));
            return;
        }

        // base case 超过目标和，停止向下遍历
        if (trackSum > target) {
            return;
        }

        // 回溯算法标准框架
        for (int i = start; i < candidates.length; i++) {
            // 选择 nums[i]
            trackSum += nums[i];
            track.add(nums[i]);
            // 递归遍历下一层回溯树
            // 同一元素可重复使用，注意参数 start 始终为 i
            backtrack(nums, i, target);
            // 撤销选择 nums[i]
            trackSum -= nums[i];
            track.removeLast();
        }
    }
}
```

## 90. Subsets II 90. 子集 II 中等

给你一个整数数组 nums ，其中可能包含重复元素，请你返回该数组所有可能的 子集（幂集）。

解集 不能 包含重复的子集。返回的解集中，子集可以按 任意顺序 排列。

示例 1：

> 输入：nums = [1,2,2]
> 
> 输出：[[],[1],[1,2],[1,2,2],[2],[2,2]]

示例 2：

> 输入：nums = [0]
> 
> 输出：[[],[0]]
 
提示：

> - 1 <= nums.length <= 10
> 
> - -10 <= nums[i] <= 10

注意：
> 做本题之前一定要先做 78.子集。
> 
> 这道题目和 78.子集 区别就是集合里有重复元素了，而且求取的子集要去重。
> 
> 关于回溯算法中的去重问题，40.组合总和II 和本题是一个套路。

```
// labuladong p278
// 时间复杂度：近似为 O(2^n)，其中 n 为候选数组长度。排序为 O(nlogn)，回溯树的遍历最多 2^n 但由于剪枝，实际会小于 2^n。
// 空间复杂度：O(n)（递归栈 + 临时路径）
class Solution {
    List<List<Integer>> res = new LinkedList<>();
    // 记录回溯算法的递归路径
    List<Integer> track = new LinkedList<>();

    public List<List<Integer>> subsetsWithDup(int[] nums) {
        // 先排序，让相同的元素靠在一起
        Arrays.sort(nums);
        backtrack(nums, 0);
        return res;
    }

    void backtrack(int[] nums, int start) {
        // 前序位置，每个节点的值都是一个子集
        res.add(new LinkedList<>(track));

        for (int i = start; i < nums.length; i++) {
            // 剪枝逻辑，值相同的相邻树枝，只遍历第一条
            if (i > start && nums[i] == nums[i - 1]) {
                continue;
            }
            track.addLast(nums[i]);
            backtrack(nums, i + 1);
            track.removeLast();
        }
    }
}
```
90. Subsets II 的 mermaid 回溯图。
![这是图片](https://github.com/joycgj/interview_hub/blob/main/pictures/90_subset.png "Magic Gardens")
