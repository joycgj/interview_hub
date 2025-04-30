# Recursion / Backtracking

题目列表
<ol>
<li>39. Combination Sum</li>
<li>39. Combination Sum</li>
<li>39. Combination Sum</li>
<li>39. Combination Sum</li>
<li>39. Combination Sum</li>
<li>39. Combination Sum</li>
<li>39. Combination Sum</li>
<li>39. Combination Sum</li>
</ol>


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
# 90. Subsets II
给定一个候选人编号的集合 candidates 和一个目标数 target ，找出 candidates 中所有可以使数字和为 target 的组合。

candidates 中的每个数字在每个组合中只能使用 一次 。

注意：解集不能包含重复的组合。 

示例 1:

> 输入: candidates = [10,1,2,7,6,1,5], target = 8,
> 
> 输出:
> [
> [1,1,6],
> [1,2,5],
> [1,7],
> [2,6]
> ]

示例 2:

> 输入: candidates = [2,5,2,1,2], target = 5,
> 
> 输出:
> [
> [1,2,2],
> [5]
> ]
 
提示:

1 <= candidates.length <= 100
1 <= candidates[i] <= 50
1 <= target <= 30
 
![这是图片](https://github.com/joycgj/interview_hub/blob/main/pictures/90_subset.png "Magic Gardens")
