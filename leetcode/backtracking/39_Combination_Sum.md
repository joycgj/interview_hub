- [Recursion / Backtracking](#recursion--backtracking)
  - [78. Subsets 子集 中等 元素无重不可复选](#78-subsets-子集-中等-元素无重不可复选)
  - [77. Combinations 组合 中等 元素无重不可复选](#77-combinations-组合-中等-元素无重不可复选)
  - [46. Permutations 全排列 中等 元素无重不可复选](#46-permutations-全排列-中等-元素无重不可复选)
  - [90. Subsets II 子集 II 中等 元素可重不可复选](#90-subsets-ii-子集-ii-中等-元素可重不可复选)
  - [40. Combination Sum II 组合总和 II 中等 元素可重不可复选](#40-combination-sum-ii-组合总和-ii-中等-元素可重不可复选)
  - [47. Permutations II 全排列 II 中等 元素可重不可复选](#47-permutations-ii-全排列-ii-中等-元素可重不可复选)
  - [39. Combination Sum 组合总和 中等 元素可重不可复选](#39-combination-sum-组合总和-中等-元素可重不可复选)
  - [491. Non-decreasing Subsequences 非递减子序列 中等 元素可重不可排序](#491-non-decreasing-subsequences-非递减子序列-中等-元素可重不可排序)

# Recursion / Backtracking

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
            // 第 k 层节点的值就是大小为 k 的排列
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

## 90. Subsets II 子集 II 中等 元素可重不可复选

给你一个整数数组 nums ，其中可能包含重复元素，请你返回该数组所有可能的 **子集**（幂集）。数组的 **子集** 是从数组中选择一些元素（可能为空）。

解集 **不能** 包含重复的子集。返回的解集中，子集可以按 **任意顺序** 排列。

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

注：
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
![90. Subsets II](../../pictures/90_Subsets_II.png "")

## 40. Combination Sum II 组合总和 II 中等 元素可重不可复选

给定一个候选人编号的集合 candidates 和一个目标数 target ，找出 candidates 中所有可以使数字和为 target 的组合。

candidates 中的每个数字在每个组合中只能使用 一次 。

**注意**：解集不能包含重复的组合。 

示例 1:

> 输入: candidates = [10,1,2,7,6,1,5], target = 8,
> 
> 输出:
> 
> [[1,1,6],[1,2,5],[1,7],[2,6]]

示例 2:

> 输入: candidates = [2,5,2,1,2], target = 5,
> 
> 输出:
> [[1,2,2],[5]] 

提示:

> - 1 <= candidates.length <= 100
> 
> - 1 <= candidates[i] <= 50
> 
> - 1 <= target <= 30

注：
> 虽然这是一个组合问题，其实换个问法就变成子集问题了，请你计算 candidates 中所有和为 target 的子集。
>
> 对比子集的问题，只要额外用一个 trackSum 变量记录回溯路径上的元素和，然后将 base case 改一改即可解决这道题。

```
// labuladong p280
class Solution {
    List<List<Integer>> res = new LinkedList<>();
    // 记录回溯算法的递归路径
    List<Integer> track = new LinkedList<>();
    // 记录track中的元素之和
    int trackSum = 0;

    public List<List<Integer>> combinationSum2(int[] candidates, int target) {
        if (candidates.length == 0) {
            return res;
        }

        // 先排序，让相同的元素靠在一起
        Arrays.sort(candidates);
        backtrack(candidates, target, 0);
        return res;
    }

    // 回溯算法主函数
    void backtrack(int[] nums, int target, int start) {
        // base case 达到目标和，找到符合条件的组合
        if (trackSum == target) {
            res.add(new LinkedList<>(track));
            return;
        }

        // base case 超过目标和，直接结束
        if (trackSum > target) {
            return;
        }
        
        // 回溯算法标准框架
        for (int i = start; i < nums.length; i++) {
            // 剪枝逻辑，值相同的树枝，只遍历第一条
            if (i > start && nums[i] == nums[i - 1]) {
                continue;
            }

            // 做选择
            track.add(nums[i]);
            trackSum += nums[i];
            // 递归遍历下一层回溯树
            backtrack(nums, target, i + 1);
            // 取消选择
            track.removeLast();
            trackSum -= nums[i];
        }
    }
}
```
## 47. Permutations II 全排列 II 中等 元素可重不可复选

给定一个可包含重复数字的序列 nums ，**按任意顺序** 返回所有不重复的全排列。

示例 1：

> 输入：nums = [1,1,2]
> 
> 输出：[[1,1,2], [1,2,1], [2,1,1]]

示例 2：

>
> 输入：nums = [1,2,3]
>
> 输出：[[1,2,3],[1,3,2],[2,1,3],[2,3,1],[3,1,2],[3,2,1]]
 
提示：

> - 1 <= nums.length <= 8
> 
> - -10 <= nums[i] <= 10

```
// labuladong p281 
// 关键：保证相同元素在排列中的相对位置保持不变 labuladong p282
class Solution {
    List<List<Integer>> res = new LinkedList<>();
    // 记录回溯算法的递归路径
    List<Integer> track = new LinkedList<>();
    // track中的元素会被标记为true
    boolean[] used;

    public List<List<Integer>> permuteUnique(int[] nums) {
        // 先排序，让相同的元素靠在一起
        Arrays.sort(nums);
        used = new boolean[nums.length];
        backtrack(nums);
        return res;
    }

    void backtrack(int[] nums) {
        if (track.size() == nums.length) {
            res.add(new LinkedList<>(track));
            return;
        }

        for (int i = 0; i < nums.length; i++) {
            if (used[i]) {
                continue;
            }

            // 新添加的剪枝逻辑，固定相同的元素在排列中的相对位置
            if (i > 0 && nums[i] == nums[i - 1] && !used[i - 1]) {
                // 如果前面的相邻相等元素没有用过，则跳过
                continue;
            }

            track.addLast(nums[i]);
            used[i] = true;
            backtrack(nums);
            track.removeLast();
            used[i] = false;
        }
    }
}
```

## 39. Combination Sum 组合总和 中等 元素可重不可复选

给你一个 **无重复元素** 的整数数组 candidates 和一个目标整数 target ，找出 candidates 中可以使数字和为目标数 target 的 所有 不同组合 ，并以列表形式返回。你可以按 任意顺序 返回这些组合。

candidates 中的 **同一个** 数字可以 **无限制重复被选取** 。如果至少一个数字的被选数量不同，则两种组合是不同的。 

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

> - 1 <= candidates.length <= 30
> 
> - 2 <= candidates[i] <= 40
> 
> - candidates 的所有元素 **互不相同**
> 
> - 1 <= target <= 40

```
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

        backtrack(candidates, target, 0);
        return res;
    }

    // 回溯算法主函数
    void backtrack(int[] nums, int target, int start) {
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
        for (int i = start; i < nums.length; i++) {
            // 选择 nums[i]
            trackSum += nums[i];
            track.add(nums[i]);
            // 递归遍历下一层回溯树
            // 同一元素可重复使用，注意参数 start 始终为 i
            backtrack(nums, target, i);
            // 撤销选择 nums[i]
            trackSum -= nums[i];
            track.removeLast();
        }
    }
}
```

## 491. Non-decreasing Subsequences 非递减子序列 中等 元素可重不可排序

给你一个整数数组 nums ，找出并返回所有该数组中不同的递增子序列，递增子序列中 **至少有两个元素** 。你可以按 **任意顺序** 返回答案。

数组中可能含有重复元素，如出现两个整数相等，也可以视作递增序列的一种特殊情况。

示例 1：

> 输入：nums = [4,6,7,7]
> 
> 输出：[[4,6],[4,6,7],[4,6,7,7],[4,7],[4,7,7],[6,7],[6,7,7],[7,7]]

示例 2：

> 输入：nums = [4,4,3,2,1]
> 
> 输出：[[4,4]]
 
提示：

> - 1 <= nums.length <= 15
> 
> - -100 <= nums[i] <= 100

```
// 代码随想录p284
class Solution {
    List<List<Integer>> res = new ArrayList<>();
    List<Integer> track = new LinkedList<>();

    public List<List<Integer>> findSubsequences(int[] nums) {
        backtrack(nums, 0);  
        return res;  
    }

    void backtrack(int[] nums, int start) {
        List<Integer> track1 = track;
        if (track.size() >= 2) {
            res.add(new ArrayList<>(track));
        }

        // 使用 set 对本层元素进行排序
        Set<Integer> used = new HashSet<>();
        for (int i = start; i < nums.length; i++) {
            if (used.contains(nums[i])) {
                continue;
            }

            if (!track.isEmpty() && nums[i] < track.getLast()) {
                continue;
            }

            // 记录这个元素在本层用过了，本层后面不能再用了
            used.add(nums[i]);
            track.add(nums[i]);
            backtrack(nums, i + 1);
            track.removeLast();
        }
    }
}
```
注：

> 90. Subsets II 需要排序，491. Non-decreasing Subsequences 不能排序
>
> 输入：nums = [4,7,6,7,8]
>
> 输出：[[4,7],[4,7,7],[4,7,7,8],[4,7,8],[4,6],[4,6,7],[4,6,7,8],[4,6,8],[4,8],[7,7],[7,7,8],[7,8],[6,7],[6,7,8],[6,8]]
>
> 如果不去重7，则会出现两次[7,8]

491. Non-decreasing Subsequences 的回溯图。
![491. Non-decreasing Subsequences](../../pictures/491_Non_Decreasing_Subsequences.png "")
