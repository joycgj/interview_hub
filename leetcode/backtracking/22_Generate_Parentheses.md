
# Recursion / Backtracking

题目列表
<ol>
<li>22. Generate Parentheses 括号生成 中等</li>
<li>77. Combinations 组合 中等 元素无重不可复选</li>
</ol>

## 22. Generate Parentheses 括号生成 中等

数字 n 代表生成括号的对数，请你设计一个函数，用于能够生成所有可能的并且 有效的 括号组合。

示例 1：

> 输入：n = 3
>
> 输出：["((()))","(()())","(())()","()(())","()()()"]

示例 2：

> 输入：n = 1
>
> 输出：["()"]
 

提示：

> - 1 <= n <= 8

```
class Solution {
    public List<String> generateParenthesis(int n) {
        List<String> res = new ArrayList<>();
        backtrack(res, new StringBuilder(), 0, 0, n);
        return res;
    }

    void backtrack(List<String> res, StringBuilder track, int open, int close, int n) {
        // 触发结束条件：当前字符串长度达到 2n
        if (track.length() == n * 2) {
            res.add(track.toString());
            return;
        }

        // open < n 排除不合法的选择
        if (open < n) {
            track.append("("); // 做选择
            backtrack(res, track, open + 1, close, n);  // 进入下一层决策树
            track.deleteCharAt(track.length() - 1); // 取消选择
        }
        // close < open 排除不合法的选择
        if (close < open) {
            track.append(")"); // 做选择
            backtrack(res, track, open, close + 1, n);  // 进入下一层决策树
            track.deleteCharAt(track.length() - 1); // 取消选择
        } 
    }
}
```


22. Generate Parentheses 的 mermaid 回溯图。
![22. Generate Parentheses](../../pictures/22_Generate_Parentheses.png "")