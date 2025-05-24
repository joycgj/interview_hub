“**荷兰国旗问题（Dutch National Flag Problem）**” 是一个经典的三路划分问题（three-way partitioning），它在算法中经常用于处理数组中的**分类、分区**操作，比如将数组划分为小于、等于、大于某个值的三段。

---

## 🧠 荷兰国旗问题简要描述：

> 给定一个数组和一个“基准值”（pivot），将数组原地重排，使得：
>
> * 所有小于 pivot 的元素在左边
> * 所有等于 pivot 的元素在中间
> * 所有大于 pivot 的元素在右边

---

## ✅ 与“荷兰国旗问题”相关的 LeetCode 题目：

以下是常考或与三路划分紧密相关的题目：

| 题号                                                                    | 标题                    | 涉及操作               |
| --------------------------------------------------------------------- | --------------------- | ------------------ |
| [75](https://leetcode.cn/problems/sort-colors/)                       | **颜色分类（Sort Colors）** | 典型的荷兰国旗问题          |
| [215](https://leetcode.cn/problems/kth-largest-element-in-an-array/)  | 数组中的第 K 个最大元素         | QuickSelect 使用三路划分 |
| [280](https://leetcode.cn/problems/wiggle-sort/)                      | 摆动排序 I                | 排列顺序需要两边波动         |
| [324](https://leetcode.cn/problems/wiggle-sort-ii/)                   | 摆动排序 II               | 需要三路划分处理中位数        |
| [905](https://leetcode.cn/problems/sort-array-by-parity/)             | 按奇偶排序数组               | 两路划分：偶数前面，奇数后面     |
| [922](https://leetcode.cn/problems/sort-array-by-parity-ii/)          | 按奇偶排序数组 II            | 按索引位置和值奇偶性划分       |
| [2404](https://leetcode.cn/problems/most-frequent-even-element/)      | 出现频率最高的偶数元素           | 虽非荷兰问题，但可用类似分区思想   |
| [148](https://leetcode.cn/problems/sort-list/)                        | 排序链表                  | 高阶：链表排序也可类比荷兰划分    |
| [786](https://leetcode.cn/problems/k-th-smallest-prime-fraction/)     | 第 K 小的质数分数            | 使用类 QuickSelect 技术 |
| [167](https://leetcode.cn/problems/two-sum-ii-input-array-is-sorted/) | 两数之和 II（输入有序）         | 可类比双指针划分策略         |

---

## ✅ 推荐刷题顺序（重点在数组划分技巧）：

```
→ 75. Sort Colors （核心原题）
→ 905. 按奇偶排序数组
→ 922. 按奇偶排序数组 II
→ 215. 第 K 大元素（使用快速选择）
→ 324. Wiggle Sort II（高级三路划分）
→ 280. Wiggle Sort I
→ 786. 第 K 小质数分数
→ 167. Two Sum II（双指针双向划分）
→ 148. Sort List（链表排序扩展）
```

---

## ✅ 推荐拓展：

* ✳️ 想深入荷兰国旗原型：查阅 **E. Dijkstra** 提出的 [Dutch National Flag Algorithm](https://en.wikipedia.org/wiki/Dutch_national_flag_problem)
* ✳️ 荷兰国旗在**快排三路优化**中非常常见，尤其适用于重复元素较多的情况
* ✳️ 常用于分治策略中“改良的快速排序 / 选择”

---

是否需要我：

* 📄 为你整理三路划分专题 PDF/笔记？
* 🔍 补充代码模板（如三路快排 / QuickSelect）？
* 🧠 制作思维图帮你理解荷兰国旗算法结构？

欢迎继续提问，我可立即生成！
