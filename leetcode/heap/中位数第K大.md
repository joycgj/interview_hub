295. Find Median from Data Stream 数据流的中位数 困难

**中位数** 是有序整数列表中的中间值。如果列表的大小是偶数，则没有中间值，中位数是两个中间值的平均值。

- 例如 arr = [2,3,4] 的中位数是 3 。
- 例如 arr = [2,3] 的中位数是 (2 + 3) / 2 = 2.5 。

实现 MedianFinder 类:

- MedianFinder() 初始化 MedianFinder 对象。

- void addNum(int num) 将数据流中的整数 num 添加到数据结构中。

- double findMedian() 返回到目前为止所有元素的中位数。与实际答案相差 10-5 以内的答案将被接受。

示例 1：

> 输入
> 
> ["MedianFinder", "addNum", "addNum", "findMedian", "addNum", "findMedian"]
[[], [1], [2], [], [3], []]
>
> 输出
[null, null, null, 1.5, null, 2.0]

解释

MedianFinder medianFinder = new MedianFinder();

medianFinder.addNum(1);    // arr = [1]

medianFinder.addNum(2);    // arr = [1, 2]
medianFinder.findMedian(); // 返回 1.5 ((1 + 2) / 2)
medianFinder.addNum(3);    // arr[1, 2, 3]
medianFinder.findMedian(); // return 2.0

提示:

- -10<sup>5</sup> <= num <= 10<sup>5</sup>
- 在调用 findMedian 之前，数据结构中至少有一个元素
- 最多 5 * 10<sup>4</sup> 次调用 addNum 和 findMedian

解法：双堆法（最优解 ✅）

**思路：**

- 用 **两个堆** 来维护数据流的中位数：
	- 最大堆 left：存储较小一半，堆顶是最大值
	- 最小堆 right：存储较大一半，堆顶是最小值
	- 保持：left.size() == right.size() 或 left.size() == right.size() + 1

```
class MedianFinder {
    // 最大堆（存左半边）
    PriorityQueue<Integer> left;
    // 最小堆（存右半边）
    PriorityQueue<Integer> right;

    public MedianFinder() {
        left = new PriorityQueue<>((a, b) -> b - a);   
        right = new PriorityQueue<>((a, b) -> a - b);     
    }
    
    public void addNum(int num) {
        // 先加入最大堆
        left.offer(num);
        // 平衡两个堆（大数移到右边）
        right.offer(left.poll());

        // 保持左边数量不少于右边（中位数在左/中间）
        if (left.size() < right.size()) {
            left.offer(right.poll());
        }
    }
    
    public double findMedian() {
        if (left.size() > right.size()) {
            return left.peek(); // 奇数个，最大堆堆顶为中位数
        } 
        // 偶数个
        return (left.peek() + right.peek()) / 2.0; 
    }
}
```