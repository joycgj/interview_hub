## 295. Find Median from Data Stream 数据流的中位数 困难

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
> 
> [[], [1], [2], [], [3], []]
>
> 输出
> 
> [null, null, null, 1.5, null, 2.0]

解释

> MedianFinder medianFinder = new MedianFinder();
>
> medianFinder.addNum(1);    // arr = [1]
>
> medianFinder.addNum(2);    // arr = [1, 2]
> 
> medianFinder.findMedian(); // 返回 1.5 ((1 + 2) / 2)
> 
> medianFinder.addNum(3);    // arr[1, 2, 3]
> 
> medianFinder.findMedian(); // return 2.0

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

- 时间复杂度：
	- addNum()	O(log n)（堆插入/平衡）
	- findMedian()	O(1)

- 空间复杂度：O(n)：堆存储了所有元素

说明：这是最优解，兼顾插入性能与实时中位数查询，面试常考。

## 480. Sliding Window Median 滑动窗口中位数 困难

**中位数** 是有序序列最中间的那个数。如果序列的长度是偶数，则没有最中间的数；此时中位数是最中间的两个数的平均数。

例如：

- [2,3,4]，中位数是 3
- [2,3]，中位数是 (2 + 3) / 2 = 2.5
  
给你一个数组 nums，有一个长度为 k 的窗口从最左端滑动到最右端。窗口中有 k 个数，每次窗口向右移动 1 位。你的任务是找出每次窗口移动后得到的新窗口中元素的中位数，并输出由它们组成的数组。

示例：

> 给出 nums = [1,3,-1,-3,5,3,6,7]，以及 k = 3。
>
> 窗口位置                      中位数
> ---------------               -----
> [1  3  -1] -3  5  3  6  7       1
>  1 [3  -1  -3] 5  3  6  7      -1
>  1  3 [-1  -3  5] 3  6  7      -1
>  1  3  -1 [-3  5  3] 6  7       3
>  1  3  -1  -3 [5  3  6] 7       5
>  1  3  -1  -3  5 [3  6  7]      6
>  因此，返回该滑动窗口的中位数数组 [1,-1,-1,3,5,6]。

提示：

- 你可以假设 k 始终有效，即：k 始终小于等于输入的非空数组的元素个数。
- 与真实值误差在 10 ^ -5 以内的答案将被视作正确答案。

通过不了用例 [-2147483648,-2147483648,2147483647,-2147483648 的，把最大堆改为 new PriorityQueue<>((o1,o2)->Integer.compare(o2,o1))

```
class Solution {
    public double[] medianSlidingWindow(int[] nums, int k) {
        int n = nums.length;
        double[] res = new double[n - k + 1];
        DualHeap dualHeap = new DualHeap(k);
        for (int i = 0; i < n; i++) {
            dualHeap.addNum(nums[i]);

            if (i >= k - 1) {
                res[i - k + 1] = dualHeap.findMedian();
                dualHeap.removeNum(nums[i - k + 1]); // 移出窗口左端元素
            }
        }

        return res;
    }

    private static class DualHeap {
        // 大根堆
        private PriorityQueue<Integer> maxHeap;
        // 小根堆
        private PriorityQueue<Integer> minHeap;
        // 窗口大小
        private int k;
        // 删除表 当窗口往后移动时 某些元素退出窗口 它们要被删除
        // 采用延迟删除策略 先在删除表中记录 只有当该元素出现在大根堆或小根堆堆顶的时候才删除
        private Map<Integer, Integer> toBeDeleted;
        //大根堆实际大小 因为存在延迟删除
        private int maxHeapSize;
        //小根堆实际大小 因为存在延迟删除
        private int minHeapSize;

        public DualHeap(int k) {
            minHeap = new PriorityQueue<>((a, b) -> a.compareTo(b));
            maxHeap = new PriorityQueue<>((a, b) -> b.compareTo(a));
            this.k = k;
            this.toBeDeleted = new HashMap<>();
            this.maxHeapSize = 0;
            this.minHeapSize = 0;
        }

        public double findMedian() {
            // 返回中位数
            return k % 2 == 0 ? ((double) minHeap.peek() + maxHeap.peek()) / 2.0 : maxHeap.peek();
        }

        public void removeNum(int num) {
            toBeDeleted.put(num, toBeDeleted.getOrDefault(num, 0) + 1);
            if (num <= maxHeap.peek()) {
                maxHeapSize--;
                // 单纯删除可能会导致移除该堆顶后堆顶的下一个数也是要延迟删除的 
                // 在makeblance时 将延迟删除的数 误当作 中位数候选数 要永远保持堆顶绝对不能是延迟删除的数
                if (num == maxHeap.peek()) {
                    prune(maxHeap);
                }
            } else {
                minHeapSize--;
                if (num == minHeap.peek()) {
                    prune(minHeap);
                }
            }
            makebalance();
        }

        public void addNum(int num) {
            // 小于等于大根堆堆顶或者大根堆为空(优先放入大根堆)
            if (maxHeap.size() == 0 || num <= maxHeap.peek()) {
                maxHeap.offer(num);
                maxHeapSize++;
            } else {
                minHeap.offer(num);
                minHeapSize++;
            }
            makebalance();
        }

        public void prune(PriorityQueue<Integer> heap) {
            // 堆顶出现了 toBeDeleted 中的值
            while (heap.size() != 0 && toBeDeleted.getOrDefault(heap.peek(), 0) > 0) {
                int top = heap.peek();
                toBeDeleted.put(top, toBeDeleted.getOrDefault(top, 0) - 1);
                if (toBeDeleted.getOrDefault(top, 0) == 0) {
                    toBeDeleted.remove(top);
                }
                heap.poll();
            }
        }

        public void makebalance() {
            if (minHeapSize > maxHeapSize) {
                maxHeap.offer(minHeap.poll());
                maxHeapSize++;
                minHeapSize--;
                prune(minHeap);
            } else if (maxHeapSize > minHeapSize + 1) {
                minHeap.offer(maxHeap.poll());
                minHeapSize++;
                maxHeapSize--;
                prune(maxHeap);
            }
        }
    }
}
```
时间 & 空间复杂度分析：
| 操作                     | 复杂度说明                  |
| ---------------------- | ---------------------- |
| `addNum` / `removeNum` | `O(log k)` 每次操作均摊堆操作   |
| `getMedian()`          | `O(1)`                 |
| 总体时间复杂度                | `O(n log k)` — 最优 ✅    |
| 空间复杂度                  | `O(k)` — 存储窗口内元素 + 延迟表 |

