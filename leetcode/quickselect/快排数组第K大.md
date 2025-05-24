- [215.  Kth Largest Element in an Array 数组中的第K个最大元素 中等](#215--kth-largest-element-in-an-array-数组中的第k个最大元素-中等)

## 215.  Kth Largest Element in an Array 数组中的第K个最大元素 中等

给定整数数组 nums 和整数 k，请返回数组中第 k 个最大的元素。

请注意，你需要找的是数组排序后的第 k 个最大的元素，而不是第 k 个不同的元素。

你必须设计并实现时间复杂度为 O(n) 的算法解决此问题。 

示例 1:

> 输入: [3,2,1,5,6,4], k = 2
> 
> 输出: 5
>

示例 2:

> 输入: [3,2,3,1,2,4,5,5,6], k = 4
> 
> 输出: 4
 
提示：

- 1 <= k <= nums.length <= 10<sup>5</sup>
- -10<sup>4</sup> <= nums[i] <= 10<sup>4</sup>

```
// labuladong 快速排序 p235
class Solution {
    public int findKthLargest(int[] nums, int k) {
        int l = 0, h = nums.length - 1;
        k = nums.length - k;
        while (l <= h) {
            int p = partition(nums, l, h);
            if (p < k) {
                l = p + 1;
            } else if (p > k) {
                h = p - 1;
            } else {
                return nums[p];
            }
        }
        return -1;
    }

    // 对nums[l..h]进行切分
    int partition(int[] nums, int l, int h) {
        int pivot = nums[l];
        // 关于区间的边界控制应格外小心，稍有不慎就会出错
        // 这里把i,j定义为开区间，同时定义
        // [l, i) <= pivot; (j, h] > pivot
        // 之后都要正确维护这个边界区间的定义
        int i = l + 1, j = h;
        // 当i>j时结束循环，以保证区间[l..h]都被覆盖
        while (i <= j) {
            while (i < h && nums[i] <= pivot) {
                i++; // 此while结束时恰好nums[i]>pivot
            }
            while (j > l && nums[j] > pivot) {
                j--; // 此while结束时恰好nums[j]<=pivot
            }
            if (i >= j) {
                break;
            }
            // 此时[l, i) <= pivot && (j, h] > pivot
            // 交换nums[j]和nums[i]
            swap(nums, i, j);
        }
        // 最后将pivot放到合适的位置，即pivot左边元素较小，右边元素较大
        swap(nums, l, j);
        return j;
    }   

    void swap(int[] nums, int i, int j) {
        int tmp = nums[i];
        nums[i] = nums[j];
        nums[j] = tmp;
    }
}
```

- 平均时间复杂度：O(n) 
- 最坏时间复杂度：O(n²)（极端情况）
- 空间复杂度：O(1)（原地划分）

说明：理论上最快，面试时非常高频，但需注意最坏情况