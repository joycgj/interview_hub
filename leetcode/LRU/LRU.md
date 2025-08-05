LRU/LFU

# 146. LRU Cache 146. LRU 缓存

请你设计并实现一个满足  **LRU (最近最少使用) 缓存** 约束的数据结构。
实现 LRUCache 类：

- LRUCache(int capacity) 以 正整数 作为容量 capacity 初始化 LRU 缓存
- int get(int key) 如果关键字 key 存在于缓存中，则返回关键字的值，否则返回 -1 。
- void put(int key, int value) 如果关键字 key 已经存在，则变更其数据值 value ；如果不存在，则向缓存中插入该组 key-value 。如果插入操作导致关键字数量超过 capacity ，则应该 **逐出** 最久未使用的关键字。

函数 get 和 put 必须以 O(1) 的平均时间复杂度运行。

示例：

> 输入
> 
> ["LRUCache", "put", "put", "get", "put", "get", "put", "get", "get", "get"]
> 
> [[2], [1, 1], [2, 2], [1], [3, 3], [2], [4, 4], [1], [3], [4]]
> 
> 输出
> 
> [null, null, null, 1, null, -1, null, -1, 3, 4]
> 
> 解释
> 
> LRUCache lRUCache = new LRUCache(2);
> 
> lRUCache.put(1, 1); // 缓存是 {1=1}
> 
> lRUCache.put(2, 2); // 缓存是 {1=1, 2=2}
> 
> lRUCache.get(1);    // 返回 1
> 
> lRUCache.put(3, 3); // 该操作会使得关键字 2 作废，缓存是 {1=1, 3=3}
> 
> lRUCache.get(2);    // 返回 -1 (未找到)
> 
> lRUCache.put(4, 4); // 该操作会使得关键字 1 作废，缓存是 {4=4, 3=3}
> 
> lRUCache.get(1);    // 返回 -1 (未找到)
> 
> lRUCache.get(3);    // 返回 3
> 
> lRUCache.get(4);    // 返回 4
 
提示：

- 1 <= capacity <= 3000
- 0 <= key <= 10000
- 0 <= value <= 10<sup>5</sup>
- 最多调用 2 * 10<sup>5</sup> 次 get 和 put

## 最优解: 哈希表 + 自定义双向链表 

- 时间复杂度:get和put均为O(1)
- 空间复杂度O(n)

核心思想:
- 自定义双向链表:维护访问顺序(头节点为最新，尾节点为最旧)
- 哈希表:存储键到节点的映射，实现 O(1) 访问
 
```
class LRUCache {
    // 自定义双向链表节点
    private static class Node {
        int key, value;
        Node next, prev;
        Node(int key, int value) {
            this.key = key;
            this.value = value;
        }
    }

    // 自定义双向链表
    private static class DoublyLinkedList {
        Node head, tail;
        int size;

        DoublyLinkedList() {
            head = new Node(-1, -1); // 虚拟头结点
            tail = new Node(-1, -1); // 虚拟尾节点
            head.next = tail;
            tail.prev = head;
            size = 0;    
        }

        // 添加节点到链表头部（最新访问）
        void addToHead(Node node) {
            node.next = head.next;
            node.prev = head;
            head.next.prev = node;
            head.next = node;
            size++;
        }   

        // 移除指定节点
        void removeNode(Node node) {
            node.prev.next = node.next;  
            node.next.prev = node.prev;
            size--;  
        }

        // 移除链表尾部节点（最久未使用）
        Node removeTail() {
            if (size == 0) {
                return null;
            }

            Node tailNode = tail.prev;
            removeNode(tailNode);
            return tailNode;
        } 
    }

    private final int capacity;
    private final Map<Integer, Node> keyMap;    // 键到节点的映射
    private final DoublyLinkedList list;        // 维护访问顺序的双向链表

    public LRUCache(int capacity) {
        this.capacity = capacity; 
        keyMap = new HashMap<>();
        list = new DoublyLinkedList();   
    }
    
    public int get(int key) {
        if (!keyMap.containsKey(key)) {
            return -1;
        }

        Node node = keyMap.get(key);
        // 移动到链表头部表示最新访问
        list.removeNode(node);
        list.addToHead(node);
        return node.value;
    }
    
    public void put(int key, int value) {
        if (keyMap.containsKey(key)) {
            // 更新值并移动到链表头部
            Node node = keyMap.get(key);
            node.value = value;
            list.removeNode(node);
            list.addToHead(node);           
        } else {
            if (keyMap.size() == capacity) {
                Node tailNode = list.removeTail();
                keyMap.remove(tailNode.key);
            }
            // 添加新节点
            Node newNode = new Node(key, value);
            keyMap.put(key, newNode);
            list.addToHead(newNode);
        }    
    }
}
```

关键点:

- 1.双向链表: DoublyLinkedList类封装了节点的插入、删除和移除逻辑。
- 2.哈希表: keyMap 实现 O(1)的键值查找。
- 3.虚拟头尾节点:简化链表操作，避免空指针检查。

优点:严格满足0(1)时间复杂度要求，代码结构清晰，


# 460. LFU Cache 460. LFU 缓存

请你为 **最不经常使用（LFU）** 缓存算法设计并实现数据结构。

实现 LFUCache 类：

- LFUCache(int capacity) - 用数据结构的容量 capacity 初始化对象
- int get(int key) - 如果键 key 存在于缓存中，则获取键的值，否则返回 -1 。
- void put(int key, int value) - 如果键 key 已存在，则变更其值；如果键不存在，请插入键值对。当缓存达到其容量 capacity 时，则应该在插入新项之前，移- 除最不经常使用的项。在此问题中，当存在平局（即两个或更多个键具有相同使用频率）时，应该去除 **最久未使用** 的键。

为了确定最不常使用的键，可以为缓存中的每个键维护一个 **使用计数器** 。使用计数最小的键是最久未使用的键。

当一个键首次插入到缓存中时，它的使用计数器被设置为 1 (由于 put 操作)。对缓存中的键执行 get 或 put 操作，使用计数器的值将会递增。

函数 get 和 put 必须以 O(1) 的平均时间复杂度运行。

示例：

> 输入：
> ["LFUCache", "put", "put", "get", "put", "get", "get", "put", "get", "get", "get"]
> 
> [[2], [1, 1], [2, 2], [1], [3, 3], [2], [3], [4, 4], [1], [3], [4]]
> 
> 输出：
> 
> [null, null, null, 1, null, -1, 3, null, -1, 3, 4]
> 
> 解释：
> 
> // cnt(x) = 键 x 的使用计数
> 
> // cache=[] 将显示最后一次使用的顺序（最左边的元素是最近的）
> 
> LFUCache lfu = new LFUCache(2);
> 
> lfu.put(1, 1);   // cache=[1,_], cnt(1)=1
> 
> lfu.put(2, 2);   // cache=[2,1], cnt(2)=1, cnt(1)=1
> 
> lfu.get(1);      // 返回 1    // cache=[1,2], cnt(2)=1, cnt(1)=2
> 
> lfu.put(3, 3);   // 去除键 2 ，因为 cnt(2)=1 ，使用计数最小   // cache=[3,1], cnt(3)=1, cnt(1)=2
> 
> lfu.get(2);      // 返回 -1（未找到）
> 
> lfu.get(3);      // 返回 3    // cache=[3,1], cnt(3)=2, cnt(1)=2
> 
> lfu.put(4, 4);   // 去除键 1 ，1 和 3 的 cnt 相同，但 1 最久未使用    // cache=[4,3], cnt(4)=1, cnt(3)=2
> 
> lfu.get(1);      // 返回 -1（未找到）
> 
> lfu.get(3);      // 返回 3    // cache=[3,4], cnt(4)=1, cnt(3)=3
> 
> lfu.get(4);      // 返回 4    // cache=[3,4], cnt(4)=2, cnt(3)=3
 
提示：

- 1 <= capacity <= 10<sup>4</sup>
- 0 <= key <= 10<sup>5</sup>
- 0 <= value <= 10<sup>9</sup>
- 最多调用 2 * 10<sup>5</sup> 次 get 和 put 方法

## 最优解: 双哈希表 + 双向链表

- 时间复杂度： get 和 put 均为 O(1)
- 空间复杂度： O(n)

核心思想：

- freqMap：按频率存储双向链表(同频率的节点按插入顺序排列)
- keyMap：存储键到节点的映射。
- minFreq 记录当前最小频率，用于快速淘汰节点

```
class LFUCache {
    private static class Node {
        int key, value, freq;
        Node next, prev;
        Node(int key, int value) {
            this.key = key;
            this.value = value;
            this.freq = 1;
        }
    }

    private static class DoublyLinkedList {
        Node head, tail;
        int size;
        DoublyLinkedList() {
            head = new Node(-1, -1);
            tail = new Node(-1, -1);
            head.next = tail;
            tail.prev = head;
            size = 0;
        }

        void addToHead(Node node) {
            node.next = head.next;
            node.prev = head;
            head.next.prev = node;
            head.next = node;
            size++;
        }

        void removeNode(Node node) {
            node.prev.next = node.next;            
            node.next.prev = node.prev;
            size--;
        }

        Node removeTail() {
            if (size == 0) {
                return null;
            }

            Node tailNode = tail.prev;
            removeNode(tailNode);
            return tailNode;
        }
    }

    private int capacity, minFreq;
    private Map<Integer, Node> keyMap;
    private Map<Integer, DoublyLinkedList> freqMap;

    public LFUCache(int capacity) {
        this.capacity = capacity;
        this.minFreq = 0;
        keyMap = new HashMap<>();
        freqMap = new HashMap<>();   
    }
    
    public int get(int key) {
        if (!keyMap.containsKey(key)) {
            return -1;
        } 
        
        Node node = keyMap.get(key);
        updateFreq(node);
        return node.value;   
    }
    
    public void put(int key, int value) {
        if (keyMap.containsKey(key)) {
            Node node = keyMap.get(key);
            node.value = value;
            updateFreq(node);
        } else {
            if (keyMap.size() == capacity) {
                DoublyLinkedList minFreqList = freqMap.get(minFreq);
                Node removedNode = minFreqList.removeTail();
                keyMap.remove(removedNode.key);
            }
            Node newNode = new Node(key, value);
            keyMap.put(key, newNode);
            freqMap.computeIfAbsent(1, k -> new DoublyLinkedList()).addToHead(newNode);
            minFreq = 1;
        }   
    }

    public void updateFreq(Node node) {
        int oldFreq = node.freq;
        DoublyLinkedList oldList = freqMap.get(oldFreq);
        oldList.removeNode(node);
        if (oldList.size == 0 && oldFreq == minFreq) {
            minFreq++;
        }
        node.freq++;
        freqMap.computeIfAbsent(node.freq, k -> new DoublyLinkedList()).addToHead(node);     
    }
}
```