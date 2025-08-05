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