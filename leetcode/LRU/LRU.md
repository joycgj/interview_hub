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

```
class LRUCache {
    private static class Node {
        int key, value;
        Node next, prev;
        Node(int key, int value) {
            this.key = key;
            this.value = value;
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
            node.next.prev = node.prev;
            node.prev.next = node.next;  
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

    private final int capacity;
    private final Map<Integer, Node> keyMap;
    private final DoublyLinkedList list;

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
        list.removeNode(node);
        list.addToHead(node);
        return node.value;
    }
    
    public void put(int key, int value) {
        if (keyMap.containsKey(key)) {
            Node node = keyMap.get(key);
            node.value = value;
            list.removeNode(node);
            list.addToHead(node);           
        } else {
            if (list.size == capacity) {
                Node node = list.removeTail();
                keyMap.remove(node.key);
            }
            Node newNode = new Node(key, value);
            keyMap.put(key, newNode);
            list.addToHead(newNode);
        }    
    }
}
```