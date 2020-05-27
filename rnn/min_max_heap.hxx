inline uint32_t log2(uint32_t x) {
  // This won't work on non x86 platforms
  // https://stackoverflow.com/questions/994593/how-to-do-an-integer-log2-in-c
#if defined(__x86_64__) || defined(__i386__)
    uint32_t y;
    asm ( "\tbsr %1, %0\n"
        : "=r"(y)
        : "r" (x)
    );
    return y;
#else
    if (x == 1) return 0;
    uint32_t ret = 0;
    while (x > 1) {
        x >>= 1;
        ret++;
    }
    return ret;
#endif
}

/**
 * This implementation of a min_max_heap was basically ripped from here:
 * https://github.com/itsjohncs/minmaxheap-cpp/blob/master/MinMaxHeap.hpp
 *
 * Along with this for reference
 * https://en.wikipedia.org/wiki/Min-max_heap
 *
 * The first layer (e.g. the layer which only contains the root node) contains the maximum element.
 **/
template<class T>
class min_max_heap {

    typedef typename std::vector<T>::const_iterator const_iterator;
    
    vector<T> heap;
    function<bool(const T&, const T&)> less_than;
    
    static inline uint32_t parent(uint32_t z) { return (z - 1) / 2; }

    static inline uint32_t left_child(uint32_t z) { return z + z + 1; }

    static inline uint32_t right_child(uint32_t z) { return z + z + 2; }

    static inline bool is_on_min_level(uint32_t z) { return log2(z + 1) % 2 == 1; }

    static inline bool is_on_max_level(uint32_t z) { return log2(z + 1) % 2 == 0; }

    template<bool max_level> void trickle_up_inner(uint32_t z) {
        if (z == 0) return;

        uint32_t z_grandparent = parent(z);
        if (z_grandparent == 0) return;
        
        z_grandparent = parent(z_grandparent);

        if (less_than(heap[z], heap[z_grandparent]) ^ max_level) {
            std::swap(heap[z], heap[z_grandparent]);
            trickle_up_inner<max_level>(z_grandparent);
        }
    }

    void trickle_up(uint32_t z) {
        if (z == 0) return;

        uint32_t z_parent = parent(z);

        if (is_on_min_level(z)) {
            if (less_than(heap[z_parent], heap[z])) {
                std::swap(heap[z_parent], heap[z]);
                trickle_up_inner<true>(z_parent);
            } else {
                trickle_up_inner<false>(z);
            }
        } else {
            if (less_than(heap[z], heap[z_parent])) {
                std::swap(heap[z], heap[z_parent]);
                trickle_up_inner<false>(z_parent);
            } else {
                trickle_up_inner<true>(z);
            }
        }
    }

    template<bool max_level> void trickle_down_inner(const uint32_t z) {
        if (z >= heap.size())
            throw std::invalid_argument("Element specified by z does not exist");

        uint32_t smallest_node = z;
        uint32_t left = left_child(z);
        uint32_t right = left + 1;

        if (left < heap.size() && (less_than(heap[left], heap[smallest_node]) ^ max_level))
            smallest_node = left;
        if (right < heap.size() && (less_than(heap[right], heap[smallest_node]) ^ max_level))
            smallest_node = right;

        uint32_t left_grandchild = left_child(left);
        for (uint32_t i = 0; i < 4 && left_grandchild + i < heap.size(); i++)
            if (less_than(heap[left_grandchild + i], heap[smallest_node]) ^ max_level)
                smallest_node = left_grandchild + i;
        
        if (z == smallest_node) return;
        
        std::swap(heap[z], heap[smallest_node]);
        if (smallest_node - left > 1) { // smallest node was a grandchild
            if (less_than(heap[parent(smallest_node)], heap[smallest_node]) ^ max_level)
                std::swap(heap[parent(smallest_node)], heap[smallest_node]);

            trickle_down_inner<max_level>(smallest_node);
        }
    }

    void trickle_down(uint32_t z) {
        if (is_on_min_level(z))
            trickle_down_inner<false>(z);
        else
            trickle_down_inner<true>(z);
    }
    
    uint32_t find_min_index() const {
        switch (heap.size()) {
            case 0:
                throw std::underflow_error("There is no minimum element because the heap is empty");
            case 1:
                return 0;
            case 2:
                return 1;
            default:
                return less_than(heap[1], heap[2]) ? 1 : 2;
        }
    }

    T delete_element(uint32_t z) {
        if (z >= (uint32_t) heap.size())
            throw std::underflow_error("Cannot delete element from the heap because it does not exist");

        if (z == heap.size() - 1) {
            T e = heap.back();
            heap.pop_back();
            return e;
        }

        std::swap(heap[z], heap[heap.size() - 1]);

        T e = heap.back();
        heap.pop_back();
        
        trickle_down(z);

        return e;
    }

public:
 
    /**
     * Creates a new min_max_heap. The only parameter is a function that will compare two elements,
     * and determine if the first is less than the second. This could probably be done in a better way
     * with generics but I'm not confident in doing so.
     **/
    min_max_heap(std::function<bool(const T&, const T&)> _less_than, uint32_t size_hint=-1) 
        : less_than(_less_than) {
        if (size_hint >= 0)
            heap.reserve(size_hint);
    }
    ~min_max_heap() { }

    bool empty() const { return heap.size() == 0; }

    uint32_t size() const { return (uint32_t) heap.size(); }

    /**
     * Adds the element e to the heap (in the correct order of course). 
     **/
    void enqueue(const T& e) {
        heap.push_back(e);
        trickle_up(heap.size() - 1);
    }

    const T& find_max() const {
        if (empty())
            throw std::underflow_error("There is no max element because the heap is empty");

        return heap[0];
    }

    const T& find_min() const {
        return heap[find_min_index()];
    }

    /**
     * Returns and removes the maximum item in this min-max heap. If the heap is emoty it will throw an
     * underflow_error
     **/
    T pop_max() {
        if (heap.size() == 0)
            throw std::underflow_error("No max element exists because the heap is empty");

        return delete_element(0);
    }

    T pop() { return pop_max(); }

    /**
     * Returns and removes the minimum item in this min-max heap. If the heap is empty, it will throw
     * an underflow_error
     **/
    T pop_min() {
        if (heap.size() == 0)
            throw std::underflow_error("No minimum element exists because the heap is empty");

        return delete_element(find_min_index());
    }

    /**
     * returns the const_iterator to the beginning of the underlying vector. There is no non-const iterator
     * because the ordering of the data structure could potentially be changed.
     **/
    const_iterator cbegin() {
        return heap.begin();
    }
    
    /**
     * returns the const_iterator pointing past the end of the underlying vector 
     * (i.e. this is an invalid reference). There is no non-const end iterator because the ordering of the
     * data structure must be preserved.
     **/
    const_iterator cend() {
        return heap.end();
    }

    /**
     * Implementation of the indexing operator to allow direct access.
     **/
    T& operator[](size_t index) {
        return heap[index];
    }

    /**
     * Allows for deletion of an element at a specified index in O(log(n)) time.
     **/
    T erase(size_t index) {
        return delete_element((uint32_t) index);
    }

    /**
     * Ensures that the underlying vector can hold at least n elements.
     **/
    void reserve(size_t n) {
        heap.reserve(n);
    }
    
    /**
     * Deletes every element in this heap.
     **/
    void clear() {
        heap.clear();
    }
};
