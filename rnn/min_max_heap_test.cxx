#include <vector>
#include <algorithm>
#include <cstdlib>
#include <iostream>
#include <stdlib.h>

using namespace std;

#include "min_max_heap.hxx"


#define N 500
void init(vector<int> *v) {
    v->clear();
    for (int i = 0; i < N; i+=1) {
        v->push_back(i);
    }

    for (int i = 0; i < N * 20; i++) {
        int i0 = rand() % N;
        int i1 = rand() % N;
        std::swap((*v)[i0], (*v)[i1]);
    }
}

void should_be_eq(vector<int> &a, vector<int> &b, char *name) {
    if (a.size() != b.size())
        goto err;
    
    for (int i = 0; i < a.size(); i++)
        if (a[i] != b[i])
            goto err;

    printf("OK: test passed '%s'\n", name);
    return;

err:
    printf("Error: test failed at line '%s'\n", name);
    return;
}

void test_pop_min(min_max_heap<int> heap, vector<int> &numbers, vector<int> &sorted) {
    for (int i = 0; i < numbers.size(); i++) {
        int n = numbers[i];
        heap.enqueue(n);
    }
    
    vector<int> should_be_sorted;
    for (int i = 0; i < numbers.size(); i++) {
        int n = heap.pop_min();
        should_be_sorted.push_back(n);
    }

    should_be_eq(sorted, should_be_sorted, "test_pop_min");
}

void test_pop_max(min_max_heap<int> heap, vector<int> &numbers, vector<int> &sorted) {
    for (int i = 0; i < numbers.size(); i++) {
        int n = numbers[i];
        heap.enqueue(n);
    }
    
    vector<int> should_be_sorted;
    for (int i = 0; i < numbers.size(); i++) {
        int n = heap.pop_max();
        should_be_sorted.push_back(n);
    }

    should_be_eq(sorted, should_be_sorted, "test_pop_max");
}

void test_replace_max(min_max_heap<int> heap, vector<int> &numbers, vector<int> &sorted) {
    for (int i = 0; i < numbers.size(); i++) {
        int n = numbers[i];
        heap.enqueue(n);
    }

    int should_be_N_minus_one = heap.replace_max(-1);
    
    if (should_be_N_minus_one != N - 1) {
        printf("Error: replaced the wrong element!\n");
    }

    vector<int> should_be_sorted;
    for (int i = 0; i < numbers.size(); i++) {
        int n = heap.pop_max();
        should_be_sorted.push_back(n);
    }

    should_be_eq(sorted, should_be_sorted, "test_replace_max");
}

int main() {
    function<bool (const int&, const int&)> less_than =
        [](const int& a, const int& b) {
            return a < b;
        };
    min_max_heap<int> heap(less_than);

    vector<int> numbers;
    init(&numbers);

    vector<int> sorted(numbers);
    std::sort(sorted.begin(), sorted.end());
    
    test_pop_min(heap, numbers, sorted);

    std::sort(sorted.begin(), sorted.end(), [](int i, int j) { return i > j; });

    test_pop_max(heap, numbers, sorted);
    
    for (int i = 0; i < N; i++)
        sorted[i] -= 1;
    
    test_replace_max(heap, numbers, sorted);
}
