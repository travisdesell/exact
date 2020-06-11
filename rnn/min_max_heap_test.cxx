#include <vector>
#include <algorithm>
#include <cstdlib>
#include <iostream>
#include <stdlib.h>
#include <chrono>

using namespace std::chrono;
using namespace std;

#include "min_max_heap.hxx"


#define N 5000000
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

#define POPULATION_SIZE 5
double examm_test_heap(min_max_heap<int> heap, vector<int> &numbers) {
    heap.clear();

    auto start = high_resolution_clock::now();
    for (int32_t i = 0; i < numbers.size(); i += 1) {
        if (heap.size() >= POPULATION_SIZE) {
            if (heap.find_max() > numbers[i])
                heap.replace_max(numbers[i]);
        } else {
            heap.enqueue(numbers[i]);
        }
    }
    auto stop = high_resolution_clock::now();
    auto duration = duration_cast<nanoseconds>(stop - start);
    long nanos = duration.count();
    double dnanos = (double) nanos;
    return dnanos * 1e-9;
}

inline void sorted_insert(vector<int> &population, int n) {
    population.insert(std::lower_bound(population.begin(), population.end(), n), n);
}

double examm_test_vector(vector<int> &numbers) {
    vector<int> population;
    population.reserve(POPULATION_SIZE);

    auto start = high_resolution_clock::now();
    for (int32_t i = 0; i < numbers.size(); i += 1) {
        if (population.size() >= POPULATION_SIZE) {
            if (population[POPULATION_SIZE - 1] > numbers[i]) {
                population.pop_back();
                sorted_insert(population, numbers[i]);
            }
        } else {
            sorted_insert(population, numbers[i]);
        }
    }
    auto stop = high_resolution_clock::now();
    auto duration = duration_cast<nanoseconds>(stop - start);
    long nanos = duration.count();
    double dnanos = (double) nanos;
    return dnanos * 1e-9;
}

int main() {
    function<bool (const int&, const int&)> less_than =
        [](const int& a, const int& b) {
            return a < b;
        };
    min_max_heap<int> heap(less_than);
    vector<int> numbers;
    init(&numbers);
    double time_vec = examm_test_vector(numbers);
    cout << "vector took " << time_vec << "s" << endl;

    double time_heap = examm_test_heap(heap, numbers);
    cout << "heap took " << time_heap << "s" << endl;

    //// std::sort(sorted.begin(), sorted.end(), [](int i, int j) { return i > j; });

    //test_pop_max(heap, numbers, sorted);
    //
    //for (int i = 0; i < N; i++)
    //    sorted[i] -= 1;
    //
    //test_replace_max(heap, numbers, sorted);
}
