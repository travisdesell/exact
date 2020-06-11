#include <vector>
#include <algorithm>
#include <cstdlib>
#include <iostream>
#include <stdlib.h>
#include <chrono>
#include <iomanip>

using namespace std::chrono;
using namespace std;

#include "min_max_heap.hxx"


// Try to insert a million members
#define N 1000000
void init(vector<int64_t> *v) {
    v->clear();
    for (int64_t i = 0; i < N; i+=1) {
        v->push_back(i);
    }

    for (int64_t i = 0; i < N * 20; i++) {
        int64_t i0 = rand() % N;
        int64_t i1 = rand() % N;
        std::swap((*v)[i0], (*v)[i1]);
    }
}

void should_be_eq(vector<int64_t> &a, vector<int64_t> &b, char *name) {
    if (a.size() != b.size())
        goto err;
    
    for (int64_t i = 0; i < a.size(); i++)
        if (a[i] != b[i])
            goto err;

    printf("OK: test passed '%s'\n", name);
    return;

err:
    printf("Error: test failed at line '%s'\n", name);
    return;
}

void test_pop_min(min_max_heap<int64_t> heap, vector<int64_t> &numbers, vector<int64_t> &sorted) {
    for (int64_t i = 0; i < numbers.size(); i++) {
        int64_t n = numbers[i];
        heap.enqueue(n);
    }
    
    vector<int64_t> should_be_sorted;
    for (int64_t i = 0; i < numbers.size(); i++) {
        int64_t n = heap.pop_min();
        should_be_sorted.push_back(n);
    }

    should_be_eq(sorted, should_be_sorted, "test_pop_min");
}

void test_pop_max(min_max_heap<int64_t> heap, vector<int64_t> &numbers, vector<int64_t> &sorted) {
    for (int64_t i = 0; i < numbers.size(); i++) {
        int64_t n = numbers[i];
        heap.enqueue(n);
    }
    
    vector<int64_t> should_be_sorted;
    for (int64_t i = 0; i < numbers.size(); i++) {
        int64_t n = heap.pop_max();
        should_be_sorted.push_back(n);
    }

    should_be_eq(sorted, should_be_sorted, "test_pop_max");
}

void test_replace_max(min_max_heap<int64_t> heap, vector<int64_t> &numbers, vector<int64_t> &sorted) {
    for (int64_t i = 0; i < numbers.size(); i++) {
        int64_t n = numbers[i];
        heap.enqueue(n);
    }

    int64_t should_be_N_minus_one = heap.replace_max(-1);
    
    if (should_be_N_minus_one != N - 1) {
        printf("Error: replaced the wrong element!\n");
    }

    vector<int64_t> should_be_sorted;
    for (int64_t i = 0; i < numbers.size(); i++) {
        int64_t n = heap.pop_max();
        should_be_sorted.push_back(n);
    }

    should_be_eq(sorted, should_be_sorted, "test_replace_max");
}

static int64_t POPULATION_SIZE = 50000;
double examm_test_heap(min_max_heap<int64_t> &heap, vector<int64_t> &numbers) {
    heap.clear();
    heap.reserve(POPULATION_SIZE);

    auto start = high_resolution_clock::now();
    for (int64_t i = 0; i < numbers.size(); i += 1) {
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

inline void sorted_insert(vector<int64_t> &population, int64_t n) {
    population.insert(std::lower_bound(population.begin(), population.end(), n), n);
}

double examm_test_vector(vector<int64_t> &numbers) {
    vector<int64_t> population;
    population.reserve(POPULATION_SIZE);

    auto start = high_resolution_clock::now();
    for (int64_t i = 0; i < numbers.size(); i += 1) {
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
    function<bool (const int64_t&, const int64_t&)> less_than =
        [](const int64_t& a, const int64_t& b) {
            return a < b;
        };
    min_max_heap<int64_t> heap(less_than);
    vector<int64_t> numbers;
    init(&numbers);

    int64_t pop_sizes[20];

    // powers of two for pop sizes
    for (int64_t i = 0; i < 15; i++) 
        pop_sizes[i] = 1 << i;

    int n_trials = 256;
    double n_trials_d = (double) n_trials;

    cout << "| pop_size | vec_time      | heap_time     |" << endl;
    cout << "|----------|---------------|---------------|" << endl;

    cout.precision(8);
    for (int64_t i = 0 ; i < 15; i++) {
        POPULATION_SIZE = pop_sizes[i];
        double sum_time_vec = 0.0;
        for (int j = 0; j < n_trials; j++)
            sum_time_vec += examm_test_vector(numbers);
        double time_vec = sum_time_vec / n_trials_d;
    
        double sum_time_heap = 0.0;
        for (int j = 0; j < n_trials; j++)
            sum_time_heap += examm_test_heap(heap, numbers);
        double time_heap = sum_time_heap / n_trials_d;

        cout << "| " << setw(8) << POPULATION_SIZE << " | " << setw(13) << time_vec << " | " << setw(13) << time_heap << " |" << endl;
    }
    //// std::sort(sorted.begin(), sorted.end(), [](int64_t i, int64_t j) { return i > j; });

    //test_pop_max(heap, numbers, sorted);
    //
    //for (int64_t i = 0; i < N; i++)
    //    sorted[i] -= 1;
    //
    //test_replace_max(heap, numbers, sorted);
}
