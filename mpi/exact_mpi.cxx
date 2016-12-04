#include <chrono>

#include <iostream>
using std::cerr;
using std::cout;
using std::endl;

#include <mutex>
using std::mutex;

#include <string>
using std::string;

#include <thread>
using std::thread;

#include <vector>
using std::vector;


//from undvc_common
#include "arguments.hxx"

#include "image_tools/image_set.hxx"

#include "strategy/exact.hxx"

mutex exact_mutex;

vector<string> arguments;

EXACT *exact;

bool finished = false;

void exact_thread(const Images &images, int id, int number_iterations) {
    for (uint32_t i = 0; i < number_iterations; i++) {
        exact_mutex.lock();
        CNN_Genome *genome = exact->generate_individual();
        exact_mutex.unlock();

        if (genome == NULL) continue;
        genome->set_name("thread_" + to_string(id));
        genome->stochastic_backpropagation(images);

        exact_mutex.lock();
        exact->insert_genome(genome);
        exact_mutex.unlock();
    }
}

void polling_thread(string polling_filename) {
    ofstream polling_file(polling_filename);

    while (true) {
        std::this_thread::sleep_for(std::chrono::seconds(60));

        cout << "printing statistics!" << endl;
        exact_mutex.lock();
        exact->print_statistics(polling_file);
        exact_mutex.unlock();

        if (finished) break;
    }

    polling_file.close();
}

int main(int argc, char** argv) {
    arguments = vector<string>(argv, argv + argc);

    int number_threads;
    get_argument(arguments, "--number_threads", true, number_threads);

    int number_iterations;
    get_argument(arguments, "--number_iterations", true, number_iterations);

    string binary_samples_filename;
    get_argument(arguments, "--samples_file", true, binary_samples_filename);

    string progress_filename;
    get_argument(arguments, "--progress_file", true, progress_filename);

    int population_size;
    get_argument(arguments, "--population_size", true, population_size);

    int min_epochs;
    get_argument(arguments, "--min_epochs", true, min_epochs);

    int max_epochs;
    get_argument(arguments, "--max_epochs", true, max_epochs);

    int improvement_required_epochs;
    get_argument(arguments, "--improvement_required_epochs", true, improvement_required_epochs);

    bool reset_edges;
    get_argument(arguments, "--reset_edges", true, reset_edges);

    Images images(binary_samples_filename);

    exact = new EXACT(images, population_size, min_epochs, max_epochs, improvement_required_epochs, reset_edges);

    vector<thread> threads;
    for (uint32_t i = 0; i < number_threads; i++) {
        threads.push_back( thread(exact_thread, images, i, number_iterations) );
    }

    thread poller(polling_thread, progress_filename);

    for (uint32_t i = 0; i < number_threads; i++) {
        threads[i].join();
    }

    finished = true;

    cout << "completed!" << endl;

    return 0;
}
