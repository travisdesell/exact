#include <chrono>

#include <condition_variable>
using std::condition_variable;

#include <iomanip>
using std::setw;

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
condition_variable polling_condition;

vector<string> arguments;

EXACT *exact;

bool finished = false;

void polling_thread(string output_directory) {
    ofstream polling_file(output_directory + "/progress.txt");

    polling_file << "#" << setw(9) << "minute";
    exact->print_statistics_header(polling_file);

    int minute = 0;
    while (true) {
        exact_mutex.lock();
        polling_file << setw(10) << minute;
        exact->print_statistics(polling_file);
        exact_mutex.unlock();

        minute++;
        std::this_thread::sleep_for( std::chrono::seconds(60) );
        if (finished) break;
    }   

    polling_file.close();
}

void exact_thread(const Images &images, int id) {
    while (true) {
        exact_mutex.lock();
        CNN_Genome *genome = exact->generate_individual();
        exact_mutex.unlock();

        if (genome == NULL) break;  //generate_individual returns NULL when the search is done

        genome->set_name("thread_" + to_string(id));
        genome->stochastic_backpropagation(images);

        exact_mutex.lock();
        exact->insert_genome(genome);
        exact_mutex.unlock();
    }
}

int main(int argc, char** argv) {
    arguments = vector<string>(argv, argv + argc);

    int number_threads;
    get_argument(arguments, "--number_threads", true, number_threads);

    string binary_samples_filename;
    get_argument(arguments, "--samples_file", true, binary_samples_filename);

    string output_directory;
    get_argument(arguments, "--output_directory", true, output_directory);

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

    int max_individuals;
    get_argument(arguments, "--max_individuals", true, max_individuals);

    Images images(binary_samples_filename);

    exact = new EXACT(images, population_size, min_epochs, max_epochs, improvement_required_epochs, reset_edges, max_individuals, output_directory);

    vector<thread> threads;
    for (uint32_t i = 0; i < number_threads; i++) {
        threads.push_back( thread(exact_thread, images, i) );
    }

    thread poller(polling_thread, output_directory);

    for (uint32_t i = 0; i < number_threads; i++) {
        threads[i].join();
    }

    finished = true;
    poller.join();

    cout << "completed!" << endl;

    return 0;
}
