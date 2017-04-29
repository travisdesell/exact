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

#include "common/arguments.hxx"

#include "image_tools/image_set.hxx"

#include "strategy/exact.hxx"

mutex exact_mutex;

vector<string> arguments;

EXACT *exact;


bool finished = false;

int images_resize;

void exact_thread(const Images &training_images, const Images &testing_images, int id) {
    while (true) {
        exact_mutex.lock();
        CNN_Genome *genome = exact->generate_individual();
        exact_mutex.unlock();

        if (genome == NULL) break;  //generate_individual returns NULL when the search is done

        genome->set_name("thread_" + to_string(id));
        genome->stochastic_backpropagation(training_images, testing_images, images_resize);

        exact_mutex.lock();
        exact->insert_genome(genome);
        exact->export_to_database();
        exact_mutex.unlock();
    }
}

int main(int argc, char** argv) {
    arguments = vector<string>(argv, argv + argc);

    int number_threads;
    get_argument(arguments, "--number_threads", true, number_threads);

    string training_filename;
    get_argument(arguments, "--training_file", true, training_filename);

    string testing_filename;
    get_argument(arguments, "--testing_file", true, testing_filename);

    int population_size;
    get_argument(arguments, "--population_size", true, population_size);

    int max_epochs;
    get_argument(arguments, "--max_epochs", true, max_epochs);

    int max_genomes;
    get_argument(arguments, "--max_genomes", true, max_genomes);

    string output_directory;
    get_argument(arguments, "--output_directory", true, output_directory);

    string search_name;
    get_argument(arguments, "--search_name", true, search_name);

    bool reset_edges;
    get_argument(arguments, "--reset_edges", true, reset_edges);

    double generalizability;
    get_argument(arguments, "--generalizability", true, generalizability);

    get_argument(arguments, "--images_resize", true, images_resize);



    Images training_images(training_filename);
    Images testing_images(testing_filename, training_images.get_average(), training_images.get_std_dev());

    exact = new EXACT(training_images, testing_images, population_size, max_epochs, max_genomes, output_directory, search_name, reset_edges, generalizability);

    //exact = new EXACT(1);

    vector<thread> threads;
    for (int32_t i = 0; i < number_threads; i++) {
        threads.push_back( thread(exact_thread, training_images, testing_images, i) );
    }

    for (int32_t i = 0; i < number_threads; i++) {
        threads[i].join();
    }

    finished = true;

    cout << "completed!" << endl;

    return 0;
}
