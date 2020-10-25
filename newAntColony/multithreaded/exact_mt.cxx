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

#include "cnn/exact.hxx"

mutex exact_mutex;

vector<string> arguments;

EXACT *exact;


bool finished = false;

int images_resize;

void exact_thread(const Images &training_images, const Images &validation_images, const Images &testing_images, int id) {
    while (true) {
        exact_mutex.lock();
        CNN_Genome *genome = exact->generate_individual();
        exact_mutex.unlock();

        if (genome == NULL) break;  //generate_individual returns NULL when the search is done

        genome->set_name("thread_" + to_string(id));
        genome->initialize();
        genome->stochastic_backpropagation(training_images, images_resize, validation_images);
        genome->evaluate_test(testing_images);

        exact_mutex.lock();
        exact->insert_genome(genome);
#ifdef _MYSQL_
        exact->export_to_database();
#endif
        exact_mutex.unlock();
    }
}

int main(int argc, char** argv) {
    arguments = vector<string>(argv, argv + argc);

    int number_threads;
    get_argument(arguments, "--number_threads", true, number_threads);

    int padding;
    get_argument(arguments, "--padding", true, padding);


    string training_filename;
    get_argument(arguments, "--training_file", true, training_filename);

    string validation_filename;
    get_argument(arguments, "--validation_file", true, validation_filename);

    string testing_filename;
    get_argument(arguments, "--testing_file", true, testing_filename);

    int population_size;
    get_argument(arguments, "--population_size", true, population_size);

    int max_epochs;
    get_argument(arguments, "--max_epochs", true, max_epochs);

    bool use_sfmp;
    get_argument(arguments, "--use_sfmp", true, use_sfmp);

    bool use_node_operations;
    get_argument(arguments, "--use_node_operations", true, use_node_operations);

    int max_genomes;
    get_argument(arguments, "--max_genomes", true, max_genomes);

    string output_directory;
    get_argument(arguments, "--output_directory", true, output_directory);

    string search_name;
    get_argument(arguments, "--search_name", true, search_name);

    bool reset_edges;
    get_argument(arguments, "--reset_edges", true, reset_edges);

    get_argument(arguments, "--images_resize", true, images_resize);



    Images training_images(training_filename, padding);
    Images validation_images(validation_filename, padding, training_images.get_average(), training_images.get_std_dev());
    Images testing_images(testing_filename, padding, training_images.get_average(), training_images.get_std_dev());

#ifdef _MYSQL_
    if (argument_exists(arguments, "--exact_id")) {
        int exact_id;
        get_argument(arguments, "--exact_id", true, exact_id);
        exact = new EXACT(exact_id);
    } else {
#endif
        exact = new EXACT(training_images, validation_images, testing_images, padding, population_size, max_epochs, use_sfmp, use_node_operations, max_genomes, output_directory, search_name, reset_edges);
#ifdef _MYSQL_
    }
#endif


    vector<thread> threads;
    for (int32_t i = 0; i < number_threads; i++) {
        threads.push_back( thread(exact_thread, training_images, validation_images, testing_images, i) );
    }

    for (int32_t i = 0; i < number_threads; i++) {
        threads[i].join();
    }

    finished = true;

    cout << "completed!" << endl;

    return 0;
}
