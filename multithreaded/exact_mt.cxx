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
        exact->export_to_database();
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

    string search_name;
    get_argument(arguments, "--search_name", true, search_name);

    int population_size;
    get_argument(arguments, "--population_size", true, population_size);

    int max_epochs;
    get_argument(arguments, "--max_epochs", true, max_epochs);

    bool reset_edges;
    get_argument(arguments, "--reset_edges", true, reset_edges);

    int max_genomes;
    get_argument(arguments, "--max_genomes", true, max_genomes);

    double learning_rate;
    get_argument(arguments, "--learning_rate", true, learning_rate);

    double learning_rate_decay;
    get_argument(arguments, "--learning_rate_decay", true, learning_rate_decay);

    double weight_decay;
    get_argument(arguments, "--weight_decay", true, weight_decay);

    double weight_decay_decay;
    get_argument(arguments, "--weight_decay_decay", true, weight_decay_decay);

    double mu;
    get_argument(arguments, "--mu", true, mu);

    double mu_decay;
    get_argument(arguments, "--mu_decay", true, mu_decay);

    double input_dropout_probability;
    get_argument(arguments, "--input_dropout_probability", true, input_dropout_probability);

    double hidden_dropout_probability;
    get_argument(arguments, "--hidden_dropout_probability", true, hidden_dropout_probability);

    int velocity_reset;
    get_argument(arguments, "--velocity_reset", true, velocity_reset);

    Images images(binary_samples_filename);

    exact = new EXACT(images, population_size, max_epochs, reset_edges, velocity_reset, mu, mu_decay, learning_rate, learning_rate_decay, weight_decay, weight_decay_decay, input_dropout_probability, hidden_dropout_probability, max_genomes, output_directory, search_name);
    //exact = new EXACT(1);

    /*
    cout << "generating individual!" << endl;

    CNN_Genome *genome = exact->generate_individual();

    cout << "exporting genome to database!" << endl;
    genome->export_to_database(1);
    cout << "inserted genome into database with id: " << genome->get_genome_id() << endl;
    genome->write_to_file("test_original.txt");

    genome->stochastic_backpropagation(images);
    */

    //CNN_Genome *from_database = new CNN_Genome(genome->get_genome_id());
    //CNN_Genome *from_database = new CNN_Genome(1);
    //from_database->stochastic_backpropagation(images);

    vector<thread> threads;
    for (uint32_t i = 0; i < number_threads; i++) {
        threads.push_back( thread(exact_thread, images, i) );
    }

    for (uint32_t i = 0; i < number_threads; i++) {
        threads[i].join();
    }

    finished = true;

    cout << "completed!" << endl;

    return 0;
}
