#include <iomanip>
using std::setw;

#include <iostream>
using std::cerr;
using std::cout;
using std::endl;

#include <string>
using std::string;

#include <vector>
using std::vector;

#include "common/arguments.hxx"

#ifdef _MYSQL_
#include "common/db_conn.hxx"
#endif

#include "cnn/exact.hxx"
#include "cnn/cnn_genome.hxx"
#include "cnn/cnn_edge.hxx"
#include "cnn/cnn_node.hxx"

int main(int argc, char **argv) {
    vector<string> arguments = vector<string>(argv, argv + argc);

    string training_data;
    get_argument(arguments, "--training_data", true, training_data);

    string testing_data;
    get_argument(arguments, "--testing_data", true, testing_data);


#ifdef _MYSQL_
    int genome_id = -1;
#endif

    CNN_Genome *genome = NULL;

    if (argument_exists(arguments, "--genome_file")) {
        string genome_filename;
        get_argument(arguments, "--genome_file", true, genome_filename);

        bool is_checkpoint = false;
        genome = new CNN_Genome(genome_filename, is_checkpoint);

#ifdef _MYSQL_
    } else if (argument_exists(arguments, "--db_file")) {
        string db_file;
        get_argument(arguments, "--db_file", true, db_file);
        set_db_info_filename(db_file);

        get_argument(arguments, "--genome_id", true, genome_id);

        genome = new CNN_Genome(genome_id);
#endif
    } else {
        cerr << "ERROR: need either --genome_file or --genome_id argument to initialize genome." << endl;
        exit(1);
    }

    Images training_images(training_data, genome->get_padding());
    Images testing_images(testing_data, genome->get_padding(), training_images.get_average(), training_images.get_std_dev());

    //float error;
    //int predictions;
    //genome->evaluate(training_images, error, predictions);

    //cout << "number enabled edges: " << genome->get_number_enabled_edges();
    //cout << "number enabled nodes: " << genome->get_number_enabled_edges();
    cout << "number enabled weights: " << genome->get_number_weights() << endl;

    genome->reset(true);
    genome->initialize();

    genome->check_gradients(training_images);

    //genome->set_to_best();
    //genome->evaluate(testing_images, error, predictions);

    //cout << "test error: " << error << endl;
    //cout << "test predictions " << predictions << endl;
}
