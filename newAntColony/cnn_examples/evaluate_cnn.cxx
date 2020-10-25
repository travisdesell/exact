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

    float error;
    int predictions;
    //genome->evaluate(training_images, error, predictions);

    if (!genome->sanity_check(SANITY_CHECK_AFTER_GENERATION)) {
        cerr << "ERROR! genome failed sanity check! This should never happen!" << endl;

        exit(1);
    }   


    genome->initialize();
    genome->set_to_best();
    genome->evaluate("testing", testing_images, error, predictions);

    cout << "test error: " << error << endl;
    cout << "test predictions " << predictions << endl;

    genome->write_to_file("./genome_" + to_string(genome->get_generation_id()));

#ifdef _MYSQL_
    if (genome_id >= 0 && argument_exists(arguments, "--update_database")) {
        ostringstream query;
        query << "UPDATE cnn_genome SET test_error = " << error << ", test_predictions = " << predictions << " WHERE id = " << genome_id;
        mysql_exact_query(query.str());
    }
#endif

}
