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

#include "strategy/exact.hxx"
#include "strategy/cnn_genome.hxx"
#include "strategy/cnn_edge.hxx"
#include "strategy/cnn_node.hxx"

int main(int argc, char **argv) {
    vector<string> arguments = vector<string>(argv, argv + argc);

    string db_file;
    get_argument(arguments, "--db_file", true, db_file);
    set_db_info_filename(db_file);

    string training_data;
    get_argument(arguments, "--training_data", true, training_data);

    string testing_data;
    get_argument(arguments, "--testing_data", true, testing_data);

    string genome_filename;
    get_argument(arguments, "--genome_file", true, genome_filename);

    bool is_checkpoint = false;
    CNN_Genome *genome_from_file = new CNN_Genome(genome_filename, is_checkpoint);

    genome_from_file->export_to_database(-1000);

    int genome_id = genome_from_file->get_genome_id();
    cout << "GENOME EXPORTED TO DATABASE WITH ID: " << genome_id << endl;

    CNN_Genome *genome_from_database = new CNN_Genome(genome_id);

    Images training_images(training_data, genome_from_file->get_padding());
    Images testing_images(testing_data, genome_from_file->get_padding(), training_images.get_average(), training_images.get_std_dev());

    float error;
    int predictions;
    //genome->evaluate(training_images, error, predictions);

    genome_from_file->set_to_best();
    //genome_from_file->evaluate(testing_images, error, predictions);

    cout << "GENOME FROM FILE test error: " << error << endl;
    cout << "GENOME FROM FILE test predictions " << predictions << endl;

    genome_from_database->set_to_best();
    //genome_from_database->evaluate(testing_images, error, predictions);
    genome_from_database->write_to_file("./exported_genome_" + to_string(genome_id));

    cout << "GENOME FROM FILE test error: " << error << endl;
    cout << "GENOME FROM FILE test predictions " << predictions << endl;


    /*
    ostringstream query;
    query << "DELETE FROM cnn_edge WHERE genome_id = " << genome_id << endl;
    query << "UPDATE cnn_genome SET test_error = " << error << ", test_predictions = " << predictions << " WHERE id = " << genome_id;
    mysql_exact_query(query.str());
    */

}
