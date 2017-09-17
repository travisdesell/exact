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
    int exact_id = -1;
#endif

    vector<CNN_Genome*> genomes;

    if (argument_exists(arguments, "--genome_files")) {
        vector<string> genome_filenames;
        get_argument_vector(arguments, "--genome_files", true, genome_filenames);

        bool is_checkpoint = false;
        for (uint32_t i = 0; i < genome_filenames.size(); i++) {
            genomes.push_back(new CNN_Genome(genome_filenames[i], is_checkpoint));
        }

#ifdef _MYSQL_
    } else if (argument_exists(arguments, "--db_file")) {
        string db_file;
        get_argument(arguments, "--db_file", true, db_file);
        set_db_info_filename(db_file);

        get_argument(arguments, "--exact_id", true, exact_id);

        EXACT *exact = new EXACT(exact_id);

        for (int32_t i = 0; i < exact->get_number_genomes(); i++) {
            genomes.push_back(exact->get_genome(i));
        }
#endif
    } else {
        cerr << "ERROR: need either --genome_files or --exact_id argument to initialize genomes." << endl;
        exit(1);
    }

    Images training_images(training_data, 0);
    Images testing_images(testing_data, 0, training_images.get_average(), training_images.get_std_dev());

    vector<int> expected_classes(testing_images.get_number_images(), 0);
    for (uint32_t i = 0; i < expected_classes.size(); i++) {
        expected_classes[i] = testing_images.get_classification(i);
    }

    vector<vector<float>> predictions(testing_images.get_number_images(), vector<float>(testing_images.get_number_classes(), 0));

    for (uint32_t i = 0; i < genomes.size(); i++) {
        cout << "evaluating predictions for genome " << i << " of " << genomes.size() << ", genome id: " << genomes[i]->get_generation_id() << endl;
        genomes[i]->set_to_best();
        genomes[i]->evaluate(testing_images, predictions);
    }

    int wrong = 0;
    for (int32_t i = 0; i < testing_images.get_number_images(); i++) {
        cout << "image " << setw(10) << i << ", expected: " << expected_classes[i] << ", predictions:";

        int max_class = 0;
        int max_value = 0;
        for (int32_t j = 0; j < testing_images.get_number_classes(); j++) {
            cout << " " << setw(3) << predictions[i][j];
            
            if (predictions[i][j] > max_value) {
                max_value = predictions[i][j];
                max_class = j;
            }
        }

        if (max_class != expected_classes[i]) {
            cout << " -- WRONG!";
            wrong++;
        }
        cout << endl;
    }
    cout << "total wrong: " << wrong << ", accuracy: " << (double)(testing_images.get_number_images() - wrong)/(double)testing_images.get_number_images() << "%" << endl;

}
