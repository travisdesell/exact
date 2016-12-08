#include <iostream>
#include <iomanip>
#include <stdio.h>
#include <stdlib.h>
#include <fstream>
#include <vector>

//from undvc_common
#include "arguments.hxx"

#include "image_tools/image_set.hxx"

#include "strategy/exact.hxx"

using namespace std;

vector<string> arguments;

int main(int argc, char** argv) {
    arguments = vector<string>(argv, argv + argc);

    string binary_samples_filename;
    get_argument(arguments, "--samples_file", true, binary_samples_filename);

    string output_filename;
    get_argument(arguments, "--output_file", true, output_filename);

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

    EXACT exact(images, population_size, min_epochs, max_epochs, improvement_required_epochs, reset_edges, max_individuals, output_directory);
    //exact..write_genomes_to_directorY(directory);

    CNN_Genome *genome = exact.get_best_genome();
    genome->write_to_file(output_filename);

    return 0;
}
