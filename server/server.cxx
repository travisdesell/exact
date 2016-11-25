#include <iostream>
#include <iomanip>
#include <stdio.h>
#include <stdlib.h>
#include <fstream>
#include <vector>

//from undvc_common
#include "arguments.hxx"

#include "image_tools/image_set.hxx"

#include "strategy/cnn_neat.hxx"

using namespace std;

vector<string> arguments;

int main(int argc, char** argv) {
    arguments = vector<string>(argv, argv + argc);

    string binary_samples_filename;
    get_argument(arguments, "--samples_file", true, binary_samples_filename);

    string output_filename;
    get_argument(arguments, "--output_file", true, output_filename);


    Images images(binary_samples_filename);

    CNN_NEAT cnn_neat(images, 50, 20);
    //cnn_neat.write_genomes_to_directorY(directory);

    CNN_Genome *genome = cnn_neat.get_best_genome();
    genome->write_to_file(output_filename);

    return 0;
}
