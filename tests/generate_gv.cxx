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

#include "strategy/exact.hxx"
#include "strategy/cnn_genome.hxx"
#include "strategy/cnn_edge.hxx"
#include "strategy/cnn_node.hxx"

int main(int argc, char **argv) {
    vector<string> arguments = vector<string>(argv, argv + argc);

    CNN_Genome *genome = new CNN_Genome(stoi(arguments[1]));

    ofstream outfile(arguments[2]);
    genome->print_graphviz(outfile);
    outfile.close();
}
