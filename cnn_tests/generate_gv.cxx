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

#include "common/db_conn.hxx"

#include "cnn/exact.hxx"
#include "cnn/cnn_genome.hxx"
#include "cnn/cnn_edge.hxx"
#include "cnn/cnn_node.hxx"

int main(int argc, char **argv) {
    vector<string> arguments = vector<string>(argv, argv + argc);

    set_db_info_filename(arguments[3]);

    CNN_Genome *genome = new CNN_Genome(stoi(arguments[1]));

    ofstream outfile(arguments[2]);
    genome->print_graphviz(outfile);
    outfile.close();
}
