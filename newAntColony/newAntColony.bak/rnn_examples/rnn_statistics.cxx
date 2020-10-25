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

#include "rnn/rnn_genome.hxx"

#include "time_series/time_series.hxx"


vector<string> arguments;

vector< vector< vector<double> > > testing_inputs;
vector< vector< vector<double> > > testing_outputs;

int main(int argc, char** argv) {
    arguments = vector<string>(argv, argv + argc);

    vector<string> rnn_filenames;
    get_argument_vector(arguments, "--rnn_filenames", true, rnn_filenames);

    double avg_nodes = 0.0;
    double avg_edges = 0.0;
    double avg_rec_edges = 0.0;
    double avg_weights = 0.0;

    for (int32_t i = 0; i < (int32_t)rnn_filenames.size(); i++) {
        cout << "reading file: " << rnn_filenames[i] << endl;
        RNN_Genome *genome = new RNN_Genome(rnn_filenames[i], true);

        int32_t nodes = genome->get_enabled_node_count();
        int32_t edges = genome->get_enabled_edge_count();
        int32_t rec_edges = genome->get_enabled_recurrent_edge_count();
        int32_t weights = genome->get_number_weights();

        cout << "RNN INFO FOR '" << rnn_filenames[i] << ", nodes: " << nodes << ", edges: " << edges << ", rec: " << rec_edges << ", weights: " << weights << endl;

        avg_nodes += nodes;
        avg_edges += edges;
        avg_rec_edges += rec_edges;
        avg_weights += weights;
    }

    avg_nodes /= rnn_filenames.size();
    avg_edges /= rnn_filenames.size();
    avg_rec_edges /= rnn_filenames.size();
    avg_weights /= rnn_filenames.size();

    cout << "AVG INFO, nodes: " << avg_nodes << ", edges: " << avg_edges << ", rec: " << avg_rec_edges << ", weights: " << avg_weights << endl;

    return 0;
}
