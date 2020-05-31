#include <chrono>

#include <condition_variable>
using std::condition_variable;

#include <iomanip>
using std::setw;

#include <mutex>
using std::mutex;

#include <string>
using std::string;

#include <thread>
using std::thread;

#include <vector>
using std::vector;

#include "common/arguments.hxx"
#include "common/log.hxx"

#include "rnn/rnn_genome.hxx"

#include "time_series/time_series.hxx"


vector<string> arguments;

vector< vector< vector<double> > > testing_inputs;
vector< vector< vector<double> > > testing_outputs;

int main(int argc, char** argv) {
    arguments = vector<string>(argv, argv + argc);

    Log::initialize(arguments);
    Log::set_id("main");

    vector<string> rnn_filenames;
    get_argument_vector(arguments, "--rnn_filenames", true, rnn_filenames);

    double avg_nodes = 0.0;
    double avg_edges = 0.0;
    double avg_rec_edges = 0.0;
    double avg_weights = 0.0;

    for (int32_t i = 0; i < (int32_t)rnn_filenames.size(); i++) {
        Log::info("reading file: %s\n", rnn_filenames[i].c_str());
        RNN_Genome *genome = new RNN_Genome(rnn_filenames[i]);

        int32_t nodes = genome->get_enabled_node_count();
        int32_t edges = genome->get_enabled_edge_count();
        int32_t rec_edges = genome->get_enabled_recurrent_edge_count();
        int32_t weights = genome->get_number_weights();

        Log::info("RNN INFO FOR '%s', nodes: %d, edges: %d, rec: %d, weights: %d\n", rnn_filenames[i].c_str(), nodes, edges, rec_edges, weights);
        Log::info("\t%s\n", genome->print_statistics_header().c_str());
        Log::info("\t%s\n", genome->print_statistics().c_str());

        avg_nodes += nodes;
        avg_edges += edges;
        avg_rec_edges += rec_edges;
        avg_weights += weights;
    }

    avg_nodes /= rnn_filenames.size();
    avg_edges /= rnn_filenames.size();
    avg_rec_edges /= rnn_filenames.size();
    avg_weights /= rnn_filenames.size();

    Log::info("AVG INFO, nodes: %d, edges: %d, rec: %d, weights: %d\n", avg_nodes, avg_edges, avg_rec_edges, avg_weights);

    Log::release_id("main");

    return 0;
}
