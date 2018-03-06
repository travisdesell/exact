#include <chrono>

#include <fstream>
using std::getline;
using std::ifstream;

#include <iostream>
using std::cout;
using std::cerr;
using std::endl;

#include <random>
using std::minstd_rand0;
using std::uniform_real_distribution;

#include <string>
using std::string;

#include <vector>
using std::vector;

#include "mpi.h"

#include "common/arguments.hxx"

#include "rnn/lstm_node.hxx"
#include "rnn/rnn_edge.hxx"
#include "rnn/rnn_genome.hxx"
#include "rnn/rnn_node.hxx"
#include "rnn/rnn_node_interface.hxx"

#include "time_series/time_series.hxx"

int main(int argc, char **argv) {
    vector<string> arguments = vector<string>(argv, argv + argc);

    string filename;

    get_argument(arguments, "--filename", true, filename);

    TimeSeriesSet *ts = new TimeSeriesSet(filename, 1.0);

    //if (rank == 0) cout << "\t\trows: " << ts->get_number_rows() << endl;
    cout << "parsed file '" << filename << "' with " << ts->get_number_rows() << " rows." << endl;

    vector<string> fields = ts->get_fields();
    for (uint32_t i = 0; i < fields.size(); i++) {
        string field = fields[i];
        ts->normalize_min_max(field, ts->get_min(field), ts->get_max(field));
    }

    vector<RNN_Node_Interface*> rnn_nodes;
    RNN_Node_Interface* input_node;
    RNN_Node_Interface* output_node;
    vector<RNN_Edge*> rnn_edges;

    int number_columns = ts->get_number_columns();
    int node_innovation_count = 0;
    int edge_innovation_count = 0;

    //only one input node
    input_node = new RNN_Node(++node_innovation_count, RNN_INPUT_NODE);
    rnn_nodes.push_back(input_node);

    output_node = new LSTM_Node(++node_innovation_count, RNN_OUTPUT_NODE);
    rnn_nodes.push_back(output_node);
    rnn_edges.push_back(new RNN_Edge(++edge_innovation_count, input_node, output_node));

    RNN_Genome *genome = new RNN_Genome(rnn_nodes, rnn_edges);

    uint32_t number_of_weights = genome->get_number_weights();

    vector<double> min_bound(number_of_weights, -1.0); 
    vector<double> max_bound(number_of_weights, 1.0); 

    cout << "Input data has " << number_columns << " columns, but we are only using 1." << endl;
    cout << "RNN has " << number_of_weights << " weights." << endl;

    vector<double> test_parameters(number_of_weights, 0.0);

    unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
    minstd_rand0 generator(seed);
    for (uint32_t i = 0; i < test_parameters.size(); i++) {
        uniform_real_distribution<double> rng(min_bound[i], max_bound[i]);
        test_parameters[i] = rng(generator);
    }

    genome->set_weights(test_parameters);

    vector<string> requested_fields;
    requested_fields.push_back("indicated_airspeed");

    vector< vector<double> > data;
    ts->export_time_series(requested_fields, data);

    double mean_squared_error = genome->predict(data, requested_fields);

    cout << "genome predicted:";
    for (uint32_t i = 0; i < requested_fields.size(); i++) {
        cout << "    '" << requested_fields[i] << "'" << endl;
    }
    cout << "with MSE: " << mean_squared_error << endl;

}
