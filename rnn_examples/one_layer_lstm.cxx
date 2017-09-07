#include <chrono>

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

#include "common/arguments.hxx"

#include "rnn/lstm_node.hxx"
#include "rnn/rnn_edge.hxx"
#include "rnn/rnn_genome.hxx"
#include "rnn/rnn_node.hxx"
#include "rnn/rnn_node_interface.hxx"

#include "time_series/time_series.hxx"

//#include "asynchronous_algorithms/particle_swarm.hxx"
//#include "asynchronous_algorithms/differential_evolution.hxx"


vector< vector< vector<double> > > series_data;
vector<double> expected_classes;

RNN_Genome *genome;

double objective_function(const vector<double> &parameters) {
    genome->set_weights(parameters);

    double total_error = 0.0;
    for (uint32_t i = 0; i < series_data.size(); i++) {
        double output = genome->predict(series_data[i], expected_classes[i]);
        double error = 0.0;

        if (expected_classes[i] == 0) {
            error = -1.0 - output;
        } else {
            error = 1.0 - output;
        }

        //cout << "output for series[" << i << "]: " << output << ", expected_class: " << expected_classes[i] << ", error: " << error << endl;
        total_error += error;
    }

    return total_error;
}

int main(int argc, char **argv) {
    vector<string> arguments = vector<string>(argv, argv + argc);

    vector<string> before_filenames;
    get_argument_vector(arguments, "--before_filenames", true, before_filenames);

    vector<string> after_filenames;
    get_argument_vector(arguments, "--after_filenames", true, after_filenames);

    vector<TimeSeriesSet*> all_time_series;

    int before_rows = 0;
    vector<TimeSeriesSet*> before_time_series;
    cout << "got before time series filenames:" << endl;
    for (uint32_t i = 0; i < before_filenames.size(); i++) {
        cout << "\t" << before_filenames[i] << endl;

        TimeSeriesSet *ts = new TimeSeriesSet(before_filenames[i], 1.0);
        before_time_series.push_back( ts );
        all_time_series.push_back( ts );

        cout << "\t\trows: " << ts->get_number_rows() << endl;
        before_rows += ts->get_number_rows();
    }
    cout << "total rows for before flights: " << before_rows << endl;

    int after_rows = 0;
    vector<TimeSeriesSet*> after_time_series;
    cout << "got after time series filenames:" << endl;
    for (uint32_t i = 0; i < after_filenames.size(); i++) {
        cout << "\t" << after_filenames[i] << endl;

        TimeSeriesSet *ts = new TimeSeriesSet(after_filenames[i], 0.0);
        after_time_series.push_back( ts );
        all_time_series.push_back( ts );

        cout << "\t\trows: " << ts->get_number_rows() << endl;
        after_rows += ts->get_number_rows();
    }
    cout << "total rows for after flights: " << after_rows << endl;

    normalize_time_series_sets(all_time_series);

    series_data.clear();
    series_data.resize(all_time_series.size());
    expected_classes.clear();
    expected_classes.resize(all_time_series.size());
    for (uint32_t i = 0; i < all_time_series.size(); i++) {
        all_time_series[i]->export_time_series(series_data[i]);
        expected_classes[i] = all_time_series[i]->get_expected_class();
    }


    /*
    for (uint32_t i = 0; i < series_data.size(); i++) {
        for (uint32_t j = 0; j < series_data[i].size(); j++) {
            for (uint32_t k = 0; k < series_data[i][j].size(); k++) {
                cout << " " << series_data[i][j][k];
            }
            cout << endl;
        }
        cout << endl << endl;
    }
    */

    /*
    for (uint32_t i = 0; i < expected_classes.size(); i++) {
        cout << "expected_classes[" << i << "]: " << expected_classes[i] << endl;
    }
    */

    vector<RNN_Node_Interface*> rnn_nodes;
    vector<RNN_Node_Interface*> layer1_nodes;
    vector<RNN_Node_Interface*> layer2_nodes;
    vector<RNN_Edge*> rnn_edges;

    int number_columns = all_time_series[0]->get_number_columns();
    int node_innovation_count = 0;
    int edge_innovation_count = 0;

    for (int32_t i = 0; i < number_columns; i++) {
        RNN_Node *node = new RNN_Node(++node_innovation_count, RNN_INPUT_NODE);
        rnn_nodes.push_back(node);
        layer1_nodes.push_back(node);
    }

    for (int32_t i = 0; i < number_columns; i++) {
        LSTM_Node *node = new LSTM_Node(++node_innovation_count, RNN_HIDDEN_NODE);
        rnn_nodes.push_back(node);
        layer2_nodes.push_back(node);

        for (int32_t j = 0; j < number_columns; j++) {
            rnn_edges.push_back(new RNN_Edge(++edge_innovation_count, layer1_nodes[j], layer2_nodes[i]));
        }
    }

    LSTM_Node *output_node = new LSTM_Node(++node_innovation_count, RNN_OUTPUT_NODE);
    rnn_nodes.push_back(output_node);
    for (int32_t i = 0; i < number_columns; i++) {
        rnn_edges.push_back(new RNN_Edge(++edge_innovation_count, layer2_nodes[i], output_node));
    }

    genome = new RNN_Genome(rnn_nodes, rnn_edges);

    uint32_t number_of_weights = genome->get_number_weights();

    vector<double> min_bound(number_of_weights, -1.0); 
    vector<double> max_bound(number_of_weights, 1.0); 

    cout << "Input data has " << number_columns << " columns." << endl;
    cout << "RNN has " << number_of_weights << " weights." << endl;

    vector<double> test_parameters(number_of_weights, 0.0);

    unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
    minstd_rand0 generator(seed);
    for (uint32_t i = 0; i < test_parameters.size(); i++) {
        uniform_real_distribution<double> rng(min_bound[i], max_bound[i]);
        test_parameters[i] = rng(generator);

        //cout << "test_parameters[" << i << "]: " << test_parameters[i] << endl;
    }

    double error = objective_function(test_parameters);

    cout << "error was: " << error << endl;

    string search_type;
    get_argument(arguments, "--search_type", true, search_type);
    if (search_type.compare("ps") == 0) {
        ParticleSwarm ps(min_bound, max_bound, arguments);
        ps.iterate(f);

    } else if (search_type.compare("de") == 0) {
        DifferentialEvolution de(min_bound, max_bound, arguments);
        de.iterate(f);


    } else {
        cerr << "Improperly specified search type: '" << search_type.c_str() <<"'" << endl;
        cerr << "Possibilities are:" << endl;
        cerr << "    de     -       differential evolution" << endl;
        cerr << "    ps     -       particle swarm optimization" << endl;
        exit(1);
    }

}
