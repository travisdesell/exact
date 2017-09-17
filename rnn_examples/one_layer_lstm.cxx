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

#include "mpi/mpi_ant_colony_optimization_new.hxx"
#include "mpi/mpi_particle_swarm.hxx"
#include "mpi/mpi_differential_evolution.hxx"

#include "asynchronous_algorithms/particle_swarm.hxx"
#include "asynchronous_algorithms/differential_evolution.hxx"


vector< vector< vector<double> > > series_data;
vector< vector< vector<double> > > test_series_data;
vector<double> expected_classes;
vector<double> test_expected_classes;

RNN_Genome *genome;

double objective_function(const vector<double> &parameters) {
    genome->set_weights(parameters);

    double total_error = 0.0;
    for (uint32_t i = 0; i < series_data.size(); i++) {
        double output = genome->predict(series_data[i], expected_classes[i]);

        double error = fabs(expected_classes[i] - output);

        //cout << "output for series[" << i << "]: " << output << ", expected_class: " << expected_classes[i] << ", error: " << error << endl;
        total_error += error;
    }

    return -total_error;
}

double test_objective_function(const vector<double> &parameters) {
    genome->set_weights(parameters);

    double total_error = 0.0;
    for (uint32_t i = 0; i < test_series_data.size(); i++) {
        double output = genome->predict(test_series_data[i], test_expected_classes[i]);

        double error = fabs(test_expected_classes[i] - output);

        cout << "output for series[" << i << "]: " << output << ", expected_class: " << test_expected_classes[i] << ", error: " << error << endl;
        total_error += error;
    }

    return -total_error;
}


int main(int argc, char **argv) {
    MPI_Init(&argc, &argv);

    int rank, max_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &max_rank);

    vector<string> arguments = vector<string>(argv, argv + argc);

    vector<string> before_filenames;
    get_argument_vector(arguments, "--before_filenames", true, before_filenames);

    vector<string> after_filenames;
    get_argument_vector(arguments, "--after_filenames", true, after_filenames);

    vector<TimeSeriesSet*> all_time_series;

    int before_rows = 0;
    vector<TimeSeriesSet*> before_time_series;
    if (rank == 0) cout << "got before time series filenames:" << endl;
    for (uint32_t i = 0; i < before_filenames.size(); i++) {
        if (rank == 0) cout << "\t" << before_filenames[i] << endl;

        TimeSeriesSet *ts = new TimeSeriesSet(before_filenames[i], 1.0);
        before_time_series.push_back( ts );
        all_time_series.push_back( ts );

        //if (rank == 0) cout << "\t\trows: " << ts->get_number_rows() << endl;
        before_rows += ts->get_number_rows();
    }
    if (rank == 0) cout << "number before files: " << before_filenames.size() << ", total rows for before flights: " << before_rows << endl;

    int after_rows = 0;
    vector<TimeSeriesSet*> after_time_series;
    if (rank == 0) cout << "got after time series filenames:" << endl;
    for (uint32_t i = 0; i < after_filenames.size(); i++) {
        if (rank == 0) cout << "\t" << after_filenames[i] << endl;

        TimeSeriesSet *ts = new TimeSeriesSet(after_filenames[i], -1.0);
        after_time_series.push_back( ts );
        all_time_series.push_back( ts );

        //if (rank == 0) cout << "\t\trows: " << ts->get_number_rows() << endl;
        after_rows += ts->get_number_rows();
    }
    if (rank == 0) cout << "number after files: " << after_filenames.size() << ", total rows for after flights: " << after_rows << endl;

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

    vector<double> min_bound(number_of_weights, -5.0); 
    vector<double> max_bound(number_of_weights, 5.0); 

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

    //double error = objective_function(test_parameters);

    //cout << "error was: " << error << endl;

    string search_type;
    get_argument(arguments, "--search_type", true, search_type);
    if (search_type.compare("test") == 0) {
        string rnn_parameter_filename;
        get_argument(arguments, "--rnn_parameter_file", true, rnn_parameter_filename);

        vector<double> parameters;

        ifstream parameter_file(rnn_parameter_filename);
        string parameter;
        while (getline(parameter_file, parameter, ',')) {
                cout << "got parameter: " << parameter << endl;
                parameters.push_back(stod(parameter));
        }

        double error = objective_function(parameters);
        cout << "total_error: " << error << endl;

        cout << "loading test time series." << endl;

        vector<string> before_test_filenames;
        get_argument_vector(arguments, "--before_test_filenames", true, before_test_filenames);

        vector<string> after_test_filenames;
        get_argument_vector(arguments, "--after_test_filenames", true, after_test_filenames);


        vector<TimeSeriesSet*> all_test_time_series;

        int before_rows = 0;
        vector<TimeSeriesSet*> before_test_time_series;
        if (rank == 0) cout << "got before time series filenames:" << endl;
        for (uint32_t i = 0; i < before_test_filenames.size(); i++) {
            if (rank == 0) cout << "\t" << before_test_filenames[i] << endl;

            TimeSeriesSet *ts = new TimeSeriesSet(before_test_filenames[i], 1.0);
            before_test_time_series.push_back( ts );
            all_test_time_series.push_back( ts );

            //if (rank == 0) cout << "\t\trows: " << ts->get_number_rows() << endl;
            before_rows += ts->get_number_rows();
        }
        if (rank == 0) cout << "number before test files: " << before_test_filenames.size() << ", total rows for before test flights: " << before_rows << endl;

        int after_rows = 0;
        vector<TimeSeriesSet*> after_test_time_series;
        if (rank == 0) cout << "got after time series filenames:" << endl;
        for (uint32_t i = 0; i < after_test_filenames.size(); i++) {
            if (rank == 0) cout << "\t" << after_test_filenames[i] << endl;

            TimeSeriesSet *ts = new TimeSeriesSet(after_test_filenames[i], -1.0);
            after_test_time_series.push_back( ts );
            all_test_time_series.push_back( ts );

            //if (rank == 0) cout << "\t\trows: " << ts->get_number_rows() << endl;
            after_rows += ts->get_number_rows();
        }
        if (rank == 0) cout << "number after test files: " << after_test_filenames.size() << ", total rows for after test flights: " << after_rows << endl;

        normalize_time_series_sets(all_test_time_series);

        test_series_data.clear();
        test_series_data.resize(all_test_time_series.size());
        test_expected_classes.clear();
        test_expected_classes.resize(all_test_time_series.size());
        for (uint32_t i = 0; i < all_test_time_series.size(); i++) {
            all_test_time_series[i]->export_time_series(test_series_data[i]);
            test_expected_classes[i] = all_test_time_series[i]->get_expected_class();
        }

        double test_error = test_objective_function(parameters);
        cout << "total_test_error: " << test_error << endl;


    } else if (search_type.compare("ps") == 0) {
        ParticleSwarm ps(min_bound, max_bound, arguments);
        ps.iterate(objective_function);

    } else if (search_type.compare("de") == 0) {
        DifferentialEvolution de(min_bound, max_bound, arguments);
        de.iterate(objective_function);

    } else if (search_type.compare("ps_mpi") == 0) {
        ParticleSwarmMPI ps(min_bound, max_bound, arguments);
        ps.go(objective_function);

    } else if (search_type.compare("de_mpi") == 0) {
        DifferentialEvolutionMPI de(min_bound, max_bound, arguments);
        de.go(objective_function);

    } else {
        cerr << "Improperly specified search type: '" << search_type.c_str() <<"'" << endl;
        cerr << "Possibilities are:" << endl;
        cerr << "    de     -       differential evolution" << endl;
        cerr << "    ps     -       particle swarm optimization" << endl;
        cerr << "    de_mpi -       MPI parallel differential evolution" << endl;
        cerr << "    ps_mpi -       MPI parallel particle swarm optimization" << endl;
        exit(1);
    }
}
