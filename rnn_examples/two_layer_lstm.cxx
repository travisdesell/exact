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
#include "rnn/bptt.hxx"

#include "time_series/time_series.hxx"

#include "mpi/mpi_ant_colony_optimization_new.hxx"
#include "mpi/mpi_particle_swarm.hxx"
#include "mpi/mpi_differential_evolution.hxx"

#include "asynchronous_algorithms/particle_swarm.hxx"
#include "asynchronous_algorithms/differential_evolution.hxx"


vector< vector< vector<double> > > training_inputs;
vector< vector< vector<double> > > training_outputs;
vector< vector< vector<double> > > test_inputs;
vector< vector< vector<double> > > test_outputs;

RNN_Genome *genome;

double objective_function(const vector<double> &parameters) {
    genome->set_weights(parameters);

    double error = 0.0;

    for (uint32_t i = 0; i < training_inputs.size(); i++) {
        error += genome->prediction_mae(training_inputs[i], training_outputs[i]);
    }

    return -error;
}

double test_objective_function(const vector<double> &parameters) {
    genome->set_weights(parameters);

    double total_error = 0.0;

    for (uint32_t i = 0; i < test_inputs.size(); i++) {
        double error = genome->prediction_mse(test_inputs[i], test_outputs[i]);
        total_error += error;

        cout << "output for series[" << i << "]: " << error << endl;
    }

    return -total_error;
}


int main(int argc, char **argv) {
    MPI_Init(&argc, &argv);

    int rank, max_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &max_rank);

    vector<string> arguments = vector<string>(argv, argv + argc);

    vector<string> training_filenames;
    get_argument_vector(arguments, "--training_filenames", true, training_filenames);

    vector<string> test_filenames;
    get_argument_vector(arguments, "--test_filenames", true, test_filenames);

    vector<TimeSeriesSet*> all_time_series;

    int training_rows = 0;
    vector<TimeSeriesSet*> training_time_series;
    if (rank == 0) cout << "got training time series filenames:" << endl;
    for (uint32_t i = 0; i < training_filenames.size(); i++) {
        if (rank == 0) cout << "\t" << training_filenames[i] << endl;

        TimeSeriesSet *ts = new TimeSeriesSet(training_filenames[i], 1.0);
        training_time_series.push_back( ts );
        all_time_series.push_back( ts );

        //if (rank == 0) cout << "\t\trows: " << ts->get_number_rows() << endl;
        training_rows += ts->get_number_rows();
    }
    if (rank == 0) cout << "number training files: " << training_filenames.size() << ", total rows for training flights: " << training_rows << endl;

    int test_rows = 0;
    vector<TimeSeriesSet*> test_time_series;
    if (rank == 0) cout << "got test time series filenames:" << endl;
    for (uint32_t i = 0; i < test_filenames.size(); i++) {
        if (rank == 0) cout << "\t" << test_filenames[i] << endl;

        TimeSeriesSet *ts = new TimeSeriesSet(test_filenames[i], -1.0);
        test_time_series.push_back( ts );
        all_time_series.push_back( ts );

        //if (rank == 0) cout << "\t\trows: " << ts->get_number_rows() << endl;
        test_rows += ts->get_number_rows();
    }
    if (rank == 0) cout << "number test files: " << test_filenames.size() << ", total rows for test flights: " << test_rows << endl;

    normalize_time_series_sets(all_time_series);

    if (rank == 0) cout << "normalized all time series" << endl;

    vector<string> input_parameter_names;
    input_parameter_names.push_back("indicated_airspeed");
    input_parameter_names.push_back("msl_altitude");
    input_parameter_names.push_back("eng_1_rpm");
    input_parameter_names.push_back("eng_1_fuel_flow");
    input_parameter_names.push_back("eng_1_oil_press");
    input_parameter_names.push_back("eng_1_oil_temp");
    input_parameter_names.push_back("eng_1_cht_1");
    input_parameter_names.push_back("eng_1_cht_2");
    input_parameter_names.push_back("eng_1_cht_3");
    input_parameter_names.push_back("eng_1_cht_4");
    input_parameter_names.push_back("eng_1_egt_1");
    input_parameter_names.push_back("eng_1_egt_2");
    input_parameter_names.push_back("eng_1_egt_3");
    input_parameter_names.push_back("eng_1_egt_4");

    vector<string> output_parameter_names;
    output_parameter_names.push_back("indicated_airspeed");
    /*
    output_parameter_names.push_back("msl_altitude");
    output_parameter_names.push_back("eng_1_rpm");
    output_parameter_names.push_back("eng_1_fuel_flow");
    output_parameter_names.push_back("eng_1_oil_press");
    output_parameter_names.push_back("eng_1_oil_temp");
    output_parameter_names.push_back("eng_1_cht_1");
    output_parameter_names.push_back("eng_1_cht_2");
    output_parameter_names.push_back("eng_1_cht_3");
    output_parameter_names.push_back("eng_1_cht_4");
    output_parameter_names.push_back("eng_1_egt_1");
    output_parameter_names.push_back("eng_1_egt_2");
    output_parameter_names.push_back("eng_1_egt_3");
    output_parameter_names.push_back("eng_1_egt_4");
    */

    int32_t time_offset = 1;

    training_inputs.resize(training_time_series.size());
    training_outputs.resize(training_time_series.size());
    for (uint32_t i = 0; i < training_time_series.size(); i++) {
        training_time_series[i]->export_time_series(training_inputs[i], input_parameter_names, -time_offset);
        training_time_series[i]->export_time_series(training_outputs[i], output_parameter_names, time_offset);
    }

    test_inputs.resize(test_time_series.size());
    test_outputs.resize(test_time_series.size());
    for (uint32_t i = 0; i < test_time_series.size(); i++) {
        test_time_series[i]->export_time_series(test_inputs[i], input_parameter_names, -time_offset);
        test_time_series[i]->export_time_series(test_outputs[i], output_parameter_names, time_offset);
    }


    vector<RNN_Node_Interface*> rnn_nodes;
    vector<RNN_Node_Interface*> layer1_nodes;
    vector<RNN_Node_Interface*> layer2_nodes;
    vector<RNN_Node_Interface*> layer3_nodes;
    vector<RNN_Edge*> rnn_edges;

    int number_columns = training_inputs[0].size();
    int node_innovation_count = 0;
    int edge_innovation_count = 0;

    cout << "number_columns: " << number_columns << endl;

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

    for (int32_t i = 0; i < number_columns; i++) {
        LSTM_Node *node = new LSTM_Node(++node_innovation_count, RNN_HIDDEN_NODE);
        rnn_nodes.push_back(node);
        layer3_nodes.push_back(node);

        for (int32_t j = 0; j < number_columns; j++) {
            rnn_edges.push_back(new RNN_Edge(++edge_innovation_count, layer2_nodes[j], layer3_nodes[i]));
        }
    }

    LSTM_Node *output_node = new LSTM_Node(++node_innovation_count, RNN_OUTPUT_NODE);
    rnn_nodes.push_back(output_node);
    for (int32_t i = 0; i < number_columns; i++) {
        rnn_edges.push_back(new RNN_Edge(++edge_innovation_count, layer3_nodes[i], output_node));
    }

    genome = new RNN_Genome(rnn_nodes, rnn_edges);

    uint32_t number_of_weights = genome->get_number_weights();

    cout << "RNN has " << number_of_weights << " weights." << endl;
    vector<double> min_bound(number_of_weights, -1.0); 
    vector<double> max_bound(number_of_weights, 1.0); 


    string search_type;
    get_argument(arguments, "--search_type", true, search_type);

    if (search_type.compare("bp") == 0) {
        int max_iterations;
        get_argument(arguments, "--max_iterations", true, max_iterations);

        genome->initialize_randomly();

        double learning_rate = 0.010;
        bool nesterov_momentum = true;
        bool adapt_learning_rate = false;
        bool reset_weights = true;
        bool use_high_norm = true;
        bool use_low_norm = true;

        string log_filename = "rnn_log.csv";

        backpropagate(genome, training_inputs, training_outputs, max_iterations, learning_rate, nesterov_momentum, adapt_learning_rate, reset_weights, use_high_norm, use_low_norm, log_filename);

        double mse, mae;

        for (uint32_t i = 0; i < training_inputs.size(); i++) {
            mse = genome->prediction_mse(training_inputs[i], training_outputs[i]);
            mae = genome->prediction_mae(training_inputs[i], training_outputs[i]);

            cout << "series[" << i << "] training MSE:  " << mse << endl;
            cout << "series[" << i << "] training MAE: " << mae << endl;
        }

        for (uint32_t i = 0; i < test_inputs.size(); i++) {
            mse = genome->prediction_mse(test_inputs[i], test_outputs[i]);
            mae = genome->prediction_mae(test_inputs[i], test_outputs[i]);

            cout << "series[" << i << "] test MSE:      " << mse << endl;
            cout << "series[" << i << "] test MAE:     " << mae << endl;
        }

    } else if (search_type.compare("ps") == 0) {
        ParticleSwarm ps(min_bound, max_bound, arguments);
        ps.iterate(objective_function);

        vector<double> parameters = ps.get_global_best();
        genome->set_weights(parameters);

        double mse, mae;

        mse = genome->prediction_mse(training_inputs[0], training_outputs[0]);
        mae = genome->prediction_mae(training_inputs[0], training_outputs[0]);

        cout << "Training Mean Squared error:  " << mse << endl;
        cout << "Training Mean Absolute error: " << mae << endl;

        mse = genome->prediction_mse(test_inputs[0], test_outputs[0]);
        mae = genome->prediction_mae(test_inputs[0], test_outputs[0]);

        cout << "Test Mean Squared error:      " << mse << endl;
        cout << "Test Mean Absolute error:     " << mae << endl;


    } else if (search_type.compare("de") == 0) {
        DifferentialEvolution de(min_bound, max_bound, arguments);
        de.iterate(objective_function);

        vector<double> parameters = de.get_global_best();
        genome->set_weights(parameters);

        double mse, mae;

        mse = genome->prediction_mse(training_inputs[0], training_outputs[0]);
        mae = genome->prediction_mae(training_inputs[0], training_outputs[0]);

        cout << "Training Mean Squared error:  " << mse << endl;
        cout << "Training Mean Absolute error: " << mae << endl;

        mse = genome->prediction_mse(test_inputs[0], test_outputs[0]);
        mae = genome->prediction_mae(test_inputs[0], test_outputs[0]);

        cout << "Test Mean Squared error:      " << mse << endl;
        cout << "Test Mean Absolute error:     " << mae << endl;


    } else if (search_type.compare("ps_mpi") == 0) {
        ParticleSwarmMPI ps(min_bound, max_bound, arguments);
        ps.go(objective_function);

        vector<double> parameters = ps.get_global_best();
        genome->set_weights(parameters);

        double mse, mae;

        mse = genome->prediction_mse(training_inputs[0], training_outputs[0]);
        mae = genome->prediction_mae(training_inputs[0], training_outputs[0]);

        cout << "Training Mean Squared error:  " << mse << endl;
        cout << "Training Mean Absolute error: " << mae << endl;

        mse = genome->prediction_mse(test_inputs[0], test_outputs[0]);
        mae = genome->prediction_mae(test_inputs[0], test_outputs[0]);

        cout << "Test Mean Squared error:      " << mse << endl;
        cout << "Test Mean Absolute error:     " << mae << endl;


    } else if (search_type.compare("de_mpi") == 0) {
        DifferentialEvolutionMPI de(min_bound, max_bound, arguments);
        de.go(objective_function);

        vector<double> parameters = de.get_global_best();
        genome->set_weights(parameters);

        double mse, mae;

        mse = genome->prediction_mse(training_inputs[0], training_outputs[0]);
        mae = genome->prediction_mae(training_inputs[0], training_outputs[0]);

        cout << "Training Mean Squared error:  " << mse << endl;
        cout << "Training Mean Absolute error: " << mae << endl;

        mse = genome->prediction_mse(test_inputs[0], test_outputs[0]);
        mae = genome->prediction_mae(test_inputs[0], test_outputs[0]);

        cout << "Test Mean Squared error:      " << mse << endl;
        cout << "Test Mean Absolute error:     " << mae << endl;


    } else {
        cerr << "Improperly specified search type: '" << search_type.c_str() <<"'" << endl;
        cerr << "Possibilities are:" << endl;
        cerr << "    bp             -       backpropagation" << endl;
        cerr << "    bp_empirical   -       empirical backpropagation" << endl;
        cerr << "    de             -       differential evolution" << endl;
        cerr << "    ps             -       particle swarm optimization" << endl;
        cerr << "    de_mpi         -       MPI parallel differential evolution" << endl;
        cerr << "    ps_mpi         -       MPI parallel particle swarm optimization" << endl;
        exit(1);
    }
}
