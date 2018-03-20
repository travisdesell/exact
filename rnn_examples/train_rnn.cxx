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

#include "rnn/generate_nn.hxx"

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
RNN* rnn;
bool using_dropout;
double dropout_probability;

double objective_function(const vector<double> &parameters) {
    rnn->set_weights(parameters);

    double error = 0.0;

    for (uint32_t i = 0; i < training_inputs.size(); i++) {
        error += rnn->prediction_mae(training_inputs[i], training_outputs[i], false, true, 0.0);
    }

    return -error;
}

double test_objective_function(const vector<double> &parameters) {
    rnn->set_weights(parameters);

    double total_error = 0.0;

    for (uint32_t i = 0; i < test_inputs.size(); i++) {
        double error = rnn->prediction_mse(test_inputs[i], test_outputs[i], false, true, 0.0);
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

    if (argument_exists(arguments, "--normalize")) {
        normalize_time_series_sets(all_time_series);
        if (rank == 0) cout << "normalized all time series" << endl;
    } else {
        if (rank == 0) cout << "not normalizing time series" << endl;
    }


    vector<string> input_parameter_names;
    /*
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
    */
    input_parameter_names.push_back("par1");
    input_parameter_names.push_back("par2");
    input_parameter_names.push_back("par3");
    input_parameter_names.push_back("par4");
    input_parameter_names.push_back("par5");
    input_parameter_names.push_back("par6");
    input_parameter_names.push_back("par7");
    input_parameter_names.push_back("par8");
    input_parameter_names.push_back("par9");
    input_parameter_names.push_back("par10");
    input_parameter_names.push_back("par11");
    input_parameter_names.push_back("par12");
    input_parameter_names.push_back("par13");
    input_parameter_names.push_back("par14");
    input_parameter_names.push_back("vib");

    vector<string> output_parameter_names;
    output_parameter_names.push_back("vib");
    //output_parameter_names.push_back("indicated_airspeed");
    //output_parameter_names.push_back("eng_1_oil_press");
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

    int32_t time_offset = 10;

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


    int number_inputs = training_inputs[0].size();
    int number_outputs = 1;

    cout << "number_inputs: " << number_inputs << ", number_outputs: " << number_outputs << endl;

    string rnn_type;
    get_argument(arguments, "--rnn_type", true, rnn_type);

    RNN_Genome *genome;
    if (rnn_type == "one_layer_lstm") {
        genome = create_lstm(number_inputs, 1, number_inputs, number_outputs);

    } else if (rnn_type == "two_layer_lstm") {
        genome = create_lstm(number_inputs, 1, number_inputs, number_outputs);

    } else if (rnn_type == "one_layer_ff") {
        genome = create_ff(number_inputs, 1, number_inputs, number_outputs);

    } else if (rnn_type == "two_layer_ff") {
        genome = create_ff(number_inputs, 1, number_inputs, number_outputs);

    } else if (rnn_type == "jordan") {
        genome = create_jordan(number_inputs, 1, number_inputs, number_outputs);

    } else if (rnn_type == "elman") {
        genome = create_elman(number_inputs, 1, number_inputs, number_outputs);

    } else {
        cerr << "ERROR: incorrect rnn type" << endl;
        cerr << "Possibilities are:" << endl;
        cerr << "    one_layer_lstm" << endl;
        cerr << "    two_layer_lstm" << endl;
        cerr << "    one_layer_ff" << endl;
        cerr << "    two_layer_ff" << endl;
        exit(1);
    }
    rnn = genome->get_rnn();

    uint32_t number_of_weights = genome->get_number_weights();

    cout << "RNN has " << number_of_weights << " weights." << endl;
    vector<double> min_bound(number_of_weights, -1.0); 
    vector<double> max_bound(number_of_weights, 1.0); 

    vector<double> best_parameters;

    string search_type;
    get_argument(arguments, "--search_type", true, search_type);

    using_dropout = false;

    if (search_type.compare("bp") == 0) {
        int max_iterations;
        get_argument(arguments, "--max_iterations", true, max_iterations);

        genome->initialize_randomly();

        double learning_rate = 0.001;
        get_argument(arguments, "--learning_rate", false, learning_rate);

        bool nesterov_momentum = true;
        bool adapt_learning_rate = false;
        bool reset_weights = false;
        bool use_high_norm = true;
        double high_threshold = 1.0;
        bool use_low_norm = true;
        double low_threshold = 0.05;
        using_dropout = false;
        dropout_probability = 0.5;

        string log_filename = "rnn_log.csv";
        if (argument_exists(arguments, "--log_filename")) {
            get_argument(arguments, "--log_filename", false, log_filename);
        }

        if (argument_exists(arguments, "--stochastic")) {
                genome->backpropagate_stochastic(training_inputs, training_outputs, max_iterations, learning_rate, nesterov_momentum, adapt_learning_rate, reset_weights, use_high_norm, high_threshold, use_low_norm, low_threshold, using_dropout, dropout_probability, log_filename);
        } else {
                genome->backpropagate(training_inputs, training_outputs, max_iterations, learning_rate, nesterov_momentum, adapt_learning_rate, reset_weights, use_high_norm, high_threshold, use_low_norm, low_threshold, using_dropout, dropout_probability, log_filename);
        }
        genome->get_weights(best_parameters);

    } else if (search_type.compare("ps") == 0) {
        ParticleSwarm ps(min_bound, max_bound, arguments);
        ps.iterate(objective_function);

        best_parameters = ps.get_global_best();

    } else if (search_type.compare("de") == 0) {
        DifferentialEvolution de(min_bound, max_bound, arguments);
        de.iterate(objective_function);

        best_parameters = de.get_global_best();

    } else if (search_type.compare("ps_mpi") == 0) {
        ParticleSwarmMPI ps(min_bound, max_bound, arguments);
        ps.go(objective_function);

        best_parameters = ps.get_global_best();

    } else if (search_type.compare("de_mpi") == 0) {
        DifferentialEvolutionMPI de(min_bound, max_bound, arguments);
        de.go(objective_function);

        best_parameters = de.get_global_best();

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

    rnn->set_weights(best_parameters);

    double mse, mae;
    double avg_mse = 0.0, avg_mae = 0.0;

    for (uint32_t i = 0; i < training_inputs.size(); i++) {
        mse = rnn->prediction_mse(training_inputs[i], training_outputs[i], using_dropout, false, dropout_probability);
        mae = rnn->prediction_mae(training_inputs[i], training_outputs[i], using_dropout, false, dropout_probability);

        avg_mse += mse;
        avg_mae += mae;

        cout << "series[" << i << "] training MSE:  " << mse << endl;
        cout << "series[" << i << "] training MAE: " << mae << endl;
    }
    avg_mse /= training_inputs.size();
    avg_mae /= training_inputs.size();
    cout << "average training MSE: " << avg_mse << endl;
    cout << "average training MAE: " << avg_mae << endl;
    cout << endl;

    avg_mse = 0.0;
    avg_mae = 0.0;
    for (uint32_t i = 0; i < test_inputs.size(); i++) {
        mse = rnn->prediction_mse(test_inputs[i], test_outputs[i], using_dropout, false, dropout_probability);
        mae = rnn->prediction_mae(test_inputs[i], test_outputs[i], using_dropout, false, dropout_probability);

        avg_mse += mse;
        avg_mae += mae;

        cout << "series[" << i << "] test MSE:      " << mse << endl;
        cout << "series[" << i << "] test MAE:     " << mae << endl;
    }
    avg_mse /= test_inputs.size();
    avg_mae /= test_inputs.size();
    cout << "average test MSE: " << avg_mse << endl;
    cout << "average test MAE: " << avg_mae << endl;
    cout << endl;

}
