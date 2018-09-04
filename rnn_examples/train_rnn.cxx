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

#include "mpi/mpi_particle_swarm.hxx"
#include "mpi/mpi_differential_evolution.hxx"

#include "asynchronous_algorithms/particle_swarm.hxx"
#include "asynchronous_algorithms/differential_evolution.hxx"


vector< vector< vector<double> > > training_inputs;
vector< vector< vector<double> > > training_outputs;
vector< vector< vector<double> > > validation_inputs;
vector< vector< vector<double> > > validation_outputs;

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

double validation_objective_function(const vector<double> &parameters) {
    rnn->set_weights(parameters);

    double total_error = 0.0;

    for (uint32_t i = 0; i < validation_inputs.size(); i++) {
        double error = rnn->prediction_mse(validation_inputs[i], validation_outputs[i], false, true, 0.0);
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

    vector<string> validation_filenames;
    get_argument_vector(arguments, "--validation_filenames", true, validation_filenames);

    int32_t time_offset = 1;
    get_argument(arguments, "--time_offset", true, time_offset);

    bool normalize = argument_exists(arguments, "--normalize");


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

    /*
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
    */

    input_parameter_names.push_back("Coyote-GROSS_GENERATOR_OUTPUT");
    input_parameter_names.push_back("Coyote-Net_Unit_Generation");
    input_parameter_names.push_back("Cyclone_-CYC__CONDITIONER_INLET_TEMP");
    input_parameter_names.push_back("Cyclone_-CYC__CONDITIONER_OUTLET_TEMP");
    input_parameter_names.push_back("Cyclone_-LIGNITE_FEEDER__RATE");
    input_parameter_names.push_back("Cyclone_-CYC__TOTAL_COMB_AIR_FLOW");
    input_parameter_names.push_back("Cyclone_-_MAIN_OIL_FLOW");
    input_parameter_names.push_back("Cyclone_-CYCLONE__MAIN_FLM_INT");


    vector<string> output_parameter_names;
    output_parameter_names.push_back("Cyclone_-_MAIN_OIL_FLOW");



    //output_parameter_names.push_back("vib");
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

    vector<TimeSeriesSet*> training_time_series, validation_time_series;
    load_time_series(training_filenames, validation_filenames, normalize, training_time_series, validation_time_series);

    export_time_series(training_time_series, input_parameter_names, output_parameter_names, time_offset, training_inputs, training_outputs);
    export_time_series(validation_time_series, input_parameter_names, output_parameter_names, time_offset, validation_inputs, validation_outputs);

    int number_inputs = training_inputs[0].size();
    int number_outputs = training_outputs[0].size();

    cout << "number_inputs: " << number_inputs << ", number_outputs: " << number_outputs << endl;

    string rnn_type;
    get_argument(arguments, "--rnn_type", true, rnn_type);

    int32_t max_recurrent_depth;
    get_argument(arguments, "--max_recurrent_depth", true, max_recurrent_depth);

    RNN_Genome *genome;
    if (rnn_type == "one_layer_lstm") {
        genome = create_lstm(number_inputs, 1, number_inputs, number_outputs, max_recurrent_depth);

    } else if (rnn_type == "two_layer_lstm") {
        genome = create_lstm(number_inputs, 1, number_inputs, number_outputs, max_recurrent_depth);

    } else if (rnn_type == "one_layer_ff") {
        genome = create_ff(number_inputs, 1, number_inputs, number_outputs, max_recurrent_depth);

    } else if (rnn_type == "two_layer_ff") {
        genome = create_ff(number_inputs, 1, number_inputs, number_outputs, max_recurrent_depth);

    } else if (rnn_type == "jordan") {
        genome = create_jordan(number_inputs, 1, number_inputs, number_outputs, max_recurrent_depth);

    } else if (rnn_type == "elman") {
        genome = create_elman(number_inputs, 1, number_inputs, number_outputs, max_recurrent_depth);

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
        genome->initialize_randomly();

        int bp_iterations;
        get_argument(arguments, "--bp_iterations", true, bp_iterations);
        genome->set_bp_iterations(bp_iterations);

        double learning_rate = 0.001;
        get_argument(arguments, "--learning_rate", false, learning_rate);

        genome->set_learning_rate(learning_rate);
        genome->set_adapt_learning_rate(false);
        genome->set_nesterov_momentum(true);
        genome->set_reset_weights(false);
        genome->enable_high_threshold(1.0);
        genome->enable_low_threshold(0.05);
        genome->disable_dropout();

        if (argument_exists(arguments, "--log_filename")) {
            string log_filename;
            get_argument(arguments, "--log_filename", false, log_filename);
            genome->set_log_filename(log_filename);
        }

        if (argument_exists(arguments, "--stochastic")) {
                genome->backpropagate_stochastic(training_inputs, training_outputs, validation_inputs, validation_outputs);
        } else {
                genome->backpropagate(training_inputs, training_outputs, validation_inputs, validation_outputs);
        }
        genome->get_weights(best_parameters);
        cout << "best validation error: " << genome->get_validation_error() << endl;

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
    cout << "TRAINING ERRORS:" << endl;
    genome->get_mse(best_parameters, training_inputs, training_outputs, true);
    cout << endl;
    genome->get_mae(best_parameters, training_inputs, training_outputs, true);
    cout << endl;

    cout << "TEST ERRORS:" << endl;
    genome->get_mse(best_parameters, validation_inputs, validation_outputs, true);
    cout << endl;
    genome->get_mae(best_parameters, validation_inputs, validation_outputs, true);
    cout << endl;
}
