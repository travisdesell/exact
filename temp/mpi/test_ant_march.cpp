#include <string>
using std::string;

#include <vector>
using std::vector;
#include "common/arguments.hxx"
#include "rnn/acnnto.hxx"
#include "time_series/time_series.hxx"

vector<string> arguments;
int main(int argc, char** argv) {
    //vector < double > demo_fitness {0.3, 0.1, 0.07, 0.05, 0.035, 0.03, 0.037, 0.033, 0.029, 0.045, 0.031} ;
    vector < double > demo_fitness {0.3} ;

    int population_size = 100 ;
    int max_genomes = 100 ;
    int bp_iterations = 20;
    double learning_rate = 0.001;
    bool use_high_threshold = false;
    double high_threshold = 0.9;
    bool use_low_threshold = false;
    double low_threshold = 0.1;
    int hidden_layer_nodes = 12 ;
    double pheromone_decay_parameter = 0.1 ;
    double pheromone_update_strength = 0.05 ;
    double pheromone_heuristic = 0.3 ;
    int max_recurrent_depth = 3 ;
    double weight_reg_parameter = 0.2 ;
    string reward_type = "fixed" ;
    // string reward_type = "regularized" ;
    arguments = vector<string>(argv, argv + argc);
    TimeSeriesSets *time_series_sets = NULL;

    string norm = "";
    get_argument(arguments, "--norm", false, norm);

    int number_of_ants= 80 ;
    get_argument(arguments, "--ants", true, number_of_ants);

    int hidden_layers_depth = 3;
    get_argument(arguments, "--hidden_layers_depth", true, hidden_layers_depth);

    string output_directory = "./" ;
    get_argument(arguments, "--output_directory", true, output_directory);

    bool bias_forward_paths = false;
    for ( int i=0; i<argc; i++) {
        if ( string( argv[i] )=="--bias_forward_paths" ) {
            bias_forward_paths = true;
            cout << "Using Forward Paths Bias\n" ;
            break;
        }
    }
    bool bias_edges = false;
    for ( int i=0; i<argc; i++) {
        if ( string( argv[i] )=="--bias_edges" ) {
            bias_edges = true;
            cout << "Using Edges Bias\n" ;
            break;
        }
    }
    bool use_two_ants_types = false;
    for ( int i=0; i<argc; i++) {
        if ( string( argv[i] )=="--use_two_ants_types" ) {
            use_two_ants_types = true;
            cout << "Using Two Ants Types\n" ;
            break;
        }
    }
    bool use_all_jumps = false;
    for ( int i=0; i<argc; i++) {
        if ( string( argv[i] )=="--use_all_jumps" ) {
            use_all_jumps = true;
            cout << "Using All Jumps\n" ;
            break;
        }
    }
    bool use_forward_social_ants = false;
    for ( int i=0; i<argc; i++) {
        if ( string( argv[i] )=="--use_forward_social_ants" ) {
            use_forward_social_ants = true;
            cout << "Using Forward Social Ants\n" ;
            break;
        }
    }
    bool use_backward_social_ants = false;
    for ( int i=0; i<argc; i++) {
        if ( string( argv[i] )=="--use_backward_social_ants" ) {
            use_backward_social_ants = true;
            cout << "Using Backward Social Ants\n" ;
            break;
        }
    }


//exit(0);
    bool use_edge_inherited_weights = false;
    bool use_pheromone_weight_update = false;
    bool use_fitness_weight_update = true;
    double const_phi = 0.0 ;

    time_series_sets = TimeSeriesSets::generate_from_arguments(arguments, true);
    ACNNTO *acnnto = new ACNNTO(population_size, max_genomes, time_series_sets->get_input_parameter_names(), time_series_sets->get_output_parameter_names(), time_series_sets->get_normalize_mins(), time_series_sets->get_normalize_maxs(), bp_iterations, learning_rate, use_high_threshold, high_threshold, use_low_threshold, low_threshold, output_directory, number_of_ants, hidden_layers_depth, hidden_layer_nodes, pheromone_decay_parameter, pheromone_update_strength, pheromone_heuristic, max_recurrent_depth, weight_reg_parameter, reward_type, norm, bias_forward_paths, bias_edges, use_edge_inherited_weights, use_pheromone_weight_update, use_fitness_weight_update, use_two_ants_types, const_phi, use_all_jumps, use_forward_social_ants, use_backward_social_ants );

    int count = 0 ;
    while ( count<population_size ) {
        RNN_Genome *genome = NULL;
        if (use_two_ants_types)
            genome = acnnto->explorer_social_ants_march();
        else
            genome = acnnto->ants_march();

        genome->set_fitness(demo_fitness[count%demo_fitness.size()]);
        genome->set_weights(genome->get_initial_parameters());
        genome->set_best_parameters(genome->get_initial_parameters());
        acnnto->insert_genome(genome);
        delete genome;
        count++;
    }

    return 0;
}
