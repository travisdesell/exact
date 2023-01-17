#include "rnn/generate_nn.hxx"

#include <cassert>

#include <utility>

#include <string>
using std::string;

#include <vector>
using std::vector;

/*
 * node_kind is the type of memory cell (e.g. LSTM, UGRNN)
 * innovation_counter - reference to an integer used to keep track if innovation numbers. it will be incremented once.
 */
RNN_Node_Interface *create_hidden_node(int32_t node_kind, int32_t &innovation_counter, double depth) {
    switch (node_kind) {
        case SIMPLE_NODE:
            return new RNN_Node(++innovation_counter, HIDDEN_LAYER, depth, SIMPLE_NODE);
        case JORDAN_NODE:
            return new RNN_Node(++innovation_counter, HIDDEN_LAYER, depth, JORDAN_NODE);
        case ELMAN_NODE:
            return new RNN_Node(++innovation_counter, HIDDEN_LAYER, depth, ELMAN_NODE);
        case UGRNN_NODE:
            return new UGRNN_Node(++innovation_counter, HIDDEN_LAYER, depth);
        case MGU_NODE:
            return new MGU_Node(++innovation_counter, HIDDEN_LAYER, depth);
        case GRU_NODE:
            return new GRU_Node(++innovation_counter, HIDDEN_LAYER, depth);
        case DELTA_NODE:
            return new Delta_Node(++innovation_counter, HIDDEN_LAYER, depth);
        case LSTM_NODE:
            return new LSTM_Node(++innovation_counter, HIDDEN_LAYER, depth);
        case ENARC_NODE:
            return new ENARC_Node(++innovation_counter, HIDDEN_LAYER, depth);
        case ENAS_DAG_NODE:
            return new ENAS_DAG_Node(++innovation_counter, HIDDEN_LAYER, depth);
        case RANDOM_DAG_NODE:
            return new RANDOM_DAG_Node(++innovation_counter, HIDDEN_LAYER, depth);
        case DNAS_NODE:
            Log::fatal("You shouldn't be creating DNAS nodes using generate_nn::create_hidden_node.\n");
            exit(1);
    }
}

DNASNode *create_dnas_node(int32_t &innovation_counter, double depth, const vector<int32_t> &node_types) {
    vector<RNN_Node_Interface *> nodes(node_types.size());

    int i = 0;
    for (auto node_type : node_types)
        nodes[i++] = create_hidden_node(node_type, innovation_counter, depth);

    DNASNode *n = new DNASNode(std::move(nodes), ++innovation_counter, HIDDEN_LAYER, depth);
    return n;
}

RNN_Genome *create_nn(const vector<string> &input_parameter_names, int32_t number_hidden_layers, int32_t number_hidden_nodes, const vector<string> &output_parameter_names, int32_t max_recurrent_depth, std::function<RNN_Node_Interface *(int32_t &, double)> make_node, WeightRules *weight_rules) {
     Log::debug("creating feed forward network with inputs: %d, hidden: %dx%d, outputs: %d, max recurrent depth: %d\n", input_parameter_names.size(), number_hidden_layers, number_hidden_nodes, output_parameter_names.size(), max_recurrent_depth);
    vector<RNN_Node_Interface*> rnn_nodes;
    vector< vector<RNN_Node_Interface*> > layer_nodes(2 + number_hidden_layers);
    vector<RNN_Edge*> rnn_edges;
    vector<RNN_Recurrent_Edge*> recurrent_edges;

    int32_t node_innovation_count = 0;
    int32_t edge_innovation_count = 0;
    int32_t current_layer = 0;

    for (int32_t i = 0; i < (int32_t)input_parameter_names.size(); i++) {
        RNN_Node *node = new RNN_Node(++node_innovation_count, INPUT_LAYER, current_layer, SIMPLE_NODE, input_parameter_names[i]);
        rnn_nodes.push_back(node);
        layer_nodes[current_layer].push_back(node);
    }
    current_layer++;

    for (int32_t i = 0; i < (int32_t)number_hidden_layers; i++) {
        for (int32_t j = 0; j < (int32_t)number_hidden_nodes; j++) {
            RNN_Node_Interface *node = make_node(node_innovation_count, current_layer);
            rnn_nodes.push_back(node);
            layer_nodes[current_layer].push_back(node);

            for (int32_t k = 0; k < (int32_t)layer_nodes[current_layer - 1].size(); k++) {
                rnn_edges.push_back(new RNN_Edge(++edge_innovation_count, layer_nodes[current_layer - 1][k], node));

                for (int32_t d = 1; d <= max_recurrent_depth; d++) {
                    recurrent_edges.push_back(new RNN_Recurrent_Edge(++edge_innovation_count, d, layer_nodes[current_layer - 1][k], node));
                }
            }
        }
        current_layer++;
    }

    for (int32_t i = 0; i < (int32_t)output_parameter_names.size(); i++) {
        RNN_Node *output_node = new RNN_Node(++node_innovation_count, OUTPUT_LAYER, current_layer, SIMPLE_NODE, output_parameter_names[i]);
        rnn_nodes.push_back(output_node);

        for (int32_t k = 0; k < (int32_t)layer_nodes[current_layer - 1].size(); k++) {
            rnn_edges.push_back(new RNN_Edge(++edge_innovation_count, layer_nodes[current_layer - 1][k], output_node));

            for (int32_t d = 1; d <= max_recurrent_depth; d++) {
                recurrent_edges.push_back(new RNN_Recurrent_Edge(++edge_innovation_count, d, layer_nodes[current_layer - 1][k], output_node));
            }
        }
    }

    RNN_Genome *genome = new RNN_Genome(rnn_nodes, rnn_edges, recurrent_edges, weight_rules);
    genome->set_parameter_names(input_parameter_names, output_parameter_names);
    return genome;
}


RNN_Genome *create_dnas_nn(const vector<string> &input_parameter_names, int32_t number_hidden_layers, int32_t number_hidden_nodes, const vector<string> &output_parameter_names, int32_t max_recurrent_depth, vector<int32_t> &node_types, WeightRules *weight_rules) {
    auto f = [&](int32_t &innovation_counter, double depth) -> RNN_Node_Interface * {
        return create_dnas_node(innovation_counter, depth, node_types);
    };

    return create_nn(input_parameter_names, number_hidden_layers, number_hidden_nodes, output_parameter_names, max_recurrent_depth, f, weight_rules);
}

RNN_Genome* get_seed_genome(const vector<string> &arguments, TimeSeriesSets *time_series_sets, WeightRules *weight_rules) {
    Log::info("Generating seed genome\n");
    RNN_Genome *seed_genome = NULL;
    string genome_file_name = "";
    string transfer_learning_version = "";
    if (get_argument(arguments, "--genome_bin", false, genome_file_name)) {
        //TODO: update this part and hopefully related arguments could be processed more organized 
        Log::info("Getting genome bin from arguments, and the seed genome is not minimal \n");
        int32_t min_recurrent_depth = 1;
        get_argument(arguments, "--min_recurrent_depth", false, min_recurrent_depth);
        int32_t max_recurrent_depth = 10;
        get_argument(arguments, "--max_recurrent_depth", false, max_recurrent_depth);
        
        Log::info("genome path is %s\n", genome_file_name.c_str());
        bool epigenetic_weights = argument_exists(arguments, "--epigenetic_weights");
        Log::info("Using epigeneratic weights is set to: %s \n", epigenetic_weights ? "True" : "False");
        seed_genome = new RNN_Genome(genome_file_name);
        seed_genome->set_normalize_bounds(time_series_sets->get_normalize_type(), time_series_sets->get_normalize_mins(), time_series_sets->get_normalize_maxs(), time_series_sets->get_normalize_avgs(), time_series_sets->get_normalize_std_devs());

        get_argument(arguments, "--transfer_learning_version", true, transfer_learning_version);
        Log::info("Transfer learning version is set to %s\n", transfer_learning_version.c_str());
        // TODO: rewrite the transfer_to() function, could take advantage of the GenomeProperty class and TimeSeriesSets class
        Log::info("Transfering seed genome\n");
        seed_genome->transfer_to(time_series_sets->get_input_parameter_names(), time_series_sets->get_output_parameter_names(), transfer_learning_version, epigenetic_weights, min_recurrent_depth, max_recurrent_depth);
        Log::info("Finished transfering seed genome\n");
    } else {
        if (seed_genome == NULL) {
            seed_genome = create_ff(time_series_sets->get_input_parameter_names(), 0, 0, time_series_sets->get_output_parameter_names(), 0, weight_rules);
            seed_genome->initialize_randomly();
            Log::info("Generated seed genome, seed genome is minimal\n");
        } 
    }

    return seed_genome;
}
