#include <algorithm>
using std::sort;
using std::upper_bound;

#include <cmath>

#include <fstream>
using std::istream;
using std::ifstream;
using std::ostream;
using std::ofstream;

#include <iomanip>
using std::setw;
using std::setfill;

#include <ios>
using std::hex;
using std::ios;

#include <iostream>
using std::endl;

#include <random>
using std::minstd_rand0;
using std::uniform_real_distribution;

#include <thread>
using std::thread;

#include <random>
using std::minstd_rand0;
using std::uniform_real_distribution;

#include <sstream>
using std::istringstream;
using std::ostringstream;

#include <string>
using std::string;
using std::to_string;

#include <vector>
using std::vector;

#include "common/random.hxx"
#include "common/color_table.hxx"
#include "common/log.hxx"

#include "rnn.hxx"
#include "rnn_node.hxx"
#include "lstm_node.hxx"
#include "gru_node.hxx"
#include "delta_node.hxx"
#include "ugrnn_node.hxx"
#include "mgu_node.hxx"
#include "rnn_genome.hxx"
#include "distribution.hxx"

string parse_fitness(double fitness) {
    if (fitness == EXAMM_MAX_DOUBLE) {
        return "UNEVAL";
    } else {
        return to_string(fitness);
    }
}


void RNN_Genome::sort_nodes_by_depth() {
    sort(nodes.begin(), nodes.end(), sort_RNN_Nodes_by_depth());
}

void RNN_Genome::sort_edges_by_depth() {
    sort(edges.begin(), edges.end(), sort_RNN_Edges_by_depth());
}

RNN_Genome::RNN_Genome(vector<RNN_Node_Interface*> &_nodes, vector<RNN_Edge*> &_edges) {
    generation_id = -1;
    group_id = -1;

    best_validation_mse = EXAMM_MAX_DOUBLE;
    best_validation_mae = EXAMM_MAX_DOUBLE;

    nodes = _nodes;
    edges = _edges;

    sort_nodes_by_depth();
    sort_edges_by_depth();

    //set default values
    bp_iterations = 20000;
    learning_rate = 0.001;
    adapt_learning_rate = false;
    use_nesterov_momentum = true;
    //use_nesterov_momentum = false;
    use_reset_weights = false;

    use_high_norm = true;
    high_threshold = 1.0;
    use_low_norm = true;
    low_threshold = 0.05;

    use_dropout = false;
    dropout_probability = 0.5;

    log_filename = "";

    uint16_t seed = std::chrono::system_clock::now().time_since_epoch().count();
    generator = minstd_rand0(seed);
    rng_0_1 = uniform_real_distribution<double>(0.0, 1.0);

    assign_reachability();
}

RNN_Genome::RNN_Genome(vector<RNN_Node_Interface*> &_nodes, vector<RNN_Edge*> &_edges, vector<RNN_Recurrent_Edge*> &_recurrent_edges) : RNN_Genome(_nodes, _edges) {
    recurrent_edges = _recurrent_edges;
    assign_reachability();
}

RNN_Genome::RNN_Genome(vector<RNN_Node_Interface*> &_nodes, vector<RNN_Edge*> &_edges, vector<RNN_Recurrent_Edge*> &_recurrent_edges, uint16_t seed) : RNN_Genome(_nodes, _edges) {
    recurrent_edges = _recurrent_edges;

    generator = minstd_rand0(seed);
    assign_reachability();
}

void RNN_Genome::set_parameter_names(const vector<string> &_input_parameter_names, const vector<string> &_output_parameter_names) {
    input_parameter_names = _input_parameter_names;
    output_parameter_names = _output_parameter_names;
}


RNN_Genome* RNN_Genome::copy() {
    vector<RNN_Node_Interface*> node_copies;
    vector<RNN_Edge*> edge_copies;
    vector<RNN_Recurrent_Edge*> recurrent_edge_copies;

    for (uint32_t i = 0; i < nodes.size(); i++) {
        node_copies.push_back( nodes[i]->copy() );
    }

    for (uint32_t i = 0; i < edges.size(); i++) {
        edge_copies.push_back( edges[i]->copy(node_copies) );
    }

    for (uint32_t i = 0; i < recurrent_edges.size(); i++) {
        recurrent_edge_copies.push_back( recurrent_edges[i]->copy(node_copies) );
    }

    RNN_Genome *other = new RNN_Genome(node_copies, edge_copies, recurrent_edge_copies);

    other->group_id = group_id;
    other->bp_iterations = bp_iterations;
    other->learning_rate = learning_rate;
    other->adapt_learning_rate = adapt_learning_rate;
    other->use_nesterov_momentum = use_nesterov_momentum;
    other->use_reset_weights = use_reset_weights;

    other->use_high_norm = use_high_norm;
    other->high_threshold = high_threshold;
    other->use_low_norm = use_low_norm;
    other->low_threshold = low_threshold;

    other->use_dropout = use_dropout;
    other->dropout_probability = dropout_probability;

    other->log_filename = log_filename;

    other->generated_by_map = generated_by_map;

    other->initial_parameters = initial_parameters;

    other->best_validation_mse = best_validation_mse;
    other->best_validation_mae = best_validation_mae;
    other->best_parameters = best_parameters;

    other->input_parameter_names = input_parameter_names;
    other->output_parameter_names = output_parameter_names;

    other->normalize_mins = normalize_mins;
    other->normalize_maxs = normalize_maxs;

    other->assign_reachability();

    return other;
}


RNN_Genome::~RNN_Genome() {
    RNN_Node_Interface *node;

    while (nodes.size() > 0) {
        node = nodes.back();
        nodes.pop_back();
        delete node;
    }

    RNN_Edge *edge;

    while (edges.size() > 0) {
        edge = edges.back();
        edges.pop_back();
        delete edge;
    }

    RNN_Recurrent_Edge *recurrent_edge;

    while (recurrent_edges.size() > 0) {
        recurrent_edge = recurrent_edges.back();
        recurrent_edges.pop_back();
        delete recurrent_edge;
    }
}

string RNN_Genome::print_statistics_header() {
    ostringstream oss;

    oss << std::left
        << setw(12) << "MSE"
        << setw(12) << "MAE"
        << setw(12) << "Edges"
        << setw(12) << "Rec Edges"
        << setw(12) << "Simple"
        << setw(12) << "Jordan"
        << setw(12) << "Elman"
        << setw(12) << "UGRNN"
        << setw(12) << "MGU"
        << setw(12) << "GRU"
        << setw(12) << "Delta"
        << setw(12) << "LSTM"
        << setw(12) << "Total"
        << "Generated";

    return oss.str();
}

string RNN_Genome::print_statistics() {
    ostringstream oss;
    oss << std::left
        << setw(12) << parse_fitness(best_validation_mse)
        << setw(12) << parse_fitness(best_validation_mae)
        << setw(12) << get_edge_count_str(false)
        << setw(12) << get_edge_count_str(true)
        << setw(12) << get_node_count_str(SIMPLE_NODE)
        << setw(12) << get_node_count_str(JORDAN_NODE)
        << setw(12) << get_node_count_str(ELMAN_NODE)
        << setw(12) << get_node_count_str(UGRNN_NODE)
        << setw(12) << get_node_count_str(MGU_NODE)
        << setw(12) << get_node_count_str(GRU_NODE)
        << setw(12) << get_node_count_str(DELTA_NODE)
        << setw(12) << get_node_count_str(LSTM_NODE)
        << setw(12) << get_node_count_str(-1)  //-1 does all nodes
        << generated_by_string();
    return oss.str();
}

double RNN_Genome::get_avg_recurrent_depth() const {
    int32_t count = 0;
    double average = 0.0;
    for (int32_t i = 0; i < recurrent_edges.size(); i++) {
        if (recurrent_edges[i]->is_reachable()) {
            average += recurrent_edges[i]->get_recurrent_depth();
            count++;
        }
    }

    //in case there are no recurrent edges
    if (count == 0) return 0;

    return average / count;
}

string RNN_Genome::get_edge_count_str(bool recurrent) {
    ostringstream oss;
    if (recurrent) {
        oss << get_enabled_recurrent_edge_count() << " (" << recurrent_edges.size() << ")";
    } else {
        oss << get_enabled_edge_count() << " (" << edges.size() << ")";
    }
    return oss.str();
}

string RNN_Genome::get_node_count_str(int node_type) {
    ostringstream oss;
    if (node_type < 0) {
        oss << get_enabled_node_count() << " (" << get_node_count() << ")";
    } else {
        int enabled_nodes = get_enabled_node_count(node_type);
        int total_nodes = get_node_count(node_type);

        if (total_nodes > 0) oss << enabled_nodes << " (" << total_nodes << ")";
    }
    return oss.str();
}


int RNN_Genome::get_enabled_node_count() {
    int32_t count = 0;

    for (int32_t i = 0; i < nodes.size(); i++) {
        if (nodes[i]->enabled) count++;
    }

    return count;
}

int RNN_Genome::get_enabled_node_count(int node_type) {
    int32_t count = 0;

    for (int32_t i = 0; i < nodes.size(); i++) {
        if (nodes[i]->enabled && nodes[i]->layer_type == HIDDEN_LAYER && nodes[i]->node_type == node_type) count++;
    }

    return count;
}

int RNN_Genome::get_node_count() {
    return nodes.size();
}


int RNN_Genome::get_node_count(int node_type) {
    int32_t count = 0;

    for (int32_t i = 0; i < nodes.size(); i++) {
        if (nodes[i]->node_type == node_type) count++;
    }

    return count;
}


void RNN_Genome::clear_generated_by() {
    generated_by_map.clear();
}

void RNN_Genome::update_generation_map(map<string, int32_t> &generation_map) {
    for (auto i = generated_by_map.begin(); i != generated_by_map.end(); i++) {
        generation_map[i->first] += i->second;
    }

}

string RNN_Genome::generated_by_string() {
    ostringstream oss;
    oss << "[";
    bool first = true;
    for (auto i = generated_by_map.begin(); i != generated_by_map.end(); i++) {
        if (!first) oss << ", ";
        oss << i->first << ":" << i->second;
        first = false;
    }
    oss << "]";

    return oss.str();
}

vector<string> RNN_Genome::get_input_parameter_names() const {
    return input_parameter_names;
}

vector<string> RNN_Genome::get_output_parameter_names() const {
    return output_parameter_names;
}

void RNN_Genome::set_normalize_bounds(const map<string,double> &_normalize_mins, const map<string,double> &_normalize_maxs) {
    normalize_mins = _normalize_mins;
    normalize_maxs = _normalize_maxs;
}

map<string,double> RNN_Genome::get_normalize_mins() const {
    return normalize_mins;
}

map<string,double> RNN_Genome::get_normalize_maxs() const {
    return normalize_maxs;
}


int32_t RNN_Genome::get_group_id() const {
    return group_id;
}

void RNN_Genome::set_group_id(int32_t _group_id) {
    group_id = _group_id;
}

int32_t RNN_Genome::get_enabled_edge_count() {
    int32_t count = 0;

    for (int32_t i = 0; i < edges.size(); i++) {
        if (edges[i]->enabled) count++;
    }

    return count;
}

int32_t RNN_Genome::get_enabled_recurrent_edge_count() {
    int32_t count = 0;

    for (int32_t i = 0; i < recurrent_edges.size(); i++) {
        if (recurrent_edges[i]->enabled) count++;
    }

    return count;
}

void RNN_Genome::set_bp_iterations(int32_t _bp_iterations) {
    bp_iterations = _bp_iterations;
}

int32_t RNN_Genome::get_bp_iterations() {
    return bp_iterations;
}

void RNN_Genome::set_learning_rate(double _learning_rate) {
    learning_rate = _learning_rate;
}

void RNN_Genome::set_adapt_learning_rate(bool _adapt_learning_rate) {
    adapt_learning_rate = _adapt_learning_rate;
}

void RNN_Genome::set_nesterov_momentum(bool _use_nesterov_momentum) {
    use_nesterov_momentum = _use_nesterov_momentum;
}

void RNN_Genome::set_reset_weights(bool _use_reset_weights) {
    use_reset_weights = _use_reset_weights;
}


void RNN_Genome::disable_high_threshold() {
    use_high_norm = false;
}

void RNN_Genome::enable_high_threshold(double _high_threshold) {
    use_high_norm = true;
    high_threshold = _high_threshold;
}

void RNN_Genome::disable_low_threshold() {
    use_low_norm = false;
}

void RNN_Genome::enable_low_threshold(double _low_threshold) {
    use_low_norm = true;
    low_threshold = _low_threshold;
}

void RNN_Genome::disable_dropout() {
    use_dropout = false;
}

void RNN_Genome::enable_dropout(double _dropout_probability) {
    dropout_probability = _dropout_probability;
}

void RNN_Genome::set_log_filename(string _log_filename) {
    log_filename = _log_filename;
}

void RNN_Genome::get_weights(vector<double> &parameters) {
    parameters.resize(get_number_weights());

    uint32_t current = 0;

    for (uint32_t i = 0; i < nodes.size(); i++) {
        nodes[i]->get_weights(current, parameters);
        //if (nodes[i]->is_reachable()) nodes[i]->get_weights(current, parameters);
    }

    for (uint32_t i = 0; i < edges.size(); i++) {
        parameters[current++] = edges[i]->weight;
        //if (edges[i]->is_reachable()) parameters[current++] = edges[i]->weight;
    }

    for (uint32_t i = 0; i < recurrent_edges.size(); i++) {
        parameters[current++] = recurrent_edges[i]->weight;
        //if (recurrent_edges[i]->is_reachable()) parameters[current++] = recurrent_edges[i]->weight;
    }
}

void RNN_Genome::set_weights(const vector<double> &parameters) {
    if (parameters.size() != get_number_weights()) {
        Log::fatal("ERROR! Trying to set weights where the RNN has %d weights, and the parameters vector has %d weights!\n", get_number_weights(), parameters.size());
        exit(1);
    }

    uint32_t current = 0;

    for (uint32_t i = 0; i < nodes.size(); i++) {
        nodes[i]->set_weights(current, parameters);
        //if (nodes[i]->is_reachable()) nodes[i]->set_weights(current, parameters);
    }

    for (uint32_t i = 0; i < edges.size(); i++) {
        edges[i]->weight = bound(parameters[current++]);
        //if (edges[i]->is_reachable()) edges[i]->weight = parameters[current++];
    }

    for (uint32_t i = 0; i < recurrent_edges.size(); i++) {
        recurrent_edges[i]->weight = bound(parameters[current++]);
        //if (recurrent_edges[i]->is_reachable()) recurrent_edges[i]->weight = parameters[current++];
    }

}

uint32_t RNN_Genome::get_number_inputs() {
    uint32_t number_inputs = 0;

    for (uint32_t i = 0; i < nodes.size(); i++) {
        if (nodes[i]->get_layer_type() == INPUT_LAYER) {
            number_inputs++;
        }
    }

    return number_inputs;
}

uint32_t RNN_Genome::get_number_outputs() {
    uint32_t number_outputs = 0;

    for (uint32_t i = 0; i < nodes.size(); i++) {
        if (nodes[i]->get_layer_type() == OUTPUT_LAYER) {
            number_outputs++;
        }
    }

    return number_outputs;
}


uint32_t RNN_Genome::get_number_weights() {
    uint32_t number_weights = 0;

    for (uint32_t i = 0; i < nodes.size(); i++) {
        number_weights += nodes[i]->get_number_weights();
        //if (nodes[i]->is_reachable()) number_weights += nodes[i]->get_number_weights();
    }

    for (uint32_t i = 0; i < edges.size(); i++) {
        number_weights++;
        //if (edges[i]->is_reachable()) number_weights++;
    }

    for (uint32_t i = 0; i < recurrent_edges.size(); i++) {
        number_weights++;
        //if (recurrent_edges[i]->is_reachable()) number_weights++;
    }

    return number_weights;
}

void RNN_Genome::initialize_randomly() {
    Log::trace("initializing genome %d of group %d randomly!\n", generation_id, group_id);
    int number_of_weights = get_number_weights();
    initial_parameters.assign(number_of_weights, 0.0);

    uniform_real_distribution<double> rng(-0.5, 0.5);
    for (uint32_t i = 0; i < initial_parameters.size(); i++) {
        initial_parameters[i] = rng(generator);
    }
    this->set_best_parameters(initial_parameters); 
}


RNN* RNN_Genome::get_rnn() {
    vector<RNN_Node_Interface*> node_copies;
    vector<RNN_Edge*> edge_copies;
    vector<RNN_Recurrent_Edge*> recurrent_edge_copies;

    for (uint32_t i = 0; i < nodes.size(); i++) {
        node_copies.push_back( nodes[i]->copy() );
        //if (nodes[i]->layer_type == INPUT_LAYER || nodes[i]->layer_type == OUTPUT_LAYER || nodes[i]->is_reachable()) node_copies.push_back( nodes[i]->copy() );
    }

    for (uint32_t i = 0; i < edges.size(); i++) {
        edge_copies.push_back( edges[i]->copy(node_copies) );
        //if (edges[i]->is_reachable()) edge_copies.push_back( edges[i]->copy(node_copies) );
    }

    for (uint32_t i = 0; i < recurrent_edges.size(); i++) {
        recurrent_edge_copies.push_back( recurrent_edges[i]->copy(node_copies) );
        //if (recurrent_edges[i]->is_reachable()) recurrent_edge_copies.push_back( recurrent_edges[i]->copy(node_copies) );
    }

    return new RNN(node_copies, edge_copies, recurrent_edge_copies);
}

vector<double> RNN_Genome::get_best_parameters() const {
    return best_parameters;
}

//INFO: ADDED BY ABDELRAHMAN TO USE FOR TRANSFER LEARNING
void RNN_Genome::set_best_parameters( vector<double> parameters) {
    best_parameters = parameters;
}

//INFO: ADDED BY ABDELRAHMAN TO USE FOR TRANSFER LEARNING
void RNN_Genome::set_initial_parameters( vector <double> parameters) {
    initial_parameters = parameters;
}

int32_t RNN_Genome::get_generation_id() const {
    return generation_id;
}

void RNN_Genome::set_generation_id(int32_t _generation_id) {
    generation_id = _generation_id;
}

double RNN_Genome::get_fitness() const {
    return best_validation_mse;
    //return best_validation_mae;
}

double RNN_Genome::get_best_validation_mse() const {
    return best_validation_mse;
}

double RNN_Genome::get_best_validation_mae() const {
    return best_validation_mae;
}



void RNN_Genome::set_generated_by(string type) {
    generated_by_map[type]++;
}

int RNN_Genome::get_generated_by(string type) {
    return generated_by_map[type];
}

bool RNN_Genome::sanity_check() {
    return true;
}


void forward_pass_thread(RNN* rnn, const vector<double> &parameters, const vector< vector<double> > &inputs, const vector< vector<double> > &outputs, uint32_t i, double *mses, bool use_dropout, bool training, double dropout_probability) {
    rnn->set_weights(parameters);
    rnn->forward_pass(inputs, use_dropout, training, dropout_probability);

    mses[i] = rnn->calculate_error_mse(outputs);
    //mses[i] = rnn->calculate_error_mae(outputs);

    Log::trace("mse[%d]: %lf\n", i, mses[i]);
}

void RNN_Genome::get_analytic_gradient(vector<RNN*> &rnns, const vector<double> &parameters, const vector< vector< vector<double> > > &inputs, const vector< vector< vector<double> > > &outputs, double &mse, vector<double> &analytic_gradient, bool training) {

    double *mses = new double[rnns.size()];
    double mse_sum = 0.0;
    vector<thread> threads;
    for (uint32_t i = 0; i < rnns.size(); i++) {
        threads.push_back( thread(forward_pass_thread, rnns[i], parameters, inputs[i], outputs[i], i, mses, use_dropout, training, dropout_probability) );
    }

    for (uint32_t i = 0; i < rnns.size(); i++) {
        threads[i].join();
        mse_sum += mses[i];
    }
    delete [] mses;

    for (uint32_t i = 0; i < rnns.size(); i++) {
        double d_mse = mse_sum * (1.0 / outputs[i][0].size()) * 2.0;
        rnns[i]->backward_pass(d_mse, use_dropout, training, dropout_probability);

        //double d_mae = mse_sum * (1.0 / outputs[i][0].size());
        //rnns[i]->backward_pass(d_mae);

    }

    mse = mse_sum;

    vector<double> current_gradients;
    analytic_gradient.assign(parameters.size(), 0.0);
    for (uint32_t k = 0; k < rnns.size(); k++) {

        uint32_t current = 0;
        for (uint32_t i = 0; i < rnns[k]->get_number_nodes(); i++) {
            rnns[k]->get_node(i)->get_gradients(current_gradients);

            for (uint32_t j = 0; j < current_gradients.size(); j++) {
                analytic_gradient[current] += current_gradients[j];
                current++;
            }
        }

        for (uint32_t i = 0; i < rnns[k]->get_number_edges(); i++) {
            analytic_gradient[current] += rnns[k]->get_edge(i)->get_gradient();
            current++;
        }
    }
}


void RNN_Genome::backpropagate(const vector< vector< vector<double> > > &inputs, const vector< vector< vector<double> > > &outputs, const vector< vector< vector<double> > > &validation_inputs, const vector< vector< vector<double> > > &validation_outputs) {

    double learning_rate = this->learning_rate / inputs.size();
    double low_threshold = sqrt(this->low_threshold * inputs.size());
    double high_threshold = sqrt(this->high_threshold * inputs.size());

    int32_t n_series = inputs.size();
    vector<RNN*> rnns;
    for (int32_t i = 0; i < n_series; i++) {
        rnns.push_back( this->get_rnn() );
    }

    vector<double> parameters = initial_parameters;

    int n_parameters = this->get_number_weights();
    vector<double> prev_parameters(n_parameters, 0.0);

    vector<double> prev_velocity(n_parameters, 0.0);
    vector<double> prev_prev_velocity(n_parameters, 0.0);

    vector<double> analytic_gradient;
    vector<double> prev_gradient(n_parameters, 0.0);

    double mu = 0.9;
    double original_learning_rate = learning_rate;

    double prev_mu;
    double prev_norm;
    double prev_learning_rate;
    double prev_mse;
    double mse;

    double parameter_norm = 0.0;
    double velocity_norm = 0.0;
    double norm = 0.0;

    //initialize the initial previous values
    get_analytic_gradient(rnns, parameters, inputs, outputs, mse, analytic_gradient, true);
    double validation_mse = get_mse(parameters, validation_inputs, validation_outputs);
    best_validation_mse = validation_mse;
    best_validation_mae = get_mae(parameters, validation_inputs, validation_outputs);
    best_parameters = parameters;

    norm = 0.0;
    for (int32_t i = 0; i < parameters.size(); i++) {
        norm += analytic_gradient[i] * analytic_gradient[i];
    }
    norm = sqrt(norm);

    ofstream *output_log = NULL;
    if (log_filename != "") {
        output_log = new ofstream(log_filename);
    }

    bool was_reset = false;
    double reset_count = 0;
    for (uint32_t iteration = 0; iteration < bp_iterations; iteration++) {
        prev_mu = mu;
        prev_norm  = norm;
        prev_mse = mse;
        prev_learning_rate = learning_rate;


        prev_gradient = analytic_gradient;

        get_analytic_gradient(rnns, parameters, inputs, outputs, mse, analytic_gradient, true);

        this->set_weights(parameters);
        validation_mse = get_mse(parameters, validation_inputs, validation_outputs);
        if (validation_mse < best_validation_mse) {
            best_validation_mse = validation_mse;
            best_validation_mae = get_mae(parameters, validation_inputs, validation_outputs);
            best_parameters = parameters;
        }

        norm = 0.0;
        velocity_norm = 0.0;
        parameter_norm = 0.0;
        for (int32_t i = 0; i < parameters.size(); i++) {
            norm += analytic_gradient[i] * analytic_gradient[i];
            velocity_norm += prev_velocity[i] * prev_velocity[i];
            parameter_norm += parameters[i] * parameters[i];
        }
        norm = sqrt(norm);
        velocity_norm = sqrt(velocity_norm);

        if (output_log != NULL) {
            (*output_log) << iteration
                << " " << mse
                << " " << validation_mse
                << " " << best_validation_mse << endl;
        }

        Log::info("iteration %10d, mse: %10lf, v_mse: %10lf, bv_mse: %10lf, lr: %lf, norm: %lf, p_norm: %lf, v_norm: %lf", iteration, mse, validation_mse, best_validation_mse, learning_rate, norm, parameter_norm, velocity_norm);

        if (use_reset_weights && prev_mse * 1.25 < mse) {
            Log::info_no_header(", RESETTING WEIGHTS %d", reset_count);
            parameters = prev_parameters;
            //prev_velocity = prev_prev_velocity;
            prev_velocity.assign(parameters.size(), 0.0);
            mse = prev_mse;
            mu = prev_mu;
            learning_rate = prev_learning_rate;
            analytic_gradient = prev_gradient;


            //learning_rate *= 0.5;
            //if (learning_rate < 0.0000001) learning_rate = 0.0000001;

            reset_count++;
            if (reset_count > 20) break;

            was_reset = true;
            continue;
        }

        if (was_reset) {
            was_reset = false;
        } else {
            reset_count -= 0.1;
            if (reset_count < 0) reset_count = 0;
            if (adapt_learning_rate) learning_rate = original_learning_rate;
        }


        if (adapt_learning_rate) {
            if (prev_mse > mse) {
                learning_rate *= 1.10;
                if (learning_rate > 1.0) learning_rate = 1.0;

                Log::info_no_header(", INCREASING LR");
            }
        }

        if (use_high_norm && norm > high_threshold) {
            double high_threshold_norm = high_threshold / norm;

            Log::info_no_header(", OVER THRESHOLD, multiplier: %lf", high_threshold_norm);

            for (int32_t i = 0; i < parameters.size(); i++) {
                analytic_gradient[i] = high_threshold_norm * analytic_gradient[i];
            }

            if (adapt_learning_rate) {
                learning_rate *= 0.5;
                if (learning_rate < 0.0000001) learning_rate = 0.0000001;
            }

        } else if (use_low_norm && norm < low_threshold) {
            double low_threshold_norm = low_threshold / norm;
            Log::info_no_header(", UNDER THRESHOLD, multiplier: %lf", low_threshold_norm);

            for (int32_t i = 0; i < parameters.size(); i++) {
                analytic_gradient[i] = low_threshold_norm * analytic_gradient[i];
            }

            if (adapt_learning_rate) {
                if (prev_mse * 1.05 < mse) {
                    Log::info_no_header(", WORSE");
                    learning_rate *= 0.5;
                    if (learning_rate < 0.0000001) learning_rate = 0.0000001;
                }
            }
        }

        if (reset_count > 0) {
            double reset_penalty = pow(5.0, -reset_count);
            Log::info_no_header(", RESET PENALTY (%d): %lf", reset_count, reset_penalty);

            for (int32_t i = 0; i < parameters.size(); i++) {
                analytic_gradient[i] = reset_penalty * analytic_gradient[i];
            }

        }

        Log::info_no_header("\n");

        if (use_nesterov_momentum) {
            for (int32_t i = 0; i < parameters.size(); i++) {
                prev_parameters[i] = parameters[i];
                prev_prev_velocity[i] = prev_velocity[i];

                double mu_v = prev_velocity[i] * prev_mu;

                prev_velocity[i] = mu_v  - (prev_learning_rate * prev_gradient[i]);
                parameters[i] += mu_v + ((mu + 1) * prev_velocity[i]);
            }
        } else {
            for (int32_t i = 0; i < parameters.size(); i++) {
                prev_parameters[i] = parameters[i];
                prev_gradient[i] = analytic_gradient[i];
                parameters[i] -= learning_rate * analytic_gradient[i];
            }
        }
    }

    RNN *g;
    while (rnns.size() > 0) {
        g = rnns.back();
        rnns.pop_back();
        delete g;

    }

    this->set_weights(best_parameters);
}

void RNN_Genome::backpropagate_stochastic(const vector< vector< vector<double> > > &inputs, const vector< vector< vector<double> > > &outputs, const vector< vector< vector<double> > > &validation_inputs, const vector< vector< vector<double> > > &validation_outputs) {

    vector<double> parameters = initial_parameters;

    int n_parameters = this->get_number_weights();
    vector<double> prev_parameters(n_parameters, 0.0);

    vector<double> prev_velocity(n_parameters, 0.0);
    vector<double> prev_prev_velocity(n_parameters, 0.0);

    vector<double> analytic_gradient;
    vector<double> prev_gradient(n_parameters, 0.0);

    double mu = 0.9;
    double original_learning_rate = learning_rate;

    int n_series = inputs.size();
    double prev_mu[n_series];
    double prev_norm[n_series];
    double prev_learning_rate[n_series];
    double prev_mse[n_series];
    double mse;

    double norm = 0.0;

    std::chrono::time_point<std::chrono::system_clock> startClock = std::chrono::system_clock::now();

    RNN* rnn = get_rnn();
    rnn->set_weights(parameters);

    //initialize the initial previous values
    for (uint32_t i = 0; i < n_series; i++) {
        Log::trace("getting analytic gradient for input/output: %d, n_series: %d, parameters.size: %d, inputs.size(): %d, outputs.size(): %d, log filename: '%s'\n", i, n_series, parameters.size(), inputs.size(), outputs.size(), log_filename.c_str());

        rnn->get_analytic_gradient(parameters, inputs[i], outputs[i], mse, analytic_gradient, use_dropout, true, dropout_probability);
        Log::trace("got analytic gradient.\n");

        norm = 0.0;
        for (int32_t j = 0; j < parameters.size(); j++) {
            norm += analytic_gradient[j] * analytic_gradient[j];
        }
        norm = sqrt(norm);
        prev_mu[i] = mu;
        prev_norm[i] = norm;
        prev_mse[i] = mse;
        prev_learning_rate[i] = learning_rate;
    }
    Log::trace("initialized previous values.\n");

    //TODO: need to get validation error on the RNN not the genome
    double validation_mse = get_mse(parameters, validation_inputs, validation_outputs);
    best_validation_mse = validation_mse;
    best_validation_mae = get_mae(parameters, validation_inputs, validation_outputs);
    best_parameters = parameters;

    Log::trace("got initial errors.\n");

    Log::trace("initial validation_mse: %lf, best validation error: %lf\n", validation_mse, best_validation_mse);
    double m = 0.0, s = 0.0;
    get_mu_sigma(parameters, m, s);
    for (int32_t i = 0; i < parameters.size(); i++) {
        Log::trace("parameters[%d]: %lf\n", i, parameters[i]);
    }

    unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
    minstd_rand0 generator(seed);
    uniform_real_distribution<double> rng(0, 1);

    int random_selection = rng(generator);
    mu = prev_mu[random_selection];
    norm = prev_norm[random_selection];
    mse = prev_mse[random_selection];
    learning_rate = prev_learning_rate[random_selection];

    ofstream *output_log = NULL;
    ostringstream memory_log;

    if (log_filename != "") {
        Log::trace("creating new log stream for '%s'\n", log_filename.c_str());
        output_log = new ofstream(log_filename);
        Log::trace("testing to see if log file is valid.\n");

        if (!output_log->is_open()) {
            Log::fatal("ERROR, could not open output log: '%s'\n", log_filename.c_str());
            exit(1);
        }

        Log::trace("opened log file '%s'\n", log_filename.c_str());
    }

    vector<int32_t> shuffle_order;
    for (int32_t i = 0; i < (int32_t)inputs.size(); i++) {
        shuffle_order.push_back(i);
    }

    bool was_reset = false;
    int reset_count = 0;

    for (uint32_t iteration = 0; iteration < bp_iterations; iteration++) {
        fisher_yates_shuffle(generator, shuffle_order);

        double avg_norm = 0.0;
        for (uint32_t k = 0; k < shuffle_order.size(); k++) {
            random_selection = shuffle_order[k];

            prev_mu[random_selection] = mu;
            prev_norm[random_selection] = norm;
            prev_mse[random_selection] = mse;
            prev_learning_rate[random_selection] = learning_rate;

            prev_gradient = analytic_gradient;

            rnn->get_analytic_gradient(parameters, inputs[random_selection], outputs[random_selection], mse, analytic_gradient, use_dropout, true, dropout_probability);

            norm = 0.0;
            for (int32_t i = 0; i < parameters.size(); i++) {
                norm += analytic_gradient[i] * analytic_gradient[i];
            }
            norm = sqrt(norm);
            avg_norm += norm;

            Log::debug("iteration %4d, series: %4d, mse: %5.10lf, lr: %lf, norm: %lf", iteration, random_selection, mse, learning_rate, norm);

            if (use_reset_weights && prev_mse[random_selection] * 2 < mse) {
                Log::debug_no_header(", RESETTING WEIGHTS");

                parameters = prev_parameters;
                //prev_velocity = prev_prev_velocity;
                prev_velocity.assign(parameters.size(), 0.0);
                mse = prev_mse[random_selection];
                mu = prev_mu[random_selection];
                learning_rate = prev_learning_rate[random_selection];
                analytic_gradient = prev_gradient;

                random_selection = rng(generator) * inputs.size();

                learning_rate *= 0.5;
                if (learning_rate < 0.0000001) learning_rate = 0.0000001;

                reset_count++;
                if (reset_count > 20) break;

                was_reset = true;
                k--;
                continue;
            }

            if (was_reset) {
                was_reset = false;
            } else {
                reset_count = 0;
                learning_rate = original_learning_rate;
            }


            if (adapt_learning_rate) {
                if (prev_mse[random_selection] > mse) {
                    learning_rate *= 1.10;
                    if (learning_rate > 1.0) learning_rate = 1.0;

                    Log::debug_no_header(", INCREASING LR");
                }
            }

            if (use_high_norm && norm > high_threshold) {
                double high_threshold_norm = high_threshold / norm;
                Log::debug_no_header(", OVER THRESHOLD, multiplier: %lf", high_threshold_norm);

                for (int32_t i = 0; i < parameters.size(); i++) {
                    analytic_gradient[i] = high_threshold_norm * analytic_gradient[i];
                }

                if (adapt_learning_rate) {
                    learning_rate *= 0.5;
                    if (learning_rate < 0.0000001) learning_rate = 0.0000001;
                }

            } else if (use_low_norm && norm < low_threshold) {
                double low_threshold_norm = low_threshold / norm;
                Log::debug_no_header(", UNDER THRESHOLD, multiplier: %lf", low_threshold_norm);

                for (int32_t i = 0; i < parameters.size(); i++) {
                    analytic_gradient[i] = low_threshold_norm * analytic_gradient[i];
                }

                if (adapt_learning_rate) {
                    if (prev_mse[random_selection] * 1.05 < mse) {
                        Log::debug_no_header(", WORSE");
                        learning_rate *= 0.5;
                        if (learning_rate < 0.0000001) learning_rate = 0.0000001;
                    }
                }
            }

            Log::debug_no_header("\n");

            if (use_nesterov_momentum) {
                for (int32_t i = 0; i < parameters.size(); i++) {
                    prev_parameters[i] = parameters[i];
                    prev_prev_velocity[i] = prev_velocity[i];

                    double mu_v = prev_velocity[i] * prev_mu[random_selection];

                    prev_velocity[i] = mu_v  - (prev_learning_rate[random_selection] * prev_gradient[i]);
                    parameters[i] += mu_v + ((mu + 1) * prev_velocity[i]);

                    if (parameters[i] < -10.0) parameters[i] = -10.0;
                    else if (parameters[i] > 10.0) parameters[i] = 10.0;
                }
            } else {
                for (int32_t i = 0; i < parameters.size(); i++) {
                    prev_parameters[i] = parameters[i];
                    prev_gradient[i] = analytic_gradient[i];
                    parameters[i] -= learning_rate * analytic_gradient[i];

                    if (parameters[i] < -10.0) parameters[i] = -10.0;
                    else if (parameters[i] > 10.0) parameters[i] = 10.0;
                }
            }
        }

        this->set_weights(parameters);

        double training_error = get_mse(parameters, inputs, outputs);
        validation_mse = get_mse(parameters, validation_inputs, validation_outputs);
        if (validation_mse < best_validation_mse) {
            best_validation_mse = validation_mse;
            best_validation_mae = get_mae(parameters, validation_inputs, validation_outputs);

            best_parameters = parameters;
        }

        if (output_log != NULL) {
            std::chrono::time_point<std::chrono::system_clock> currentClock = std::chrono::system_clock::now();
            long milliseconds = std::chrono::duration_cast<std::chrono::milliseconds>(currentClock - startClock).count();

            //make sure the output log is good
            if ( !output_log->good() ) {
                output_log->close();
                delete output_log;

                output_log = new ofstream(log_filename, std::ios_base::app);
                Log::trace("testing to see if log file valid for '%s'\n", log_filename.c_str());

                if (!output_log->is_open()) {
                    Log::fatal("ERROR, could not open output log: '%s'\n", log_filename.c_str());
                    exit(1);
                }
            }

            (*output_log) << iteration
                << "," << milliseconds
                << "," << training_error
                << "," << validation_mse
                << "," << best_validation_mse
                << "," << best_validation_mae
                << "," << avg_norm << endl;

            memory_log << iteration
                << "," << milliseconds
                << "," << training_error
                << "," << validation_mse
                << "," << best_validation_mse
                << "," << best_validation_mae
                << "," << avg_norm << endl;
        }


        Log::info("iteration %4d, mse: %5.10lf, v_mse: %5.10lf, bv_mse: %5.10lf, avg_norm: %5.10lf\n", iteration, training_error, validation_mse, best_validation_mse, avg_norm);

    }

    if (log_filename != "") {
        ofstream memory_log_file(log_filename + "_mem");
        memory_log_file << memory_log.str();
        memory_log_file.close();
    }

    delete rnn;

    this->set_weights(best_parameters);
    Log::trace("backpropagation completed, getting mu/sigma\n");
    double _mu, _sigma;
    get_mu_sigma(best_parameters, _mu, _sigma);
}

double RNN_Genome::get_mse(const vector<double> &parameters, const vector< vector< vector<double> > > &inputs, const vector< vector< vector<double> > > &outputs) {
    RNN *rnn = get_rnn();
    rnn->set_weights(parameters);

    double mse = 0.0;
    double avg_mse = 0.0;

    for (uint32_t i = 0; i < inputs.size(); i++) {
        mse = rnn->prediction_mse(inputs[i], outputs[i], use_dropout, false, dropout_probability);

        avg_mse += mse;

        Log::trace("series[%5d]: MSE: %5.10lf\n", i, mse);
    }

    delete rnn;

    avg_mse /= inputs.size();
    Log::trace("average MSE: %5.10lf\n", avg_mse);
    return avg_mse;
}

double RNN_Genome::get_mae(const vector<double> &parameters, const vector< vector< vector<double> > > &inputs, const vector< vector< vector<double> > > &outputs) {
    RNN *rnn = get_rnn();
    rnn->set_weights(parameters);

    double mae;
    double avg_mae = 0.0;

    for (uint32_t i = 0; i < inputs.size(); i++) {
        mae = rnn->prediction_mae(inputs[i], outputs[i], use_dropout, false, dropout_probability);

        avg_mae += mae;

        Log::debug("series[%5d] MAE: %5.10lf\n", i, mae);
    }

    delete rnn;

    avg_mae /= inputs.size();
    Log::debug("average MAE: %5.10lf\n", avg_mae);
    return avg_mae;
}

vector< vector<double> > RNN_Genome::get_predictions(const vector<double> &parameters, const vector< vector< vector<double> > > &inputs, const vector< vector< vector<double> > > &outputs) {
    RNN *rnn = get_rnn();
    rnn->set_weights(parameters);

    vector< vector<double> > all_results;

    //one input vector per testing file
    for (uint32_t i = 0; i < inputs.size(); i++) {
        all_results.push_back(rnn->get_predictions(inputs[i], outputs[i], use_dropout, dropout_probability));
    }

    delete rnn;

    return all_results;
}


void RNN_Genome::write_predictions(const vector<string> &input_filenames, const vector<double> &parameters, const vector< vector< vector<double> > > &inputs, const vector< vector< vector<double> > > &outputs) {
    RNN *rnn = get_rnn();
    rnn->set_weights(parameters);

    for (uint32_t i = 0; i < inputs.size(); i++) {
        Log::info("input filename[%5d]: '%s'\n", i, input_filenames[i].c_str());

        string output_filename = "predictions_" + std::to_string(i) + ".txt";
        Log::info("output filename: '%s'\n", output_filename.c_str());

        rnn->write_predictions(output_filename, input_parameter_names, output_parameter_names, inputs[i], outputs[i], use_dropout, dropout_probability);
    }

    delete rnn;
}

bool RNN_Genome::has_node_with_innovation(int32_t innovation_number) const {
    for (int32_t i = 0; i < (int32_t)nodes.size(); i++) {
        if (nodes[i]->get_innovation_number() == innovation_number) return true;
    }

    return false;
}

bool RNN_Genome::equals(RNN_Genome* other) {

    if (nodes.size() != other->nodes.size()) return false;
    if (edges.size() != other->edges.size()) return false;
    if (recurrent_edges.size() != other->recurrent_edges.size()) return false;

    for (int32_t i = 0; i < (int32_t)nodes.size(); i++) {
        if (!nodes[i]->equals(other->nodes[i])) return false;
    }

    for (int32_t i = 0; i < (int32_t)edges.size(); i++) {
        if (!edges[i]->equals(other->edges[i])) return false;
    }


    for (int32_t i = 0; i < (int32_t)recurrent_edges.size(); i++) {
        if (!recurrent_edges[i]->equals(other->recurrent_edges[i])) return false;
    }

    return true;
}

void RNN_Genome::assign_reachability() {
    Log::trace("assigning reachability!\n");
    Log::trace("%6d nodes, %6d edges, %6d recurrent edges\n", nodes.size(), edges.size(), recurrent_edges.size());

    for (int32_t i = 0; i < (int32_t)nodes.size(); i++) {
        nodes[i]->forward_reachable = false;
        nodes[i]->backward_reachable = false;
        nodes[i]->total_inputs = 0;
        nodes[i]->total_outputs = 0;

        //set enabled input nodes as reachable
        if (nodes[i]->layer_type == INPUT_LAYER && nodes[i]->enabled) {
            nodes[i]->forward_reachable = true;
            nodes[i]->total_inputs = 1;

            Log::trace("\tsetting input node[%5d] reachable\n", i);
        }

        if (nodes[i]->layer_type == OUTPUT_LAYER) {
            nodes[i]->backward_reachable = true;
            nodes[i]->total_outputs = 1;
        }
    }

    for (int32_t i = 0; i < (int32_t)edges.size(); i++) {
        edges[i]->forward_reachable = false;
        edges[i]->backward_reachable = false;
    }

    for (int32_t i = 0; i < (int32_t)recurrent_edges.size(); i++) {
        recurrent_edges[i]->forward_reachable = false;
        recurrent_edges[i]->backward_reachable = false;
    }

    //do forward reachability
    vector<RNN_Node_Interface*> nodes_to_visit;
    for (int32_t i = 0; i < (int32_t)nodes.size(); i++) {
        if (nodes[i]->layer_type == INPUT_LAYER && nodes[i]->enabled) {
            nodes_to_visit.push_back(nodes[i]);
        }
    }

    while (nodes_to_visit.size() > 0) {
        RNN_Node_Interface *current = nodes_to_visit.back();
        nodes_to_visit.pop_back();

        //if the node is not enabled, we don't need to do anything
        if (!current->enabled) continue;

        for (int32_t i = 0; i < (int32_t)edges.size(); i++) {
            if (edges[i]->input_innovation_number == current->innovation_number &&
                    edges[i]->enabled) {
                //this is an edge coming out of this node

                if (edges[i]->output_node->enabled) {
                    edges[i]->forward_reachable = true;

                    if (edges[i]->output_node->forward_reachable == false) {
                        if (edges[i]->output_node->innovation_number == edges[i]->input_node->innovation_number) {
                            Log::fatal("ERROR, forward edge was circular -- this should never happen");
                            exit(1);
                        }
                        edges[i]->output_node->forward_reachable = true;
                        nodes_to_visit.push_back(edges[i]->output_node);
                    }
                }
            }
        }

        for (int32_t i = 0; i < (int32_t)recurrent_edges.size(); i++) {
            if (recurrent_edges[i]->forward_reachable) continue;

            if (recurrent_edges[i]->input_innovation_number == current->innovation_number &&
                    recurrent_edges[i]->enabled) {
                //this is an recurrent_edge coming out of this node

                if (recurrent_edges[i]->output_node->enabled) {
                    recurrent_edges[i]->forward_reachable = true;

                    if (recurrent_edges[i]->output_node->forward_reachable == false) {
                        recurrent_edges[i]->output_node->forward_reachable = true;

                        //handle the edge case when a recurrent edge loops back on itself
                        nodes_to_visit.push_back(recurrent_edges[i]->output_node);
                    }
                }
            }
        }
    }

    //do backward reachability
    nodes_to_visit.clear();
    for (int32_t i = 0; i < (int32_t)nodes.size(); i++) {
        if (nodes[i]->layer_type == OUTPUT_LAYER && nodes[i]->enabled) {
            nodes_to_visit.push_back(nodes[i]);
        }
    }

    while (nodes_to_visit.size() > 0) {
        RNN_Node_Interface *current = nodes_to_visit.back();
        nodes_to_visit.pop_back();

        //if the node is not enabled, we don't need to do anything
        if (!current->enabled) continue;

        for (int32_t i = 0; i < (int32_t)edges.size(); i++) {
            if (edges[i]->output_innovation_number == current->innovation_number &&
                    edges[i]->enabled) {
                //this is an edge coming out of this node

                if (edges[i]->input_node->enabled) {
                    edges[i]->backward_reachable = true;
                    if (edges[i]->input_node->backward_reachable == false) {
                        edges[i]->input_node->backward_reachable = true;
                        nodes_to_visit.push_back(edges[i]->input_node);
                    }
                }
            }
        }

        for (int32_t i = 0; i < (int32_t)recurrent_edges.size(); i++) {
            if (recurrent_edges[i]->output_innovation_number == current->innovation_number &&
                    recurrent_edges[i]->enabled) {
                //this is an recurrent_edge coming out of this node

                if (recurrent_edges[i]->input_node->enabled) {
                    recurrent_edges[i]->backward_reachable = true;
                    if (recurrent_edges[i]->input_node->backward_reachable == false) {
                        recurrent_edges[i]->input_node->backward_reachable = true;
                        nodes_to_visit.push_back(recurrent_edges[i]->input_node);
                    }
                }
            }
        }
    }

    //set inputs/outputs
    for (int32_t i = 0; i < (int32_t)edges.size(); i++) {
        if (edges[i]->is_reachable()) {
            edges[i]->input_node->total_outputs++;
            edges[i]->output_node->total_inputs++;
        }
    }

    for (int32_t i = 0; i < (int32_t)recurrent_edges.size(); i++) {
        if (recurrent_edges[i]->is_reachable()) {
            recurrent_edges[i]->input_node->total_outputs++;
            recurrent_edges[i]->output_node->total_inputs++;
        }
    }

    if (Log::at_level(Log::TRACE)) {
        Log::trace("node reachabiltity:\n");
        for (int32_t i = 0; i < nodes.size(); i++) {
            RNN_Node_Interface *n = nodes[i];
            Log::trace("node %5d, e: %d, fr: %d, br: %d, ti: %5d, to: %5d\n", n->innovation_number, n->enabled, n->forward_reachable, n->backward_reachable, n->total_inputs, n->total_outputs);
        }

        Log::trace("edge reachabiltity:\n");
        for (int32_t i = 0; i < edges.size(); i++) {
            RNN_Edge *e = edges[i];
            Log::trace("edge %5d, e: %d, fr: %d, br: %d\n", e->innovation_number, e->enabled, e->forward_reachable, e->backward_reachable);
        }

        Log::trace("recurrent edge reachabiltity:\n");
        for (int32_t i = 0; i < recurrent_edges.size(); i++) {
            RNN_Recurrent_Edge *e = recurrent_edges[i];
            Log::trace("recurrent edge %5d, e: %d, fr: %d, br: %d\n", e->innovation_number, e->enabled, e->forward_reachable, e->backward_reachable);
        }
    }
}


bool RNN_Genome::outputs_unreachable() {
    assign_reachability();

    for (int32_t i = 0; i < (int32_t)nodes.size(); i++) {
        if (nodes[i]->layer_type == OUTPUT_LAYER && !nodes[i]->is_reachable()) return true;
    }

    return false;
}

void RNN_Genome::get_mu_sigma(const vector<double> &p, double &mu, double &sigma) {
    if (p.size() == 0) {
        mu = 0.0;
        sigma = 0.25;
        Log::debug("\tmu: %lf, sigma: %lf, parameters.size() == 0\n", mu, sigma);
        return;
    }

    mu = 0.0;
    sigma = 0.0;

    for (int32_t i = 0; i < p.size(); i++) {
        /*
        if (p[i] < -10 || p[i] > 10) {
            Log::fatal("ERROR in get_mu_sigma, parameter[%d] was out of bounds: %lf\n", i, p[i]);
            Log::fatal("all parameters:\n");
            for (int32_t i = 0; i < (int32_t)p.size(); i++) {
                Log::fatal("\t%lf\n", p[i]);
            }
            exit(1);
        }
        */

        if (p[i] < -10) mu += -10.0;
        else if (p[i] > 10) mu += 10.0;
        else mu += p[i];
    }
    mu /= p.size();

    double temp;
    for (int32_t i = 0; i < p.size(); i++) {
        temp = (mu - p[i]) * (mu - p[i]);
        sigma += temp;
    }

    sigma /= (p.size() - 1);
    sigma = sqrt(sigma);

    Log::debug("\tmu: %lf, sigma: %lf, parameters.size(): %d\n", mu, sigma, p.size());
    if (std::isnan(mu) || std::isinf(mu) || std::isnan(sigma) || std::isinf(sigma)) {
        Log::fatal("mu or sigma was not a number, all parameters:\n");
        for (int32_t i = 0; i < (int32_t)p.size(); i++) {
            Log::fatal("\t%lf", p[i]);
        }
        exit(1);
    }

    if (mu < -11.0 || mu > 11.0 || sigma < -30.0 || sigma > 30.0) {
        Log::fatal("mu or sigma exceeded possible bounds (11 or 30), all parameters:\n");
        for (int32_t i = 0; i < (int32_t)p.size(); i++) {
            Log::fatal("\t%lf", p[i]);
        }
        exit(1);
    }
}


RNN_Node_Interface* RNN_Genome::create_node(double mu, double sigma, int node_type, int32_t &node_innovation_count, double depth) {
    RNN_Node_Interface *n = NULL;

    Log::info("CREATING NODE, type: '%s'\n", NODE_TYPES[node_type].c_str());
    if (node_type == LSTM_NODE) {
        n = new LSTM_Node(++node_innovation_count, HIDDEN_LAYER, depth);
    } else if (node_type == DELTA_NODE) {
        n = new Delta_Node(++node_innovation_count, HIDDEN_LAYER, depth);
    } else if (node_type == GRU_NODE) {
        n = new GRU_Node(++node_innovation_count, HIDDEN_LAYER, depth);
    } else if (node_type == MGU_NODE) {
        n = new MGU_Node(++node_innovation_count, HIDDEN_LAYER, depth);
    } else if (node_type == UGRNN_NODE) {
        n = new UGRNN_Node(++node_innovation_count, HIDDEN_LAYER, depth);
    } else if (node_type == SIMPLE_NODE || node_type == JORDAN_NODE || node_type == ELMAN_NODE) {
        n = new RNN_Node(++node_innovation_count, HIDDEN_LAYER, depth, node_type);
    } else {
        Log::fatal("ERROR: attempted to create a node with an unknown node type: %d\n", node_type);
        exit(1);
    }

    n->initialize_randomly(generator, normal_distribution, mu, sigma);

    return n;
}

bool RNN_Genome::attempt_edge_insert(RNN_Node_Interface *n1, RNN_Node_Interface *n2, double mu, double sigma, int32_t &edge_innovation_count) {
    Log::info("\tadding edge between nodes %d and %d\n", n1->innovation_number, n2->innovation_number);

    if (n1->depth == n2->depth) {
        Log::info("\tcannot add edge between nodes as their depths are the same: %lf and %lf\n", n1->depth, n2->depth);
        return false;
    }

    if (n2->depth < n1->depth) {
        //swap the nodes so that the lower one is first
        RNN_Node_Interface *temp = n2;
        n2 = n1;
        n1 = temp;
        Log::info("\tswaping nodes, because n2->depth < n1->depth\n");
    }


    //check to see if an edge between the two nodes already exists
    for (int32_t i = 0; i < (int32_t)edges.size(); i++) {
        if (edges[i]->input_innovation_number == n1->innovation_number &&
                edges[i]->output_innovation_number == n2->innovation_number) {
            if (!edges[i]->enabled) {
                //edge was disabled so we can enable it
                Log::info("\tedge already exists but was disabled, enabling it.\n");
                edges[i]->enabled = true;
                return true;
            } else {
                Log::info("\tedge already exists, not adding.\n");
                //edge was already enabled, so there will not be a change
                return false;
            }
        }
    }

    RNN_Edge *e = new RNN_Edge(++edge_innovation_count, n1, n2);
    e->weight = bound(normal_distribution.random(generator, mu, sigma));

    Log::info("\tadding edge between nodes %d and %d, new edge weight: %lf\n", e->input_innovation_number, e->output_innovation_number, e->weight);

    edges.insert( upper_bound(edges.begin(), edges.end(), e, sort_RNN_Edges_by_depth()), e);
    return true;
}

bool RNN_Genome::attempt_recurrent_edge_insert(RNN_Node_Interface *n1, RNN_Node_Interface *n2, double mu, double sigma, Distribution* dist, int32_t &edge_innovation_count) {
    Log::info("\tadding recurrent edge between nodes %d and %d\n", n1->innovation_number, n2->innovation_number);

    //int32_t recurrent_depth = 1 + (rng_0_1(generator) * (max_recurrent_depth - 1));
    int32_t recurrent_depth = dist->sample();

    //check to see if an edge between the two nodes already exists
    for (int32_t i = 0; i < (int32_t)recurrent_edges.size(); i++) {
        if (recurrent_edges[i]->input_innovation_number == n1->innovation_number &&
                recurrent_edges[i]->output_innovation_number == n2->innovation_number
                && recurrent_edges[i]->recurrent_depth == recurrent_depth) {

            if (!recurrent_edges[i]->enabled) {
                //edge was disabled so we can enable it
                Log::info("\trecurrent edge already exists but was disabled, enabling it.\n");
                recurrent_edges[i]->enabled = true;
                return true;
            } else {
                Log::info("\tenabled recurrent edge already existed between selected nodes %d and %d at recurrent depth: %d\n", n1->innovation_number, n2->innovation_number, recurrent_depth);
                //edge was already enabled, so there will not be a change
                return false;
            }
        }
    }

    RNN_Recurrent_Edge *e = new RNN_Recurrent_Edge(++edge_innovation_count, recurrent_depth, n1, n2);
    e->weight = bound(normal_distribution.random(generator, mu, sigma));

    Log::info("\tadding recurrent edge between nodes %d and %d, new edge weight: %d\n", e->input_innovation_number, e->output_innovation_number, e->weight);

    recurrent_edges.insert( upper_bound(recurrent_edges.begin(), recurrent_edges.end(), e, sort_RNN_Recurrent_Edges_by_depth()), e);
    return true;
}

void RNN_Genome::generate_recurrent_edges(RNN_Node_Interface *node, double mu, double sigma, Distribution *dist, int32_t &edge_innovation_count) {

    if (node->node_type == JORDAN_NODE) {
        for (int32_t i = 0; i < (int32_t)edges.size(); i++) {
            if (edges[i]->input_innovation_number == node->innovation_number && edges[i]->enabled) {
                attempt_recurrent_edge_insert(edges[i]->output_node, node, mu, sigma, dist, edge_innovation_count);
            }
        }

    } else if (node->node_type == ELMAN_NODE) {
        //elman nodes have a circular reference to themselves
        attempt_recurrent_edge_insert(node, node, mu, sigma, dist, edge_innovation_count);
    }
}



bool RNN_Genome::add_edge(double mu, double sigma, int32_t &edge_innovation_count) {
    Log::info("\tattempting to add edge!\n");
    vector<RNN_Node_Interface*> reachable_nodes;
    for (int32_t i = 0; i < (int32_t)nodes.size(); i++) {
        if (nodes[i]->is_reachable()) reachable_nodes.push_back(nodes[i]);
    }
    Log::info("\treachable_nodes.size(): %d\n", reachable_nodes.size());

    int position = rng_0_1(generator) * reachable_nodes.size();

    RNN_Node_Interface *n1 = reachable_nodes[position];
    Log::info("\tselected first node %d with depth %d\n", n1->innovation_number, n1->depth);
    //printf("pos: %d, size: %d\n", position, reachable_nodes.size());

    for (int i = 0; i < reachable_nodes.size();) {
        auto it = reachable_nodes[i];
        if (it->depth == n1->depth) {
            reachable_nodes.erase(reachable_nodes.begin() + i);        
        } else {
            i++;
        }
    }

    // for (auto i = reachable_nodes.begin(); i < reachable_nodes.end();) {
    //     if ((*i)->depth == n1->depth) {
    //         Log::info("\t\terasing node %d with depth %d\n", (*i)->innovation_number, (*i)->depth);
    //         reachable_nodes.erase(i);
    //     } else {
    //         Log::info("\t\tkeeping node %d with depth %d\n", (*i)->innovation_number, (*i)->depth);
    //         i++;
    //     }
    // }

    Log::info("\treachable_nodes.size(): %d\n", reachable_nodes.size());


    position = rng_0_1(generator) * reachable_nodes.size();
    RNN_Node_Interface *n2 = reachable_nodes[position];
    Log::info("\tselected second node %d with depth %d\n", n2->innovation_number, n2->depth);

    return attempt_edge_insert(n1, n2, mu, sigma, edge_innovation_count);
}

bool RNN_Genome::add_recurrent_edge(double mu, double sigma, Distribution* dist, int32_t &edge_innovation_count) {
    Log::info("\tattempting to add recurrent edge!\n");

    vector<RNN_Node_Interface*> possible_input_nodes;
    vector<RNN_Node_Interface*> possible_output_nodes;
    for (int32_t i = 0; i < (int32_t)nodes.size(); i++) {
        if (nodes[i]->is_reachable()) {
            possible_input_nodes.push_back(nodes[i]);

            if (nodes[i]->layer_type != INPUT_LAYER) {
                possible_output_nodes.push_back(nodes[i]);
            }
        }
    }

    Log::info("\tpossible_input_nodes.size(): %d\n", possible_input_nodes.size());
    Log::info("\tpossible_output_nodes.size(): %d\n", possible_output_nodes.size());

    if (possible_input_nodes.size() == 0) return false;
    if (possible_output_nodes.size() == 0) return false;

    int p1 = rng_0_1(generator) * possible_input_nodes.size();
    int p2 = rng_0_1(generator) * possible_output_nodes.size();
    //no need to swap the nodes as recurrent connections can go backwards

    RNN_Node_Interface *n1 = possible_input_nodes[p1];
    Log::info("\tselected first node %d with depth %d\n", n1->innovation_number, n1->depth);

    RNN_Node_Interface *n2 = possible_output_nodes[p2];
    Log::info("\tselected second node %d with depth %d\n", n2->innovation_number, n2->depth);

    return attempt_recurrent_edge_insert(n1, n2, mu, sigma, dist, edge_innovation_count);
}


//TODO: should probably change these to enable/disable path
bool RNN_Genome::disable_edge() {
    //TODO: edge should be reachable
    vector<RNN_Edge*> enabled_edges;
    for (int32_t i = 0; i < edges.size(); i++) {
        if (edges[i]->enabled) enabled_edges.push_back(edges[i]);
    }

    vector<RNN_Recurrent_Edge*> enabled_recurrent_edges;
    for (int32_t i = 0; i < recurrent_edges.size(); i++) {
        if (recurrent_edges[i]->enabled) enabled_recurrent_edges.push_back(recurrent_edges[i]);
    }

    if ((enabled_edges.size() + enabled_recurrent_edges.size()) == 0) {
        return false;
    }


    int32_t position = (enabled_edges.size() + enabled_recurrent_edges.size()) * rng_0_1(generator);

    if (position < enabled_edges.size()) {
        enabled_edges[position]->enabled = false;
        return true;
    } else {
        position -= enabled_edges.size();
        enabled_recurrent_edges[position]->enabled = false;
        return true;
    }
}

bool RNN_Genome::enable_edge() {
    //TODO: edge should be reachable
    vector<RNN_Edge*> disabled_edges;
    for (int32_t i = 0; i < edges.size(); i++) {
        if (!edges[i]->enabled) disabled_edges.push_back(edges[i]);
    }

    vector<RNN_Recurrent_Edge*> disabled_recurrent_edges;
    for (int32_t i = 0; i < recurrent_edges.size(); i++) {
        if (!recurrent_edges[i]->enabled) disabled_recurrent_edges.push_back(recurrent_edges[i]);
    }

    if ((disabled_edges.size() + disabled_recurrent_edges.size()) == 0) {
        return false;
    }

    int32_t position = (disabled_edges.size() + disabled_recurrent_edges.size()) * rng_0_1(generator);

    if (position < disabled_edges.size()) {
        disabled_edges[position]->enabled = true;
        return true;
    } else {
        position -= disabled_edges.size();
        disabled_recurrent_edges[position]->enabled = true;
        return true;
    }
}


bool RNN_Genome::split_edge(double mu, double sigma, int node_type, Distribution *dist, int32_t &edge_innovation_count, int32_t &node_innovation_count) {
    Log::info("\tattempting to split an edge!\n");
    vector<RNN_Edge*> enabled_edges;
    for (int32_t i = 0; i < edges.size(); i++) {
        if (edges[i]->enabled) enabled_edges.push_back(edges[i]);
    }

    vector<RNN_Recurrent_Edge*> enabled_recurrent_edges;
    for (int32_t i = 0; i < recurrent_edges.size(); i++) {
        if (recurrent_edges[i]->enabled) enabled_recurrent_edges.push_back(recurrent_edges[i]);
    }

    int32_t position = rng_0_1(generator) * (enabled_edges.size() + enabled_recurrent_edges.size());

    bool was_forward_edge = false;
    RNN_Node_Interface *n1 = NULL;
    RNN_Node_Interface *n2 = NULL;
    if (position < enabled_edges.size()) {
        RNN_Edge *edge = enabled_edges[position];
        n1 = edge->input_node;
        n2 = edge->output_node;
        edge->enabled = false;
        was_forward_edge = true;
    } else {
        position -= enabled_edges.size();
        RNN_Recurrent_Edge *recurrent_edge = enabled_recurrent_edges[position];
        n1 = recurrent_edge->input_node;
        n2 = recurrent_edge->output_node;
        recurrent_edge->enabled = false;
    }

    double new_depth = (n1->get_depth() + n2->get_depth()) / 2.0;
    RNN_Node_Interface *new_node = create_node(mu, sigma, node_type, node_innovation_count, new_depth);

    nodes.insert( upper_bound(nodes.begin(), nodes.end(), new_node, sort_RNN_Nodes_by_depth()), new_node);

    if (was_forward_edge) {
        attempt_edge_insert(n1, new_node, mu, sigma, edge_innovation_count);
        attempt_edge_insert(new_node, n2, mu, sigma, edge_innovation_count);
    } else {
        attempt_recurrent_edge_insert(n1, new_node, mu, sigma, dist, edge_innovation_count);
        attempt_recurrent_edge_insert(new_node, n2, mu, sigma, dist, edge_innovation_count);
    }

    if (node_type == JORDAN_NODE || node_type == ELMAN_NODE) generate_recurrent_edges(new_node, mu, sigma, dist, edge_innovation_count);

    return true;
}

bool RNN_Genome::connect_new_input_node(double mu, double sigma, RNN_Node_Interface *new_node, Distribution *dist, int32_t &edge_innovation_count) {
    Log::info("\tattempting to connect a new input node for transfer learning!\n");

    vector<RNN_Node_Interface*> possible_outputs;

    int32_t enabled_count = 0;
    double avg_outputs = 0.0;

    for (int32_t i = 0; i < (int32_t)nodes.size(); i++) {
        //can connect to output or hidden nodes
        if (nodes[i]->get_layer_type() == OUTPUT_LAYER || (nodes[i]->get_layer_type() == HIDDEN_LAYER && nodes[i]->is_reachable())) {
            Log::info("\tpotential connection node[%d], depth: %lf, total_inputs: %d, total_outputs: %d\n", nodes[i]->get_innovation_number(), nodes[i]->get_depth(), nodes[i]->get_total_inputs(), nodes[i]->get_total_outputs());
            possible_outputs.push_back(nodes[i]);
        }

        if (nodes[i]->enabled) {
            enabled_count++;
            avg_outputs += nodes[i]->total_outputs;
        }
    }

    avg_outputs /= enabled_count;

    double output_sigma = 0.0;
    double temp;
    for (int32_t i = 0; i < (int32_t)nodes.size(); i++) {
        if (nodes[i]->enabled) {
            temp = (avg_outputs - nodes[i]->total_outputs);
            temp = temp * temp;
            output_sigma += temp;
        }
    }

    output_sigma /= (enabled_count - 1);
    output_sigma = sqrt(output_sigma);

    int32_t max_outputs = fmax(1, 2.0 + normal_distribution.random(generator, avg_outputs, output_sigma));
    Log::info("\tadd new input node, max_outputs: %d\n", max_outputs);

    int32_t enabled_edges = get_enabled_edge_count();
    int32_t enabled_recurrent_edges = get_enabled_recurrent_edge_count();

    double recurrent_probability = (double)enabled_recurrent_edges / (double)(enabled_recurrent_edges + enabled_edges);
    //recurrent_probability = fmax(0.2, recurrent_probability);

    Log::info("\tadd new node for transfer recurrent probability: %lf\n", recurrent_probability);

    while (possible_outputs.size() > max_outputs) {
        int32_t position = rng_0_1(generator) * possible_outputs.size();
        possible_outputs.erase(possible_outputs.begin() + position);
    }

    for (int32_t i = 0; i < possible_outputs.size(); i++) {
        //TODO: remove after running tests without recurrent edges
        //recurrent_probability = 0;

        if (rng_0_1(generator) < recurrent_probability) {
            attempt_recurrent_edge_insert(new_node, possible_outputs[i], mu, sigma, dist, edge_innovation_count);
        } else {
            attempt_edge_insert(new_node, possible_outputs[i], mu, sigma, edge_innovation_count);
        }
    }

    return true;
}

bool RNN_Genome::connect_new_output_node(double mu, double sigma, RNN_Node_Interface *new_node, Distribution *dist, int32_t &edge_innovation_count) {
    Log::info("\tattempting to connect a new output node for transfer learning!\n");

    vector<RNN_Node_Interface*> possible_inputs;

    int32_t enabled_count = 0;
    double avg_inputs = 0.0;

    for (int32_t i = 0; i < (int32_t)nodes.size(); i++) {
        //can connect to input or hidden nodes
        if (nodes[i]->get_layer_type() == INPUT_LAYER || (nodes[i]->get_layer_type() == HIDDEN_LAYER && nodes[i]->is_reachable())) {
            possible_inputs.push_back(nodes[i]);
            Log::info("\tpotential connection node[%d], depth: %lf, total_inputs: %d, total_outputs: %d\n", nodes[i]->get_innovation_number(), nodes[i]->get_depth(), nodes[i]->get_total_inputs(), nodes[i]->get_total_outputs());
        }

        if (nodes[i]->enabled) {
            enabled_count++;
            avg_inputs += nodes[i]->total_inputs;
        }
    }

    avg_inputs /= enabled_count;

    double input_sigma = 0.0;
    double temp;
    for (int32_t i = 0; i < (int32_t)nodes.size(); i++) {
        if (nodes[i]->enabled) {
            temp = (avg_inputs - nodes[i]->total_inputs);
            temp = temp * temp;
            input_sigma += temp;
        }
    }

    input_sigma /= (enabled_count - 1);
    input_sigma = sqrt(input_sigma);

    int32_t max_inputs = fmax(1, 2.0 + normal_distribution.random(generator, avg_inputs, input_sigma));
    Log::info("\tadd new output node, max_inputs: %d\n", max_inputs);

    int32_t enabled_edges = get_enabled_edge_count();
    int32_t enabled_recurrent_edges = get_enabled_recurrent_edge_count();

    double recurrent_probability = (double)enabled_recurrent_edges / (double)(enabled_recurrent_edges + enabled_edges);
    //recurrent_probability = fmax(0.2, recurrent_probability);

    Log::info("\tadd new node for transfer recurrent probability: %lf\n", recurrent_probability);

    while (possible_inputs.size() > max_inputs) {
        int32_t position = rng_0_1(generator) * possible_inputs.size();
        possible_inputs.erase(possible_inputs.begin() + position);
    }

    for (int32_t i = 0; i < possible_inputs.size(); i++) {
        //TODO: remove after running tests without recurrent edges
        //recurrent_probability = 0;

        if (rng_0_1(generator) < recurrent_probability) {
            attempt_recurrent_edge_insert(possible_inputs[i], new_node, mu, sigma, dist, edge_innovation_count);
        } else {
            attempt_edge_insert(possible_inputs[i], new_node, mu, sigma, edge_innovation_count);
        }
    }

    return true;
}



//INFO: ADDED BY ABDELRAHMAN TO USE FOR TRANSFER LEARNING
bool RNN_Genome::connect_node_to_hid_nodes( double mu, double sig, RNN_Node_Interface *new_node, Distribution *dist, int32_t &edge_innovation_count, bool from_input ) {

    vector<RNN_Node_Interface*> candidate_nodes;

    int32_t enabled_count   = 0;
    double avg_candidates   = 0.0;

    for (int32_t i = 0; i < (int32_t)nodes.size(); i++) {
        if (nodes[i]->get_layer_type()==HIDDEN_LAYER && nodes[i]->is_reachable()){
            if (nodes[i]->enabled) {
                candidate_nodes.push_back(nodes[i]);
                enabled_count++;
                if (from_input)
                    avg_candidates += nodes[i]->total_inputs;
                else
                    avg_candidates += nodes[i]->total_outputs;
            }
        }
    }

    avg_candidates /= enabled_count;

    double sigma = 0.0;
    double temp;


    for (int32_t i = 0; i < (int32_t)nodes.size(); i++) {
        if ( nodes[i]->enabled && nodes[i]->get_layer_type()==HIDDEN_LAYER ) {
            if (from_input)
                temp = (avg_candidates - nodes[i]->total_inputs);
            else
                temp = (avg_candidates - nodes[i]->total_outputs);
            temp = temp * temp;
            sigma += temp;
        }
    }
    if (enabled_count!=1)
        sigma /= (enabled_count - 1);
    else
        sigma /= 1 ;
    sigma = sqrt(sigma);

    int32_t max_candidates = fmax(1, 2.0 + normal_distribution.random(generator, avg_candidates, sigma));

    int32_t enabled_edges = get_enabled_edge_count();
    int32_t enabled_recurrent_edges = get_enabled_recurrent_edge_count();

    double recurrent_probability = (double)enabled_recurrent_edges / (double)(enabled_recurrent_edges + enabled_edges);

    while (candidate_nodes.size() > max_candidates) {
        int32_t position = rng_0_1(generator) * candidate_nodes.size();
        candidate_nodes.erase(candidate_nodes.begin() + position);
    }


    for (auto node: candidate_nodes) {
        if (rng_0_1(generator) < recurrent_probability) {
            int32_t recurrent_depth = dist->sample();
            RNN_Recurrent_Edge *e;
            if (from_input)
                e = new RNN_Recurrent_Edge(++edge_innovation_count, recurrent_depth, new_node, node);
            else
                e = new RNN_Recurrent_Edge(++edge_innovation_count, recurrent_depth, node, new_node);
            e->weight = bound(normal_distribution.random(generator, mu, sigma));
            Log::debug("\tadding recurrent edge between nodes %d and %d, new edge weight: %d\n", e->input_innovation_number, e->output_innovation_number, e->weight);
            recurrent_edges.insert( upper_bound(recurrent_edges.begin(), recurrent_edges.end(), e, sort_RNN_Recurrent_Edges_by_depth()), e);

            initial_parameters.push_back(e->weight);
            best_parameters.push_back(e->weight);

            // attempt_recurrent_edge_insert(new_node, node, mu, sigma, dist, edge_innovation_count);
        }
        else {
            RNN_Edge *e;
            if (from_input)
                e = new RNN_Edge(++edge_innovation_count, new_node, node);
            else{
                e = new RNN_Edge(++edge_innovation_count, node, new_node);
            }
            e->weight = bound(normal_distribution.random(generator, mu, sigma));
            Log::info("\tadding edge between nodes %d and %d, new edge weight: %lf\n", e->input_innovation_number, e->output_innovation_number, e->weight);
            edges.insert( upper_bound(edges.begin(), edges.end(), e, sort_RNN_Edges_by_depth()), e);

            initial_parameters.push_back(e->weight);
            best_parameters.push_back(e->weight);

            // attempt_edge_insert(node, new_node, mu, sig, edge_innovation_count);
        }
    }
    return true;
}

/*   ################# ################# ################# */

bool RNN_Genome::add_node(double mu, double sigma, int node_type, Distribution *dist, int32_t &edge_innovation_count, int32_t &node_innovation_count) {
    Log::info("\tattempting to add a node!\n");
    double split_depth = rng_0_1(generator);

    vector<RNN_Node_Interface*> possible_inputs;
    vector<RNN_Node_Interface*> possible_outputs;

    int32_t enabled_count = 0;
    double avg_inputs = 0.0;
    double avg_outputs = 0.0;

    for (int32_t i = 0; i < (int32_t)nodes.size(); i++) {
        if (nodes[i]->depth < split_depth && nodes[i]->is_reachable()) possible_inputs.push_back(nodes[i]);
        else if (nodes[i]->is_reachable()) possible_outputs.push_back(nodes[i]);

        if (nodes[i]->enabled) {
            enabled_count++;
            avg_inputs += nodes[i]->total_inputs;
            avg_outputs += nodes[i]->total_outputs;
        }
    }

    avg_inputs /= enabled_count;
    avg_outputs /= enabled_count;

    double input_sigma = 0.0;
    double output_sigma = 0.0;
    double temp;
    for (int32_t i = 0; i < (int32_t)nodes.size(); i++) {
        if (nodes[i]->enabled) {
            temp = (avg_inputs - nodes[i]->total_inputs);
            temp = temp * temp;
            input_sigma += temp;

            temp = (avg_outputs - nodes[i]->total_outputs);
            temp = temp * temp;
            output_sigma += temp;
        }
    }

    input_sigma /= (enabled_count - 1);
    input_sigma = sqrt(input_sigma);

    output_sigma /= (enabled_count - 1);
    output_sigma = sqrt(output_sigma);

    int32_t max_inputs = fmax(1, 2.0 + normal_distribution.random(generator, avg_inputs, input_sigma));
    int32_t max_outputs = fmax(1, 2.0 + normal_distribution.random(generator, avg_outputs, output_sigma));
    Log::info("\tadd node, split depth: %lf, max_inputs: %d, max_outputs: %d\n", split_depth, max_inputs, max_outputs);

    int32_t enabled_edges = get_enabled_edge_count();
    int32_t enabled_recurrent_edges = get_enabled_recurrent_edge_count();

    double recurrent_probability = (double)enabled_recurrent_edges / (double)(enabled_recurrent_edges + enabled_edges);
    //recurrent_probability = fmax(0.2, recurrent_probability);

    Log::info("\tadd node recurrent probability: %lf\n", recurrent_probability);

    while (possible_inputs.size() > max_inputs) {
        int32_t position = rng_0_1(generator) * possible_inputs.size();
        possible_inputs.erase(possible_inputs.begin() + position);
    }

    while (possible_outputs.size() > max_outputs) {
        int32_t position = rng_0_1(generator) * possible_outputs.size();
        possible_outputs.erase(possible_outputs.begin() + position);
    }

    RNN_Node_Interface *new_node = create_node(mu, sigma, node_type, node_innovation_count, split_depth);
    nodes.insert( upper_bound(nodes.begin(), nodes.end(), new_node, sort_RNN_Nodes_by_depth()), new_node);

    for (int32_t i = 0; i < possible_inputs.size(); i++) {
        //TODO: remove after running tests without recurrent edges
        //recurrent_probability = 0;

        if (rng_0_1(generator) < recurrent_probability) {
            attempt_recurrent_edge_insert(possible_inputs[i], new_node, mu, sigma, dist, edge_innovation_count);
        } else {
            attempt_edge_insert(possible_inputs[i], new_node, mu, sigma, edge_innovation_count);
        }
    }

    for (int32_t i = 0; i < possible_outputs.size(); i++) {
        //TODO: remove after running tests without recurrent edges
        //recurrent_probability = 0;

        if (rng_0_1(generator) < recurrent_probability) {
            attempt_recurrent_edge_insert(new_node, possible_outputs[i], mu, sigma, dist, edge_innovation_count);
        } else {
            attempt_edge_insert(new_node, possible_outputs[i], mu, sigma, edge_innovation_count);
        }
    }

    if (node_type == JORDAN_NODE || node_type == ELMAN_NODE) generate_recurrent_edges(new_node, mu, sigma, dist, edge_innovation_count);

    return true;
}

bool RNN_Genome::enable_node() {
    Log::info("\tattempting to enable a node!\n");
    vector<RNN_Node_Interface*> possible_nodes;
    for (int32_t i = 0; i < (int32_t)nodes.size(); i++) {
        if (!nodes[i]->enabled) possible_nodes.push_back(nodes[i]);
    }

    if (possible_nodes.size() == 0) return false;

    int position = rng_0_1(generator) * possible_nodes.size();
    possible_nodes[position]->enabled = true;
    Log::info("\tenabling node %d at depth %lf\n", possible_nodes[position]->innovation_number, possible_nodes[position]->depth);

    return true;
}

bool RNN_Genome::disable_node() {
    Log::info("\tattempting to disable a node!\n");
    vector<RNN_Node_Interface*> possible_nodes;
    for (int32_t i = 0; i < (int32_t)nodes.size(); i++) {
        if (nodes[i]->layer_type != OUTPUT_LAYER && nodes[i]->enabled) possible_nodes.push_back(nodes[i]);
    }

    if (possible_nodes.size() == 0) return false;

    int position = rng_0_1(generator) * possible_nodes.size();
    possible_nodes[position]->enabled = false;
    Log::info("\tdisabling node %d at depth %lf\n", possible_nodes[position]->innovation_number, possible_nodes[position]->depth);

    return true;
}

bool RNN_Genome::split_node(double mu, double sigma, int node_type, Distribution *dist, int32_t &edge_innovation_count, int32_t &node_innovation_count) {
    Log::info("\tattempting to split a node!\n");
    vector<RNN_Node_Interface*> possible_nodes;
    for (int32_t i = 0; i < (int32_t)nodes.size(); i++) {
        if (nodes[i]->layer_type != INPUT_LAYER && nodes[i]->layer_type != OUTPUT_LAYER &&
                nodes[i]->is_reachable()) {
            possible_nodes.push_back(nodes[i]);
        }
    }

    if (possible_nodes.size() == 0) return false;

    int position = rng_0_1(generator) * possible_nodes.size();
    RNN_Node_Interface *selected_node = possible_nodes[position];
    Log::info("\tselected node: %d at depth %lf\n", selected_node->innovation_number, selected_node->depth);

    vector<RNN_Edge*> input_edges;
    vector<RNN_Edge*> output_edges;
    for (int32_t i = 0; i < (int32_t)edges.size(); i++) {
        if (edges[i]->output_innovation_number == selected_node->innovation_number) {
            input_edges.push_back(edges[i]);
        }

        if (edges[i]->input_innovation_number == selected_node->innovation_number) {
            output_edges.push_back(edges[i]);
        }
    }

    vector<RNN_Recurrent_Edge*> recurrent_edges_1;
    vector<RNN_Recurrent_Edge*> recurrent_edges_2;

    for (int32_t i = 0; i < (int32_t)recurrent_edges.size(); i++) {
        if (recurrent_edges[i]->output_innovation_number == selected_node->innovation_number
                || recurrent_edges[i]->input_innovation_number == selected_node->innovation_number) {

            if (rng_0_1(generator) < 0.5) recurrent_edges_1.push_back(recurrent_edges[i]);
            if (rng_0_1(generator) < 0.5) recurrent_edges_2.push_back(recurrent_edges[i]);
        }
    }
    Log::info("\t\trecurrent_edges_1.size(): %d, recurrent_edges_2.size(): %d, input_edges.size(): %d, output_edges.size(): %d\n", recurrent_edges_1.size(), recurrent_edges_2.size(), input_edges.size(), output_edges.size());

    if (input_edges.size() == 0 || output_edges.size() == 0) {
        Log::warning("\tthe input or output edges size was 0 for the selected node, we cannot split it\n");
        //write_graphviz("error_genome.gv");
        //exit(1);
        return false;
    }

    vector<RNN_Edge*> input_edges_1;
    vector<RNN_Edge*> input_edges_2;

    for (int32_t i = 0; i < (int32_t)input_edges.size(); i++) {
        if (rng_0_1(generator) < 0.5) input_edges_1.push_back(input_edges[i]);
        if (rng_0_1(generator) < 0.5) input_edges_2.push_back(input_edges[i]);
    }

    //make sure there is at least one input edge
    if (input_edges_1.size() == 0 && input_edges.size() > 0) {
        int position = rng_0_1(generator) * input_edges.size();
        input_edges_1.push_back(input_edges[position]);
    }

    if (input_edges_2.size() == 0 && input_edges.size() > 0) {
        int position = rng_0_1(generator) * input_edges.size();
        input_edges_2.push_back(input_edges[position]);
    }

    vector<RNN_Edge*> output_edges_1;
    vector<RNN_Edge*> output_edges_2;

    for (int32_t i = 0; i < (int32_t)output_edges.size(); i++) {
        if (rng_0_1(generator) < 0.5) output_edges_1.push_back(output_edges[i]);
        if (rng_0_1(generator) < 0.5) output_edges_2.push_back(output_edges[i]);
    }

    //make sure there is at least one output edge
    if (output_edges_1.size() == 0 && output_edges.size() > 0) {
        int position = rng_0_1(generator) * output_edges.size();
        output_edges_1.push_back(output_edges[position]);
    }

    if (output_edges_2.size() == 0 && output_edges.size() > 0) {
        int position = rng_0_1(generator) * output_edges.size();
        output_edges_2.push_back(output_edges[position]);
    }

    //create the two new nodes
    double n1_avg_input = 0.0, n1_avg_output = 0.0;
    double n2_avg_input = 0.0, n2_avg_output = 0.0;

    for (int32_t i = 0; i < (int32_t)input_edges_1.size(); i++) {
        n1_avg_input += input_edges_1[i]->input_node->depth;
    }
    n1_avg_input /= input_edges_1.size();

    for (int32_t i = 0; i < (int32_t)output_edges_1.size(); i++) {
        n1_avg_output += output_edges_1[i]->output_node->depth;
    }
    n1_avg_output /= output_edges_1.size();

    for (int32_t i = 0; i < (int32_t)input_edges_2.size(); i++) {
        n2_avg_input += input_edges_2[i]->input_node->depth;
    }
    n2_avg_input /= input_edges_2.size();

    for (int32_t i = 0; i < (int32_t)output_edges_2.size(); i++) {
        n2_avg_output += output_edges_2[i]->output_node->depth;
    }
    n2_avg_output /= output_edges_2.size();

    double new_depth_1 = (n1_avg_input + n1_avg_output) / 2.0;
    double new_depth_2 = (n2_avg_input + n2_avg_output) / 2.0;

    RNN_Node_Interface *new_node_1 = create_node(mu, sigma, node_type, node_innovation_count, new_depth_1);
    RNN_Node_Interface *new_node_2 = create_node(mu, sigma, node_type, node_innovation_count, new_depth_2);

    //create the new edges
    for (int32_t i = 0; i < (int32_t)input_edges_1.size(); i++) {
        attempt_edge_insert(input_edges_1[i]->input_node, new_node_1, mu, sigma, edge_innovation_count);
    }

    for (int32_t i = 0; i < (int32_t)output_edges_1.size(); i++) {
        attempt_edge_insert(new_node_1, output_edges_1[i]->output_node, mu, sigma, edge_innovation_count);
    }

    for (int32_t i = 0; i < (int32_t)input_edges_2.size(); i++) {
        attempt_edge_insert(input_edges_2[i]->input_node, new_node_2, mu, sigma, edge_innovation_count);
    }

    for (int32_t i = 0; i < (int32_t)output_edges_2.size(); i++) {
        attempt_edge_insert(new_node_2, output_edges_2[i]->output_node, mu, sigma, edge_innovation_count);
    }

    Log::debug("\tattempting recurrent edge inserts for split node\n");

    for (int32_t i = 0; i < (int32_t)recurrent_edges_1.size(); i++) {
        if (recurrent_edges_1[i]->input_innovation_number == selected_node->innovation_number) {
            attempt_recurrent_edge_insert(new_node_1, recurrent_edges_1[i]->output_node, mu, sigma, dist, edge_innovation_count);
        } else if (recurrent_edges_1[i]->output_innovation_number == selected_node->innovation_number) {
            attempt_recurrent_edge_insert(recurrent_edges_1[i]->input_node, new_node_1, mu, sigma, dist, edge_innovation_count);
        } else {
            Log::fatal("\trecurrent edge list for split had an edge which was not connected to the selected node! This should never happen.\n");
            exit(1);
        }
        //disable the old recurrent edges
        recurrent_edges_1[i]->enabled = false;
    }

    for (int32_t i = 0; i < (int32_t)recurrent_edges_2.size(); i++) {
        if (recurrent_edges_2[i]->input_innovation_number == selected_node->innovation_number) {
            attempt_recurrent_edge_insert(new_node_2, recurrent_edges_2[i]->output_node, mu, sigma, dist, edge_innovation_count);
        } else if (recurrent_edges_2[i]->output_innovation_number == selected_node->innovation_number) {
            attempt_recurrent_edge_insert(recurrent_edges_2[i]->input_node, new_node_2, mu, sigma, dist, edge_innovation_count);
        } else {
            Log::fatal("\trecurrent edge list for split had an edge which was not connected to the selected node! This should never happen.\n");
            exit(1);
        }
        //disable the old recurrent edges
        recurrent_edges_2[i]->enabled = false;
    }

    nodes.insert( upper_bound(nodes.begin(), nodes.end(), new_node_1, sort_RNN_Nodes_by_depth()), new_node_1);
    nodes.insert( upper_bound(nodes.begin(), nodes.end(), new_node_2, sort_RNN_Nodes_by_depth()), new_node_2);

    //disable the selected node and it's edges
    for (int32_t i = 0; i < (int32_t)input_edges.size(); i++) {
        input_edges[i]->enabled = false;
    }

    for (int32_t i = 0; i < (int32_t)output_edges.size(); i++) {
        output_edges[i]->enabled = false;
    }

    selected_node->enabled = false;

    if (node_type == JORDAN_NODE || node_type == ELMAN_NODE) generate_recurrent_edges(new_node_1, mu, sigma, dist, edge_innovation_count);

    if (node_type == JORDAN_NODE || node_type == ELMAN_NODE) generate_recurrent_edges(new_node_2, mu, sigma, dist, edge_innovation_count);

    return true;
}

bool RNN_Genome::merge_node(double mu, double sigma, int node_type, Distribution *dist, int32_t &edge_innovation_count, int32_t &node_innovation_count) {
    Log::info("\tattempting to merge a node!\n");
    vector<RNN_Node_Interface*> possible_nodes;
    for (int32_t i = 0; i < (int32_t)nodes.size(); i++) {
        if (nodes[i]->layer_type != INPUT_LAYER && nodes[i]->layer_type != OUTPUT_LAYER) possible_nodes.push_back(nodes[i]);
    }

    if (possible_nodes.size() < 2) return false;

    while (possible_nodes.size() > 2) {
        int32_t position = rng_0_1(generator) * possible_nodes.size();
        possible_nodes.erase(possible_nodes.begin() + position);
    }

    RNN_Node_Interface *n1 = possible_nodes[0];
    RNN_Node_Interface *n2 = possible_nodes[1];
    n1->enabled = false;
    n2->enabled = false;

    double new_depth = (n1->depth + n2->depth) / 2.0;

    RNN_Node_Interface *new_node = create_node(mu, sigma, node_type, node_innovation_count, new_depth);
    nodes.insert( upper_bound(nodes.begin(), nodes.end(), new_node, sort_RNN_Nodes_by_depth()), new_node);

    vector<RNN_Edge*> merged_edges;
    for (int32_t i = 0; i < (int32_t)edges.size(); i++) {
        RNN_Edge *e = edges[i];

        if (e->input_innovation_number == n1->innovation_number ||
                e->input_innovation_number == n2->innovation_number ||
                e->output_innovation_number == n1->innovation_number ||
                e->output_innovation_number == n2->innovation_number) {

            //if the edge is between the two merged nodes just disasble it
            if ((e->input_innovation_number == n1->innovation_number &&
                    e->output_innovation_number == n2->innovation_number)
                    ||
                    (e->input_innovation_number == n2->innovation_number &&
                    e->output_innovation_number == n1->innovation_number)) {
                e->enabled = false;
            }

            if (e->enabled) {
                e->enabled = false;
                merged_edges.push_back(e);
            }
        }
    }

    for (int32_t i = 0; i < (int32_t)merged_edges.size(); i++) {
        RNN_Edge *e = merged_edges[i];

        RNN_Node_Interface *input_node = NULL;
        RNN_Node_Interface *output_node = NULL;

        if (e->input_innovation_number == n1->innovation_number ||
                e->input_innovation_number == n2->innovation_number) {
            input_node = new_node;
        } else {
            input_node = e->input_node;
        }

        if (e->output_innovation_number == n1->innovation_number ||
                e->output_innovation_number == n2->innovation_number) {
            output_node = new_node;
        } else {
            output_node = e->output_node;
        }

        if (input_node->depth == output_node->depth) {
            Log::info("\tskipping merged edge because the input and output nodes are the same depth\n");
            continue;
        }

        //swap the edges becasue the input node is deeper than the output node
        if (input_node->depth > output_node->depth) {
            RNN_Node_Interface *tmp = input_node;
            input_node = output_node;
            output_node = tmp;
        }

        attempt_edge_insert(input_node, output_node, mu, sigma, edge_innovation_count);
    }

    vector<RNN_Recurrent_Edge*> merged_recurrent_edges;
    for (int32_t i = 0; i < (int32_t)recurrent_edges.size(); i++) {
        RNN_Recurrent_Edge *e = recurrent_edges[i];

        if (e->input_innovation_number == n1->innovation_number ||
                e->input_innovation_number == n2->innovation_number ||
                e->output_innovation_number == n1->innovation_number ||
                e->output_innovation_number == n2->innovation_number) {

            if (e->enabled) {
                e->enabled = false;
                merged_recurrent_edges.push_back(e);
            }
        }
    }

    //add recurrent edges to merged node
    for (int32_t i = 0; i < (int32_t)merged_recurrent_edges.size(); i++) {
        RNN_Recurrent_Edge *e = merged_recurrent_edges[i];

        RNN_Node_Interface *input_node = NULL;
        RNN_Node_Interface *output_node = NULL;

        if (e->input_innovation_number == n1->innovation_number ||
                e->input_innovation_number == n2->innovation_number) {
            input_node = new_node;
        } else {
            input_node = e->input_node;
        }

        if (e->output_innovation_number == n1->innovation_number ||
                e->output_innovation_number == n2->innovation_number) {
            output_node = new_node;
        } else {
            output_node = e->output_node;
        }

        attempt_recurrent_edge_insert(input_node, output_node, mu, sigma, dist, edge_innovation_count);
    }

    if (node_type == JORDAN_NODE || node_type == ELMAN_NODE) generate_recurrent_edges(new_node, mu, sigma, dist, edge_innovation_count);

    return false;
}

string RNN_Genome::get_color(double weight, bool is_recurrent) {
    double max = 0.0;
    double min = 0.0;

    ostringstream oss;

    if (!is_recurrent) {
        for (int32_t i = 0; i < (int32_t)edges.size(); i++) {
            if (edges[i]->weight > max) {
                max = edges[i]->weight;
            } else if (edges[i]->weight < min) {
                min = edges[i]->weight;
            }
        }

    } else {
        for (int32_t i = 0; i < (int32_t)recurrent_edges.size(); i++) {
            if (recurrent_edges[i]->weight > max) {
                max = recurrent_edges[i]->weight;
            } else if (recurrent_edges[i]->weight < min) {
                min = recurrent_edges[i]->weight;
            }
        }
    }

    double value;
    if (weight <= 0) {
        value = -((weight / min) / 2.0) + 0.5;
    } else {
        value = ((weight / max) / 2.0) + 0.5;
    }
    Color color = get_colormap(value);

    Log::debug("weight: %lf, converted to value: %lf\n", weight, value);

    oss << hex << setw(2) << setfill('0') << color.red
        << hex << setw(2) << setfill('0') << color.green
        << hex << setw(2) << setfill('0') << color.blue;

    return oss.str();
}


void RNN_Genome::write_graphviz(string filename) {
    ofstream outfile(filename);

    outfile << "digraph RNN {" << endl;
    outfile << "labelloc=\"t\";" << endl;
    outfile << "label=\"Genome Fitness: " << best_validation_mae * 100.0 << "% MAE\";" << endl;
    outfile << endl;

    outfile << "\tgraph [pad=\"0.01\", nodesep=\"0.05\", ranksep=\"0.9\"];" << endl;

    int32_t input_name_index = 0;
    outfile << "\t{" << endl;
    outfile << "\t\trank = source;" << endl;
    for (uint32_t i = 0; i < nodes.size(); i++) {
        if (nodes[i]->layer_type != INPUT_LAYER) continue;
        input_name_index++;
        if (nodes[i]->total_outputs == 0) continue;
        outfile << "\t\tnode" << nodes[i]->innovation_number
            << " [shape=box,color=green,label=\"input " << nodes[i]->innovation_number
            << "\\ndepth " << nodes[i]->depth;

        if (input_parameter_names.size() != 0) {
            outfile << "\\n" << input_parameter_names[input_name_index - 1];
        }

        outfile << "\"];" << endl;
    }
    outfile << "\t}" << endl;
    outfile << endl;

    int32_t output_count = 0;
    int32_t input_count = 0;
    for (uint32_t i = 0; i < nodes.size(); i++) {
        if (nodes[i]->layer_type == OUTPUT_LAYER) output_count++;
        if (nodes[i]->layer_type == INPUT_LAYER) input_count++;
    }

    int32_t output_name_index = 0;
    outfile << "\t{" << endl;
    outfile << "\t\trank = sink;" << endl;
    for (uint32_t i = 0; i < nodes.size(); i++) {
        if (nodes[i]->layer_type != OUTPUT_LAYER) continue;
        output_name_index++;
        outfile << "\t\tnode" << nodes[i]->get_innovation_number()
            << " [shape=box,color=blue,label=\"output " << nodes[i]->innovation_number
            << "\\ndepth " << nodes[i]->depth;

        if (output_parameter_names.size() != 0) {
            outfile << "\\n" << output_parameter_names[output_name_index - 1];
        }

        outfile << "\"];" << endl;
    }
    outfile << "\t}" << endl;
    outfile << endl;

    bool printed_first = false;

    if (input_count > 1) {
        for (uint32_t i = 0; i < nodes.size(); i++) {
            if (nodes[i]->layer_type != INPUT_LAYER) continue;
            if (nodes[i]->total_outputs == 0) continue;

            if (!printed_first) {
                printed_first = true;
                outfile << "\tnode" << nodes[i]->get_innovation_number();
            } else {
                outfile << " -> node" << nodes[i]->get_innovation_number();
            }
        }
        outfile << " [style=invis];" << endl << endl;

        outfile << endl;
    }

    if (output_count > 1) {
        printed_first = false;
        for (uint32_t i = 0; i < nodes.size(); i++) {
            if (nodes[i]->layer_type != OUTPUT_LAYER) continue;

            if (!printed_first) {
                printed_first = true;
                outfile << "\tnode" << nodes[i]->get_innovation_number();
            } else {
                outfile << " -> node" << nodes[i]->get_innovation_number();
            }
        }
        outfile << " [style=invis];" << endl << endl;
        outfile << endl;
    }

    //draw the hidden nodes
    for (uint32_t i = 0; i < nodes.size(); i++) {
        if (nodes[i]->layer_type != HIDDEN_LAYER) continue;
        if (!nodes[i]->is_reachable()) continue;

        string color = "black";
        string node_type = NODE_TYPES[ nodes[i]->node_type ];

        outfile << "\t\tnode" << nodes[i]->get_innovation_number() << " [shape=box,color=" << color << ",label=\"" << node_type << " node #" << nodes[i]->get_innovation_number() << "\\ndepth " << nodes[i]->depth << "\"];" << endl;
    }
    outfile << endl;

    //draw the enabled edges
    for (uint32_t i = 0; i < edges.size(); i++) {
        if (!edges[i]->is_reachable()) continue;

        outfile << "\tnode" << edges[i]->get_input_node()->get_innovation_number() << " -> node" << edges[i]->get_output_node()->get_innovation_number() << " [color=\"#" << get_color(edges[i]->weight, false) << "\"]; /* weight: " << edges[i]->weight << " */" << endl;
    }
    outfile << endl;

    //draw the enabled recurrent edges
    for (uint32_t i = 0; i < recurrent_edges.size(); i++) {
        if (!recurrent_edges[i]->is_reachable()) continue;

        outfile << "\tnode" << recurrent_edges[i]->get_input_node()->get_innovation_number() << " -> node" << recurrent_edges[i]->get_output_node()->get_innovation_number() << " [color=\"#" << get_color(recurrent_edges[i]->weight, true) << "\",style=dotted]; /* weight: " << recurrent_edges[i]->weight << ", recurrent_depth: " << recurrent_edges[i]->recurrent_depth << " */" << endl;
    }
    outfile << endl;


    outfile << "}" << endl;
    outfile.close();

}

void read_map(istream &in, map<string, double> &m) {
    int map_size;
    in >> map_size;
    for (int i = 0; i < map_size; i++) {
        string key;
        in >> key;
        double value;
        in >> value;

        m[key] = value;
    }
}

void write_map(ostream &out, map<string, double> &m) {
    out << m.size();

    for (auto iterator = m.begin(); iterator != m.end(); iterator++) {
        out << " " << iterator->first;
        out << " " << iterator->second;
    }
}

void read_map(istream &in, map<string, int> &m) {
    int map_size;
    in >> map_size;
    for (int i = 0; i < map_size; i++) {
        string key;
        in >> key;
        int value;
        in >> value;

        m[key] = value;
    }
}

void write_map(ostream &out, map<string, int> &m) {
    out << m.size();
    for (auto iterator = m.begin(); iterator != m.end(); iterator++) {

        out << " "<< iterator->first;
        out << " "<< iterator->second;
    }
}



void write_binary_string(ostream &out, string s, string name) {
    int32_t n = s.size();
    Log::debug("writing %d %s characters '%s'\n", n, name.c_str(), s.c_str());
    out.write((char*)&n, sizeof(int32_t));
    out.write((char*)&s[0], sizeof(char) * s.size());
}

void read_binary_string(istream &in, string &s, string name) {
    int32_t n;
    in.read((char*)&n, sizeof(int32_t));

    Log::debug("reading %d %s characters.\n", n, name.c_str());
    char* s_v = new char[n];
    in.read((char*)s_v, sizeof(char) * n);
    s.assign(s_v, s_v + n);
    delete [] s_v;

    Log::debug("read %d %s characters '%s'\n", n, name.c_str(), s.c_str());
}


RNN_Genome::RNN_Genome(string binary_filename) {
    ifstream bin_infile(binary_filename, ios::in | ios::binary);

    if (!bin_infile.good()) {
        Log::fatal("ERROR: could not open RNN genome file '%s' for reading.\n", binary_filename.c_str());
        exit(1);
    }

    read_from_stream(bin_infile);
    bin_infile.close();
}

RNN_Genome::RNN_Genome(char *array, int32_t length) {
    read_from_array(array, length);
}

RNN_Genome::RNN_Genome(istream &bin_infile) {
    read_from_stream(bin_infile);
}

void RNN_Genome::read_from_array(char *array, int32_t length) {
    string array_str;
    for (uint32_t i = 0; i < length; i++) {
        array_str.push_back(array[i]);
    }

    istringstream iss(array_str);
    read_from_stream(iss);
}

void RNN_Genome::read_from_stream(istream &bin_istream) {
    Log::debug("READING GENOME FROM STREAM\n");

    bin_istream.read((char*)&generation_id, sizeof(int32_t));
    bin_istream.read((char*)&group_id, sizeof(int32_t));
    bin_istream.read((char*)&bp_iterations, sizeof(int32_t));
    bin_istream.read((char*)&learning_rate, sizeof(double));
    bin_istream.read((char*)&adapt_learning_rate, sizeof(bool));
    bin_istream.read((char*)&use_nesterov_momentum, sizeof(bool));
    bin_istream.read((char*)&use_reset_weights, sizeof(bool));

    bin_istream.read((char*)&use_high_norm, sizeof(bool));
    bin_istream.read((char*)&high_threshold, sizeof(double));
    bin_istream.read((char*)&use_low_norm, sizeof(bool));
    bin_istream.read((char*)&low_threshold, sizeof(double));

    bin_istream.read((char*)&use_dropout, sizeof(bool));
    bin_istream.read((char*)&dropout_probability, sizeof(double));

    Log::debug("generation_id: %d\n", generation_id);
    Log::debug("bp_iterations: %d\n", bp_iterations);
    Log::debug("learning_rate: %lf\n", learning_rate);
    Log::debug("adapt_learning_rate: %d\n", adapt_learning_rate);
    Log::debug("use_nesterov_momentum: %d\n", use_nesterov_momentum);
    Log::debug("use_reset_weights: %d\n", use_reset_weights);

    Log::debug("use_high_norm: %d\n", use_high_norm);
    Log::debug("high_threshold: %lf\n", high_threshold);
    Log::debug("use_low_norm: %d\n", use_low_norm);
    Log::debug("low_threshold: %lf\n", low_threshold);

    Log::debug("use_dropout: %d\n", use_dropout);
    Log::debug("dropout_probability: %lf\n", dropout_probability);

    read_binary_string(bin_istream, log_filename, "log_filename");
    string generator_str;
    read_binary_string(bin_istream, generator_str, "generator");
    istringstream generator_iss(generator_str);
    generator_iss >> generator;

    string rng_0_1_str;
    read_binary_string(bin_istream, rng_0_1_str, "rng_0_1");
    // So for some reason this was serialized incorrectly for some genomes,
    // but the value should always be the same so we really don't need to de-serialize it anways and can just
    // assign it a constant value
    rng_0_1 = uniform_real_distribution<double>(0.0, 1.0);
    // Formerly:
    // istringstream rng_0_1_iss(rng_0_1_str);
    //rng_0_1_iss >> rng_0_1;
     

    string generated_by_map_str;
    read_binary_string(bin_istream, generated_by_map_str, "generated_by_map");
    istringstream generated_by_map_iss(generated_by_map_str);
    read_map(generated_by_map_iss, generated_by_map);

    bin_istream.read((char*)&best_validation_mse, sizeof(double));
    bin_istream.read((char*)&best_validation_mae, sizeof(double));

    int32_t n_initial_parameters;
    bin_istream.read((char*)&n_initial_parameters, sizeof(int32_t));
    Log::debug("reading %d initial parameters.\n", n_initial_parameters);
    double* initial_parameters_v = new double[n_initial_parameters];
    bin_istream.read((char*)initial_parameters_v, sizeof(double) * n_initial_parameters);
    initial_parameters.assign(initial_parameters_v, initial_parameters_v + n_initial_parameters);
    delete [] initial_parameters_v;

    int32_t n_best_parameters;
    bin_istream.read((char*)&n_best_parameters, sizeof(int32_t));
    Log::debug("reading %d best parameters.\n", n_best_parameters);
    double* best_parameters_v = new double[n_best_parameters];
    bin_istream.read((char*)best_parameters_v, sizeof(double) * n_best_parameters);
    best_parameters.assign(best_parameters_v, best_parameters_v + n_best_parameters);
    delete [] best_parameters_v;


    input_parameter_names.clear();
    int32_t n_input_parameter_names;
    bin_istream.read((char*)&n_input_parameter_names, sizeof(int32_t));
    Log::debug("reading %d input parameter names.\n", n_input_parameter_names);
    for (int32_t i = 0; i < n_input_parameter_names; i++) {
        string input_parameter_name;
        read_binary_string(bin_istream, input_parameter_name, "input_parameter_names[" + std::to_string(i) + "]");
        input_parameter_names.push_back(input_parameter_name);
    }

    output_parameter_names.clear();
    int32_t n_output_parameter_names;
    bin_istream.read((char*)&n_output_parameter_names, sizeof(int32_t));
    Log::debug("reading %d output parameter names.\n", n_output_parameter_names);
    for (int32_t i = 0; i < n_output_parameter_names; i++) {
        string output_parameter_name;
        read_binary_string(bin_istream, output_parameter_name, "output_parameter_names[" + std::to_string(i) + "]");
        output_parameter_names.push_back(output_parameter_name);
    }



    int32_t n_nodes;
    bin_istream.read((char*)&n_nodes, sizeof(int32_t));
    Log::debug("reading %d nodes.\n", n_nodes);

    nodes.clear();
    for (int32_t i = 0; i < n_nodes; i++) {
        int32_t innovation_number;
        int32_t type;
        int32_t node_type;
        double depth;
        bool enabled;

        bin_istream.read((char*)&innovation_number, sizeof(int32_t));
        bin_istream.read((char*)&type, sizeof(int32_t));
        bin_istream.read((char*)&node_type, sizeof(int32_t));
        bin_istream.read((char*)&depth, sizeof(double));
        bin_istream.read((char*)&enabled, sizeof(bool));

        Log::debug("NODE: %d %d %lf %d\n", innovation_number, type, node_type, depth, enabled);

        RNN_Node_Interface *node;
        if (node_type == LSTM_NODE) {
            node = new LSTM_Node(innovation_number, type, depth);
        } else if (node_type == DELTA_NODE) {
            node = new Delta_Node(innovation_number, type, depth);
        } else if (node_type == GRU_NODE) {
            node = new GRU_Node(innovation_number, type, depth);
        } else if (node_type == MGU_NODE) {
            node = new MGU_Node(innovation_number, type, depth);
        } else if (node_type == UGRNN_NODE) {
            node = new UGRNN_Node(innovation_number, type, depth);
        } else if (node_type == SIMPLE_NODE || node_type == JORDAN_NODE || node_type == ELMAN_NODE) {
            node = new RNN_Node(innovation_number, type, depth, node_type);
        } else {
            Log::fatal("Error reading node from stream, unknown node_type: %d\n", node_type);
            exit(1);
        }

        node->enabled = enabled;
        nodes.push_back(node);
    }


    int32_t n_edges;
    bin_istream.read((char*)&n_edges, sizeof(int32_t));
    Log::debug("reading %d edges.\n", n_edges);

    edges.clear();
    for (int32_t i = 0; i < n_edges; i++) {
        int32_t innovation_number;
        int32_t input_innovation_number;
        int32_t output_innovation_number;
        bool enabled;

        bin_istream.read((char*)&innovation_number, sizeof(int32_t));
        bin_istream.read((char*)&input_innovation_number, sizeof(int32_t));
        bin_istream.read((char*)&output_innovation_number, sizeof(int32_t));
        bin_istream.read((char*)&enabled, sizeof(bool));

        Log::debug("EDGE: %d %d %d %d\n", innovation_number, input_innovation_number, output_innovation_number, enabled);

        RNN_Edge *edge = new RNN_Edge(innovation_number, input_innovation_number, output_innovation_number, nodes);
        edge->enabled = enabled;
        edges.push_back(edge);
    }


    int32_t n_recurrent_edges;
    bin_istream.read((char*)&n_recurrent_edges, sizeof(int32_t));
    Log::debug("reading %d recurrent_edges.\n", n_recurrent_edges);

    recurrent_edges.clear();
    for (int32_t i = 0; i < n_recurrent_edges; i++) {
        int32_t innovation_number;
        int32_t recurrent_depth;
        int32_t input_innovation_number;
        int32_t output_innovation_number;
        bool enabled;

        bin_istream.read((char*)&innovation_number, sizeof(int32_t));
        bin_istream.read((char*)&recurrent_depth, sizeof(int32_t));
        bin_istream.read((char*)&input_innovation_number, sizeof(int32_t));
        bin_istream.read((char*)&output_innovation_number, sizeof(int32_t));
        bin_istream.read((char*)&enabled, sizeof(bool));

        Log::debug("RECURRENT EDGE: %d %d %d %d %d\n", innovation_number, recurrent_depth, input_innovation_number, output_innovation_number, enabled);

        RNN_Recurrent_Edge *recurrent_edge = new RNN_Recurrent_Edge(innovation_number, recurrent_depth, input_innovation_number, output_innovation_number, nodes);
        recurrent_edge->enabled = enabled;
        recurrent_edges.push_back(recurrent_edge);
    }

    string normalize_mins_str;
    read_binary_string(bin_istream, normalize_mins_str, "normalize_mins");
    istringstream normalize_mins_iss(normalize_mins_str);
    read_map(normalize_mins_iss, normalize_mins);

    string normalize_maxs_str;
    read_binary_string(bin_istream, normalize_maxs_str, "normalize_maxs");
    istringstream normalize_maxs_iss(normalize_maxs_str);
    read_map(normalize_maxs_iss, normalize_maxs);

    assign_reachability();
}

void RNN_Genome::write_to_array(char **bytes, int32_t &length) {
    ostringstream oss;
    write_to_stream(oss);

    string bytes_str = oss.str();
    length = bytes_str.size();
    (*bytes) = (char*)malloc(length * sizeof(char));
    for (uint32_t i = 0; i < length; i++) {
        (*bytes)[i] = bytes_str[i];
    }
}

void RNN_Genome::write_to_file(string bin_filename) {
    ofstream bin_outfile(bin_filename, ios::out | ios::binary);
    write_to_stream(bin_outfile);
    bin_outfile.close();
}

#define checkpoint(name) ; //printf("checkpoint %d\n", name++)

void RNN_Genome::write_to_stream(ostream &bin_ostream) {
    Log::debug("WRITING GENOME TO STREAM\n");
    int x = 0;
    checkpoint(x);
    bin_ostream.write((char*)&generation_id, sizeof(int32_t));
    bin_ostream.write((char*)&group_id, sizeof(int32_t));
    bin_ostream.write((char*)&bp_iterations, sizeof(int32_t));
    bin_ostream.write((char*)&learning_rate, sizeof(double));
    bin_ostream.write((char*)&adapt_learning_rate, sizeof(bool));
    bin_ostream.write((char*)&use_nesterov_momentum, sizeof(bool));
    bin_ostream.write((char*)&use_reset_weights, sizeof(bool));

    checkpoint(x);
    bin_ostream.write((char*)&use_high_norm, sizeof(bool));
    bin_ostream.write((char*)&high_threshold, sizeof(double));
    bin_ostream.write((char*)&use_low_norm, sizeof(bool));
    bin_ostream.write((char*)&low_threshold, sizeof(double));

    checkpoint(x);
    bin_ostream.write((char*)&use_dropout, sizeof(bool));
    bin_ostream.write((char*)&dropout_probability, sizeof(double));

    Log::debug("generation_id: %d\n", generation_id);
    Log::debug("bp_iterations: %d\n", bp_iterations);
    Log::debug("learning_rate: %lf\n", learning_rate);
    Log::debug("adapt_learning_rate: %d\n", adapt_learning_rate);
    Log::debug("use_nesterov_momentum: %d\n", use_nesterov_momentum);
    Log::debug("use_reset_weights: %d\n", use_reset_weights);

    Log::debug("use_high_norm: %d\n", use_high_norm);
    Log::debug("high_threshold: %lf\n", high_threshold);
    Log::debug("use_low_norm: %d\n", use_low_norm);
    Log::debug("low_threshold: %lf\n", low_threshold);

    Log::debug("use_dropout: %d\n", use_dropout);
    Log::debug("dropout_probability: %lf\n", dropout_probability);

    checkpoint(x);
    write_binary_string(bin_ostream, log_filename, "log_filename");

    ostringstream generator_oss;
    generator_oss << generator;
    string generator_str = generator_oss.str();
    write_binary_string(bin_ostream, generator_str, "generator");

    checkpoint(x);
    ostringstream rng_0_1_oss;
    rng_0_1_oss << rng_0_1;
    string rng_0_1_str = rng_0_1_oss.str();
    write_binary_string(bin_ostream, rng_0_1_str, "rng_0_1");

    checkpoint(x);
    ostringstream generated_by_map_oss;
    write_map(generated_by_map_oss, generated_by_map);
    string generated_by_map_str = generated_by_map_oss.str();
    write_binary_string(bin_ostream, generated_by_map_str, "generated_by_map");

    bin_ostream.write((char*)&best_validation_mse, sizeof(double));
    bin_ostream.write((char*)&best_validation_mae, sizeof(double));

    int32_t n_initial_parameters = initial_parameters.size();
    Log::debug("writing %d initial parameters.\n", n_initial_parameters);
    bin_ostream.write((char*)&n_initial_parameters, sizeof(int32_t));
    bin_ostream.write((char*)&initial_parameters[0], sizeof(double) * initial_parameters.size());

    checkpoint(x);
    int32_t n_best_parameters = best_parameters.size();
    bin_ostream.write((char*)&n_best_parameters, sizeof(int32_t));
    if (n_best_parameters)
        bin_ostream.write((char*)&best_parameters[0], sizeof(double) * best_parameters.size());


    int32_t n_input_parameter_names = input_parameter_names.size();
    bin_ostream.write((char*)&n_input_parameter_names, sizeof(int32_t));
    for (int32_t i = 0; i < input_parameter_names.size(); i++) {
        write_binary_string(bin_ostream, input_parameter_names[i], "input_parameter_names[" + std::to_string(i) + "]");
    }

    checkpoint(x);
    int32_t n_output_parameter_names = output_parameter_names.size();
    bin_ostream.write((char*)&n_output_parameter_names, sizeof(int32_t));
    for (int32_t i = 0; i < output_parameter_names.size(); i++) {
        write_binary_string(bin_ostream, output_parameter_names[i], "output_parameter_names[" + std::to_string(i) + "]");
    }

    checkpoint(x);
    int32_t n_nodes = nodes.size();
    bin_ostream.write((char*)&n_nodes, sizeof(int32_t));
    Log::debug("writing %d nodes.\n", n_nodes);

    for (int32_t i = 0; i < nodes.size(); i++) {
        Log::debug("NODE: %d %d %d %d\n", nodes[i]->innovation_number, nodes[i]->layer_type, nodes[i]->node_type, nodes[i]->depth);
        nodes[i]->write_to_stream(bin_ostream);
    }

    checkpoint(x);

    int32_t n_edges = edges.size();
    bin_ostream.write((char*)&n_edges, sizeof(int32_t));
    Log::debug("writing %d edges.\n", n_edges);

    for (int32_t i = 0; i < edges.size(); i++) {
        Log::debug("EDGE: %d %d %d\n", edges[i]->innovation_number, edges[i]->input_innovation_number, edges[i]->output_innovation_number);
        edges[i]->write_to_stream(bin_ostream);
    }


    checkpoint(x);
    int32_t n_recurrent_edges = recurrent_edges.size();
    bin_ostream.write((char*)&n_recurrent_edges, sizeof(int32_t));
    Log::debug("writing %d recurrent edges.\n", n_recurrent_edges);

    for (int32_t i = 0; i < recurrent_edges.size(); i++) {
        Log::debug("RECURRENT EDGE: %d %d %d %d\n", recurrent_edges[i]->innovation_number, recurrent_edges[i]->recurrent_depth, recurrent_edges[i]->input_innovation_number, recurrent_edges[i]->output_innovation_number);

        recurrent_edges[i]->write_to_stream(bin_ostream);
    }

    checkpoint(x);
    ostringstream normalize_mins_oss;
    write_map(normalize_mins_oss, normalize_mins);
    string normalize_mins_str = normalize_mins_oss.str();
    write_binary_string(bin_ostream, normalize_mins_str, "normalize_mins");

    ostringstream normalize_maxs_oss;
    write_map(normalize_maxs_oss, normalize_maxs);
    string normalize_maxs_str = normalize_maxs_oss.str();
    write_binary_string(bin_ostream, normalize_maxs_str, "normalize_maxs");
}

void RNN_Genome::update_innovation_counts(int32_t &node_innovation_count, int32_t &edge_innovation_count) {
    int32_t max_node_innovation_count = -1;

    for (int32_t i = 0; i < this->nodes.size(); i += 1) {
        RNN_Node_Interface *node = this->nodes[i];
        max_node_innovation_count = std::max(max_node_innovation_count, node->innovation_number);
    }

    int32_t max_edge_innovation_count = -1;
    for (int32_t i = 0; i < this->edges.size(); i += 1) {
        RNN_Edge *edge = this->edges[i];
        max_edge_innovation_count = std::max(max_edge_innovation_count, edge->innovation_number);
    }
    for (int32_t i = 0; i < this->recurrent_edges.size(); i += 1) {
        RNN_Recurrent_Edge *redge = this->recurrent_edges[i];
        max_edge_innovation_count = std::max(max_edge_innovation_count, redge->innovation_number);
    }

    if (max_node_innovation_count == -1) {
        // Fatal log message
        Log::fatal("Seed genome had max node innovation number of -1 - this should never happen (unless the genome is empty :)");
    }
    if (max_edge_innovation_count == -1) {
        // Fatal log message
        Log::fatal("Seed genome had max node innovation number of -1 - this should never happen (and the genome isn't empty since max_node_innovation_count > -1)");
    }

    // One more than the highest we've seen should be good enough.
    node_innovation_count = max_node_innovation_count + 1;
    edge_innovation_count = max_edge_innovation_count + 1;
}
