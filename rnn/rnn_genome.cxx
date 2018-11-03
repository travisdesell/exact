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
using std::cout;
using std::cerr;
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

#include <vector>
using std::vector;

#include "common/random.hxx"
#include "common/color_table.hxx"

#include "rnn.hxx"
#include "rnn_node.hxx"
#include "lstm_node.hxx"
#include "rnn_genome.hxx"

void RNN_Genome::sort_nodes_by_depth() {
    sort(nodes.begin(), nodes.end(), sort_RNN_Nodes_by_depth());
}

void RNN_Genome::sort_edges_by_depth() {
    sort(edges.begin(), edges.end(), sort_RNN_Edges_by_depth());
}

RNN_Genome::RNN_Genome(vector<RNN_Node_Interface*> &_nodes, vector<RNN_Edge*> &_edges) {
    generation_id = -1;
    best_validation_error = EXALT_MAX_DOUBLE;
    best_validation_mae = EXALT_MAX_DOUBLE;

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

    other->best_validation_error = best_validation_error;
    other->best_validation_mae = best_validation_mae;
    other->best_parameters = best_parameters;

    other->input_parameter_names = input_parameter_names;
    other->output_parameter_names = output_parameter_names;


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

int32_t RNN_Genome::get_enabled_node_count() {
    int32_t count = 0;

    for (int32_t i = 0; i < nodes.size(); i++) {
        if (nodes[i]->enabled) count++;
    }

    return count;
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

string RNN_Genome::generated_by_string() {
    ostringstream oss;
    oss << "[";
    bool first = true;
    for (auto i = generated_by_map.begin(); i != generated_by_map.end(); i++) {
        if (!first) oss << ", ";
        oss << i->first << ":" << generated_by_map[i->first];
        first = false;
    }
    oss << "]";

    return oss.str();
}

void RNN_Genome::set_bp_iterations(int32_t _bp_iterations) {
    bp_iterations = _bp_iterations;
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
        cerr << "ERROR! Trying to set weights where the RNN has " << get_number_weights() << " weights, and the parameters vector has << " << parameters.size() << " weights!" << endl;
        exit(1);
    }

    uint32_t current = 0;

    for (uint32_t i = 0; i < nodes.size(); i++) {
        nodes[i]->set_weights(current, parameters);
        //if (nodes[i]->is_reachable()) nodes[i]->set_weights(current, parameters);
    }

    for (uint32_t i = 0; i < edges.size(); i++) {
        edges[i]->weight = parameters[current++];
        //if (edges[i]->is_reachable()) edges[i]->weight = parameters[current++];
    }

    for (uint32_t i = 0; i < recurrent_edges.size(); i++) {
        recurrent_edges[i]->weight = parameters[current++];
        //if (recurrent_edges[i]->is_reachable()) recurrent_edges[i]->weight = parameters[current++];
    }

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
    //cout << "initializing randomly!" << endl;
    int number_of_weights = get_number_weights();
    initial_parameters.assign(number_of_weights, 0.0);

    uniform_real_distribution<double> rng(-0.5, 0.5);
    for (uint32_t i = 0; i < initial_parameters.size(); i++) {
        initial_parameters[i] = rng(generator);
    }
}


RNN* RNN_Genome::get_rnn() {
    vector<RNN_Node_Interface*> node_copies;
    vector<RNN_Edge*> edge_copies;
    vector<RNN_Recurrent_Edge*> recurrent_edge_copies;

    for (uint32_t i = 0; i < nodes.size(); i++) {
        node_copies.push_back( nodes[i]->copy() );
        //if (nodes[i]->type == RNN_INPUT_NODE || nodes[i]->type == RNN_OUTPUT_NODE || nodes[i]->is_reachable()) node_copies.push_back( nodes[i]->copy() );
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

int32_t RNN_Genome::get_generation_id() const {
    return generation_id;
}

void RNN_Genome::set_generation_id(int32_t _generation_id) {
    generation_id = _generation_id;
}

double RNN_Genome::get_validation_error() const {
    return best_validation_error;
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
    //cout << "mse[" << i << "]: " << mse_current << endl;
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
    double validation_error = get_mse(parameters, validation_inputs, validation_outputs);
    best_validation_error = validation_error;
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
        validation_error = get_mse(parameters, validation_inputs, validation_outputs);
        if (validation_error < best_validation_error) {
            best_validation_error = validation_error;
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
                << " " << validation_error
                << " " << best_validation_error << endl;
        }

        cout << "iteration " << setw(10) << iteration
             << ", mse: " << setw(10) << mse 
             << ", v_mse: " << setw(10) << validation_error 
             << ", bv_mse: " << setw(10) << best_validation_error 
             << ", lr: " << setw(10) << learning_rate 
             << ", norm: " << setw(10) << norm
             << ", p_norm: " << setw(10) << parameter_norm
             << ", v_norm: " << setw(10) << velocity_norm;

        if (use_reset_weights && prev_mse * 1.25 < mse) {
            cout << ", RESETTING WEIGHTS " << reset_count << endl;
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

                cout << ", INCREASING LR";
            }
        }

        if (use_high_norm && norm > high_threshold) {
            double high_threshold_norm = high_threshold / norm;

            cout << ", OVER THRESHOLD, multiplier: " << high_threshold_norm;

            for (int32_t i = 0; i < parameters.size(); i++) {
                analytic_gradient[i] = high_threshold_norm * analytic_gradient[i];
            }

            if (adapt_learning_rate) {
                learning_rate *= 0.5;
                if (learning_rate < 0.0000001) learning_rate = 0.0000001;
            }

        } else if (use_low_norm && norm < low_threshold) {
            double low_threshold_norm = low_threshold / norm;
            cout << ", UNDER THRESHOLD, multiplier: " << low_threshold_norm;

            for (int32_t i = 0; i < parameters.size(); i++) {
                analytic_gradient[i] = low_threshold_norm * analytic_gradient[i];
            }

            if (adapt_learning_rate) {
                if (prev_mse * 1.05 < mse) {
                    cout << ", WORSE";
                    learning_rate *= 0.5;
                    if (learning_rate < 0.0000001) learning_rate = 0.0000001;
                }
            }
        }

        if (reset_count > 0) {
            double reset_penalty = pow(5.0, -reset_count);
            cout << ", RESET PENALTY (" << reset_count << "): " << reset_penalty;

            for (int32_t i = 0; i < parameters.size(); i++) {
                analytic_gradient[i] = reset_penalty * analytic_gradient[i];
            }

        }


        cout << endl;

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
        //cout << "getting analytic gradient for input/output: " << i << ", n_series: " << n_series << ", parameters.size: " << parameters.size() << ", log filename: " << log_filename << endl;
        //cout << "inputs.size(): " << inputs.size()  << ", outputs.size(): " << outputs.size() << ", log filename: " << log_filename << endl;

        rnn->get_analytic_gradient(parameters, inputs[i], outputs[i], mse, analytic_gradient, use_dropout, true, dropout_probability);
        //cout << "got analytic gradient, inputs.size(): " << inputs.size()  << ", outputs.size(): " << outputs.size() << ", log filename: " << log_filename << endl;

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
    //cout << "initialized previous values on: " << log_filename << endl;

    //TODO: need to get validation error on the RNN not the genome
    double validation_error = get_mse(parameters, validation_inputs, validation_outputs, false);
    best_validation_error = validation_error;
    best_validation_mae = get_mae(parameters, validation_inputs, validation_outputs);
    best_parameters = parameters;

    //cout << "got initial errors on: " << log_filename << endl;

    /*
    cout << "initial validation_error: " << validation_error << endl;
    cout << "best validation error: " << best_validation_error << endl;
    double m = 0.0, s = 0.0;
    get_mu_sigma(parameters, m, s);
    for (int32_t i = 0; i < parameters.size(); i++) {
        cout << "parameters[" << i << "]: " << parameters[i] << endl;
    }
    */

    unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
    minstd_rand0 generator(seed);
    uniform_real_distribution<double> rng(0, 1);

    int random_selection = rng(generator);
    mu = prev_mu[random_selection];
    norm = prev_norm[random_selection];
    mse = prev_mse[random_selection];
    learning_rate = prev_learning_rate[random_selection];

    ofstream *output_log = NULL;
    
    if (log_filename != "") {
        //cout << "craeting new log stream for " << log_filename << endl;
        output_log = new ofstream(log_filename);
        //cout << "testing to see if log file valid for " << log_filename << endl;

        if (!output_log->is_open()) {
            cerr << "ERROR, could not open output log: '" << log_filename << "'" << endl;
            exit(1);
        }

        //cout << "opened log file '" << log_filename << "'" << endl;
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

            /*
            cout << "iteration " << iteration
                << ", series: " << random_selection
                << ", mse: " << mse 
                << ", lr: " << learning_rate 
                << ", norm: " << norm;
                */

            if (use_reset_weights && prev_mse[random_selection] * 2 < mse) {
                //cout << ", RESETTING WEIGHTS" << endl;
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

                    //cout << ", INCREASING LR";
                }
            }

            if (use_high_norm && norm > high_threshold) {
                double high_threshold_norm = high_threshold / norm;
                //cout << ", OVER THRESHOLD, multiplier: " << high_threshold_norm;

                for (int32_t i = 0; i < parameters.size(); i++) {
                    analytic_gradient[i] = high_threshold_norm * analytic_gradient[i];
                }

                if (adapt_learning_rate) {
                    learning_rate *= 0.5;
                    if (learning_rate < 0.0000001) learning_rate = 0.0000001;
                }

            } else if (use_low_norm && norm < low_threshold) {
                double low_threshold_norm = low_threshold / norm;
                //cout << ", UNDER THRESHOLD, multiplier: " << low_threshold_norm;

                for (int32_t i = 0; i < parameters.size(); i++) {
                    analytic_gradient[i] = low_threshold_norm * analytic_gradient[i];
                }

                if (adapt_learning_rate) {
                    if (prev_mse[random_selection] * 1.05 < mse) {
                        cout << ", WORSE";
                        learning_rate *= 0.5;
                        if (learning_rate < 0.0000001) learning_rate = 0.0000001;
                    }
                }
            }

            //cout << endl;

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
        validation_error = get_mse(parameters, validation_inputs, validation_outputs);
        if (validation_error < best_validation_error) {
            best_validation_error = validation_error;
            best_validation_mae = get_mae(parameters, validation_inputs, validation_outputs);

            best_parameters = parameters;
        }

        if (output_log != NULL) {
            std::chrono::time_point<std::chrono::system_clock> currentClock = std::chrono::system_clock::now();
            long milliseconds = std::chrono::duration_cast<std::chrono::milliseconds>(currentClock - startClock).count();

            (*output_log) << iteration
                << "," << milliseconds
                << "," << training_error
                << "," << validation_error
                << "," << best_validation_error
                << "," << best_validation_mae
                << "," << avg_norm << endl;

        }


        cout << "iteration " << setw(5) << iteration << ", mse: " << training_error << ", v_mse: " << validation_error << ", bv_mse: " << best_validation_error << ", avg_norm: " << avg_norm << endl;

    }

    delete rnn;

    this->set_weights(best_parameters);
    cout << "backpropagation completed, getting mu/sigma" << endl;
    double _mu, _sigma;
    get_mu_sigma(best_parameters, _mu, _sigma);
}

double RNN_Genome::get_mse(const vector<double> &parameters, const vector< vector< vector<double> > > &inputs, const vector< vector< vector<double> > > &outputs, bool verbose) {
    RNN *rnn = get_rnn();
    rnn->set_weights(parameters);

    double mse = 0.0;
    double avg_mse = 0.0;

    int32_t width = ceil(log10(inputs.size()));
    for (uint32_t i = 0; i < inputs.size(); i++) {
        mse = rnn->prediction_mse(inputs[i], outputs[i], use_dropout, false, dropout_probability);

        avg_mse += mse;

        if (verbose) {
            cout << "series[" << setw(width) << i << "] MSE:  " << mse << endl;
        }
    }

    delete rnn;

    avg_mse /= inputs.size();
    if (verbose) {
        cout << "average MSE:   " << string(width, ' ') << avg_mse << endl;
    }
    return avg_mse;
}

double RNN_Genome::get_mae(const vector<double> &parameters, const vector< vector< vector<double> > > &inputs, const vector< vector< vector<double> > > &outputs, bool verbose) {
    RNN *rnn = get_rnn();
    rnn->set_weights(parameters);

    double mae;
    double avg_mae = 0.0;

    int32_t width = ceil(log10(inputs.size()));
    for (uint32_t i = 0; i < inputs.size(); i++) {
        mae = rnn->prediction_mae(inputs[i], outputs[i], use_dropout, false, dropout_probability);

        avg_mae += mae;

        if (verbose) {
            cout << "series[" << setw(width) << i << "] MAE:  " << mae << endl;
        }
    }

    delete rnn;

    avg_mae /= inputs.size();
    if (verbose) {
        cout << "average MAE:   " << string(width, ' ') << avg_mae << endl;
    }
    return avg_mae;
}

void RNN_Genome::write_predictions(const vector<string> &input_filenames, const vector<double> &parameters, const vector< vector< vector<double> > > &inputs, const vector< vector< vector<double> > > &outputs) {
    RNN *rnn = get_rnn();
    rnn->set_weights(parameters);

    for (uint32_t i = 0; i < inputs.size(); i++) {
        cout << "input filename[" << i << "]: " << input_filenames[i] << endl;

        string output_filename = "predictions_" + std::to_string(i) + ".txt";
        cout << "output filename: " << output_filename << endl;

        rnn->write_predictions(output_filename, input_parameter_names, output_parameter_names, inputs[i], outputs[i], use_dropout, dropout_probability);
    }

    delete rnn;
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
    //cout << "assigning reachability!" << endl;
    //cout << nodes.size() << " nodes, " << edges.size() << " edges, " << recurrent_edges.size() << " recurrent_edges" << endl;

    for (int32_t i = 0; i < (int32_t)nodes.size(); i++) {
        nodes[i]->forward_reachable = false;
        nodes[i]->backward_reachable = false;
        nodes[i]->total_inputs = 0;
        nodes[i]->total_outputs = 0;

        //set enabled input nodes as reachable
        if (nodes[i]->type == RNN_INPUT_NODE && nodes[i]->enabled) {
            nodes[i]->forward_reachable = true;
            nodes[i]->total_inputs = 1;
            
            //cout << "\tsetting input node[" << i << "] reachable" << endl;
        }

        if (nodes[i]->type == RNN_OUTPUT_NODE) {
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
        if (nodes[i]->type == RNN_INPUT_NODE && nodes[i]->enabled) {
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
                            cerr << "ERROR, forward edge was circular -- this should never happen" << endl;
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
        if (nodes[i]->type == RNN_OUTPUT_NODE && nodes[i]->enabled) {
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

    /*
    cout << "node reachabiltity:" << endl;
    for (int32_t i = 0; i < nodes.size(); i++) {
        RNN_Node_Interface *n = nodes[i];
        cout << "node " << n->innovation_number << ", e: " << n->enabled << ", fr: " << n->forward_reachable << ", br: " << n->backward_reachable << ", ti: " << n->total_inputs << ", to: " << n->total_outputs << endl;
    }

    cout << "edge reachability:" << endl;
    for (int32_t i = 0; i < edges.size(); i++) {
        RNN_Edge *e = edges[i];
        cout << "edge " << e->innovation_number << ", e: " << e->enabled << ", fr: " << e->forward_reachable << ", br: " << e->backward_reachable << endl;
    }

    cout << "recurrent edge reachability:" << endl;
    for (int32_t i = 0; i < recurrent_edges.size(); i++) {
        RNN_Recurrent_Edge *e = recurrent_edges[i];
        cout << "recurrent edge " << e->innovation_number << ", e: " << e->enabled << ", fr: " << e->forward_reachable << ", br: " << e->backward_reachable << endl;
    }
    */
}


bool RNN_Genome::outputs_unreachable() {
    assign_reachability();

    for (int32_t i = 0; i < (int32_t)nodes.size(); i++) {
        if (nodes[i]->type == RNN_OUTPUT_NODE && !nodes[i]->is_reachable()) return true;
    }

    return false;
}

void RNN_Genome::get_mu_sigma(const vector<double> &p, double &mu, double &sigma) {
    if (p.size() == 0) {
        mu = 0.0;
        sigma = 0.25;
        cout << "\tmu: " << mu << ", sigma: " << sigma << ", parameters.size() == 0" << endl;
        return;
    }

    mu = 0.0;
    sigma = 0.0;
    
    for (int32_t i = 0; i < p.size(); i++) {
        if (p[i] < -10 || p[i] > 10) {
            cerr << "ERROR in get_mu_sigma, parameter[" << i << "]: was out of bounds: " << p[i] << endl;
            cerr << "all parameters: " << endl;
            for (int32_t i = 0; i < (int32_t)p.size(); i++) { 
                cerr << "\t" << p[i] << endl;
            }
            exit(1);
        }

        mu += p[i];
    }
    mu /= p.size();

    double temp;
    for (int32_t i = 0; i < p.size(); i++) {
        temp = (mu - p[i]) * (mu - p[i]);
        sigma += temp;
    }

    sigma /= (p.size() - 1);
    sigma = sqrt(sigma);

    cout << "\tmu: " << mu << ", sigma: " << sigma << ", parameters.size(): " << p.size() << endl;
    if (std::isnan(mu) || std::isinf(mu) || std::isnan(sigma) || std::isinf(sigma)) {
        cerr << "mu or sigma was not a number, best parameters: " << endl;
        for (int32_t i = 0; i < (int32_t)p.size(); i++) { 
            cerr << "\t" << p[i] << endl;
        }

        exit(1);
    }

    if (mu < -5.0 || mu > 5.0 || sigma < -5.0 || sigma > 5.0) {
        cerr << "mu or sigma exceeded possible bounds, best parameters: " << endl;
        cerr << "mu: " << mu << endl;
        cerr << "sigma: " << mu << endl;
        for (int32_t i = 0; i < (int32_t)p.size(); i++) { 
            cerr << "\t" << p[i] << endl;
        }

        exit(1);
    }
}


RNN_Node_Interface* RNN_Genome::create_node(double mu, double sigma, double lstm_node_rate, int32_t &node_innovation_count, double depth) {
    RNN_Node_Interface *n = NULL;

    if (rng_0_1(generator) < lstm_node_rate) {
        n = new LSTM_Node(++node_innovation_count, RNN_HIDDEN_NODE, depth);
    } else {
        n = new RNN_Node(++node_innovation_count, RNN_HIDDEN_NODE, depth);
    }

    n->initialize_randomly(generator, normal_distribution, mu, sigma);

    return n;
}

bool RNN_Genome::attempt_edge_insert(RNN_Node_Interface *n1, RNN_Node_Interface *n2, double mu, double sigma, int32_t &edge_innovation_count) {
    cout << "\tadding edge between nodes " << n1->innovation_number << " and " << n2->innovation_number << endl;

    if (n1->depth == n2->depth) {
        cout << "\tcannot add edge between nodes as their depths are the same: " << n1->depth << " and " << n2->depth << endl;
        return false;
    }

    if (n2->depth < n1->depth) {
        //swap the nodes so that the lower one is first
        RNN_Node_Interface *temp = n2;
        n2 = n1;
        n1 = temp;
        cout << "\tswaping nodes, because n2->depth < n1->depth" << endl;
    }


    //check to see if an edge between the two nodes already exists
    for (int32_t i = 0; i < (int32_t)edges.size(); i++) {
        if (edges[i]->input_innovation_number == n1->innovation_number &&
                edges[i]->output_innovation_number == n2->innovation_number) {
            if (!edges[i]->enabled) {
                //edge was disabled so we can enable it
                cout << "\tedge already exists but was disabled, enabling it." << endl;
                edges[i]->enabled = true;
                return true;
            } else {
                cout << "\tedge already exists, not adding." << endl;
                //edge was already enabled, so there will not be a change
                return false;
            }
        }
    }

    RNN_Edge *e = new RNN_Edge(++edge_innovation_count, n1, n2);
    e->weight = normal_distribution.random(generator, mu, sigma);
    if (e->weight <= -10.0) e->weight = -10.0;
    if (e->weight >= 10.0) e->weight = 10.0;

    cout << "\tadding edge between nodes " << e->input_innovation_number << " and " << e->output_innovation_number << ", new edge weight: " << e->weight << endl;

    edges.insert( upper_bound(edges.begin(), edges.end(), e, sort_RNN_Edges_by_depth()), e);
    return true;
}

bool RNN_Genome::attempt_recurrent_edge_insert(RNN_Node_Interface *n1, RNN_Node_Interface *n2, double mu, double sigma, int32_t max_recurrent_depth, int32_t &edge_innovation_count) {
    cout << "\tadding recurrent edge between nodes " << n1->innovation_number << " and " << n2->innovation_number << endl;

    //check to see if an edge between the two nodes already exists
    for (int32_t i = 0; i < (int32_t)recurrent_edges.size(); i++) {
        if (recurrent_edges[i]->input_innovation_number == n1->innovation_number &&
                recurrent_edges[i]->output_innovation_number == n2->innovation_number) {
            if (!recurrent_edges[i]->enabled) {
                //edge was disabled so we can enable it
                cout << "\tedge already exists but was disabled, enabling it." << endl;
                recurrent_edges[i]->enabled = true;
                return true;
            } else {
                cout << "\tedge already exists, not adding." << endl;
                //edge was already enabled, so there will not be a change
                return false;
            }
        }
    }

    int32_t recurrent_depth = 1 + (rng_0_1(generator) * (max_recurrent_depth - 1));

    RNN_Recurrent_Edge *e = new RNN_Recurrent_Edge(++edge_innovation_count, recurrent_depth, n1, n2);
    e->weight = normal_distribution.random(generator, mu, sigma);
    if (e->weight <= -10.0) e->weight = -10.0;
    if (e->weight >= 10.0) e->weight = 10.0;

    cout << "\tadding recurrent edge between nodes " << e->input_innovation_number << " and " << e->output_innovation_number << ", new edge weight: " << e->weight << endl;

    recurrent_edges.insert( upper_bound(recurrent_edges.begin(), recurrent_edges.end(), e, sort_RNN_Recurrent_Edges_by_depth()), e);
    return true;
}



bool RNN_Genome::add_edge(double mu, double sigma, int32_t &edge_innovation_count) {
    cout << "\tattempting to add edge!" << endl;
    vector<RNN_Node_Interface*> reachable_nodes;
    for (int32_t i = 0; i < (int32_t)nodes.size(); i++) {
        if (nodes[i]->is_reachable()) reachable_nodes.push_back(nodes[i]);
    }
    cout << "\treachable_nodes.size(): " << reachable_nodes.size() << endl;

    int position = rng_0_1(generator) * reachable_nodes.size();

    RNN_Node_Interface *n1 = reachable_nodes[position];
    cout << "\tselected first node " << n1->innovation_number << " with depth " << n1->depth << endl;

    for (auto i = reachable_nodes.begin(); i != reachable_nodes.end();) {
        if ((*i)->depth == n1->depth) {
            cout << "\t\terasing node " << (*i)->innovation_number << " with depth " << (*i)->depth << endl;
            reachable_nodes.erase(i);
        } else {
            cout << "\t\tkeeping node " << (*i)->innovation_number << " with depth " << (*i)->depth << endl;
            i++;
        }
    }
    cout << "\treachable_nodes.size(): " << reachable_nodes.size() << endl;


    position = rng_0_1(generator) * reachable_nodes.size();
    RNN_Node_Interface *n2 = reachable_nodes[position];
    cout << "\tselected second node " << n2->innovation_number << " with depth " << n2->depth << endl;

    return attempt_edge_insert(n1, n2, mu, sigma, edge_innovation_count);
}

bool RNN_Genome::add_recurrent_edge(double mu, double sigma, int32_t max_recurrent_depth, int32_t &edge_innovation_count) {
    cout << "\tattempting to add recurrent edge!" << endl;

    vector<RNN_Node_Interface*> possible_input_nodes;
    vector<RNN_Node_Interface*> possible_output_nodes;
    for (int32_t i = 0; i < (int32_t)nodes.size(); i++) {
        if (nodes[i]->is_reachable()) {
            possible_input_nodes.push_back(nodes[i]);

            if (nodes[i]->type != RNN_INPUT_NODE) {
                possible_output_nodes.push_back(nodes[i]);
            }
        }
    }
 
    cout << "\tpossible_input_nodes.size(): " << possible_input_nodes.size() << endl;
    cout << "\tpossible_output_nodes.size(): " << possible_output_nodes.size() << endl;
    if (possible_input_nodes.size() == 0) return false;
    if (possible_output_nodes.size() == 0) return false;

    int p1 = rng_0_1(generator) * possible_input_nodes.size();
    int p2 = rng_0_1(generator) * possible_output_nodes.size();

    RNN_Node_Interface *n1 = possible_input_nodes[p1];
    cout << "\tselected first node " << n1->innovation_number << " with depth " << n1->depth << endl;

    RNN_Node_Interface *n2 = possible_output_nodes[p2];
    cout << "\tselected second node " << n2->innovation_number << " with depth " << n2->depth << endl;

    //no need to swap the nodes as recurrent connections can go backwards

    int32_t recurrent_depth = 1 + (rng_0_1(generator) * (max_recurrent_depth - 1));

    //check to see if an edge between the two nodes already exists
    for (int32_t i = 0; i < (int32_t)recurrent_edges.size(); i++) {
        if (recurrent_edges[i]->input_innovation_number == n1->innovation_number &&
                recurrent_edges[i]->output_innovation_number == n2->innovation_number
                && recurrent_edges[i]->recurrent_depth == recurrent_depth) {
            if (!recurrent_edges[i]->enabled) {
                recurrent_edges[i]->enabled = true;
                cout << "\tenabling recurrent edge between nodes " << n1->innovation_number << " and " << n2->innovation_number << endl;
                return true;
            } else {
                cout << "\tenabled edge already existed between selected nodes " << n1->innovation_number << " and " << n2->innovation_number << endl;

                return false;
            }
        }
    }

    cout << "\tadding recurrent edge between nodes " << n1->innovation_number << " and " << n2->innovation_number << " with recurrent depth: " << recurrent_depth << endl;

    //edge with same input/output did not exist, now we can create it
    RNN_Recurrent_Edge *recurrent_edge = new RNN_Recurrent_Edge(++edge_innovation_count, recurrent_depth, n1, n2);
    recurrent_edge->weight = normal_distribution.random(generator, mu, sigma);
    if (recurrent_edge->weight <= -10.0) recurrent_edge->weight = -10.0;
    if (recurrent_edge->weight >= 10.0) recurrent_edge->weight = 10.0;

    recurrent_edges.insert( upper_bound(recurrent_edges.begin(), recurrent_edges.end(), recurrent_edge, sort_RNN_Recurrent_Edges_by_depth()), recurrent_edge);

    return true;
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


bool RNN_Genome::split_edge(double mu, double sigma, double lstm_node_rate, int32_t max_recurrent_depth, int32_t &edge_innovation_count, int32_t &node_innovation_count) {
    cout << "\tattempting to split an edge!" << endl;
    vector<RNN_Edge*> enabled_edges;
    for (int32_t i = 0; i < edges.size(); i++) {
        if (edges[i]->enabled) enabled_edges.push_back(edges[i]);
    }

    vector<RNN_Recurrent_Edge*> enabled_recurrent_edges;
    for (int32_t i = 0; i < recurrent_edges.size(); i++) {
        if (recurrent_edges[i]->enabled) enabled_recurrent_edges.push_back(recurrent_edges[i]);
    }

    int32_t position = rng_0_1(generator) * (enabled_edges.size() + enabled_recurrent_edges.size());

    RNN_Node_Interface *n1 = NULL;
    RNN_Node_Interface *n2 = NULL;
    if (position < enabled_edges.size()) {
        RNN_Edge *edge = enabled_edges[position];
        n1 = edge->input_node;
        n2 = edge->output_node;
        edge->enabled = false;
    } else {
        position -= enabled_edges.size();
        RNN_Recurrent_Edge *recurrent_edge = enabled_recurrent_edges[position];
        n1 = recurrent_edge->input_node;
        n2 = recurrent_edge->output_node;
        recurrent_edge->enabled = false;
    }

    double new_depth = (n1->get_depth() + n2->get_depth()) / 2.0;
    RNN_Node_Interface *new_node = create_node(mu, sigma, lstm_node_rate, node_innovation_count, new_depth);

    nodes.insert( upper_bound(nodes.begin(), nodes.end(), new_node, sort_RNN_Nodes_by_depth()), new_node);

    double recurrent_probability = (double)enabled_recurrent_edges.size() / (double)(enabled_recurrent_edges.size() + enabled_edges.size());

    if (rng_0_1(generator) < recurrent_probability) {
        attempt_recurrent_edge_insert(n1, new_node, mu, sigma, max_recurrent_depth, edge_innovation_count);
    } else {
        attempt_edge_insert(n1, new_node, mu, sigma, edge_innovation_count);
    }

    if (rng_0_1(generator) < recurrent_probability) {
        attempt_recurrent_edge_insert(new_node, n2, mu, sigma, max_recurrent_depth, edge_innovation_count);
    } else {
        attempt_edge_insert(new_node, n2, mu, sigma, edge_innovation_count);
    }

    return true;
}

bool RNN_Genome::add_node(double mu, double sigma, double lstm_node_rate, int32_t max_recurrent_depth, int32_t &edge_innovation_count, int32_t &node_innovation_count) {
    double split_depth = rng_0_1(generator);

    vector<RNN_Node_Interface*> possible_inputs;
    vector<RNN_Node_Interface*> possible_outputs;

    int32_t enabled_count = 0;
    double avg_inputs = 0.0;
    double avg_outputs = 0.0;

    for (int32_t i = 0; i < (int32_t)nodes.size(); i++) {
        if (nodes[i]->depth < split_depth) possible_inputs.push_back(nodes[i]);
        else possible_outputs.push_back(nodes[i]);

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

    int32_t max_inputs = fmax(1, 4.0 + normal_distribution.random(generator, avg_inputs, input_sigma));
    int32_t max_outputs = fmax(1, 4.0 + normal_distribution.random(generator, avg_outputs, output_sigma));
    cout << "\tadd node max_inputs: " << max_inputs << endl;
    cout << "\tadd node max_outputs: " << max_outputs << endl;

    int32_t enabled_edges = get_enabled_edge_count();
    int32_t enabled_recurrent_edges = get_enabled_recurrent_edge_count();

    double recurrent_probability = (double)enabled_recurrent_edges / (double)(enabled_recurrent_edges + enabled_edges);
    //recurrent_probability = fmax(0.2, recurrent_probability);

    cout << "\tadd node recurrent probability: " << recurrent_probability << endl;

    while (possible_inputs.size() > max_inputs) {
        int32_t position = rng_0_1(generator) * possible_inputs.size();
        possible_inputs.erase(possible_inputs.begin() + position);
    }

    while (possible_outputs.size() > max_outputs) {
        int32_t position = rng_0_1(generator) * possible_outputs.size();
        possible_outputs.erase(possible_outputs.begin() + position);
    }

    RNN_Node_Interface *new_node = create_node(mu, sigma, lstm_node_rate, node_innovation_count, split_depth);
    nodes.insert( upper_bound(nodes.begin(), nodes.end(), new_node, sort_RNN_Nodes_by_depth()), new_node);

    for (int32_t i = 0; i < possible_inputs.size(); i++) {
        if (rng_0_1(generator) < recurrent_probability) {
            attempt_recurrent_edge_insert(possible_inputs[i], new_node, mu, sigma, max_recurrent_depth, edge_innovation_count);
        } else {
            attempt_edge_insert(possible_inputs[i], new_node, mu, sigma, edge_innovation_count);
        }
    }

    for (int32_t i = 0; i < possible_outputs.size(); i++) {
        if (rng_0_1(generator) < recurrent_probability) {
            attempt_recurrent_edge_insert(new_node, possible_outputs[i], mu, sigma, max_recurrent_depth, edge_innovation_count);
        } else {
            attempt_edge_insert(new_node, possible_outputs[i], mu, sigma, edge_innovation_count);
        }
    }

    return true;
}

bool RNN_Genome::enable_node() {
    cout << "\tattempting to enable a node!" << endl;
    vector<RNN_Node_Interface*> possible_nodes;
    for (int32_t i = 0; i < (int32_t)nodes.size(); i++) {
        if (!nodes[i]->enabled) possible_nodes.push_back(nodes[i]);
    }

    if (possible_nodes.size() == 0) return false;

    int position = rng_0_1(generator) * possible_nodes.size();
    possible_nodes[position]->enabled = true;
    cout << "\tenabling node " << possible_nodes[position]->innovation_number << " at depth " << possible_nodes[position]->depth << endl;

    return true;
}

bool RNN_Genome::disable_node() {
    cout << "\tattempting to disable a node!" << endl;
    vector<RNN_Node_Interface*> possible_nodes;
    for (int32_t i = 0; i < (int32_t)nodes.size(); i++) {
        if (nodes[i]->type != RNN_OUTPUT_NODE && nodes[i]->enabled) possible_nodes.push_back(nodes[i]);
    }

    if (possible_nodes.size() == 0) return false;

    int position = rng_0_1(generator) * possible_nodes.size();
    possible_nodes[position]->enabled = false;
    cout << "\tdisabling node " << possible_nodes[position]->innovation_number << " at depth " << possible_nodes[position]->depth << endl;

    return true;
}

bool RNN_Genome::split_node(double mu, double sigma, double lstm_node_rate, int32_t max_recurrent_depth, int32_t &edge_innovation_count, int32_t &node_innovation_count) {
    cout << "\tattempting to split a node!" << endl;
    vector<RNN_Node_Interface*> possible_nodes;
    for (int32_t i = 0; i < (int32_t)nodes.size(); i++) {
        if (nodes[i]->type != RNN_INPUT_NODE && nodes[i]->type != RNN_OUTPUT_NODE &&
                nodes[i]->is_reachable()) {
            possible_nodes.push_back(nodes[i]);
        }
    }

    if (possible_nodes.size() == 0) return false;

    int position = rng_0_1(generator) * possible_nodes.size();
    RNN_Node_Interface *selected_node = possible_nodes[position];
    cout << "\tselected node: " << selected_node->innovation_number << " at depth " << selected_node->depth << endl;

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
    cout << "\t[split node] recurrent_edges_1.size(): " << recurrent_edges_1.size() << endl;
    cout << "\t[split node] recurrent_edges_2.size(): " << recurrent_edges_2.size() << endl;
    cout << "\t[split node] input_edges.size(): " << input_edges.size() << endl;
    cout << "\t[split node] output_edges.size(): " << output_edges.size() << endl;

    if (input_edges.size() == 0 || output_edges.size() == 0) {
        cout << "\t[split node] error, input or output edges size was 0, cannot create a node" << endl;
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

    RNN_Node_Interface *new_node_1 = create_node(mu, sigma, lstm_node_rate, node_innovation_count, new_depth_1);
    RNN_Node_Interface *new_node_2 = create_node(mu, sigma, lstm_node_rate, node_innovation_count, new_depth_2);

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

    cout << "\t[split node] attempting recurrent edge inserts" << endl;

    for (int32_t i = 0; i < (int32_t)recurrent_edges_1.size(); i++) {
        if (recurrent_edges_1[i]->input_innovation_number == selected_node->innovation_number) {
            attempt_recurrent_edge_insert(new_node_1, recurrent_edges_1[i]->output_node, mu, sigma, max_recurrent_depth, edge_innovation_count);
        } else if (recurrent_edges_1[i]->output_innovation_number == selected_node->innovation_number) {
            attempt_recurrent_edge_insert(recurrent_edges_1[i]->input_node, new_node_1, mu, sigma, max_recurrent_depth, edge_innovation_count);
        } else {
            cerr << "\trecurrent edge list for split had an edge which was not connected to the selected node! This should never happen." << endl;
            exit(1);
        }
        //disable the old recurrent edges
        recurrent_edges_1[i]->enabled = false;
    }

    for (int32_t i = 0; i < (int32_t)recurrent_edges_2.size(); i++) {
        if (recurrent_edges_2[i]->input_innovation_number == selected_node->innovation_number) {
            attempt_recurrent_edge_insert(new_node_2, recurrent_edges_2[i]->output_node, mu, sigma, max_recurrent_depth, edge_innovation_count);
        } else if (recurrent_edges_2[i]->output_innovation_number == selected_node->innovation_number) {
            attempt_recurrent_edge_insert(recurrent_edges_2[i]->input_node, new_node_2, mu, sigma, max_recurrent_depth, edge_innovation_count);
        } else {
            cerr << "\trecurrent edge list for split had an edge which was not connected to the selected node! This should never happen." << endl;
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

    return true;
}

bool RNN_Genome::merge_node(double mu, double sigma, double lstm_node_rate, int32_t max_recurrent_depth, int32_t &edge_innovation_count, int32_t &node_innovation_count) {
    cout << "\tattempting to merge a node!" << endl;
    vector<RNN_Node_Interface*> possible_nodes;
    for (int32_t i = 0; i < (int32_t)nodes.size(); i++) {
        if (nodes[i]->type != RNN_INPUT_NODE && nodes[i]->type != RNN_OUTPUT_NODE) possible_nodes.push_back(nodes[i]);
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

    RNN_Node_Interface *new_node = create_node(mu, sigma, lstm_node_rate, node_innovation_count, new_depth);
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
            cout << "\tskipping merged edge because the input and output nodes are the same depth" << endl;
            continue;
        }

        //swap the edges becasue the input node is deeper than the output node
        if (input_node->depth > output_node->depth) {
            RNN_Node_Interface *tmp = input_node;
            input_node = output_node;
            output_node = tmp;
        }

        attempt_edge_insert(input_node, output_node, mu, sigma, edge_innovation_count);

        /*
        if (merged_edges[i]->input_node->depth > new_node->depth) {
            cout << "\tinput node depth " << merged_edges[i]->input_node->depth << " > new node depth " << new_node->depth << endl;
            attempt_edge_insert(new_node, merged_edges[i]->output_node, mu, sigma, edge_innovation_count);

        } else if (merged_edges[i]->output_node->depth < new_node->depth) {
            cout << "\toutput node depth " << merged_edges[i]->output_node->depth << " < new node depth " << new_node->depth << endl;
            attempt_edge_insert(merged_edges[i]->input_node, new_node, mu, sigma, edge_innovation_count);

        } else {
            if (merged_edges[i]->input_innovation_number == n1->innovation_number ||
                    merged_edges[i]->input_innovation_number == n2->innovation_number) {
                //merged edge was an output edge
                cout << "\tthis was an output edge" << endl;
                attempt_edge_insert(new_node, merged_edges[i]->output_node, mu, sigma, edge_innovation_count);
            } else if (merged_edges[i]->output_innovation_number == n1->innovation_number ||
                    merged_edges[i]->output_innovation_number == n2->innovation_number) {
                //mergerd edge was an input edge
                cout << "\tthis was an input edge" << endl;
                attempt_edge_insert(merged_edges[i]->input_node, new_node, mu, sigma, edge_innovation_count);

            } else {
                cerr << "ERROR in merge node, reached statement that should never happen!" << endl;
                exit(1);
            }
        }
        */
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

        attempt_recurrent_edge_insert(input_node, output_node, mu, sigma, max_recurrent_depth, edge_innovation_count);
    }

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

    //cout << "weight: " << weight << ", converted to value: " << value << endl;

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
        if (nodes[i]->type != RNN_INPUT_NODE) continue;
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
        if (nodes[i]->type == RNN_OUTPUT_NODE) output_count++;
        if (nodes[i]->type == RNN_INPUT_NODE) input_count++;
    }

    int32_t output_name_index = 0;
    outfile << "\t{" << endl;
    outfile << "\t\trank = sink;" << endl;
    for (uint32_t i = 0; i < nodes.size(); i++) {
        if (nodes[i]->type != RNN_OUTPUT_NODE) continue;
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
            if (nodes[i]->type != RNN_INPUT_NODE) continue;
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
            if (nodes[i]->type != RNN_OUTPUT_NODE) continue;

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
        if (nodes[i]->type != RNN_HIDDEN_NODE) continue;
        if (!nodes[i]->is_reachable()) continue;

        string color;
        if (nodes[i]->node_type == LSTM_NODE) {
            color = "orange";
        } else {
            color = "black";
        }
        outfile << "\t\tnode" << nodes[i]->get_innovation_number() << " [shape=box,color=" << color << ",label=\"node " << nodes[i]->get_innovation_number() << "\\ndepth " << nodes[i]->depth << "\"];" << endl;
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

void write_map(ostream &out, map<string, int> &m) {
    out << m.size();
    for (auto iterator = m.begin(); iterator != m.end(); iterator++) {

        out << " "<< iterator->first;
        out << " "<< iterator->second;
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


void write_binary_string(ostream &out, string s, string name, bool verbose) {
    int32_t n = s.size();
    if (verbose) cout << "writing " << n << " " << name << " characters '" << s << "'" << endl;
    out.write((char*)&n, sizeof(int32_t));
    out.write((char*)&s[0], sizeof(char) * s.size());
}

void read_binary_string(istream &in, string &s, string name, bool verbose) {

    int32_t n;
    in.read((char*)&n, sizeof(int32_t));

    if (verbose) cout << "reading " << n << " " << name << " characters." << endl;
    char* s_v = new char[n];
    in.read((char*)s_v, sizeof(char) * n);
    s.assign(s_v, s_v + n);
    delete [] s_v;

    if (verbose) cout << "read " << n << " " << name << " characters '" << s << "'" << endl;
}


RNN_Genome::RNN_Genome(string binary_filename, bool verbose) {
    ifstream bin_infile(binary_filename, ios::in | ios::binary);

    if (!bin_infile.good()) {
        cerr << "ERROR: could not open RNN genome file '" << binary_filename << "' for reading." << endl;
        exit(1);
    }

    read_from_stream(bin_infile, verbose);
    bin_infile.close();
}

RNN_Genome::RNN_Genome(char *array, int32_t length, bool verbose) {
    read_from_array(array, length, verbose);
}

RNN_Genome::RNN_Genome(istream &bin_infile, bool verbose) {
    read_from_stream(bin_infile, verbose);
}

void RNN_Genome::read_from_array(char *array, int32_t length, bool verbose) {
    string array_str;
    for (uint32_t i = 0; i < length; i++) {
        array_str.push_back(array[i]);
    }

    istringstream iss(array_str);
    read_from_stream(iss, verbose);
}

void RNN_Genome::read_from_stream(istream &bin_istream, bool verbose) {
    if (verbose) cout << "READING GENOME FROM STREAM" << endl;
    bin_istream.read((char*)&generation_id, sizeof(int32_t));
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

    if (verbose) {
        cout << "generation_id: " << generation_id << endl;
        cout << "bp_iterations: " << bp_iterations << endl;
        cout << "learning_rate: " << learning_rate << endl;
        cout << "adapt_learning_rate: " << adapt_learning_rate << endl;
        cout << "use_nesterov_momentum: " << use_nesterov_momentum << endl;
        cout << "use_reset_weights: " << use_reset_weights << endl;

        cout << "use_high_norm: " << use_high_norm << endl;
        cout << "high_threshold: " << high_threshold << endl;
        cout << "use_low_norm: " << use_low_norm << endl;
        cout << "low_threshold: " << low_threshold << endl;

        cout << "use_dropout: " << use_dropout << endl;
        cout << "dropout_probability: " << dropout_probability << endl;
    }

    read_binary_string(bin_istream, log_filename, "log_filename", verbose);

    string generator_str;
    read_binary_string(bin_istream, generator_str, "generator", verbose);
    istringstream generator_iss(generator_str);
    generator_iss >> generator;

    string rng_0_1_str;
    read_binary_string(bin_istream, rng_0_1_str, "rng_0_1", verbose);
    istringstream rng_0_1_iss;
    rng_0_1_iss >> rng_0_1;


    string generated_by_map_str;
    read_binary_string(bin_istream, generated_by_map_str, "generated_by_map", verbose);
    istringstream generated_by_map_iss(generated_by_map_str);
    read_map(generated_by_map_iss, generated_by_map);

    bin_istream.read((char*)&best_validation_error, sizeof(double));
    bin_istream.read((char*)&best_validation_mae, sizeof(double));

    int32_t n_initial_parameters;
    bin_istream.read((char*)&n_initial_parameters, sizeof(int32_t));
    if (verbose) cout << "reading " << n_initial_parameters << " initial parameters." << endl;
    double* initial_parameters_v = new double[n_initial_parameters];
    bin_istream.read((char*)initial_parameters_v, sizeof(double) * n_initial_parameters);
    initial_parameters.assign(initial_parameters_v, initial_parameters_v + n_initial_parameters);
    delete [] initial_parameters_v;

    int32_t n_best_parameters;
    bin_istream.read((char*)&n_best_parameters, sizeof(int32_t));
    if (verbose) cout << "reading " << n_best_parameters << " best parameters." << endl;
    double* best_parameters_v = new double[n_best_parameters];
    bin_istream.read((char*)best_parameters_v, sizeof(double) * n_best_parameters);
    best_parameters.assign(best_parameters_v, best_parameters_v + n_best_parameters);
    delete [] best_parameters_v;


    input_parameter_names.clear();
    int32_t n_input_parameter_names;
    bin_istream.read((char*)&n_input_parameter_names, sizeof(int32_t));
    if (verbose) cout << "reading " << n_input_parameter_names << " input parameter names." << endl;
    for (int32_t i = 0; i < n_input_parameter_names; i++) {
        string input_parameter_name;
        read_binary_string(bin_istream, input_parameter_name, "input_parameter_names[" + std::to_string(i) + "]", verbose);
        input_parameter_names.push_back(input_parameter_name);
    }

    output_parameter_names.clear();
    int32_t n_output_parameter_names;
    bin_istream.read((char*)&n_output_parameter_names, sizeof(int32_t));
    if (verbose) cout << "reading " << n_output_parameter_names << " output parameter names." << endl;
    for (int32_t i = 0; i < n_output_parameter_names; i++) {
        string output_parameter_name;
        read_binary_string(bin_istream, output_parameter_name, "output_parameter_names[" + std::to_string(i) + "]", verbose);
        output_parameter_names.push_back(output_parameter_name);
    }



    int32_t n_nodes;
    bin_istream.read((char*)&n_nodes, sizeof(int32_t));
    if (verbose) cout << "reading " << n_nodes << " nodes." << endl;

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

        if (verbose) cout << "NODE: " << innovation_number << " " << type << " " << node_type << " " << depth << " " << enabled << endl;

        RNN_Node_Interface *node;
        if (node_type == LSTM_NODE) {
            node = new LSTM_Node(innovation_number, type, depth);
        } else if (node_type == RNN_NODE) {
            node = new RNN_Node(innovation_number, type, depth);
        } else {
            cerr << "Error reading node from stream, unknown node_type: " << node_type << endl;
            exit(1);
        }
        node->enabled = enabled;
        nodes.push_back(node);
    }


    int32_t n_edges;
    bin_istream.read((char*)&n_edges, sizeof(int32_t));
    if (verbose) cout << "reading " << n_edges << " edges." << endl;

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

        if (verbose) cout << "EDGE: " << innovation_number << " " << input_innovation_number << " " << output_innovation_number << " " << enabled << endl;

        RNN_Edge *edge = new RNN_Edge(innovation_number, input_innovation_number, output_innovation_number, nodes);
        edge->enabled = enabled;
        edges.push_back(edge);
    }


    int32_t n_recurrent_edges;
    bin_istream.read((char*)&n_recurrent_edges, sizeof(int32_t));
    if (verbose) cout << "reading " << n_recurrent_edges << " recurrent_edges." << endl;

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

        if (verbose) cout << "RECURRENT EDGE: " << innovation_number << " " << recurrent_depth << " " << input_innovation_number << " " << output_innovation_number << " " << enabled << endl;

        RNN_Recurrent_Edge *recurrent_edge = new RNN_Recurrent_Edge(innovation_number, recurrent_depth, input_innovation_number, output_innovation_number, nodes);
        recurrent_edge->enabled = enabled;
        recurrent_edges.push_back(recurrent_edge);
    }

    assign_reachability();
}

void RNN_Genome::write_to_array(char **bytes, int32_t &length, bool verbose) {
    ostringstream oss;
    write_to_stream(oss, verbose);

    string bytes_str = oss.str();
    length = bytes_str.size();
    (*bytes) = (char*)malloc(length * sizeof(char));
    for (uint32_t i = 0; i < length; i++) {
        (*bytes)[i] = bytes_str[i];
    }
}

void RNN_Genome::write_to_file(string bin_filename, bool verbose) {
    ofstream bin_outfile(bin_filename, ios::out | ios::binary);
    write_to_stream(bin_outfile, verbose);
    bin_outfile.close();
}

void RNN_Genome::write_to_stream(ostream &bin_ostream, bool verbose) {
    if (verbose) cout << "WRITING GENOME TO STREAM" << endl;

    bin_ostream.write((char*)&generation_id, sizeof(int32_t));
    bin_ostream.write((char*)&bp_iterations, sizeof(int32_t));
    bin_ostream.write((char*)&learning_rate, sizeof(double));
    bin_ostream.write((char*)&adapt_learning_rate, sizeof(bool));
    bin_ostream.write((char*)&use_nesterov_momentum, sizeof(bool));
    bin_ostream.write((char*)&use_reset_weights, sizeof(bool));

    bin_ostream.write((char*)&use_high_norm, sizeof(bool));
    bin_ostream.write((char*)&high_threshold, sizeof(double));
    bin_ostream.write((char*)&use_low_norm, sizeof(bool));
    bin_ostream.write((char*)&low_threshold, sizeof(double));

    bin_ostream.write((char*)&use_dropout, sizeof(bool));
    bin_ostream.write((char*)&dropout_probability, sizeof(double));

    if (verbose) {
        cout << "generation_id: " << generation_id << endl;
        cout << "bp_iterations: " << bp_iterations << endl;
        cout << "learning_rate: " << learning_rate << endl;
        cout << "adapt_learning_rate: " << adapt_learning_rate << endl;
        cout << "use_nesterov_momentum: " << use_nesterov_momentum << endl;
        cout << "use_reset_weights: " << use_reset_weights << endl;

        cout << "use_high_norm: " << use_high_norm << endl;
        cout << "high_threshold: " << high_threshold << endl;
        cout << "use_low_norm: " << use_low_norm << endl;
        cout << "low_threshold: " << low_threshold << endl;

        cout << "use_dropout: " << use_dropout << endl;
        cout << "dropout_probability: " << dropout_probability << endl;
    }

    write_binary_string(bin_ostream, log_filename, "log_filename", verbose);

    ostringstream generator_oss;
    generator_oss << generator;
    string generator_str = generator_oss.str();
    write_binary_string(bin_ostream, generator_str, "generator", verbose);

    ostringstream rng_0_1_oss;
    rng_0_1_oss << rng_0_1;
    string rng_0_1_str = rng_0_1_oss.str();
    write_binary_string(bin_ostream, rng_0_1_str, "rng_0_1", verbose);

    ostringstream generated_by_map_oss;
    write_map(generated_by_map_oss, generated_by_map);
    string generated_by_map_str = generated_by_map_oss.str();
    write_binary_string(bin_ostream, generated_by_map_str, "generated_by_map", verbose);

    bin_ostream.write((char*)&best_validation_error, sizeof(double));
    bin_ostream.write((char*)&best_validation_mae, sizeof(double));

    int32_t n_initial_parameters = initial_parameters.size();
    if (verbose) cout << "writing " << n_initial_parameters << " initial parameters." << endl;
    bin_ostream.write((char*)&n_initial_parameters, sizeof(int32_t));
    bin_ostream.write((char*)&initial_parameters[0], sizeof(double) * initial_parameters.size());

    int32_t n_best_parameters = best_parameters.size();
    if (verbose) cout << "writing " << n_best_parameters << " best parameters." << endl;
    bin_ostream.write((char*)&n_best_parameters, sizeof(int32_t));
    bin_ostream.write((char*)&best_parameters[0], sizeof(double) * best_parameters.size());

    int32_t n_input_parameter_names = input_parameter_names.size();
    bin_ostream.write((char*)&n_input_parameter_names, sizeof(int32_t));
    for (int32_t i = 0; i < input_parameter_names.size(); i++) {
        write_binary_string(bin_ostream, input_parameter_names[i], "input_parameter_names[" + std::to_string(i) + "]", verbose);
    }

    int32_t n_output_parameter_names = output_parameter_names.size();
    bin_ostream.write((char*)&n_output_parameter_names, sizeof(int32_t));
    for (int32_t i = 0; i < output_parameter_names.size(); i++) {
        write_binary_string(bin_ostream, output_parameter_names[i], "output_parameter_names[" + std::to_string(i) + "]", verbose);
    }

    int32_t n_nodes = nodes.size();
    bin_ostream.write((char*)&n_nodes, sizeof(int32_t));
    if (verbose) cout << "writing " << n_nodes << " nodes." << endl;

    for (int32_t i = 0; i < nodes.size(); i++) {
        if (verbose) cout << "NODE: " << nodes[i]->innovation_number << " " << nodes[i]->type << " " << nodes[i]->node_type << " " << nodes[i]->depth << endl;
        nodes[i]->write_to_stream(bin_ostream);
    }


    int32_t n_edges = edges.size();
    bin_ostream.write((char*)&n_edges, sizeof(int32_t));
    if (verbose) cout << "writing " << n_edges << " edges." << endl;

    for (int32_t i = 0; i < edges.size(); i++) {
        if (verbose) cout << "EDGE: " << edges[i]->innovation_number << " " << edges[i]->input_innovation_number << " " << edges[i]->output_innovation_number << endl;
        edges[i]->write_to_stream(bin_ostream);
    }


    int32_t n_recurrent_edges = recurrent_edges.size();
    bin_ostream.write((char*)&n_recurrent_edges, sizeof(int32_t));
    if (verbose) cout << "writing " << n_recurrent_edges << " recurrent_edges." << endl;

    for (int32_t i = 0; i < recurrent_edges.size(); i++) {
        if (verbose) cout << "RECURRENT EDGE: " << recurrent_edges[i]->innovation_number << " " << recurrent_edges[i]->recurrent_depth << " " << recurrent_edges[i]->input_innovation_number << " " << recurrent_edges[i]->output_innovation_number << endl;

        recurrent_edges[i]->write_to_stream(bin_ostream);
    }
}
