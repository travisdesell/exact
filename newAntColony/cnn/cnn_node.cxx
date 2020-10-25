#include <cmath>
//using std::isnan;
//using std::isinf;

#include <chrono>

#include <cstdio>

#include <fstream>
using std::ofstream;
using std::ifstream;
using std::ios;
using std::ios_base;

#include <iomanip>
using std::defaultfloat;
using std::hexfloat;
using std::setw;
using std::setprecision;

#include <iostream>
using std::cout;
using std::cerr;
using std::endl;
using std::ostream;
using std::istream;

#include <limits>
using std::numeric_limits;

#include <random>
using std::minstd_rand0;
using std::uniform_real_distribution;

#include <stdexcept>
using std::runtime_error;

#include <sstream>
using std::ostringstream;
using std::istringstream;

#include <string>
using std::string;

#include <vector>
using std::vector;

#include "image_tools/image_set.hxx"
#include "common/random.hxx"
#include "common/exp.hxx"
#include "comparison.hxx"
#include "cnn_genome.hxx"
#include "cnn_edge.hxx"
#include "cnn_node.hxx"

#include "stdint.h"

float read_hexfloat(istream &infile) {
#ifdef _WIN32
	float result;
	infile >> std::hexfloat >> result >> std::defaultfloat;
	return result;
#else
    string s;
    infile >> s;

    //cout << "read hexfloat: '" << s << "'" << endl;

    if (s.compare("-nan(ind)") == 0 || s.compare("-nan") == 0 || s.compare("nan(ind)") == 0 || s.compare("nan") == 0) {
        return EXACT_MAX_FLOAT;
    } else {
        return stod(s);
    }
#endif
}

void write_hexfloat(ostream &outfile, float value) {
#ifdef _WIN32
    char hf[32];
    sprintf(hf, "%a", value);
    outfile << hf;
#else
    outfile << std::hexfloat << value << std::defaultfloat;
#endif
}

CNN_Node::CNN_Node() {
    node_id = -1;
    exact_id = -1;
    genome_id = -1;

    innovation_number = -1;

    forward_visited = false;
    reverse_visited = false;
    needs_initialization = true;

    disabled = false;

    weight_count = 0;
    inverse_variance = 0;

    gamma = 1;
    best_gamma = 1;
    previous_velocity_gamma = 0;

    beta = 0;
    best_beta = 0;
    previous_velocity_beta = 0;

    running_mean = 0;
    best_running_mean = 0;
    running_variance = 1.0;
    best_running_variance = 1.0;
}

CNN_Node::CNN_Node(int _innovation_number, float _depth, int _batch_size, int _size_x, int _size_y, int _type) {
    node_id = -1;
    exact_id = -1;
    genome_id = -1;

    innovation_number = _innovation_number;
    depth = _depth;
    type = _type;

    batch_size = _batch_size;

    size_x = _size_x;
    size_y = _size_y;

    inverse_variance = 0;
    total_inputs = 0;
    inputs_fired = 0;

    total_outputs = 0;
    outputs_fired = 0;

    weight_count = 0;

    forward_visited = false;
    reverse_visited = false;

    needs_initialization = true;

    disabled = false;

    gamma = 1;
    best_gamma = 1;
    previous_velocity_gamma = 0;

    beta = 0;
    best_beta = 0;
    previous_velocity_beta = 0;

    running_mean = 0;
    best_running_mean = 0;
    running_variance = 1.0;
    best_running_variance = 1.0;

    total_size = batch_size * size_y * size_x;

    values_in = new float[total_size]();
    errors_in = new float[total_size]();

    values_out = new float[total_size]();
    errors_out = new float[total_size]();
    relu_gradients = new float[total_size]();
    pool_gradients = new float[total_size]();
}

#ifdef _MYSQL_
CNN_Node::CNN_Node(int _node_id) {
    node_id = _node_id;

    ostringstream query;
    query << "SELECT * FROM cnn_node WHERE id = " << node_id;

    mysql_exact_query(query.str());

    MYSQL_RES *result = mysql_store_result(exact_db_conn);

    if (result != NULL) {
        MYSQL_ROW row = mysql_fetch_row(result);

        int column = 0;

        exact_id = atoi(row[++column]);
        genome_id = atoi(row[++column]);
        innovation_number = atoi(row[++column]);
        depth = atof(row[++column]);

        batch_size = atoi(row[++column]);
        size_x = atoi(row[++column]);
        size_y = atoi(row[++column]);

        total_size = batch_size * size_y * size_x;

        type = atoi(row[++column]);

        //need to reset these because it will be modified when edges are set
        total_inputs = 0;
        inputs_fired = 0;

        total_outputs = 0;
        outputs_fired = 0;

        forward_visited = atoi(row[++column]);
        reverse_visited = atoi(row[++column]);
        weight_count = atoi(row[++column]);
        needs_initialization = atoi(row[++column]);

        disabled = atoi(row[++column]);

        istringstream batch_norm_parameters_iss(row[++column]);

        gamma = read_hexfloat(batch_norm_parameters_iss);
        best_gamma = read_hexfloat(batch_norm_parameters_iss);
        previous_velocity_gamma = read_hexfloat(batch_norm_parameters_iss);

        beta = read_hexfloat(batch_norm_parameters_iss);
        best_beta = read_hexfloat(batch_norm_parameters_iss);
        previous_velocity_beta = read_hexfloat(batch_norm_parameters_iss);

        running_mean = read_hexfloat(batch_norm_parameters_iss);
        best_running_mean = read_hexfloat(batch_norm_parameters_iss);
        running_variance = read_hexfloat(batch_norm_parameters_iss);
        best_running_variance = read_hexfloat(batch_norm_parameters_iss);

        mysql_free_result(result);
    } else {
        ostringstream error_message;
        error_message << "ERROR! Could not find cnn_node in database with node id: " << node_id << endl;
        throw runtime_error(error_message.str());
    }

    weight_count = 0;

    //initialize arrays not stored to database
    values_in = new float[total_size]();
    errors_in = new float[total_size]();

    values_out = new float[total_size]();
    errors_out = new float[total_size]();
    relu_gradients = new float[total_size]();
    pool_gradients = new float[total_size]();

    //cout << "read node!" << endl;
    //cout << this << endl;
}

void CNN_Node::export_to_database(int _exact_id, int _genome_id) {
    ostringstream query;

    exact_id = _exact_id;
    genome_id = _genome_id;

    //cout << "inserting node with exact_id: " << exact_id << " and genome id: " << genome_id << endl;

    if (node_id >= 0) {
        query << "REPLACE INTO cnn_node SET id = " << node_id << ",";
    } else {
        query << "INSERT INTO cnn_node SET";
    }

    query << " exact_id = " << exact_id
        << ", genome_id = " << genome_id
        << ", innovation_number = " << innovation_number
        << ", depth = " << depth
        << ", batch_size = " << batch_size
        << ", size_x = " << size_x
        << ", size_y = " << size_y
        << ", type = " << type
        << ", forward_visited = " << forward_visited
        << ", reverse_visited = " << reverse_visited
        << ", weight_count = " << weight_count
        << ", needs_initialization = " << needs_initialization
        << ", disabled = " << disabled
        << ", batch_norm_parameters = '";

    write_hexfloat(query, gamma);
    query << " ";
    write_hexfloat(query, best_gamma);
    query << " ";
    write_hexfloat(query, previous_velocity_gamma);
    query << " ";
    write_hexfloat(query, beta);
    query << " ";
    write_hexfloat(query, best_beta);
    query << " ";
    write_hexfloat(query, previous_velocity_beta);
    query << " ";
    write_hexfloat(query, running_mean);
    query << " ";
    write_hexfloat(query, best_running_mean);
    query << " ";
    write_hexfloat(query, running_variance);
    query << " ";
    write_hexfloat(query, best_running_variance);
    query << "'";

    mysql_exact_query(query.str());

    if (node_id < 0) {
        node_id = mysql_exact_last_insert_id();
        //cout << "set node id to: " << node_id << endl;
    }
}

int CNN_Node::get_node_id() const {
    return node_id;
}
#endif

CNN_Node::~CNN_Node() {
    delete [] values_out;
    delete [] errors_out;
    delete [] relu_gradients;
    delete [] pool_gradients;

    delete [] values_in;
    delete [] errors_in;
}

CNN_Node* CNN_Node::copy() const {
    CNN_Node *copy = new CNN_Node();

    copy->node_id = -1;
    copy->genome_id = genome_id;

    copy->innovation_number = innovation_number;
    copy->depth = depth;
    copy->batch_size = batch_size;
    copy->size_x = size_x;
    copy->size_y = size_y;
    copy->total_size = total_size;

    copy->type = type;

    copy->total_inputs = 0; //this will be updated when edges are set
    copy->inputs_fired = inputs_fired;

    copy->total_outputs = 0; //this will be updated when edges are set
    copy->outputs_fired = outputs_fired;

    copy->forward_visited = forward_visited;
    copy->reverse_visited = reverse_visited;
    copy->weight_count = weight_count;
    copy->needs_initialization = needs_initialization;

    copy->disabled = disabled;

    copy->gamma = gamma;
    copy->best_gamma = best_gamma;
    copy->previous_velocity_gamma = previous_velocity_gamma;

    copy->best_beta = best_beta;
    copy->previous_velocity_beta = previous_velocity_beta;

    copy->running_mean = running_mean;
    copy->best_running_mean = best_running_mean;
    copy->running_variance = running_variance;
    copy->best_running_variance = best_running_variance;

    copy->values_in = new float[total_size]();
    copy->errors_in = new float[total_size]();

    copy->values_out = new float[total_size]();
    copy->errors_out = new float[total_size]();
    copy->relu_gradients = new float[total_size]();
    copy->pool_gradients = new float[total_size]();

    for (uint32_t i = 0; i < total_size; i++) {
        copy->values_in[i] = values_in[i];
        copy->errors_in[i] = errors_in[i];

        copy->values_out[i] = values_out[i];
        copy->errors_out[i] = errors_out[i];
        copy->relu_gradients[i] = relu_gradients[i];
        copy->pool_gradients[i] = pool_gradients[i];
    }

    return copy;
}

bool CNN_Node::needs_init() const {
    return needs_initialization;
}

void CNN_Node::reset_weight_count() {
    weight_count = 0;
}

void CNN_Node::initialize() {
    gamma = 1.0;
    beta = 0.0;
    needs_initialization = false;

    delete [] values_in;
    delete [] errors_in;

    delete [] values_out;
    delete [] errors_out;
    delete [] relu_gradients;
    delete [] pool_gradients;

    values_in = new float[total_size]();
    errors_in = new float[total_size]();

    values_out = new float[total_size]();
    errors_out = new float[total_size]();
    relu_gradients = new float[total_size]();
    pool_gradients = new float[total_size]();
}

void CNN_Node::reset_velocities() {
    previous_velocity_gamma = 0;
    previous_velocity_beta = 0;
}

void CNN_Node::add_weight_count(int _weight_count) {
    weight_count += _weight_count;
    
    //cerr << "node " << innovation_number << " setting weight count to: " << weight_count << endl;
}

int CNN_Node::get_weight_count() const {
    return weight_count;
}

bool CNN_Node::is_fixed() const {
    return type != INPUT_NODE && type != OUTPUT_NODE && type != SOFTMAX_NODE;
}

bool CNN_Node::is_hidden() const {
    return type == HIDDEN_NODE;
}


bool CNN_Node::is_input() const {
    return type == INPUT_NODE;
}


bool CNN_Node::is_output() const {
    return type == OUTPUT_NODE;
}


bool CNN_Node::is_enabled() const {
    return disabled == false;
}

bool CNN_Node::is_disabled() const {
    return disabled == true;
}

void CNN_Node::disable() {
    disabled = true;
}

void CNN_Node::enable() {
    disabled = false;
}

bool CNN_Node::is_softmax() const {
    return type == SOFTMAX_NODE;
}


bool CNN_Node::is_reachable() const {
    return !disabled && forward_visited && reverse_visited;
}

bool CNN_Node::is_forward_visited() const {
    return forward_visited;
}

bool CNN_Node::is_reverse_visited() const {
    return reverse_visited;
}

void CNN_Node::forward_visit() {
    forward_visited = true;
}

void CNN_Node::reverse_visit() {
    reverse_visited = true;
}

void CNN_Node::set_unvisited() {
    total_inputs = 0;
    total_outputs = 0;
    forward_visited = false;
    reverse_visited = false;
}

int CNN_Node::get_batch_size() const {
    return batch_size;
}

bool CNN_Node::vectors_correct() const {
    if (total_size != batch_size * size_y * size_x) {
        cerr << "ERROR! vectors incorrect on node " << innovation_number << endl;
        cerr << "total_size: " << total_size << " and batch size: " << batch_size << ", size_y: " << size_y << ", size_x: " << size_x << endl;
        return false;
    }

    return true;
}

int CNN_Node::get_size_x() const {
    return size_x;
}

int CNN_Node::get_size_y() const {
    return size_y;
}

int CNN_Node::get_innovation_number() const {
    return innovation_number;
}

float CNN_Node::get_depth() const {
    return depth;
}

void CNN_Node::resize_arrays() {
    delete [] values_in;
    delete [] errors_in;

    delete [] values_out;
    delete [] errors_out;
    delete [] relu_gradients;
    delete [] pool_gradients;

    values_in = new float[total_size]();
    errors_in = new float[total_size]();

    values_out = new float[total_size]();
    errors_out = new float[total_size]();
    relu_gradients = new float[total_size]();
    pool_gradients = new float[total_size]();

    needs_initialization = true;
}


float CNN_Node::get_value_in(int batch_number, int y, int x) {
    return values_in[(batch_number * size_y * size_x) + (size_x * y) + x];
    //return values_in[batch_number][y][x];
}

void CNN_Node::set_value_in(int batch_number, int y, int x, float value) {
    values_in[(batch_number * size_y * size_x) + (size_x * y) + x] = value;
    //values_in[batch_number][y][x] = value;
}

float* CNN_Node::get_values_in() {
    return values_in;
}

float CNN_Node::get_value_out(int batch_number, int y, int x) {
    return values_out[(batch_number * size_y * size_x) + (size_x * y) + x];
    //return values_out[batch_number][y][x];
}

void CNN_Node::set_value_out(int batch_number, int y, int x, float value) {
    values_out[(batch_number * size_y * size_x) + (size_x * y) + x] = value;
    //values_out[batch_number][y][x] = value;
}

float* CNN_Node::get_values_out() {
    return values_out;
}


void CNN_Node::set_error_in(int batch_number, int y, int x, float error) {
    errors_in[(batch_number * size_y * size_x) + (size_x * y) + x] = error;
    //errors_in[batch_number][y][x] = error;
}

float* CNN_Node::get_errors_in() {
    return errors_in;
}

void CNN_Node::set_error_out(int batch_number, int y, int x, float error) {
    errors_out[(batch_number * size_y * size_x) + (size_x * y) + x] = error;
    //errors_out[batch_number][y][x] = error;
}

float* CNN_Node::get_errors_out() {
    return errors_out;
}

float* CNN_Node::get_relu_gradients() {
    return relu_gradients;
}

float* CNN_Node::get_pool_gradients() {
    return pool_gradients;
}


void CNN_Node::print(ostream &out) {
    out << "CNN_Node " << innovation_number << ", at depth: " << depth << " of input size x: " << size_x << ", y: " << size_y << endl;

    for (int32_t batch_number = 0; batch_number < batch_size; batch_number++) {
        out << "    batch_number: " << batch_number << endl;
        out << "    values_in:" << endl;
        int current = batch_number * size_y * size_x;
        for (int32_t i = 0; i < size_y; i++) {
            out << "    ";
            for (int32_t j = 0; j < size_x; j++) {
                out << setw(13) << setprecision(8) << values_in[current];
                current++;
            }
            out << endl;
        }

        out << "    errors_in:" << endl;
        current = batch_number * size_y * size_x;
        for (int32_t i = 0; i < size_y; i++) {
            out << "    ";
            for (int32_t j = 0; j < size_x; j++) {
                out << setw(13) << setprecision(8) << errors_in[current];
                current++;
            }
            out << endl;
        }

        out << "    values_out:" << endl;
        current = batch_number * size_y * size_x;
        for (int32_t i = 0; i < size_y; i++) {
            out << "    ";
            for (int32_t j = 0; j < size_x; j++) {
                out << setw(13) << setprecision(8) << values_out[current];
                current++;
            }
            out << endl;
        }

        out << "    errors_out:" << endl;
        current = batch_number * size_y * size_x;
        for (int32_t i = 0; i < size_y; i++) {
            out << "    ";
            for (int32_t j = 0; j < size_x; j++) {
                out << setw(13) << setprecision(8) << errors_out[current];
                current++;
            }
            out << endl;
        }

        out << "    relu_gradients:" << endl;
        current = batch_number * size_y * size_x;
        for (int32_t i = 0; i < size_y; i++) {
            out << "    ";
            for (int32_t j = 0; j < size_x; j++) {
                out << setw(13) << setprecision(8) << relu_gradients[current];
                current++;
            }
            out << endl;
        }

        out << "    pool_gradients:" << endl;
        current = batch_number * size_y * size_x;
        for (int32_t i = 0; i < size_y; i++) {
            out << "    ";
            for (int32_t j = 0; j < size_x; j++) {
                out << setw(13) << setprecision(8) << pool_gradients[current];
                current++;
            }
            out << endl;
        }
    }
}

void CNN_Node::reset_times() {
    input_fired_time = 0.0;
    output_fired_time = 0.0;
}

void CNN_Node::accumulate_times(float &total_input_time, float &total_output_time) {
    total_input_time += input_fired_time;
    total_output_time += output_fired_time;
}

void CNN_Node::reset() {
    inputs_fired = 0;
    outputs_fired = 0;

    if (is_reachable()) {
        for (int32_t current = 0; current < total_size; current++) {
            values_in[current] = 0.0;
            errors_in[current] = 0.0;

            values_out[current] = 0.0;
            errors_out[current] = 0.0;
            relu_gradients[current] = 0.0;
            pool_gradients[current] = 0.0;
        }
    }
}


void CNN_Node::save_best_weights() {
    best_gamma = gamma;
    best_beta = beta;

    best_running_mean = running_mean;
    best_running_variance = running_variance;
}

void CNN_Node::set_weights_to_best() {
    gamma = best_gamma;
    beta = best_beta;

    running_mean = best_running_mean;
    running_variance = best_running_variance;
}

void CNN_Node::update_batch_size(int new_batch_size) {
    batch_size = new_batch_size;
    total_size = batch_size * size_y * size_x;
    resize_arrays();
}

bool CNN_Node::modify_size_x(int change) {
    int previous_size_x = size_x;

    size_x += change;

    //make sure the size doesn't drop below 1
    if (size_x <= 0) size_x = 1;
    if (size_x == previous_size_x) return false;

    total_size = batch_size * size_y * size_x;
    resize_arrays();

    return true;
}

bool CNN_Node::modify_size_y(int change) {
    int previous_size_y = size_y;

    size_y += change;

    //make sure the size doesn't drop below 1
    if (size_y <= 0) size_y = 1;
    if (size_y == previous_size_y) return false;

    total_size = batch_size * size_y * size_x;
    resize_arrays();

    return true;
}

void CNN_Node::add_input() {
    total_inputs++;
    //cout << "\t\tadding input on node: " << innovation_number << ", total inputs: " << total_inputs << endl;
}

void CNN_Node::disable_input() {
    total_inputs--;
    //cout << "\t\tdisabling input on node: " << innovation_number << ", total inputs: " << total_inputs << endl;
}

int CNN_Node::get_number_inputs() const {
    return total_inputs;
}

int CNN_Node::get_inputs_fired() const {
    return inputs_fired;
}

void CNN_Node::add_output() {
    total_outputs++;
    //cout << "\t\tadding output on node: " << innovation_number << ", total outputs: " << total_outputs << endl;
}

void CNN_Node::disable_output() {
    total_outputs--;
    //cout << "\t\tdisabling output on node: " << innovation_number << ", total outputs: " << total_outputs << endl;
}

int CNN_Node::get_number_outputs() const {
    return total_outputs;
}

int CNN_Node::get_outputs_fired() const {
    return outputs_fired;
}

void CNN_Node::zero_test_statistics() {
    if (type == INPUT_NODE || type == SOFTMAX_NODE) return;

    running_mean = 0;
    running_variance = 0;
}

void CNN_Node::divide_test_statistics(int number_batches) {
    if ( !(type == INPUT_NODE || type == SOFTMAX_NODE)) {
        running_mean /= number_batches;
        running_variance /= number_batches;
    }

    cout << "node " << innovation_number << ", final test statistics, running_mean: " << running_mean << ", best_running_mean: " << best_running_mean << ", running_variance: " << running_variance << ", best_running_variance: " << best_running_variance << endl;
}

void CNN_Node::print_batch_statistics() {
    cerr << "batch_mean: " << batch_mean << ", running_mean: " << running_mean << ", batch_variance: " << batch_variance << ", running_variance: " << running_variance << ", gamma: " << gamma << ", beta: " << beta << endl;
}


void CNN_Node::batch_normalize(bool training, bool accumulating_test_statistics, float epsilon, float alpha) {
    //normalize the batch
    if (training || accumulating_test_statistics) {
        batch_mean = 0.0;

        //cout << "pre-batch normalization on node: " << innovation_number << endl;

        for (int32_t current = 0; current < total_size; current++) {
            batch_mean += values_in[current];
        }
        batch_mean /= (uint64_t)batch_size * (uint64_t)size_y * (uint64_t)size_x;

        batch_variance = 0.0;
        float diff;
        for (int32_t current = 0; current < total_size; current++) {
            diff = values_in[current] - batch_mean;
            batch_variance += diff * diff;
        }
        batch_variance /= (uint64_t)batch_size * (uint64_t)size_y * (uint64_t)size_x;

        batch_std_dev = exact_sqrt(batch_variance + epsilon);

        inverse_variance = 1.0 / batch_std_dev;

        //cout << endl;
        //cout << "post-batch normalization on node: " << innovation_number << endl;
        float temp;
        for (int32_t current = 0; current < total_size; current++) {
            temp = (values_in[current] - batch_mean) * inverse_variance;
            values_in[current] = temp;   //values in becomes x_hat
            values_out[current] = (gamma * temp) + beta;
        }

#ifdef NAN_CHECKS
        if (std::isnan(batch_mean) || std::isinf(batch_mean) || std::isnan(batch_variance) || std::isinf(batch_variance)) {
            cerr << "ERROR! NAN or INF batch_mean or batch_variance on node " << innovation_number << "!" << endl;
            cerr << "gamma: " << gamma << ", beta: " << beta << endl;

            int current = 0;
            for (int32_t batch_number = 0; batch_number < batch_size; batch_number++) {
                for (int32_t y = 0; y < size_y; y++) {
                    for (int32_t x = 0; x < size_x; x++) {
                        cout << setw(10) << std::fixed << values_in[current];
                        current++;
                    }
                    cout << endl;
                }
                cout << endl;
            }

            throw runtime_error("batch_normalize resulted in NAN or INF when calculating batch_mean or batch_variance");
        }
#endif

        batch_variance = (batch_size / (batch_size - 1)) * batch_variance;

        if (accumulating_test_statistics) {
            running_mean += batch_mean;
            running_variance += batch_variance;
            //running_mean = (batch_mean * alpha) + ((1.0 - alpha) * running_mean);
            //running_variance = (batch_variance * alpha) + ((1.0 - alpha) * running_variance);
 
            //cout << "node " << innovation_number << " accumulating test statistics, batch_mean: " << batch_mean << ", batch_variance: " << batch_variance << endl;
        } else {
            running_mean = (batch_mean * alpha) + ((1.0 - alpha) * running_mean);
            running_variance = (batch_variance * alpha) + ((1.0 - alpha) * running_variance);
        }

        //cout << "\tnode " << innovation_number << ", batch_mean: " << batch_mean << ", batch_variance: " << batch_variance << ", batch_std_dev: " << batch_std_dev << ", running_mean: " << running_mean << ", running_variance: " << running_variance << endl;

    } else { //testing
        float term1 =  gamma / exact_sqrt(running_variance + epsilon);
        float term2 = beta - ((gamma * running_mean) / exact_sqrt(running_variance + epsilon));

        for (int32_t current = 0; current < total_size; current++) {
            values_out[current] = (term1 * values_in[current]) + term2;

#ifdef NAN_CHECKS
            if (std::isnan(values_out[current]) || std::isinf(values_out[current])) {
                cerr << "ERROR! NAN or INF batch_mean or batch_variance on node " << innovation_number << "!" << endl;
                cerr << "values_out[" << current << "]: " << values_out[current] << ", values_in[" << current << "]: " << values_in[current] << endl;
                cerr << "gamma: " << gamma << ", beta: " << beta << endl;
                cerr << "term1: " << term1 << ", term2: " << term2 << endl;

                int current = 0;
                for (int32_t batch_number = 0; batch_number < batch_size; batch_number++) {
                    for (int32_t y = 0; y < size_y; y++) {
                        for (int32_t x = 0; x < size_x; x++) {
                            cout << setw(10) << std::fixed << values_in[current];
                            current++;
                        }
                        cout << endl;
                    }
                    cout << endl;
                }

                throw runtime_error("batch_normalize resulted in NAN or INF when calculating values_out");
            }
#endif

        }
    }
}

void CNN_Node::apply_relu(float* values, float* gradients) {
    for (int32_t current = 0; current < total_size; current++) {
        //cout << "values_out for node " << innovation_number << " now " << values_out[batch_number][y][x] << " after adding bias: " << bias[batch_number][y][x] << endl;

        //apply activation function
        if (values[current] <= RELU_MIN) {
            values[current] = values[current] * RELU_MIN_LEAK;
            gradients[current] = RELU_MIN_LEAK; 
        } else if (values[current] > RELU_MAX) {
            values[current] = RELU_MAX;
            gradients[current] = RELU_MAX_LEAK; 
        } else {
            gradients[current] = 1.0;
        }
    }
}

void CNN_Node::apply_dropout(float* values, float* gradients, bool perform_dropout, bool accumulate_test_statistics, float dropout_probability, minstd_rand0 &generator) {
    if (perform_dropout && !accumulate_test_statistics) {
        for (int32_t current = 0; current < total_size; current++) {
            //cout << "values for node " << innovation_number << " now " << values[batch_number][y][x] << " after adding bias: " << bias[batch_number][y][x] << endl;

            if (random_0_1(generator) < dropout_probability) {
                values[current] = 0.0;
                gradients[current] = 0.0;
            }
        }

    } else {
        float dropout_scale = 1.0 - dropout_probability;
        for (int32_t current = 0; current < total_size; current++) {
            values[current] *= dropout_scale;
        }
    }
}

void CNN_Node::backpropagate_relu(float* errors, float* gradients) {
    for (int32_t current = 0; current < total_size; current++) {
        errors[current] *= gradients[current];
    }
}


void CNN_Node::backpropagate_batch_normalization(bool training, float mu, float learning_rate, float epsilon) {
    //backprop  batch normalization here
    float delta_beta = 0.0;
    float delta_gamma = 0.0;

    float value_in;
    float value_hat;
    //float value_out;
    float delta_out;
    //float delta_values_hat;
    //float delta_values_hat_sum = 0.0;
    //float delta_values_hat_x_values_sum = 0.0;

    float derr_dvariance = 0.0;
    float derr_dmean = 0.0;

    float derr_dmean_term1 = 0.0;
    float derr_dmean_term2 = 0.0;

    float m = (uint64_t)batch_size * (uint64_t)size_y * (uint64_t)size_x;
    float diff;

    for (int32_t current = 0; current < total_size; current++) {
        delta_out = errors_out[current];
        value_hat = values_in[current];

        value_in = (value_hat + batch_mean) * batch_std_dev;
        diff = value_in - batch_mean;

        delta_beta += delta_out;
        delta_gamma += value_hat * delta_out;

        derr_dvariance += diff * delta_out * gamma;

        derr_dmean_term1 += delta_out * gamma;
        derr_dmean_term2 += diff;
    }

    float inv_m = 1.0 / m;
    float inv_m_x_2 = 2.0 * inv_m;

    float bv_pow_32 = -0.5 / (batch_std_dev * batch_std_dev * batch_std_dev);

    //derr_dvariance *= -0.5 * pow(batch_variance + epsilon, -1.5);
    derr_dvariance *= bv_pow_32;

    derr_dmean_term1 *= -inverse_variance;
    derr_dmean_term2 *= -inv_m_x_2 * derr_dvariance;
    derr_dmean = derr_dmean_term1 + derr_dmean_term2;

    for (int32_t current = 0; current < total_size; current++) {
        delta_out = errors_out[current] * relu_gradients[current];
        value_hat = values_in[current];
        value_in = (value_hat + batch_mean) * batch_std_dev;

        errors_in[current] = (delta_out * inverse_variance) + (derr_dvariance * inv_m_x_2 * (value_in - batch_mean)) + (derr_dmean * inv_m);


#ifdef NAN_CHECKS
        if (std::isnan(errors_in[current]) || std::isinf(errors_in[current])) {
            cerr << "ERROR! errors_in[" << current << "] became: " << errors_in[current] << "!" << endl;
            cerr << "value_in: " << value_in << endl;
            cerr << "derr_dmean: " << derr_dmean << endl;
            cerr << "inv_m: " << inv_m << endl;
            cerr << "derr_dvariance: " << derr_dvariance << endl;
            cerr << "inv_m_x_2: " << inv_m_x_2 << endl;
            cerr << "inverse_variance: " << inverse_variance << endl;
            cerr << "batch_mean: " << batch_mean << endl;
            cerr << "batch_size: " << batch_size << endl;
            cerr << "gamma: " << gamma << endl;
            cerr << "delta_out: " << delta_out << endl;
            cerr << "values_in[" << current << "]: " << values_in[current] << endl;

            for (int32_t current = 0; current < total_size; current++) {
                cout << "errors_out[" << current << "]: " << errors_out[current] << ", values_in[" << current << "]: " << values_in[current] << endl;
            }

            throw runtime_error("backpropagate_batch_normalization resulted in NAN or INF");
        }
#endif
    }

    if (training) {
        //backpropagate beta
        float pv_beta = previous_velocity_beta;

        float velocity_beta = (mu * pv_beta) - (learning_rate / batch_size) * delta_beta;
        beta += velocity_beta + mu * (velocity_beta - pv_beta);
        //beta += velocity_beta;

        //beta += (-mu * pv_beta + (1 + mu) * velocity_beta);
        //beta -= (beta * weight_decay);

        previous_velocity_beta = velocity_beta;

        if (beta <= -50) {
            cerr << "beta hit: " << beta << endl;
            beta = -50.0;
            previous_velocity_beta = 0.0;
        } else if (beta >= 50.0) {
            cerr << "beta hit: " << beta << endl;
            beta = 50.0;
            previous_velocity_beta = 0.0;
        }

        //backpropagate gamma
        float pv_gamma = previous_velocity_gamma;

        float velocity_gamma = (mu * pv_gamma) - (learning_rate / batch_size) * delta_gamma;
        //gamma += velocity_gamma;
        gamma += velocity_gamma + mu * (velocity_gamma - pv_gamma);
        //gamma += (-mu * pv_gamma + (1 + mu) * velocity_gamma);
        //gamma -= (gamma * weight_decay);

        previous_velocity_gamma = velocity_gamma;

        if (gamma <= -50) {
            cerr << "gamma hit: " << gamma << endl;
            gamma = -50.0;
            previous_velocity_gamma = 0.0;
        } else if (gamma >= 50.0) {
            cerr << "gamma hit: " << gamma << endl;
            gamma = 50.0;
            previous_velocity_gamma = 0.0;
        }
    }

    //cout << "\tnode " << innovation_number << ", delta_gamma: " << delta_gamma << ", delta_beta: " << delta_beta << ", gamma now: " << gamma << ", beta now: " << beta << endl;
}


void CNN_Node::set_values(const ImagesInterface &images, const vector<int> &batch, int channel, bool perform_dropout, bool accumulate_test_statistics, float input_dropout_probability, minstd_rand0 &generator) {
    //images.size() may be less than batch size, in the case when the total number of images is not divisible by the batch_size
    if (batch.size() > batch_size) {
        ostringstream error_message;
        error_message << "ERROR: number of batch images: " << batch.size() << " > batch_size of input node: " << batch_size << endl;
        throw runtime_error(error_message.str());
    }

    if (images.get_image_height() != size_y) {
        ostringstream error_message;
        error_message << "ERROR: height of input image: " << images.get_image_height() << " != size_y of input node: " << size_y << endl;
        throw runtime_error(error_message.str());
    }

    if (images.get_image_width() != size_x) {
        ostringstream error_message;
        error_message << "ERROR: width of input image: " << images.get_image_width() << " != size_x of input node: " << size_x << endl;
        throw runtime_error(error_message.str());
    }

    //images.size() may be less than batch size, in the case when the total number of images is not divisible by the batch_size
    int current = 0;
    for (int32_t batch_number = 0; batch_number < batch.size(); batch_number++) {
        for (int32_t y = 0; y < size_y; y++) {
            for (int32_t x = 0; x < size_x; x++) {
                values_out[current] = images.get_pixel(batch[batch_number], channel, y, x);
                current++;
            }
        }
    }

    if (input_dropout_probability > 0) apply_dropout(values_out, relu_gradients, perform_dropout, accumulate_test_statistics, input_dropout_probability, generator);
}


void CNN_Node::input_fired(bool training, bool accumulate_test_statistics, float epsilon, float alpha, bool perform_dropout, float hidden_dropout_probability, minstd_rand0 &generator) {

    using namespace std::chrono;
    high_resolution_clock::time_point input_fired_start_time = high_resolution_clock::now();

    inputs_fired++;

    //cout << "input fired on node: " << innovation_number << ", inputs fired: " << inputs_fired << ", total_inputs: " << total_inputs << endl;

    if (inputs_fired == total_inputs) {
        if (type != SOFTMAX_NODE) {
            apply_relu(values_in, relu_gradients);

            if (hidden_dropout_probability > 0) apply_dropout(values_in, relu_gradients, perform_dropout, accumulate_test_statistics, hidden_dropout_probability, generator);

            batch_normalize(training, accumulate_test_statistics, epsilon, alpha);
        }

    } else if (inputs_fired > total_inputs) {
        cerr << "ERROR! inputs_fired > total_inputs" << endl;

        cerr << "inputs_fired: " << inputs_fired << endl;
        cerr << "total_inputs: " << total_inputs << endl;

        cerr << "node: " << endl;
        print(cerr);

        throw runtime_error("Error in input fired, inputs_fired > total_inputs");
    }

    high_resolution_clock::time_point input_fired_end_time = high_resolution_clock::now();
    duration<float, std::milli> time_span = input_fired_end_time - input_fired_start_time;

    input_fired_time += time_span.count() / 1000.0;
}

void CNN_Node::output_fired(bool training, float mu, float learning_rate, float epsilon) {
    using namespace std::chrono;
    high_resolution_clock::time_point output_fired_start_time = high_resolution_clock::now();


    outputs_fired++;

    //cout << "output fired on node: " << innovation_number << ", outputs fired: " << outputs_fired << ", total_outputs: " << total_outputs << endl;

    if (outputs_fired == total_outputs) {
        if (type != SOFTMAX_NODE && type != INPUT_NODE) {
            backpropagate_batch_normalization(training, mu, learning_rate, epsilon);
            backpropagate_relu(errors_in, relu_gradients);
        }

    } else if (outputs_fired > total_outputs) {
        cerr << "ERROR! outputs_fired > total_outputs" << endl;

        cerr << "outputs_fired: " << outputs_fired << endl;
        cerr << "total_outputs: " << total_outputs << endl;

        cerr << "node: " << endl;
        print(cerr);

        throw runtime_error("Error in output fired, outputs_fired > total_outputs");
    }


    high_resolution_clock::time_point output_fired_end_time = high_resolution_clock::now();
    duration<float, std::milli> time_span = output_fired_end_time - output_fired_start_time;

    output_fired_time += time_span.count() / 1000.0;
}


bool CNN_Node::has_nan() const {
    for (int32_t current = 0; current < total_size; current++) {
        if (std::isnan(values_in[current]) || std::isinf(values_in[current])) return true;
        if (std::isnan(errors_in[current]) || std::isinf(errors_in[current])) return true;

        if (std::isnan(values_out[current]) || std::isinf(values_out[current])) return true;
        if (std::isnan(errors_out[current]) || std::isinf(errors_out[current])) return true;
        if (std::isnan(relu_gradients[current]) || std::isinf(relu_gradients[current])) return true;
        if (std::isnan(pool_gradients[current]) || std::isinf(pool_gradients[current])) return true;
    }

    return false;
}

void CNN_Node::print_statistics() {
    cerr << "node " << setw(4) << innovation_number;
    cerr << ", gamma: " << gamma << ", beta: " << beta;
    cerr << "\tINPUTS: ";
    print_statistics(values_in, errors_in, relu_gradients);
    cerr << "\tOUTPUTS: ";
    print_statistics(values_out, errors_out, relu_gradients);
}

void CNN_Node::print_statistics(const float* values, const float* errors, const float* gradients) {
    float value_min = std::numeric_limits<float>::max(), value_max = -std::numeric_limits<float>::max(), value_avg = 0.0;
    float error_min = std::numeric_limits<float>::max(), error_max = -std::numeric_limits<float>::max(), error_avg = 0.0;
    float gradient_min = std::numeric_limits<float>::max(), gradient_max = -std::numeric_limits<float>::max(), gradient_avg = 0.0;

    for (int32_t current = 0; current < total_size; current++) {
        if (values[current] < value_min) value_min = values[current];
        if (values[current] > value_max) value_max = values[current];
        value_avg += values[current];

        if (gradients[current] < gradient_min) gradient_min = gradients[current];
        if (gradients[current] > gradient_max) gradient_max = gradients[current];
        gradient_avg += gradients[current];

        if (errors[current] < error_min) error_min = errors[current];
        if (errors[current] > error_max) error_max = errors[current];
        error_avg += errors[current];
    }

    error_avg /= batch_size * size_y * size_x;
    gradient_avg /= batch_size * size_y * size_x;
    value_avg /= batch_size * size_y * size_x;

    cerr << "v_min: " << value_min << ", v_avg: " << value_avg << ", v_max: " << value_max;
    cerr << ", gradient_min: " << gradient_min << ", gradient_avg: " << gradient_avg << ", gradient_max: " << gradient_max << endl;
    cerr << ", error_min: " << error_min << ", error_avg: " << error_avg << ", error_max: " << error_max << endl;
}

ostream &operator<<(ostream &os, const CNN_Node* node) {
    os << node->node_id << " ";
    os << node->exact_id << " ";
    os << node->genome_id << " ";
    os << node->innovation_number << " ";
    os << node->depth << " ";
    os << node->batch_size << " ";
    os << node->size_x << " ";
    os << node->size_y << " ";
    os << node->type << " ";
    os << node->weight_count << " ";
    os << node->needs_initialization << " ";
    os << node->disabled << endl;

    write_hexfloat(os, node->gamma);
    os << endl;

    write_hexfloat(os, node->best_gamma);
    os << endl;

    write_hexfloat(os, node->previous_velocity_gamma);
    os << endl;

    write_hexfloat(os, node->beta);
    os << endl;

    write_hexfloat(os, node->best_beta);
    os << endl;

    write_hexfloat(os, node->previous_velocity_beta);
    os << endl;

    write_hexfloat(os, node->running_mean);
    os << endl;

    write_hexfloat(os, node->best_running_mean);
    os << endl;

    write_hexfloat(os, node->running_variance);
    os << endl;

    write_hexfloat(os, node->best_running_variance);

    return os;
}

std::istream &operator>>(std::istream &is, CNN_Node* node) {
    is >> node->node_id;
    is >> node->exact_id;
    is >> node->genome_id;
    is >> node->innovation_number;
    is >> node->depth;
    is >> node->batch_size;
    is >> node->size_x;
    is >> node->size_y;
    is >> node->type;
    is >> node->weight_count;
    is >> node->needs_initialization;
    is >> node->disabled;

    node->total_size = node->batch_size * node->size_y * node->size_x;

    node->gamma = read_hexfloat(is);
    node->best_gamma = read_hexfloat(is);
    node->previous_velocity_gamma = read_hexfloat(is);
    node->beta = read_hexfloat(is);
    node->best_beta = read_hexfloat(is);
    node->previous_velocity_beta = read_hexfloat(is);
    node->running_mean = read_hexfloat(is);
    node->best_running_mean = read_hexfloat(is);
    node->running_variance = read_hexfloat(is);
    node->best_running_variance = read_hexfloat(is);

    /*
    cerr << "node " << node->innovation_number
         << " " << node->gamma
         << " " << node->best_gamma
         << " " << node->previous_velocity_gamma
         << " " << node->beta
         << " " << node->best_beta
         << " " << node->previous_velocity_beta
         << " " << node->running_mean
         << " " << node->best_running_mean
         << " " << node->running_variance
         << " " << node->best_running_variance << endl;
    */

    node->total_inputs = 0;
    node->inputs_fired = 0;

    node->total_outputs = 0;
    node->outputs_fired = 0;

    node->forward_visited = false;
    node->reverse_visited = false;

    node->values_in = new float[node->total_size]();
    node->errors_in = new float[node->total_size]();

    node->values_out = new float[node->total_size]();
    node->errors_out = new float[node->total_size]();
    node->relu_gradients = new float[node->total_size]();
    node->pool_gradients = new float[node->total_size]();

    return is;
}


bool CNN_Node::is_identical(const CNN_Node *other, bool testing_checkpoint) {
    if (are_different("node_id", node_id, other->node_id)) return false;
    if (are_different("exact_id", exact_id, other->exact_id)) return false;
    if (are_different("genome_id", genome_id, other->genome_id)) return false;

    if (are_different("innovation_number", innovation_number, other->innovation_number)) return false;
    if (are_different("depth", depth, other->depth)) return false;

    if (are_different("batch_size", batch_size, other->batch_size)) return false;
    if (are_different("size_y", size_y, other->size_y)) return false;
    if (are_different("size_x", size_x, other->size_x)) return false;
    if (are_different("total_size", total_size, other->total_size)) return false;

    if (are_different("weight_count", weight_count, other->weight_count)) return false;

    if (are_different("type", type, other->type)) return false;

    if (are_different("total_inputs", total_inputs, other->total_inputs)) return false;
    //if (are_different("inputs_fired", inputs_fired, other->inputs_fired)) return false;

    if (are_different("total_outputs", total_outputs, other->total_outputs)) return false;
    if (are_different("outputs_fired", outputs_fired, other->outputs_fired)) return false;

    if (are_different("forward_visited", forward_visited, other->forward_visited)) return false;
    if (are_different("reverse_visited", reverse_visited, other->reverse_visited)) return false;
    if (are_different("needs_initialization", needs_initialization, other->needs_initialization)) return false;

    if (are_different("disabled", disabled, other->disabled)) return false;

    if (are_different("gamma", gamma, other->gamma)) return false;
    if (are_different("best_gamma", best_gamma, other->best_gamma)) return false;
    if (are_different("previous_velocity_gamma", previous_velocity_gamma, other->previous_velocity_gamma)) return false;

    if (are_different("beta", beta, other->beta)) return false;
    if (are_different("best_beta", best_beta, other->best_beta)) return false;
    if (are_different("previous_velocity_beta", previous_velocity_beta, other->previous_velocity_beta)) return false;

    //these are all reset and recalculated on each batch
    //if (are_different("batch_mean", batch_mean, other->batch_mean)) return false;
    //if (are_different("batch_variance", batch_variance, other->batch_variance)) return false;
    //if (are_different("batch_std_dev", batch_std_dev, other->batch_std_dev)) return false;
    //if (are_different("inverse_variance", inverse_variance, other->inverse_variance)) return false;

    if (are_different("running_mean", running_mean, other->running_mean)) return false;
    if (are_different("best_running_mean", best_running_mean, other->best_running_mean)) return false;

    if (are_different("running_variance", running_variance, other->running_variance)) return false;
    if (are_different("best_running_variance", best_running_variance, other->best_running_variance)) return false;

    //values, errors_out, relu_gradients, pool_gradients, values_in, errors_in, input_fired_time and output_fired time
    //are all reset to 0 before every pass so don't need to be compared

    return true;
}

