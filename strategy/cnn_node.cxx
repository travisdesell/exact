#include <cmath>
using std::isnan;
using std::isinf;

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

#include <sstream>
using std::ostringstream;
using std::istringstream;

#include <string>
using std::string;

#include <vector>
using std::vector;

#include "image_tools/image_set.hxx"
#include "common/random.hxx"
#include "cnn_edge.hxx"
#include "cnn_node.hxx"

#include "stdint.h"

double read_hexfloat(istream &infile) {
#ifdef _WIN32
	double result;
	infile >> std::hexfloat >> result >> std::defaultfloat;
	return result;
#else
    string s;
    infile >> s;
    return stod(s);
#endif
}

void write_hexfloat(ostream &outfile, double value) {
#ifdef _WIN32
    char hf[32];
    sprintf(hf, "%la", value);
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

    visited = false;
    needs_initialization = true;

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

CNN_Node::CNN_Node(int _innovation_number, double _depth, int _batch_size, int _size_x, int _size_y, int _type) {
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

    visited = false;

    needs_initialization = true;

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

    values_in = vector< vector< vector<double > > >(batch_size, vector< vector<double> >(size_y, vector<double>(size_x, 0.0)));
    values_hat = vector< vector< vector<double > > >(batch_size, vector< vector<double> >(size_y, vector<double>(size_x, 0.0)));
    values_out = vector< vector< vector<double > > >(batch_size, vector< vector<double> >(size_y, vector<double>(size_x, 0.0)));
    deltas = vector< vector< vector<double > > >(batch_size, vector< vector<double> >(size_y, vector<double>(size_x, 0.0)));

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
        
        type = atoi(row[++column]);

        //need to reset these because it will be modified when edges are set
        total_inputs = 0;
        inputs_fired = 0;

        total_outputs = 0;
        outputs_fired = 0;

        visited = atoi(row[++column]);
        weight_count = atoi(row[++column]);
        needs_initialization = atoi(row[++column]);

        gamma = atoi(row[++column]);
        best_gamma = atoi(row[++column]);
        previous_velocity_gamma = atoi(row[++column]);

        beta = atoi(row[++column]);
        best_beta = atoi(row[++column]);
        previous_velocity_beta = atoi(row[++column]);

        running_mean = atoi(row[++column]);
        best_running_mean = atoi(row[++column]);
        running_variance = atoi(row[++column]);
        best_running_variance = atoi(row[++column]);

        mysql_free_result(result);
    } else {
        cerr << "ERROR! Could not find cnn_node in database with node id: " << node_id << endl;
        exit(1);
    }

    weight_count = 0;

    //initialize arrays not stored to database
    values_in = vector< vector< vector<double > > >(batch_size, vector< vector<double> >(size_y, vector<double>(size_x, 0.0)));
    values_hat = vector< vector< vector<double > > >(batch_size, vector< vector<double> >(size_y, vector<double>(size_x, 0.0)));
    values_out = vector< vector< vector<double > > >(batch_size, vector< vector<double> >(size_y, vector<double>(size_x, 0.0)));
    deltas = vector< vector< vector<double > > >(batch_size, vector< vector<double> >(size_y, vector<double>(size_x, 0.0)));

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
        << ", size_y = " << size_y;
        << ", type = " << type
        << ", visited = " << visited
        << ", weight_count = " << weight_count
        << ", needs_initialization = " << needs_initialization
        << ", gamma = " << gamma
        << ", best_gamma = " << best_gamma
        << ", previous_velocity_gamma = " << previous_velocity_gamma
        << ", beta = " << beta
        << ", best_beta = " << best_beta
        << ", previous_velocity_beta = " << previous_velocity_beta
        << ", running_mean = " << running_mean
        << ", best_running_mean = " << best_running_mean
        << ", running_variance = " << running_variance
        << ", best_running_variance = " << best_running_variance;

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

CNN_Node* CNN_Node::copy() const {
    CNN_Node *copy = new CNN_Node();

    copy->node_id = -1;
    copy->genome_id = genome_id;

    copy->innovation_number = innovation_number;
    copy->depth = depth;
    copy->batch_size = batch_size;
    copy->size_x = size_x;
    copy->size_y = size_y;

    copy->type = type;

    copy->total_inputs = 0; //this will be updated when edges are set
    copy->inputs_fired = inputs_fired;

    copy->total_outputs = 0; //this will be updated when edges are set
    copy->outputs_fired = outputs_fired;

    copy->visited = visited;
    copy->weight_count = weight_count;
    copy->needs_initialization = needs_initialization;

    copy->gamma = gamma;
    copy->best_gamma = best_gamma;
    copy->previous_velocity_gamma = previous_velocity_gamma;

    copy->best_beta = best_beta;
    copy->previous_velocity_beta = previous_velocity_beta;

    copy->running_mean = running_mean;
    copy->best_running_mean = best_running_mean;
    copy->running_variance = running_variance;
    copy->best_running_variance = best_running_variance;

    copy->values_in = values_in;
    copy->values_hat = values_hat;
    copy->values_out = values_out;
    copy->deltas = deltas;

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


bool CNN_Node::is_softmax() const {
    return type == SOFTMAX_NODE;
}


bool CNN_Node::is_visited() const {
    return visited;
}

void CNN_Node::visit() {
    visited = true;
}

void CNN_Node::set_unvisited() {
    visited = false;
}

int CNN_Node::get_batch_size() const {
    return batch_size;
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

double CNN_Node::get_depth() const {
    return depth;
}

void CNN_Node::resize_arrays() {
    values_in = vector< vector< vector<double> > >(batch_size, vector< vector<double> >(size_y, vector<double>(size_x, 0.0)));
    values_hat = vector< vector< vector<double> > >(batch_size, vector< vector<double> >(size_y, vector<double>(size_x, 0.0)));
    values_out = vector< vector< vector<double> > >(batch_size, vector< vector<double> >(size_y, vector<double>(size_x, 0.0)));
    deltas = vector< vector< vector<double> > >(batch_size, vector< vector<double> >(size_y, vector<double>(size_x, 0.0)));
    needs_initialization = true;
}


void CNN_Node::set_input_values(const vector<const Image> &images, int channel, bool perform_dropout, double input_dropout_probability, minstd_rand0 &generator) {
    if (images.size() != batch_size) {
        cerr << "ERROR: number of batch images: " << images.size() << " != batch_size of input node: " << batch_size << endl;
        exit(1);
    }

    if (images[0].get_rows() != size_y) {
        cerr << "ERROR: rows of input image: " << images[0].get_rows() << " != size_y of input node: " << size_y << endl;
        exit(1);
    }

    if (images[0].get_cols() != size_x) {
        cerr << "ERROR: cols of input image: " << images[0].get_cols() << " != size_x of input node: " << size_x << endl;
        exit(1);
    }

    if (perform_dropout) {
        for (int32_t batch_number = 0; batch_number < batch_size; batch_number++) {
            //cout << "setting input image[" << batch_number << "]: " << endl;
            for (int32_t y = 0; y < size_y; y++) {
                for (int32_t x = 0; x < size_x; x++) {
                    if (rng_double(generator) < input_dropout_probability) {
                        values_out[batch_number][y][x] = 0.0;
                    } else {
                        values_out[batch_number][y][x] = images[batch_number].get_pixel(channel, y, x);
                    }
                    //cout << setw(5) << values[y][x];
                }
            }
            //cout << endl;
        }
    } else {
        double dropout_scale = 1.0 - input_dropout_probability;
        for (int32_t batch_number = 0; batch_number < batch_size; batch_number++) {
            //cout << "setting input image[" << batch_number << "]: " << endl;
            for (int32_t y = 0; y < size_y; y++) {
                for (int32_t x = 0; x < size_x; x++) {
                    values_out[batch_number][y][x] = images[batch_number].get_pixel(channel, y, x);
                    //cout << setw(5) << values[y][x];
                }
            }
            //cout << endl;
        }
    }
}

double CNN_Node::get_input_value(int batch_number, int y, int x) {
    return values_in[batch_number][y][x];
}

void CNN_Node::set_input_value(int batch_number, int y, int x, double value) {
    values_in[batch_number][y][x] = value;
}

vector< vector< vector<double> >>& CNN_Node::get_input_values() {
    return values_in;
}


double CNN_Node::get_output_value(int batch_number, int y, int x) {
    return values_out[batch_number][y][x];
}

void CNN_Node::set_output_value(int batch_number, int y, int x, double value) {
    values_out[batch_number][y][x] = value;
}


void CNN_Node::set_delta(int batch_number, int y, int x, double delta) {
    deltas[batch_number][y][x] = delta;
}


vector< vector< vector<double> >>& CNN_Node::get_output_values() {
    return values_out;
}

vector<vector< vector<double> > >& CNN_Node::get_deltas() {
    return deltas;
}

void CNN_Node::print(ostream &out) {
    out << "CNN_Node " << innovation_number << ", at depth: " << depth << " of input size x: " << size_x << ", y: " << size_y << endl;

    for (int32_t batch_number = 0; batch_number < batch_size; batch_number++) {
        out << "    batch_number: " << batch_number << endl;
        out << "    values_in:" << endl;
        for (int32_t i = 0; i < size_y; i++) {
            out << "    ";
            for (int32_t j = 0; j < size_x; j++) {
                out << setw(13) << setprecision(8) << values_in[batch_number][i][j];
            }
            out << endl;
        }

        out << "    values_hat:" << endl;
        for (int32_t i = 0; i < size_y; i++) {
            out << "    ";
            for (int32_t j = 0; j < size_x; j++) {
                out << setw(13) << setprecision(8) << values_hat[batch_number][i][j];
            }
            out << endl;
        }

        out << "    values_out:" << endl;
        for (int32_t i = 0; i < size_y; i++) {
            out << "    ";
            for (int32_t j = 0; j < size_x; j++) {
                out << setw(13) << setprecision(8) << values_out[batch_number][i][j];
            }
            out << endl;
        }

        out << "    deltas:" << endl;
        for (int32_t i = 0; i < size_y; i++) {
            out << "    ";
            for (int32_t j = 0; j < size_x; j++) {
                out << setw(13) << setprecision(8) << deltas[batch_number][i][j];
            }
            out << endl;
        }
    }
}

void CNN_Node::reset() {
    inputs_fired = 0;
    outputs_fired = 0;

    for (int32_t batch_number = 0; batch_number < batch_size; batch_number++) {
        for (int32_t y = 0; y < size_y; y++) {
            for (int32_t x = 0; x < size_x; x++) {
                values_in[batch_number][y][x] = 0;
                values_hat[batch_number][y][x] = 0;
                values_out[batch_number][y][x] = 0;
                deltas[batch_number][y][x] = 0;
            }
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

void CNN_Node::batch_normalize(bool training, double epsilon, double alpha) {
    //normalize the batch
    if (training) {
        double batch_mean = 0.0;

        //cout << "pre-batch normalization on node: " << innovation_number << endl;

        for (int32_t batch_number = 0; batch_number < batch_size; batch_number++) {
            for (int32_t y = 0; y < size_y; y++) {
                for (int32_t x = 0; x < size_x; x++) {
                    batch_mean += values_in[batch_number][y][x];
                    //cout << setw(10) << std::fixed << values_in[batch_number][y][x];
                }
                //cout << endl;
            }
            //cout << endl;
        }
        batch_mean /= (uint64_t)batch_size * (uint64_t)size_y * (uint64_t)size_x;

        double batch_variance = 0.0;
        double diff;
        for (int32_t batch_number = 0; batch_number < batch_size; batch_number++) {
            for (int32_t y = 0; y < size_y; y++) {
                for (int32_t x = 0; x < size_x; x++) {
                    diff = values_in[batch_number][y][x] - batch_mean;
                    batch_variance += diff * diff;
                }
            }
        }
        batch_variance /= (uint64_t)batch_size * (uint64_t)size_y * (uint64_t)size_x;

        double batch_std_dev = sqrt(batch_variance + epsilon);

        //cout << endl;
        //cout << "post-batch normalization on node: " << innovation_number << endl;
        double temp;
        for (int32_t batch_number = 0; batch_number < batch_size; batch_number++) {
            for (int32_t y = 0; y < size_y; y++) {
                for (int32_t x = 0; x < size_x; x++) {
                    temp = (values_in[batch_number][y][x] - batch_mean) / batch_std_dev;
                    values_hat[batch_number][y][x] = temp;
                    values_out[batch_number][y][x] = (gamma * temp) + beta;

                    //cout << setw(10) << std::fixed << values_out[batch_number][y][x];
                }
                //cout << endl;
            }
            //cout << endl;
        }

//#ifdef NAN_CHECKS
        if (isnan(batch_mean) || isinf(batch_mean) || isnan(batch_variance) || isinf(batch_variance)) {
            cerr << "ERROR! NAN or INF batch_mean or batch_variance on node " << innovation_number << "!" << endl;
            cerr << "gamma: " << gamma << ", beta: " << beta << endl;

            for (int32_t batch_number = 0; batch_number < batch_size; batch_number++) {
                for (int32_t y = 0; y < size_y; y++) {
                    for (int32_t x = 0; x < size_x; x++) {
                        cout << setw(10) << std::fixed << values_in[batch_number][y][x];
                    }
                    cout << endl;
                }
                cout << endl;
            }

            exit(1);
        }
//#endif

        inverse_variance = 1.0 / batch_std_dev;

        batch_variance = batch_size / (batch_size - 1);

        running_mean = (batch_mean * alpha) + ((1.0 - alpha) * running_mean);
        running_variance = (batch_variance * alpha) + ((1.0 - alpha) * running_variance);

        //cout << "\tnode " << innovation_number << ", batch_mean: " << batch_mean << ", batch_variance: " << batch_variance << ", batch_std_dev: " << batch_std_dev << ", running_mean: " << running_mean << ", running_variance: " << running_variance << endl;

    } else { //testing
        double term1 =  gamma / sqrt(running_variance + epsilon);
        double term2 = beta - ((gamma * running_mean) / sqrt(running_variance + epsilon));

        for (int32_t batch_number = 0; batch_number < batch_size; batch_number++) {

            for (int32_t y = 0; y < size_y; y++) {
                for (int32_t x = 0; x < size_x; x++) {
                    values_out[batch_number][y][x] = (term1 * values_in[batch_number][y][x]) + term2;
                }
            }
        }
    }
}

bool CNN_Node::modify_size_x(int change) {
    int previous_size_x = size_x;

    size_x += change;

    //make sure the size doesn't drop below 1
    if (size_x <= 0) size_x = 1;
    if (size_x == previous_size_x) return false;

    resize_arrays();

    return true;
}

bool CNN_Node::modify_size_y(int change) {
    int previous_size_y = size_y;

    size_y += change;

    //make sure the size doesn't drop below 1
    if (size_y <= 0) size_y = 1;
    if (size_y == previous_size_y) return false;

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

void CNN_Node::input_fired(bool training, double epsilon, double alpha) {
    inputs_fired++;

    //cout << "input fired on node: " << innovation_number << ", inputs fired: " << inputs_fired << ", total_inputs: " << total_inputs << endl;

    if (inputs_fired == total_inputs) {
        if (type != SOFTMAX_NODE) {
            batch_normalize(training, epsilon, alpha);

            for (int32_t batch_number = 0; batch_number < batch_size; batch_number++) {
                for (int32_t y = 0; y < size_y; y++) {
                    for (int32_t x = 0; x < size_x; x++) {
                        //cout << "values for node " << innovation_number << " now " << values[batch_number][y][x] << " after adding bias: " << bias[batch_number][y][x] << endl;

                        //apply activation function
                        if (values_in[batch_number][y][x] <= RELU_MIN) {
                            values_in[batch_number][y][x] *= RELU_MIN_LEAK;
                        } else if (values_in[batch_number][y][x] > RELU_MAX) {
                            values_in[batch_number][y][x] = RELU_MAX;
                        }
                    }
                }
            }

        }

    } else if (inputs_fired > total_inputs) {
        cerr << "ERROR! inputs_fired > total_inputs" << endl;

        cerr << "inputs_fired: " << inputs_fired << endl;
        cerr << "total_inputs: " << total_inputs << endl;

        cerr << "node: " << endl;
        print(cerr);

        exit(1);
    }
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


void CNN_Node::output_fired(double mu, double learning_rate) {
    outputs_fired++;

    //cout << "output fired on node: " << innovation_number << ", outputs fired: " << outputs_fired << ", total_outputs: " << total_outputs << endl;

    if (outputs_fired == total_outputs) {
        if (type != SOFTMAX_NODE && type != INPUT_NODE) {

            //backprop relu
            double gradient;
            for (int32_t batch_number = 0; batch_number < batch_size; batch_number++) {
                for (int32_t y = 0; y < size_y; y++) {
                    for (int32_t x = 0; x < size_x; x++) {
                        if (values_in[batch_number][y][x] <= RELU_MIN) gradient = RELU_MIN_LEAK;
                        else if (values_in[batch_number][y][x] > RELU_MAX) gradient = RELU_MAX_LEAK;
                        else gradient = 1.0;

                        //deltas now delta before REL
                        deltas[batch_number][y][x] *= gradient;
                    }
                }
            }

            //backprop  batch normalization here
            double delta_beta = 0.0;
            double delta_gamma = 0.0;

            double value_hat, value_out;
            double delta_out, delta_values_hat;
            double delta_values_hat_sum = 0.0;
            double delta_values_hat_x_values_sum = 0.0;

            for (int32_t batch_number = 0; batch_number < batch_size; batch_number++) {
                for (int32_t y = 0; y < size_y; y++) {
                    for (int32_t x = 0; x < size_x; x++) {
                        delta_out = deltas[batch_number][y][x];
                        value_hat = values_hat[batch_number][y][x];
                        value_out = values_out[batch_number][y][x];

                        delta_values_hat = gamma * delta_out;

                        delta_beta += delta_out;
                        delta_gamma += value_hat * delta_out;

                        delta_values_hat_sum += delta_values_hat;
                        delta_values_hat_x_values_sum += delta_values_hat * value_hat;
                    }
                }
            }

            double inv_var_div_batch = inverse_variance / batch_size;
            double batch_x_gamma = batch_size * gamma;

            for (int32_t batch_number = 0; batch_number < batch_size; batch_number++) {
                for (int32_t y = 0; y < size_y; y++) {
                    for (int32_t x = 0; x < size_x; x++) {
                        delta_out = deltas[batch_number][y][x];
                        delta_values_hat = gamma * delta_out;


                        //this makes delta = delta_in
                        deltas[batch_number][y][x] = inv_var_div_batch * ((batch_x_gamma * delta_out) - delta_values_hat_sum - (values_hat[batch_number][y][x] * delta_values_hat_x_values_sum));
//#ifdef NAN_CHECKS
                        if (isnan(deltas[batch_number][y][x]) || isinf(deltas[batch_number][y][x])) {
                            cerr << "ERROR! deltas[" << batch_number << "][" << y << "][" << x << "] became: " << deltas[batch_number][y][x] << "!" << endl;
                            cerr << "inverse_variance: " << inverse_variance << endl;
                            cerr << "batch_size: " << batch_size << endl;
                            cerr << "gamma: " << gamma << endl;
                            cerr << "delta_out: " << delta_out << endl;
                            cerr << "delta_values_hat_sum: " << delta_values_hat_sum << endl;
                            cerr << "values_hat[" << batch_number << "][" << y << "][" << x << "]: " << values_hat[batch_number][y][x] << endl;
                            cerr << "delta_values_hat_x_values_sum: " << delta_values_hat_x_values_sum << endl;

                            exit(1);
                        }
//#endif
                    }
                }
            }
            //deltas now delta_in

            //backpropagate beta
            double pv_beta = previous_velocity_beta;

            double velocity_beta = (mu * pv_beta) - learning_rate * delta_beta;
            beta += (-mu * pv_beta + (1 + mu) * velocity_beta);
            //beta -= (beta * weight_decay);

            previous_velocity_beta = velocity_beta;

            //backpropagate gamma
            double pv_gamma = previous_velocity_gamma;

            double velocity_gamma = (mu * pv_gamma) - learning_rate * delta_gamma;
            gamma += (-mu * pv_gamma + (1 + mu) * velocity_gamma);
            //gamma -= (gamma * weight_decay);

            //cout << "\tnode " << innovation_number << ", delta_gamma: " << delta_gamma << ", delta_beta: " << delta_beta << ", gamma now: " << gamma << ", beta now: " << beta << endl;

            previous_velocity_gamma = velocity_gamma;
        }

    } else if (outputs_fired > total_outputs) {
        cerr << "ERROR! outputs_fired > total_outputs" << endl;

        cerr << "outputs_fired: " << outputs_fired << endl;
        cerr << "total_outputs: " << total_outputs << endl;

        cerr << "node: " << endl;
        print(cerr);

        exit(1);
    }
}


bool CNN_Node::has_nan() const {
    for (int32_t batch_number = 0; batch_number < batch_size; batch_number++) {
        for (int32_t y = 0; y < size_y; y++) {
            for (int32_t x = 0; x < size_x; x++) {
                if (isnan(values_in[batch_number][y][x]) || isinf(values_in[batch_number][y][x])) return true;
                if (isnan(values_hat[batch_number][y][x]) || isinf(values_hat[batch_number][y][x])) return true;
                if (isnan(values_out[batch_number][y][x]) || isinf(values_out[batch_number][y][x])) return true;
                if (isnan(deltas[batch_number][y][x]) || isinf(deltas[batch_number][y][x])) return true;
            }
        }
    }

    return false;
}

void CNN_Node::print_statistics() {
    double value_in_min = std::numeric_limits<double>::max(), value_in_max = -std::numeric_limits<double>::max(), value_in_avg = 0.0;
    double value_hat_min = std::numeric_limits<double>::max(), value_hat_max = -std::numeric_limits<double>::max(), value_hat_avg = 0.0;
    double value_out_min = std::numeric_limits<double>::max(), value_out_max = -std::numeric_limits<double>::max(), value_out_avg = 0.0;
    double delta_min = std::numeric_limits<double>::max(), delta_max = -std::numeric_limits<double>::max(), delta_avg = 0.0;

    for (int batch_number = 0; batch_number < batch_size; batch_number++) {
        for (int y = 0; y < size_y; y++) {
            for (int x = 0; x < size_x; x++) {
                if (values_in[batch_number][y][x] < value_in_min) value_in_min = values_in[batch_number][y][x];
                if (values_in[batch_number][y][x] > value_in_max) value_in_max = values_in[batch_number][y][x];
                value_in_avg += values_in[batch_number][y][x];

                if (values_hat[batch_number][y][x] < value_hat_min) value_hat_min = values_hat[batch_number][y][x];
                if (values_hat[batch_number][y][x] > value_hat_max) value_hat_max = values_hat[batch_number][y][x];
                value_hat_avg += values_hat[batch_number][y][x];

                if (values_out[batch_number][y][x] < value_out_min) value_out_min = values_out[batch_number][y][x];
                if (values_out[batch_number][y][x] > value_out_max) value_out_max = values_out[batch_number][y][x];
                value_out_avg += values_out[batch_number][y][x];

                if (deltas[batch_number][y][x] < delta_min) delta_min = deltas[batch_number][y][x];
                if (deltas[batch_number][y][x] > delta_max) delta_max = deltas[batch_number][y][x];
                delta_avg += deltas[batch_number][y][x];
            }
        }
    }

    delta_avg /= batch_size * size_y * size_x;
    value_in_avg /= batch_size * size_y * size_x;
    value_hat_avg /= batch_size * size_y * size_x;
    value_out_avg /= batch_size * size_y * size_x;

    cerr << "node " << setw(4) << innovation_number;
    cerr << ", gamma: " << gamma << ", beta: " << beta;
    cerr << ", v_in_min: " << value_in_min << ", v_in_avg: " << value_in_avg << ", v_in_max: " << value_in_max;
    cerr << ", v_hat_min: " << value_hat_min << ", v_hat_avg: " << value_hat_avg << ", v_hat_max: " << value_hat_max;
    cerr << ", v_out_min: " << value_out_min << ", v_out_avg: " << value_out_avg << ", v_out_max: " << value_out_max;
    cerr << ", delta_min: " << delta_min << ", delta_avg: " << delta_avg << ", delta_max: " << delta_max << endl;
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
    os << node->visited << " ";
    os << node->weight_count << " ";
    os << node->needs_initialization << " ";

    write_hexfloat(os, node->gamma);
    write_hexfloat(os, node->best_gamma);
    write_hexfloat(os, node->previous_velocity_gamma);
    write_hexfloat(os, node->beta);
    write_hexfloat(os, node->best_beta);
    write_hexfloat(os, node->previous_velocity_beta);
    write_hexfloat(os, node->running_mean);
    write_hexfloat(os, node->best_running_mean);
    write_hexfloat(os, node->running_variance);
    write_hexfloat(os, node->best_running_variance);

    return os;
}

std::istream &operator>>(std::istream &is, CNN_Node* node) {
    is >> node->node_id;
    is >> node->exact_id;
    is >> node->genome_id;
    is >> node->innovation_number;
    is >> node->depth;
    is >> node->size_x;
    is >> node->size_y;
    is >> node->type;
    is >> node->visited;
    is >> node->weight_count;
    is >> node->needs_initialization;

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

    node->total_inputs = 0;
    node->inputs_fired = 0;

    node->visited = false;

    node->values_in = vector< vector< vector<double> > >(node->batch_size, vector< vector<double> >(node->size_y, vector<double>(node->size_x, 0.0)));
    node->values_hat = vector< vector< vector<double> > >(node->batch_size, vector< vector<double> >(node->size_y, vector<double>(node->size_x, 0.0)));
    node->values_out = vector< vector< vector<double> > >(node->batch_size, vector< vector<double> >(node->size_y, vector<double>(node->size_x, 0.0)));
    node->deltas = vector< vector< vector<double> > >(node->batch_size, vector< vector<double> >(node->size_y, vector<double>(node->size_x, 0.0)));

    return is;
}
