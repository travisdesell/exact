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
}

CNN_Node::CNN_Node(int _innovation_number, double _depth, int _size_x, int _size_y, int _type) {
    node_id = -1;
    exact_id = -1;
    genome_id = -1;

    innovation_number = _innovation_number;
    depth = _depth;
    type = _type;

    size_x = _size_x;
    size_y = _size_y;

    total_inputs = 0;
    inputs_fired = 0;

    weight_count = 0;

    visited = false;

    values = vector< vector<double> >(size_y, vector<double>(size_x, 0.0));
    errors = vector< vector<double> >(size_y, vector<double>(size_x, 0.0));
    gradients = vector< vector<double> >(size_y, vector<double>(size_x, 0.0));
    bias = vector< vector<double> >(size_y, vector<double>(size_x, 0.0));
    best_bias = vector< vector<double> >(size_y, vector<double>(size_x, 0.0));
    bias_velocity = vector< vector<double> >(size_y, vector<double>(size_x, 0.0));
    best_bias_velocity = vector< vector<double> >(size_y, vector<double>(size_x, 0.0));

    needs_initialization = true;
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

        exact_id = atoi(row[1]);
        genome_id = atoi(row[2]);
        innovation_number = atoi(row[3]);
        depth = atof(row[4]);

        size_x = atoi(row[5]);
        size_y = atoi(row[6]);
        
        istringstream bias_iss(row[9]);
        parse_vector_2d(bias, bias_iss, size_x, size_y);

        istringstream best_bias_iss(row[10]);
        parse_vector_2d(best_bias, best_bias_iss, size_x, size_y);

        istringstream bias_velocity_iss(row[11]);
        parse_vector_2d(bias_velocity, bias_velocity_iss, size_x, size_y);

        istringstream best_bias_velocity_iss(row[12]);
        parse_vector_2d(best_bias_velocity, best_bias_velocity_iss, size_x, size_y);

        type = atoi(row[13]);

        //need to reset these because it will be modified when edges are set
        total_inputs = 0;
        inputs_fired = 0;

        visited = atoi(row[14]);
        weight_count = atoi(row[15]);
        needs_initialization = atoi(row[16]);

        mysql_free_result(result);
    } else {
        cerr << "ERROR! Could not find cnn_node in database with node id: " << node_id << endl;
        exit(1);
    }

    weight_count = 0;

    //initialize arrays not stored to database
    values = vector< vector<double> >(size_y, vector<double>(size_x, 0.0));
    errors = vector< vector<double> >(size_y, vector<double>(size_x, 0.0));
    gradients = vector< vector<double> >(size_y, vector<double>(size_x, 0.0));

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
        << ", size_x = " << size_x
        << ", size_y = " << size_y;

    query << ", bias = '";
    for (int32_t y = 0; y < size_y; y++) {
        for (int32_t x = 0; x < size_x; x++) {
            if (x != 0) query << " ";
            query << setprecision(15) << bias[y][x];
        }
        if (y != size_y - 1) query << "\n";
    }

    query << "', best_bias = '";
    for (int32_t y = 0; y < size_y; y++) {
        for (int32_t x = 0; x < size_x; x++) {
            if (x != 0) query << " ";
            query << setprecision(15) << best_bias[y][x];
        }
        if (y != size_y - 1) query << "\n";
    }

    query << "', bias_velocity = '";
    for (int32_t y = 0; y < size_y; y++) {
        for (int32_t x = 0; x < size_x; x++) {
            if (x != 0) query << " ";
            query << setprecision(15) << bias_velocity[y][x];
        }
        if (y != size_y - 1) query << "\n";
    }

    query << "', best_bias_velocity = '";
    for (int32_t y = 0; y < size_y; y++) {
        for (int32_t x = 0; x < size_x; x++) {
            if (x != 0) query << " ";
            query << setprecision(15) << best_bias_velocity[y][x];
        }
        if (y != size_y - 1) query << "\n";
    }


    query << "', type = " << type
        << ", visited = " << visited
        << ", weight_count = " << weight_count
        << ", needs_initialization = " << needs_initialization;

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
    copy->size_x = size_x;
    copy->size_y = size_y;

    copy->type = type;

    copy->total_inputs = 0; //this will be updated when edges are set
    copy->inputs_fired = inputs_fired;

    copy->visited = visited;
    copy->weight_count = weight_count;
    copy->needs_initialization = needs_initialization;

    copy->values = values;
    copy->errors = errors;
    copy->gradients = gradients;
    copy->bias = bias;
    copy->best_bias = best_bias;
    copy->bias_velocity = bias_velocity;
    copy->best_bias_velocity = best_bias_velocity;

    return copy;
}

bool CNN_Node::needs_init() const {
    return needs_initialization;
}

void CNN_Node::reset_weight_count() {
    weight_count = 0;
}

void CNN_Node::add_weight_count(int _weight_count) {
    weight_count += _weight_count;
    
    //cerr << "node " << innovation_number << " setting weight count to: " << weight_count << endl;
}

int CNN_Node::get_weight_count() const {
    return weight_count;
}

void CNN_Node::initialize_bias(minstd_rand0 &generator, NormalDistribution &normal_distribution) {
    if (type == INPUT_NODE) {
        weight_count = size_x * size_y;
    } else if (weight_count == 0) {
        cerr << "ERROR! Initializing bias without having set node weight_counts yet!" << endl;
        exit(1);
    }

    double mu = 0.0;
    double sigma = sqrt(2.0 / weight_count);

    for (int32_t y = 0; y < size_y; y++) {
        for (int32_t x = 0; x < size_x; x++) {
            bias[y][x] = normal_distribution.random(generator, mu, sigma);
            best_bias[y][x] = 0.0;
            bias_velocity[y][x] = 0.0;
            best_bias_velocity[y][x] = 0.0;
            //cout << "node " << innovation_number << " bias[" << i << "][" << j <<"]: " << bias[i][j] << endl;
        }
    }
    needs_initialization = false;
}

void CNN_Node::reset_velocities() {
    for (int32_t y = 0; y < size_y; y++) {
        for (int32_t x = 0; x < size_x; x++) {
            bias_velocity[y][x] = 0.0;
        }
    }
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

double CNN_Node::get_value(int y, int x) {
    return values[y][x];
}

void CNN_Node::set_value(int y, int x, double value) {
    values[y][x] = value;
}

vector< vector<double> >& CNN_Node::get_values() {
    return values;
}

void CNN_Node::set_error(int y, int x, double value) {
    errors[y][x] = value;
}

double CNN_Node::get_error(int y, int x) {
    return errors[y][x];
}

vector< vector<double> >& CNN_Node::get_errors() {
    return errors;
}

void CNN_Node::set_gradient(int y, int x, double gradient) {
    gradients[y][x] = gradient;
}

vector< vector<double> >& CNN_Node::get_gradients() {
    return gradients;
}

void CNN_Node::print(ostream &out) {
    out << "CNN_Node " << innovation_number << ", at depth: " << depth << " of input size x: " << size_x << ", y: " << size_y << endl;

    out << "    values:" << endl;
    for (int32_t i = 0; i < size_y; i++) {
        out << "    ";
        for (int32_t j = 0; j < size_x; j++) {
            out << setw(13) << setprecision(8) << values[i][j];
        }
        out << endl;
    }

    out << "    errors:" << endl;
    for (int32_t i = 0; i < size_y; i++) {
        out << "    ";
        for (int32_t j = 0; j < size_x; j++) {
            out << setw(13) << setprecision(8) << errors[i][j];
        }
        out << endl;
    }

    out << "    gradients:" << endl;
    for (int32_t i = 0; i < size_y; i++) {
        out << "    ";
        for (int32_t j = 0; j < size_x; j++) {
            out << setw(13) << setprecision(8) << gradients[i][j];
        }
        out << endl;
    }

    out << "    bias:" << endl;
    for (int32_t i = 0; i < size_y; i++) {
        out << "    ";
        for (int32_t j = 0; j < size_x; j++) {
            out << setw(13) << setprecision(8) << bias[i][j];
        }
        out << endl;
    }

    out << "    bias_velocity:" << endl;
    for (int32_t i = 0; i < size_y; i++) {
        out << "    ";
        for (int32_t j = 0; j < size_x; j++) {
            out << setw(13) << setprecision(8) << bias_velocity[i][j];
        }
        out << endl;
    }
}

void CNN_Node::reset() {
    inputs_fired = 0;

    for (int32_t y = 0; y < size_y; y++) {
        for (int32_t x = 0; x < size_x; x++) {
            values[y][x] = 0;
            errors[y][x] = 0;
            gradients[y][x] = 0;
        }
    }
}

void CNN_Node::set_values(const Image &image, int rows, int cols, bool perform_dropout, uniform_real_distribution<double> &rng_double, minstd_rand0 &generator, double input_dropout_probability) {
    if (rows != size_y) {
        cerr << "ERROR: rows of input image: " << rows << " != size_y of input node: " << size_y << endl;
        exit(1);
    }

    if (cols != size_x) {
        cerr << "ERROR: cols of input image: " << cols << " != size_x of input node: " << size_x << endl;
        exit(1);
    }

    //cout << "setting input image: " << endl;
    int current = 0;
    if (perform_dropout) {
        for (int32_t y = 0; y < size_y; y++) {
            for (int32_t x = 0; x < size_x; x++) {
                if (rng_double(generator) < input_dropout_probability) {
                    values[y][x] = 0.0;
                } else {
                    values[y][x] = image.get_pixel(x, y);
                    current++;
                    //cout << setw(5) << values[y][x];
                }
            }
            //cout << endl;
        }
    } else {
        double dropout_scale = 1.0 - input_dropout_probability;
        for (int32_t y = 0; y < size_y; y++) {
            for (int32_t x = 0; x < size_x; x++) {
                values[y][x] = image.get_pixel(x, y) * dropout_scale;
                current++;
            }
        }
    }
}

void CNN_Node::save_best_bias() {
    for (int32_t y = 0; y < size_y; y++) {
        for (int32_t x = 0; x < size_x; x++) {
            best_bias[y][x] = bias[y][x];
            best_bias_velocity[y][x] = bias_velocity[y][x];
        }
    }
}

void CNN_Node::set_bias_to_best() {
    for (int32_t y = 0; y < size_y; y++) {
        for (int32_t x = 0; x < size_x; x++) {
            bias[y][x] = best_bias[y][x];
            //bias_velocity[y][x] = best_bias_velocity[y][x];
            bias_velocity[y][x] = 0;
        }
    }
}


void CNN_Node::resize_arrays() {
    values = vector< vector<double> >(size_y, vector<double>(size_x, 0.0));
    errors = vector< vector<double> >(size_y, vector<double>(size_x, 0.0));
    gradients = vector< vector<double> >(size_y, vector<double>(size_x, 0.0));
    bias = vector< vector<double> >(size_y, vector<double>(size_x, 0.0));
    best_bias = vector< vector<double> >(size_y, vector<double>(size_x, 0.0));
    bias_velocity = vector< vector<double> >(size_y, vector<double>(size_x, 0.0));
    best_bias_velocity = vector< vector<double> >(size_y, vector<double>(size_x, 0.0));

    needs_initialization = true;
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


bool CNN_Node::has_zero_bias() const {
    double bias_sum = 0.0;
    for (int32_t y = 0; y < size_y; y++) {
        for (int32_t x = 0; x < size_x; x++) {
            bias_sum += (bias[y][x] * bias[y][x]);
        }
    }

    return !is_softmax() && bias_sum == 0;
}

bool CNN_Node::has_zero_best_bias() const {
    double best_bias_sum = 0.0;
    for (int32_t y = 0; y < size_y; y++) {
        for (int32_t x = 0; x < size_x; x++) {
            best_bias_sum += (best_bias[y][x] * best_bias[y][x]);
        }
    }

    return !is_softmax() && best_bias_sum == 0;
}



int CNN_Node::get_number_inputs() const {
    return total_inputs;
}

int CNN_Node::get_inputs_fired() const {
    return inputs_fired;
}

void CNN_Node::input_fired(bool perform_dropout, uniform_real_distribution<double> &rng_double, minstd_rand0 &generator, double hidden_dropout_probability) {
    inputs_fired++;

    //cout << "input fired on node: " << innovation_number << ", inputs fired: " << inputs_fired << ", total_inputs: " << total_inputs << endl;

    if (inputs_fired == total_inputs) {
        if (type != SOFTMAX_NODE) {
            if (perform_dropout) {
                for (int32_t y = 0; y < size_y; y++) {
                    for (int32_t x = 0; x < size_x; x++) {
                        if (rng_double(generator) < hidden_dropout_probability) {
                            values[y][x] = 0.0;
                            gradients[y][x] = 0.0;
                        } else {
                            values[y][x] += bias[y][x];
                            //cout << "values for node " << innovation_number << " now " << values[y][x] << " after adding bias: " << bias[y][x] << endl;

                            //apply activation function
                            if (values[y][x] <= RELU_MIN) {
                                values[y][x] *= RELU_MIN_LEAK;
                                gradients[y][x] = RELU_MIN_LEAK;
                            } else if (values[y][x] > RELU_MAX) {
                                //values[y][x] = ((values[y][x] - RELU_MAX) * RELU_MAX_LEAK) + RELU_MAX;
                                //gradients[y][x] = RELU_MAX_LEAK;
                                values[y][x] = RELU_MAX;
                                gradients[y][x] = 0.0;
                            } else {
                                gradients[y][x] = 1.0;
                            }
                        }
                    }
                }

            } else {
                double dropout_scale = 1.0 - hidden_dropout_probability;

                for (int32_t y = 0; y < size_y; y++) {
                    for (int32_t x = 0; x < size_x; x++) {
                        values[y][x] += bias[y][x];
                        //cout << "values for node " << innovation_number << " now " << values[y][x] << " after adding bias: " << bias[y][x] << endl;

                        //apply activation function
                        if (values[y][x] <= RELU_MIN) {
                            values[y][x] *= RELU_MIN_LEAK;
                            gradients[y][x] = RELU_MIN_LEAK;
                        } else if (values[y][x] > RELU_MAX) {
                            //values[y][x] = ((values[y][x] - RELU_MAX) * RELU_MAX_LEAK) + RELU_MAX;
                            //gradients[y][x] = RELU_MAX_LEAK;
                            values[y][x] = RELU_MAX;
                            gradients[y][x] = 0.0;
                        } else {
                            gradients[y][x] = 1.0;
                        }

                        values[y][x] *= dropout_scale;
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

void CNN_Node::propagate_bias(double mu, double learning_rate, double weight_decay) {
    double dx, pv, velocity, b, previous_bias;

    for (int32_t y = 0; y < size_y; y++) {
        for (int32_t x = 0; x < size_x; x++) {
            //dx = gradients[y][x] * errors[y][x] * bias[y][x];
            dx = gradients[y][x] * errors[y][x];
            pv = bias_velocity[y][x];

            velocity = (mu * pv) - (learning_rate * dx);

            b = bias[y][x];
            previous_bias = b;
            b += -mu * pv + (1 + mu) * velocity;
            b -= (b * weight_decay);
            bias[y][x] = b;

            bias_velocity[y][x] = velocity;

#ifdef NAN_CHECKS
            if (isnan(bias[y][x]) || isinf(bias[y][x])) {
                cerr << "ERROR! bias became " << bias[y][x] << " in node: " << innovation_number << endl;
                cerr << "\tdx: " << dx << endl;
                cerr << "\tpv: " << pv << endl;
                cerr << "\tvelocity: " << velocity << endl;
                cerr << "\tprevious_bias: " << previous_bias << endl;
                exit(1);
            }
#endif

            if (bias[y][x] < -100.0) bias[y][x] = -100.0;
            else if (bias[y][x] > 100.0) bias[y][x] = 100.0;
        }
    }
}

bool CNN_Node::has_nan() const {
    for (int32_t y = 0; y < size_y; y++) {
        for (int32_t x = 0; x < size_x; x++) {
            if (isnan(values[y][x]) || isinf(values[y][x])) return true;
            if (isnan(errors[y][x]) || isinf(errors[y][x])) return true;
            if (isnan(gradients[y][x]) || isinf(gradients[y][x])) return true;
            if (isnan(bias[y][x]) || isinf(bias[y][x])) return true;
            if (isnan(bias_velocity[y][x]) || isinf(bias_velocity[y][x])) return true;
        }
    }

    return false;
}

void CNN_Node::print_statistics() {
    double value_min = std::numeric_limits<double>::max(), value_max = -std::numeric_limits<double>::max(), value_avg = 0.0;
    double error_min = std::numeric_limits<double>::max(), error_max = -std::numeric_limits<double>::max(), error_avg = 0.0;
    double bias_min = std::numeric_limits<double>::max(), bias_max = -std::numeric_limits<double>::max(), bias_avg = 0.0;
    double bias_velocity_min = std::numeric_limits<double>::max(), bias_velocity_max = -std::numeric_limits<double>::max(), bias_velocity_avg = 0.0;

    for (int y = 0; y < size_y; y++) {
        for (int x = 0; x < size_x; x++) {
            if (values[y][x] < value_min) value_min = values[y][x];
            if (values[y][x] > value_max) value_max = values[y][x];
            value_avg += values[y][x];

            if (errors[y][x] < error_min) error_min = errors[y][x];
            if (errors[y][x] > error_max) error_max = errors[y][x];
            error_avg += errors[y][x];

            if (bias[y][x] < bias_min) bias_min = bias[y][x];
            if (bias[y][x] > bias_max) bias_max = bias[y][x];
            bias_avg += bias[y][x];

            if (bias_velocity[y][x] < bias_velocity_min) bias_velocity_min = bias_velocity[y][x];
            if (bias_velocity[y][x] > bias_velocity_max) bias_velocity_max = bias_velocity[y][x];
            bias_velocity_avg += bias_velocity[y][x];
        }
    }

    error_avg /= size_y * size_x;
    value_avg /= size_y * size_x;

    cerr << "node " << setw(4) << innovation_number << ", v_min: " << value_min << ", v_avg: " << value_avg << ", v_max: " << value_max << ", e_min: " << error_min << ", e_avg: " << error_avg << ", e_max: " << error_max << ", b_min: " << bias_min << ", b_avg: " << bias_avg << ", b_max: " << bias_max << ", bv_min: " << bias_velocity_min << ", bv_avg: " << bias_velocity_avg << ", bv_max: " << bias_velocity_max << endl;
}

ostream &operator<<(ostream &os, const CNN_Node* node) {
    os << node->node_id << " ";
    os << node->exact_id << " ";
    os << node->genome_id << " ";
    os << node->innovation_number << " ";
    os << node->depth << " ";
    os << node->size_x << " ";
    os << node->size_y << " ";
    os << node->type << " ";
    os << node->visited << " ";
    os << node->weight_count << " ";
    os << node->needs_initialization << endl;

    os << "BIAS" << endl;
    for (int32_t y = 0; y < node->size_y; y++) {
        for (int32_t x = 0; x < node->size_x; x++) {
            if (y > 0 || x > 0) os << " ";
            write_hexfloat(os, node->bias[y][x]);
        }
    }
    os << endl;

    os << "BEST_BIAS" << endl;
    for (int32_t y = 0; y < node->size_y; y++) {
        for (int32_t x = 0; x < node->size_x; x++) {
            if (y > 0 || x > 0) os << " ";
            write_hexfloat(os, node->best_bias[y][x]);
        }
    }
    os << endl;

    os << "BIAS_VELOCITY" << endl;
    for (int32_t y = 0; y < node->size_y; y++) {
        for (int32_t x = 0; x < node->size_x; x++) {
            if (y > 0 || x > 0) os << " ";
            write_hexfloat(os, node->bias_velocity[y][x]);
        }
    }
    os << endl;

    os << "BEST_BIAS_VELOCITY" << endl;
    for (int32_t y = 0; y < node->size_y; y++) {
        for (int32_t x = 0; x < node->size_x; x++) {
            if (y > 0 || x > 0) os << " ";
            write_hexfloat(os, node->best_bias_velocity[y][x]);
        }
    }

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

    node->total_inputs = 0;
    node->inputs_fired = 0;

    node->visited = false;

    node->values = vector< vector<double> >(node->size_y, vector<double>(node->size_x, 0.0));
    node->errors = vector< vector<double> >(node->size_y, vector<double>(node->size_x, 0.0));
    node->gradients = vector< vector<double> >(node->size_y, vector<double>(node->size_x, 0.0));
    node->bias = vector< vector<double> >(node->size_y, vector<double>(node->size_x, 0.0));
    node->best_bias = vector< vector<double> >(node->size_y, vector<double>(node->size_x, 0.0));
    node->bias_velocity = vector< vector<double> >(node->size_y, vector<double>(node->size_x, 0.0));
    node->best_bias_velocity = vector< vector<double> >(node->size_y, vector<double>(node->size_x, 0.0));

    string line, prev_line;
    getline(is, prev_line);
    getline(is, line);
    if (line.compare("BIAS") != 0) {
        cerr << "ERROR: invalid input file, expected line to be 'BIAS' but line was '" << line << "'" << endl;
        cerr << "prev_line: '" << prev_line << "'" << endl;
        exit(1);
    }


    for (int32_t y = 0; y < node->size_y; y++) {
        for (int32_t x = 0; x < node->size_x; x++) {
            node->bias[y][x] = read_hexfloat(is);
            //cout << "reading node bias[" << y << "][" << x << "]: " << b << endl;
        }
    }

    getline(is, line);
    getline(is, line);
    if (line.compare("BEST_BIAS") != 0) {
        cerr << "ERROR: invalid input file, expected line to be 'BEST_BIAS' but line was '" << line << "'" << endl;
        exit(1);
    }

    for (int32_t y = 0; y < node->size_y; y++) {
        for (int32_t x = 0; x < node->size_x; x++) {
            node->best_bias[y][x] = read_hexfloat(is);
            //cout << "reading node best_bias[" << y << "][" << x << "]: " << b << endl;
        }
    }

    getline(is, line);
    getline(is, line);
    if (line.compare("BIAS_VELOCITY") != 0) {
        cerr << "ERROR: invalid input file, expected line to be 'BIAS_VELOCITY' but line was '" << line << "'" << endl;
        exit(1);
    }

    for (int32_t y = 0; y < node->size_y; y++) {
        for (int32_t x = 0; x < node->size_x; x++) {
            node->bias_velocity[y][x] = read_hexfloat(is);
            //cout << "reading node bias_velocity[" << y << "][" << x << "]: " << b << endl;
        }
    }

    getline(is, line);
    getline(is, line);
    if (line.compare("BEST_BIAS_VELOCITY") != 0) {
        cerr << "ERROR: invalid input file, expected line to be 'BEST_BIAS_VELOCITY' but line was '" << line << "'" << endl;
        exit(1);
    }

    for (int32_t y = 0; y < node->size_y; y++) {
        for (int32_t x = 0; x < node->size_x; x++) {
            node->best_bias_velocity[y][x] = read_hexfloat(is);
            //cout << "reading node best_bias_velocity[" << y << "][" << x << "]: " << b << endl;
        }
    }

    return is;
}
