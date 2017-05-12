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

    forward_visited = false;
    reverse_visited = false;
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

    forward_visited = false;
    reverse_visited = false;

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
    errors_in = vector< vector< vector<double > > >(batch_size, vector< vector<double> >(size_y, vector<double>(size_x, 0.0)));
    gradients_in = vector< vector< vector<double > > >(batch_size, vector< vector<double> >(size_y, vector<double>(size_x, 0.0)));

    values_out = vector< vector< vector<double > > >(batch_size, vector< vector<double> >(size_y, vector<double>(size_x, 0.0)));
    errors_out = vector< vector< vector<double > > >(batch_size, vector< vector<double> >(size_y, vector<double>(size_x, 0.0)));
    gradients_out = vector< vector< vector<double > > >(batch_size, vector< vector<double> >(size_y, vector<double>(size_x, 0.0)));
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

        forward_visited = atoi(row[++column]);
        reverse_visited = atoi(row[++column]);
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
    gradients_in = vector< vector< vector<double > > >(batch_size, vector< vector<double> >(size_y, vector<double>(size_x, 0.0)));
    errors_in = vector< vector< vector<double > > >(batch_size, vector< vector<double> >(size_y, vector<double>(size_x, 0.0)));

    values_out = vector< vector< vector<double > > >(batch_size, vector< vector<double> >(size_y, vector<double>(size_x, 0.0)));
    gradients_out = vector< vector< vector<double > > >(batch_size, vector< vector<double> >(size_y, vector<double>(size_x, 0.0)));
    errors_out = vector< vector< vector<double > > >(batch_size, vector< vector<double> >(size_y, vector<double>(size_x, 0.0)));

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

    copy->forward_visited = forward_visited;
    copy->reverse_visited = reverse_visited;
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
    copy->errors_in = errors_in;
    copy->gradients_in = gradients_in;

    copy->values_out = values_out;
    copy->errors_out = errors_out;
    copy->gradients_out = gradients_out;

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

    values_in = vector< vector< vector<double> > >(batch_size, vector< vector<double> >(size_y, vector<double>(size_x, 0.0)));
    errors_in = vector< vector< vector<double> > >(batch_size, vector< vector<double> >(size_y, vector<double>(size_x, 0.0)));
    gradients_in = vector< vector< vector<double> > >(batch_size, vector< vector<double> >(size_y, vector<double>(size_x, 0.0)));

    values_out = vector< vector< vector<double> > >(batch_size, vector< vector<double> >(size_y, vector<double>(size_x, 0.0)));
    errors_out = vector< vector< vector<double> > >(batch_size, vector< vector<double> >(size_y, vector<double>(size_x, 0.0)));
    gradients_out = vector< vector< vector<double> > >(batch_size, vector< vector<double> >(size_y, vector<double>(size_x, 0.0)));
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


bool CNN_Node::is_reachable() const {
    return forward_visited && reverse_visited;
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
    if (values_in.size() != batch_size) {
        cerr << "ERROR! vectors incorrect on node " << innovation_number << endl;
        cerr << "values_in.size(): " << values_in.size() << " and batch size: " << batch_size << endl;
        return false;
    }

    for (uint32_t i = 0; i < values_in.size(); i++) {
        if (values_in[i].size() != size_y) {
            cerr << "ERROR! vectors incorrect on node " << innovation_number << endl;
            cerr << "values_in[" << i << "].size(): " << values_in[i].size() << " and size_y: " << size_x << endl;
            return false;
        }

        for (uint32_t j = 0; j < values_in[i].size(); j++) {
            if (values_in[i][j].size() != size_x) {
                cerr << "ERROR! vectors incorrect on node " << innovation_number << endl;
                cerr << "values_in[" << i << "][" << j << "].size(): " << values_in[i][j].size() << " and size_x: " << size_x << endl;
                return false;
            }
        }
    }

    if (errors_in.size() != batch_size) {
        cerr << "ERROR! vectors incorrect on node " << innovation_number << endl;
        cerr << "errors_in.size(): " << errors_in.size() << " and batch size: " << batch_size << endl;
        return false;
    }

    for (uint32_t i = 0; i < errors_in.size(); i++) {
        if (errors_in[i].size() != size_y) {
            cerr << "ERROR! vectors incorrect on node " << innovation_number << endl;
            cerr << "errors_in[" << i << "].size(): " << errors_in[i].size() << " and size_y: " << size_x << endl;
            return false;
        }

        for (uint32_t j = 0; j < errors_in[i].size(); j++) {
            if (errors_in[i][j].size() != size_x) {
                cerr << "ERROR! vectors incorrect on node " << innovation_number << endl;
                cerr << "errors_in[" << i << "][" << j << "].size(): " << errors_in[i][j].size() << " and size_x: " << size_x << endl;
                return false;
            }
        }
    }

    if (gradients_in.size() != batch_size) {
        cerr << "ERROR! vectors incorrect on node " << innovation_number << endl;
        cerr << "gradients_in.size(): " << gradients_in.size() << " and batch size: " << batch_size << endl;
        return false;
    }

    for (uint32_t i = 0; i < gradients_in.size(); i++) {
        if (gradients_in[i].size() != size_y) {
            cerr << "ERROR! vectors incorrect on node " << innovation_number << endl;
            cerr << "gradients_in[" << i << "].size(): " << gradients_in[i].size() << " and size_y: " << size_x << endl;
            return false;
        }

        for (uint32_t j = 0; j < gradients_in[i].size(); j++) {
            if (gradients_in[i][j].size() != size_x) {
                cerr << "ERROR! vectors incorrect on node " << innovation_number << endl;
                cerr << "gradients_in[" << i << "][" << j << "].size(): " << gradients_in[i][j].size() << " and size_x: " << size_x << endl;
                return false;
            }
        }
    }

    if (values_out.size() != batch_size) {
        cerr << "ERROR! vectors incorrect on node " << innovation_number << endl;
        cerr << "values_out.size(): " << values_out.size() << " and batch size: " << batch_size << endl;
        return false;
    }

    for (uint32_t i = 0; i < values_out.size(); i++) {
        if (values_out[i].size() != size_y) {
            cerr << "ERROR! vectors incorrect on node " << innovation_number << endl;
            cerr << "values_out[" << i << "].size(): " << values_out[i].size() << " and size_y: " << size_x << endl;
            return false;
        }

        for (uint32_t j = 0; j < values_out[i].size(); j++) {
            if (values_out[i][j].size() != size_x) {
                cerr << "ERROR! vectors incorrect on node " << innovation_number << endl;
                cerr << "values_out[" << i << "][" << j << "].size(): " << values_out[i][j].size() << " and size_x: " << size_x << endl;
                return false;
            }
        }
    }

    if (errors_out.size() != batch_size) {
        cerr << "ERROR! vectors incorrect on node " << innovation_number << endl;
        cerr << "errors_out.size(): " << errors_out.size() << " and batch size: " << batch_size << endl;
        return false;
    }

    for (uint32_t i = 0; i < errors_out.size(); i++) {
        if (errors_out[i].size() != size_y) {
            cerr << "ERROR! vectors incorrect on node " << innovation_number << endl;
            cerr << "errors_out[" << i << "].size(): " << errors_out[i].size() << " and size_y: " << size_x << endl;
            return false;
        }

        for (uint32_t j = 0; j < errors_out[i].size(); j++) {
            if (errors_out[i][j].size() != size_x) {
                cerr << "ERROR! vectors incorrect on node " << innovation_number << endl;
                cerr << "errors_out[" << i << "][" << j << "].size(): " << errors_out[i][j].size() << " and size_x: " << size_x << endl;
                return false;
            }
        }
    }

    if (gradients_out.size() != batch_size) {
        cerr << "ERROR! vectors incorrect on node " << innovation_number << endl;
        cerr << "gradients_out.size(): " << gradients_out.size() << " and batch size: " << batch_size << endl;
        return false;
    }

    for (uint32_t i = 0; i < gradients_out.size(); i++) {
        if (gradients_out[i].size() != size_y) {
            cerr << "ERROR! vectors incorrect on node " << innovation_number << endl;
            cerr << "gradients_out[" << i << "].size(): " << gradients_out[i].size() << " and size_y: " << size_x << endl;
            return false;
        }

        for (uint32_t j = 0; j < gradients_out[i].size(); j++) {
            if (gradients_out[i][j].size() != size_x) {
                cerr << "ERROR! vectors incorrect on node " << innovation_number << endl;
                cerr << "gradients_out[" << i << "][" << j << "].size(): " << gradients_out[i][j].size() << " and size_x: " << size_x << endl;
                return false;
            }
        }
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

double CNN_Node::get_depth() const {
    return depth;
}

void CNN_Node::resize_arrays() {
    values_in = vector< vector< vector<double> > >(batch_size, vector< vector<double> >(size_y, vector<double>(size_x, 0.0)));
    errors_in = vector< vector< vector<double> > >(batch_size, vector< vector<double> >(size_y, vector<double>(size_x, 0.0)));
    gradients_in = vector< vector< vector<double> > >(batch_size, vector< vector<double> >(size_y, vector<double>(size_x, 0.0)));

    values_out = vector< vector< vector<double> > >(batch_size, vector< vector<double> >(size_y, vector<double>(size_x, 0.0)));
    errors_out = vector< vector< vector<double> > >(batch_size, vector< vector<double> >(size_y, vector<double>(size_x, 0.0)));
    gradients_out = vector< vector< vector<double> > >(batch_size, vector< vector<double> >(size_y, vector<double>(size_x, 0.0)));

    needs_initialization = true;
}


double CNN_Node::get_value_in(int batch_number, int y, int x) {
    return values_in[batch_number][y][x];
}

void CNN_Node::set_value_in(int batch_number, int y, int x, double value) {
    values_in[batch_number][y][x] = value;
}

vector< vector< vector<double> >>& CNN_Node::get_values_in() {
    return values_in;
}

double CNN_Node::get_value_out(int batch_number, int y, int x) {
    return values_out[batch_number][y][x];
}

void CNN_Node::set_value_out(int batch_number, int y, int x, double value) {
    values_out[batch_number][y][x] = value;
}

vector< vector< vector<double> >>& CNN_Node::get_values_out() {
    return values_out;
}



void CNN_Node::set_error_in(int batch_number, int y, int x, double error) {
    errors_in[batch_number][y][x] = error;
}

vector<vector< vector<double> > >& CNN_Node::get_errors_in() {
    return errors_in;
}

void CNN_Node::set_error_out(int batch_number, int y, int x, double error) {
    errors_out[batch_number][y][x] = error;
}

vector<vector< vector<double> > >& CNN_Node::get_errors_out() {
    return errors_out;
}



void CNN_Node::set_gradient_in(int batch_number, int y, int x, double gradient) {
    gradients_in[batch_number][y][x] = gradient;
}

vector<vector< vector<double> > >& CNN_Node::get_gradients_in() {
    return gradients_in;
}

void CNN_Node::set_gradient_out(int batch_number, int y, int x, double gradient) {
    gradients_out[batch_number][y][x] = gradient;
}

vector<vector< vector<double> > >& CNN_Node::get_gradients_out() {
    return gradients_out;
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

        out << "    errors_in:" << endl;
        for (int32_t i = 0; i < size_y; i++) {
            out << "    ";
            for (int32_t j = 0; j < size_x; j++) {
                out << setw(13) << setprecision(8) << errors_in[batch_number][i][j];
            }
            out << endl;
        }

        out << "    gradients_in:" << endl;
        for (int32_t i = 0; i < size_y; i++) {
            out << "    ";
            for (int32_t j = 0; j < size_x; j++) {
                out << setw(13) << setprecision(8) << gradients_in[batch_number][i][j];
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

        out << "    errors_out:" << endl;
        for (int32_t i = 0; i < size_y; i++) {
            out << "    ";
            for (int32_t j = 0; j < size_x; j++) {
                out << setw(13) << setprecision(8) << errors_out[batch_number][i][j];
            }
            out << endl;
        }

        out << "    gradients_out:" << endl;
        for (int32_t i = 0; i < size_y; i++) {
            out << "    ";
            for (int32_t j = 0; j < size_x; j++) {
                out << setw(13) << setprecision(8) << gradients_out[batch_number][i][j];
            }
            out << endl;
        }
    }
}

void CNN_Node::reset_times() {
    input_fired_time = 0.0;
    output_fired_time = 0.0;
}

void CNN_Node::accumulate_times(double &total_input_time, double &total_output_time) {
    total_input_time += input_fired_time;
    total_output_time += output_fired_time;
}

void CNN_Node::reset() {
    inputs_fired = 0;
    outputs_fired = 0;

    if (is_reachable()) {
        for (int32_t batch_number = 0; batch_number < batch_size; batch_number++) {
            for (int32_t y = 0; y < size_y; y++) {
                for (int32_t x = 0; x < size_x; x++) {
                    values_in[batch_number][y][x] = 0;
                    errors_in[batch_number][y][x] = 0;
                    gradients_in[batch_number][y][x] = 0;

                    values_out[batch_number][y][x] = 0;
                    errors_out[batch_number][y][x] = 0;
                    gradients_out[batch_number][y][x] = 0;
                }
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

void CNN_Node::update_batch_size(int new_batch_size) {
    batch_size = new_batch_size;
    resize_arrays();
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


void CNN_Node::batch_normalize(bool training, bool accumulating_test_statistics, double epsilon, double alpha) {
    //normalize the batch
    if (training || accumulating_test_statistics) {
        batch_mean = 0.0;

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

        batch_variance = 0.0;
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

        batch_std_dev = sqrt(batch_variance + epsilon);

        inverse_variance = 1.0 / batch_std_dev;

        //cout << endl;
        //cout << "post-batch normalization on node: " << innovation_number << endl;
        double temp;
        for (int32_t batch_number = 0; batch_number < batch_size; batch_number++) {
            for (int32_t y = 0; y < size_y; y++) {
                for (int32_t x = 0; x < size_x; x++) {
                    temp = (values_in[batch_number][y][x] - batch_mean) * inverse_variance;
                    values_in[batch_number][y][x] = temp;   //values in becomes x_hat
                    values_out[batch_number][y][x] = (gamma * temp) + beta;

                    //cout << setw(10) << std::fixed << values_out[batch_number][y][x];
                }
                //cout << endl;
            }
            //cout << endl;
        }

#ifdef NAN_CHECKS
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

void CNN_Node::apply_relu(vector< vector< vector<double> > > &values, vector< vector< vector<double> > > &gradients) {
    for (int32_t batch_number = 0; batch_number < batch_size; batch_number++) {
        for (int32_t y = 0; y < size_y; y++) {
            for (int32_t x = 0; x < size_x; x++) {
                //cout << "values_out for node " << innovation_number << " now " << values_out[batch_number][y][x] << " after adding bias: " << bias[batch_number][y][x] << endl;

                //apply activation function
                if (values[batch_number][y][x] <= RELU_MIN) {
                    values[batch_number][y][x] = values[batch_number][y][x] * RELU_MIN_LEAK;
                    gradients[batch_number][y][x] = RELU_MIN_LEAK; 
                } else if (values[batch_number][y][x] > RELU_MAX) {
                    values[batch_number][y][x] = RELU_MAX;
                    gradients[batch_number][y][x] = RELU_MAX_LEAK; 
                } else {
                    gradients[batch_number][y][x] = 1.0;
                }
            }
        }
    }
}

void CNN_Node::apply_dropout(vector< vector< vector<double> > > &values, vector< vector< vector<double> > > &gradients, bool perform_dropout, bool accumulate_test_statistics, double dropout_probability, minstd_rand0 &generator) {
    if (perform_dropout && !accumulate_test_statistics) {
        for (int32_t batch_number = 0; batch_number < batch_size; batch_number++) {
            for (int32_t y = 0; y < size_y; y++) {
                for (int32_t x = 0; x < size_x; x++) {
                    //cout << "values for node " << innovation_number << " now " << values[batch_number][y][x] << " after adding bias: " << bias[batch_number][y][x] << endl;

                    if (random_0_1(generator) < dropout_probability) {
                        values[batch_number][y][x] = 0.0;
                        gradients[batch_number][y][x] = 0.0;
                    }
                }
            }
        }

    } else {
        double dropout_scale = 1.0 - dropout_probability;
        for (int32_t batch_number = 0; batch_number < batch_size; batch_number++) {
            for (int32_t y = 0; y < size_y; y++) {
                for (int32_t x = 0; x < size_x; x++) {
                    values[batch_number][y][x] *= dropout_scale;
                }
            }
        }
    }
}

/*
void CNN_Node::backpropagate_dropout() {
    for (int32_t y = 0; y < size_y; y++) {
        for (int32_t x = 0; x < size_x; x++) {
            if (dropped_out[y][x]) {
                for (int32_t batch_number = 0; batch_number < batch_size; batch_number++) {
                    deltas[batch_number][y][x] = 0.0;
                } 
            }
        }
    }
}

void CNN_Node::backpropagate_relu() {
    for (int32_t batch_number = 0; batch_number < batch_size; batch_number++) {
        for (int32_t y = 0; y < size_y; y++) {
            for (int32_t x = 0; x < size_x; x++) {
                gradients_in[batch_number][y][x] = gradients_out[batch_number][y][x];
                errors_in[batch_number][y][x] = errors_out[batch_number][y][x];
            }
        }
    }
}
*/


void CNN_Node::backpropagate_batch_normalization(double mu, double learning_rate, double epsilon) {
    //backprop  batch normalization here
    double delta_beta = 0.0;
    double delta_gamma = 0.0;

    double value_in;
    double value_hat;
    //double value_out;
    double delta_out;
    //double delta_values_hat;
    //double delta_values_hat_sum = 0.0;
    //double delta_values_hat_x_values_sum = 0.0;

    double derr_dvariance = 0.0;
    double derr_dmean = 0.0;

    double derr_dmean_term1 = 0.0;
    double derr_dmean_term2 = 0.0;

    double m = (uint64_t)batch_size * (uint64_t)size_y * (uint64_t)size_x;
    double diff;

    for (int32_t batch_number = 0; batch_number < batch_size; batch_number++) {
        for (int32_t y = 0; y < size_y; y++) {
            for (int32_t x = 0; x < size_x; x++) {
                delta_out = errors_out[batch_number][y][x] * gradients_out[batch_number][y][x];
                value_hat = values_in[batch_number][y][x];

                value_in = (value_hat + batch_mean) * batch_std_dev;
                diff = value_in - batch_mean;

                delta_beta += delta_out;
                delta_gamma += value_hat * delta_out;

                derr_dvariance += diff * delta_out * gamma;

                derr_dmean_term1 += delta_out * gamma;
                derr_dmean_term2 += diff;
            }
        }
    }

    double inv_m = 1.0 / m;
    double inv_m_x_2 = 2.0 * inv_m;

    derr_dvariance *= -0.5 * pow(batch_variance + epsilon, -1.5);
    derr_dmean_term1 *= -inverse_variance;
    derr_dmean_term2 *= -inv_m_x_2 * derr_dvariance;
    derr_dmean = derr_dmean_term1 + derr_dmean_term2;

    for (int32_t batch_number = 0; batch_number < batch_size; batch_number++) {
        for (int32_t y = 0; y < size_y; y++) {
            for (int32_t x = 0; x < size_x; x++) {
                delta_out = errors_out[batch_number][y][x] * gradients_out[batch_number][y][x];
                value_hat = values_in[batch_number][y][x];
                value_in = (value_hat + batch_mean) * batch_std_dev;

                //gradients_in[batch_number][y][x] = inverse_variance;
                errors_in[batch_number][y][x] = (delta_out * inverse_variance) + (derr_dvariance * inv_m_x_2 * (value_in - batch_mean)) + (derr_dmean * inv_m);

                //errors_in[batch_number][y][x] = delta_out * gamma;
                //errors_in[batch_number][y][x] = errors_out[batch_number][y][x];
                //errors_in[batch_number][y][x] = 1.0;

#ifdef NAN_CHECKS
                if (isnan(errors_in[batch_number][y][x]) || isinf(errors_in[batch_number][y][x])) {
                    cerr << "ERROR! errors_in[" << batch_number << "][" << y << "][" << x << "] became: " << errors_in[batch_number][y][x] << "!" << endl;
                    cerr << "inverse_variance: " << inverse_variance << endl;
                    cerr << "batch_size: " << batch_size << endl;
                    cerr << "gamma: " << gamma << endl;
                    cerr << "delta_out: " << delta_out << endl;
                    cerr << "delta_values_hat_sum: " << delta_values_hat_sum << endl;
                    cerr << "values_in[" << batch_number << "][" << y << "][" << x << "]: " << values_in[batch_number][y][x] << endl;
                    cerr << "delta_values_hat_x_values_sum: " << delta_values_hat_x_values_sum << endl;

                    exit(1);
                }
#endif
            }
        }
    }

    //backpropagate beta
    double pv_beta = previous_velocity_beta;

    double velocity_beta = (mu * pv_beta) - learning_rate * delta_beta;
    beta += velocity_beta + mu * (velocity_beta - pv_beta);
    //beta += (-mu * pv_beta + (1 + mu) * velocity_beta);
    //beta -= (beta * weight_decay);

    previous_velocity_beta = velocity_beta;

    //backpropagate gamma
    double pv_gamma = previous_velocity_gamma;

    double velocity_gamma = (mu * pv_gamma) - learning_rate * delta_gamma;
    gamma += velocity_gamma + mu * (velocity_gamma - pv_gamma);
    //gamma += (-mu * pv_gamma + (1 + mu) * velocity_gamma);
    //gamma -= (gamma * weight_decay);

    previous_velocity_gamma = velocity_gamma;

    //cout << "\tnode " << innovation_number << ", delta_gamma: " << delta_gamma << ", delta_beta: " << delta_beta << ", gamma now: " << gamma << ", beta now: " << beta << endl;
}


void CNN_Node::set_values(const vector<Image> &images, int channel, bool perform_dropout, bool accumulate_test_statistics, double input_dropout_probability, minstd_rand0 &generator) {
    //images.size() may be less than batch size, in the case when the total number of images is not divisible by the batch_size
    if (images.size() > batch_size) {
        cerr << "ERROR: number of batch images: " << images.size() << " > batch_size of input node: " << batch_size << endl;
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

    //images.size() may be less than batch size, in the case when the total number of images is not divisible by the batch_size
    for (int32_t batch_number = 0; batch_number < images.size(); batch_number++) {
        for (int32_t y = 0; y < size_y; y++) {
            for (int32_t x = 0; x < size_x; x++) {
                values_out[batch_number][y][x] = images[batch_number].get_pixel(channel, y, x);
            }
        }
    }

    if (input_dropout_probability > 0) apply_dropout(values_out, gradients_out, perform_dropout, accumulate_test_statistics, input_dropout_probability, generator);
}


void CNN_Node::input_fired(bool training, bool accumulate_test_statistics, double epsilon, double alpha, bool perform_dropout, double hidden_dropout_probability, minstd_rand0 &generator) {
    double input_fired_start_time = time(NULL);

    inputs_fired++;

    //cout << "input fired on node: " << innovation_number << ", inputs fired: " << inputs_fired << ", total_inputs: " << total_inputs << endl;

    if (inputs_fired == total_inputs) {
        if (type != SOFTMAX_NODE) {
            batch_normalize(training, accumulate_test_statistics, epsilon, alpha);
            apply_relu(values_out, gradients_out);
            if (hidden_dropout_probability > 0) apply_dropout(values_out, gradients_out, perform_dropout, accumulate_test_statistics, hidden_dropout_probability, generator);
        }

    } else if (inputs_fired > total_inputs) {
        cerr << "ERROR! inputs_fired > total_inputs" << endl;

        cerr << "inputs_fired: " << inputs_fired << endl;
        cerr << "total_inputs: " << total_inputs << endl;

        cerr << "node: " << endl;
        print(cerr);

        exit(1);
    }

    input_fired_time += time(NULL) - input_fired_start_time;
}

void CNN_Node::output_fired(double mu, double learning_rate, double epsilon) {
    double output_fired_start_time = time(NULL);

    outputs_fired++;

    //cout << "output fired on node: " << innovation_number << ", outputs fired: " << outputs_fired << ", total_outputs: " << total_outputs << endl;

    if (outputs_fired == total_outputs) {
        if (type != SOFTMAX_NODE && type != INPUT_NODE) {
            //backpropagate_dropout();
            //backpropagate_relu();
            backpropagate_batch_normalization(mu, learning_rate, epsilon);
        }

    } else if (outputs_fired > total_outputs) {
        cerr << "ERROR! outputs_fired > total_outputs" << endl;

        cerr << "outputs_fired: " << outputs_fired << endl;
        cerr << "total_outputs: " << total_outputs << endl;

        cerr << "node: " << endl;
        print(cerr);

        exit(1);
    }

    output_fired_time += time(NULL) - output_fired_start_time;
}


bool CNN_Node::has_nan() const {
    for (int32_t batch_number = 0; batch_number < batch_size; batch_number++) {
        for (int32_t y = 0; y < size_y; y++) {
            for (int32_t x = 0; x < size_x; x++) {
                if (isnan(values_in[batch_number][y][x]) || isinf(values_in[batch_number][y][x])) return true;
                if (isnan(errors_in[batch_number][y][x]) || isinf(errors_in[batch_number][y][x])) return true;
                if (isnan(gradients_in[batch_number][y][x]) || isinf(gradients_in[batch_number][y][x])) return true;

                if (isnan(values_out[batch_number][y][x]) || isinf(values_out[batch_number][y][x])) return true;
                if (isnan(errors_out[batch_number][y][x]) || isinf(errors_out[batch_number][y][x])) return true;
                if (isnan(gradients_out[batch_number][y][x]) || isinf(gradients_out[batch_number][y][x])) return true;
            }
        }
    }

    return false;
}

void CNN_Node::print_statistics() {
    cerr << "node " << setw(4) << innovation_number;
    cerr << ", gamma: " << gamma << ", beta: " << beta;
    cerr << "\tINPUTS: ";
    print_statistics(values_in, errors_in, gradients_in);
    cerr << "\tOUTPUTS: ";
    print_statistics(values_out, errors_out, gradients_out);
}

void CNN_Node::print_statistics(const vector< vector< vector<double> > > &values, const vector< vector< vector<double> > > &errors, const vector< vector< vector<double> > > &gradients) {
    double value_min = std::numeric_limits<double>::max(), value_max = -std::numeric_limits<double>::max(), value_avg = 0.0;
    double error_min = std::numeric_limits<double>::max(), error_max = -std::numeric_limits<double>::max(), error_avg = 0.0;
    double gradient_min = std::numeric_limits<double>::max(), gradient_max = -std::numeric_limits<double>::max(), gradient_avg = 0.0;

    for (int batch_number = 0; batch_number < batch_size; batch_number++) {
        for (int y = 0; y < size_y; y++) {
            for (int x = 0; x < size_x; x++) {
                if (values[batch_number][y][x] < value_min) value_min = values[batch_number][y][x];
                if (values[batch_number][y][x] > value_max) value_max = values[batch_number][y][x];
                value_avg += values[batch_number][y][x];

                if (gradients[batch_number][y][x] < gradient_min) gradient_min = gradients[batch_number][y][x];
                if (gradients[batch_number][y][x] > gradient_max) gradient_max = gradients[batch_number][y][x];
                gradient_avg += gradients[batch_number][y][x];

                if (errors[batch_number][y][x] < error_min) error_min = errors[batch_number][y][x];
                if (errors[batch_number][y][x] > error_max) error_max = errors[batch_number][y][x];
                error_avg += errors[batch_number][y][x];
            }
        }
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

    node->total_outputs = 0;
    node->outputs_fired = 0;

    node->forward_visited = false;
    node->reverse_visited = false;

    node->values_in = vector< vector< vector<double> > >(node->batch_size, vector< vector<double> >(node->size_y, vector<double>(node->size_x, 0.0)));
    node->errors_in = vector< vector< vector<double> > >(node->batch_size, vector< vector<double> >(node->size_y, vector<double>(node->size_x, 0.0)));
    node->gradients_in = vector< vector< vector<double> > >(node->batch_size, vector< vector<double> >(node->size_y, vector<double>(node->size_x, 0.0)));

    node->values_out = vector< vector< vector<double> > >(node->batch_size, vector< vector<double> >(node->size_y, vector<double>(node->size_x, 0.0)));
    node->errors_out = vector< vector< vector<double> > >(node->batch_size, vector< vector<double> >(node->size_y, vector<double>(node->size_x, 0.0)));
    node->gradients_out = vector< vector< vector<double> > >(node->batch_size, vector< vector<double> >(node->size_y, vector<double>(node->size_x, 0.0)));

    return is;
}
