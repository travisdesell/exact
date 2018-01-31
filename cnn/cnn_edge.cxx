#include <cmath>
using std::isnan;
using std::isinf;

#include <chrono>

#include <fstream>
using std::ofstream;
using std::ifstream;
using std::ios;

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

#include <random>
using std::minstd_rand0;
using std::normal_distribution;

#include <thread>

#include <sstream>
using std::ostringstream;

#include <stdexcept>
using std::runtime_error;

#include <string>
using std::string;

#include <vector>
using std::vector;

#include "common/random.hxx"
#include "image_tools/image_set.hxx"
#include "comparison.hxx"
#include "cnn_edge.hxx"
#include "cnn_node.hxx"
#include "pooling.hxx"
#include "propagation.hxx"

#include "stdint.h"


int random_edge_type(float random_value) {
    if (random_value < 0.5) {
        return CONVOLUTIONAL;
    } else {
        return POOLING;
    }
}

CNN_Edge::CNN_Edge() {
    edge_id = -1;
    exact_id = -1;
    genome_id = -1;

    type = -1;

    innovation_number = -1;

    input_node_innovation_number = -1;
    output_node_innovation_number = -1;

    input_node = NULL;
    output_node = NULL;

    needs_initialization = true;

    weights = NULL;
    weight_updates = NULL;
    best_weights = NULL;

    previous_velocity = NULL;
    best_velocity = NULL;

    scale = 1.0;
    best_scale = 1.0;
    previous_velocity_scale = 0.0;
    best_velocity_scale = 0.0;
}

CNN_Edge::CNN_Edge(CNN_Node *_input_node, CNN_Node *_output_node, bool _fixed, int _innovation_number, int _type) {
    edge_id = -1;
    exact_id = -1;
    genome_id = -1;

    type = _type;
    fixed = _fixed;
    innovation_number = _innovation_number;
    disabled = false;
    forward_visited = false;
    reverse_visited = false;

    reverse_filter_x = false;
    reverse_filter_y = false;
    needs_initialization = true;

    scale = 1.0;
    best_scale = 1.0;
    previous_velocity_scale = 0.0;
    best_velocity_scale = 0.0;

    input_node = _input_node;
    output_node = _output_node;

    input_node_innovation_number = input_node->get_innovation_number();
    output_node_innovation_number = output_node->get_innovation_number();

    batch_size = _input_node->get_batch_size();

    if (output_node->get_size_x() <= input_node->get_size_x()) {
        filter_x = (input_node->get_size_x() - output_node->get_size_x()) + 1;
    } else {
        reverse_filter_x = true;
        filter_x = (output_node->get_size_x() - input_node->get_size_x()) + 1;
    }

    if (output_node->get_size_y() <= input_node->get_size_y()) {
        filter_y = (input_node->get_size_y() - output_node->get_size_y()) + 1;
    } else {
        reverse_filter_y = true;
        filter_y = (output_node->get_size_y() - input_node->get_size_y()) + 1;
    }

    filter_size = filter_y * filter_x;

    //cout << "edge " << innovation_number << " set filter_size to: " << filter_size << endl;

    //cout << "\t\tcreated edge " << innovation_number << " (node " << input_node_innovation_number << " to " << output_node_innovation_number << ") with filter_x: " << filter_x << " (input: " << input_node->get_size_x() << ", output: " << output_node->get_size_x() << ") and filter_y: " << filter_y << " (input: " << input_node->get_size_y() << ", output: " << output_node->get_size_y() << "), reverse filter: " << reverse_filter_x << ", reverse_filter_y: " << reverse_filter_y << endl;

    weights = new float[filter_size]();
    weight_updates = new float[filter_size]();
    best_weights = new float[filter_size]();

    previous_velocity = new float[filter_size]();
    best_velocity = new float[filter_size]();

    initialize_pools(y_pools, y_pool_offset, input_node->get_size_y(), output_node->get_size_y());
    initialize_pools(x_pools, x_pool_offset, input_node->get_size_x(), output_node->get_size_x());
}

CNN_Edge::~CNN_Edge() {
    delete [] weights;
    delete [] weight_updates;
    delete [] best_weights;

    delete [] previous_velocity;
    delete [] best_velocity;

    input_node = NULL;
    output_node = NULL;
}

void parse_float_2d(float **output, istringstream &iss, int filter_x, int filter_y) {
    //if (*output != NULL) delete [] (*output);
    (*output) = new float[filter_y * filter_x];

    int current = 0;

    float val;
    while(iss >> val || !iss.eof()) {
        if (iss.fail()) {
            iss.clear();
            string dummy;
            iss >> dummy;
            continue;
        }

        //cout << "output[" << current_x << "][" << current_y << "]: " << val << endl;
        (*output)[current] = val;

        current++;
    }
}


void parse_vector_2d(vector< vector<float> > &output, istringstream &iss, int filter_x, int filter_y) {
    output.clear();
    output = vector< vector<float> >(filter_y, vector<float>(filter_x));

    int current_x = 0, current_y = 0;

    float val;
    while(iss >> val || !iss.eof()) {
        if (iss.fail()) {
            iss.clear();
            string dummy;
            iss >> dummy;
            continue;
        }

        //cout << "output[" << current_x << "][" << current_y << "]: " << val << endl;
        output.at(current_y).at(current_x) = val;

        current_x++;

        if (current_x >= filter_x) {
            current_x = 0;
            current_y++;
        }
    }
}



#ifdef _MYSQL_
CNN_Edge::CNN_Edge(int _edge_id) {
    edge_id = _edge_id;

    ostringstream query;
    query << "SELECT * FROM cnn_edge WHERE id = " << edge_id;

    mysql_exact_query(query.str());

    MYSQL_RES *result = mysql_store_result(exact_db_conn);

    if (result != NULL) {
        MYSQL_ROW row = mysql_fetch_row(result);

        int column = 0;

        exact_id = atoi(row[++column]);
        genome_id = atoi(row[++column]);

        type = atoi(row[++column]);
        innovation_number = atoi(row[++column]);

        input_node_innovation_number = atoi(row[++column]);
        output_node_innovation_number = atoi(row[++column]);

        batch_size = atoi(row[++column]);
        filter_x = atoi(row[++column]);
        filter_y = atoi(row[++column]);
        filter_size = filter_y * filter_x;

        //cout << "reading weights for exact edge" << endl;
        istringstream weights_iss(row[++column]);
        //parse_float_2d(&weights, weights_iss, filter_x, filter_y);
        weights = new float[filter_y * filter_x];
        for (int32_t i = 0; i < filter_y * filter_x; i++) {
            weights[i] = read_hexfloat(weights_iss);
        }

        //cout << "reading best weights for exact edge" << endl;

        istringstream best_weights_iss(row[++column]);
        //parse_float_2d(&best_weights, best_weights_iss, filter_x, filter_y);
        best_weights = new float[filter_y * filter_x];
        for (int32_t i = 0; i < filter_y * filter_x; i++) {
            best_weights[i] = read_hexfloat(best_weights_iss);
        }
        //cout << "success!" << endl;

        fixed = atoi(row[++column]);
        disabled = atoi(row[++column]);
        forward_visited = atoi(row[++column]);
        reverse_visited = atoi(row[++column]);
        reverse_filter_x = atoi(row[++column]);
        reverse_filter_y = atoi(row[++column]);
        needs_initialization = atoi(row[++column]);

        istringstream scale_iss(row[++column]);
        scale = read_hexfloat(scale_iss);
        best_scale = read_hexfloat(scale_iss);
        previous_velocity_scale = read_hexfloat(scale_iss);
        best_velocity_scale = read_hexfloat(scale_iss);


        mysql_free_result(result);
    } else {
        ostringstream error_message;
        error_message << "ERROR! Could not find cnn_edge in database with edge id: " << edge_id;
        throw runtime_error(error_message.str());
    }

    weight_updates = new float[filter_size]();

    previous_velocity = new float[filter_size]();
    best_velocity = new float[filter_size]();

    //pools will be initialized after nodes are set

    //cout << "read edge!" << endl;
    //cout << this << endl;
}

void CNN_Edge::export_to_database(int _exact_id, int _genome_id) {
    ostringstream query;

    genome_id = _genome_id;
    exact_id = _exact_id;

    //cout << "inserting edge with exact_id: " << exact_id << " and genome id: " << genome_id << endl;

    if (edge_id >= 0) {
        query << "REPLACE INTO cnn_edge SET id = " << edge_id << ",";
    } else {
        query << "INSERT INTO cnn_edge SET";
    }

    query << " exact_id = " << exact_id
        << ", genome_id = " << genome_id
        << ", type = " << type
        << ", innovation_number = " << innovation_number
        << ", input_node_innovation_number = " << input_node_innovation_number
        << ", output_node_innovation_number = " << output_node_innovation_number
        << ", batch_size = " << batch_size
        << ", filter_x = " << filter_x
        << ", filter_y = " << filter_y
        << ", fixed = " << fixed
        << ", disabled = " << disabled
        << ", forward_visited = " << forward_visited
        << ", reverse_visited = " << reverse_visited
        << ", reverse_filter_x = " << reverse_filter_x
        << ", reverse_filter_y = " << reverse_filter_y
        << ", needs_initialization = " << needs_initialization
        << ", weights = '";

    int current = 0;
    for (int32_t y = 0; y < filter_y; y++) {
        for (int32_t x = 0; x < filter_x; x++) {
            if (x != 0) query << " ";
            write_hexfloat(query, weights[current]);
            //query << setprecision(15) << weights[current];
            current++;
        }
        if (y != filter_y - 1) query << "\n";
    }

    query << "', best_weights = '";
    current = 0;
    for (int32_t y = 0; y < filter_y; y++) {
        for (int32_t x = 0; x < filter_x; x++) {
            if (x != 0) query << " ";
            write_hexfloat(query, best_weights[current]);
            //query << setprecision(15) << best_weights[current];
            current++;
        }
        if (y != filter_y - 1) query << "\n";
    }
    query << "', scale_values = '";
    write_hexfloat(query, scale);
    query << " ";
    write_hexfloat(query, best_scale);
    query << " ";
    write_hexfloat(query, previous_velocity_scale);
    query << " ";
    write_hexfloat(query, best_velocity_scale);
    query <<"'";

    /*
    query << "', previous_velocity = '";
    for (int32_t y = 0; y < filter_y; y++) {
        for (int32_t x = 0; x < filter_x; x++) {
            if (x != 0) query << " ";
            query << setprecision(15) << previous_velocity[y][x];
        }
        if (y != filter_y - 1) query << "\n";
    }

    query << "', best_velocity = '";
    for (int32_t y = 0; y < filter_y; y++) {
        for (int32_t x = 0; x < filter_x; x++) {
            if (x != 0) query << " ";
            query << setprecision(15) << best_velocity[y][x];
        }
        if (y != filter_y - 1) query << "\n";
    }
    query << "'";
    */
    //query << "', previous_velocity = '', best_velocity = ''";

    mysql_exact_query(query.str());

    if (edge_id < 0) {
        edge_id = mysql_exact_last_insert_id();
        //cout << "set edge id to " << edge_id << endl;
    }
}

int CNN_Edge::get_edge_id() const {
    return edge_id;
}
#endif

bool CNN_Edge::equals(CNN_Edge *other) const {
    return filter_x == other->filter_x && filter_y == other->filter_y && disabled == other->disabled && reverse_filter_x == other->reverse_filter_x && reverse_filter_y == other->reverse_filter_y && type == other->type;
}

bool CNN_Edge::needs_init() const {
    return needs_initialization;
}

void CNN_Edge::set_needs_init() {
    needs_initialization = true;
}


void CNN_Edge::reset_times() {
    propagate_forward_time = 0.0;
    propagate_backward_time = 0.0;
    weight_update_time = 0.0;
}

void CNN_Edge::accumulate_times(float &total_forward_time, float &total_backward_time, float &total_weight_update_time) {
    total_forward_time += propagate_forward_time;
    total_backward_time += propagate_backward_time;
    total_weight_update_time += weight_update_time;
}

int CNN_Edge::get_type() const {
    return type;
}

int CNN_Edge::get_filter_size() const {
    return filter_x * filter_y;
}

int CNN_Edge::get_filter_x() const {
    return filter_x;
}

int CNN_Edge::get_filter_y() const {
    return filter_y;
}

bool CNN_Edge::is_reverse_filter_x() const {
    return reverse_filter_x;
}

bool CNN_Edge::is_reverse_filter_y() const {
    return reverse_filter_y;
}

float CNN_Edge::get_weight(int i) const {
    return weights[i];
}

float CNN_Edge::get_weight_update(int i) const {
    return weight_updates[i];
}

void CNN_Edge::update_weight(int i, float diff) {
    weights[i] += diff;
}

float CNN_Edge::get_scale() const {
    return scale;
}

void CNN_Edge::propagate_weight_count() {
    if (type == CONVOLUTIONAL) {
        output_node->add_weight_count(filter_x * filter_y);
    }
}

void CNN_Edge::initialize_weights(minstd_rand0 &generator, NormalDistribution &normal_distribution) {
    if (type == POOLING) {
        needs_initialization = false;
        return;
    }

    int edge_size = output_node->get_weight_count();
    if (edge_size == 0) {
        cerr << "ERROR! Initializing weights on an edge when node weight counts have not yet been set!" << endl;
        cerr << "edge innovation number: " << innovation_number << endl;
        cerr << "output node innovation number: " << output_node_innovation_number << endl;
        cerr << "input node innovation number: " << input_node_innovation_number << endl;
        cerr << "edge type: " << type << endl;
        throw runtime_error("ERROR! Initializing weights on an edge when node weight counts have not yet been set!");
    }

    float mu = 0.0;
    float sigma = sqrt(2.0 / edge_size);

    //discard the first
    normal_distribution.random(generator, mu, sigma);

    for (uint32_t i = 0; i < filter_size; i++) {
        weights[i] = normal_distribution.random(generator, mu, sigma);
        best_weights[i] = 0.0;
        previous_velocity[i] = 0.0;
    }
    //cout << "initialized weights for edge " << innovation_number << ", weights[0][0]: " << weights[0][0] << endl;

    needs_initialization = false;

    scale = 1.0;
    best_scale = 1.0;
    previous_velocity_scale = 0.0;
    best_velocity_scale = 0.0;
}

void CNN_Edge::reset_velocities() {
    int current = 0;
    for (uint32_t y = 0; y < filter_y; y++) {
        for (uint32_t x = 0; x < filter_x; x++) {
            previous_velocity[current] = 0.0;
            current++;
        }
    }

    previous_velocity_scale = 0.0;
    best_velocity_scale = 0.0;
}

void CNN_Edge::resize() {
    //this may have changed from a regular to reverse filter
    if (output_node->get_size_x() <= input_node->get_size_x()) {
        reverse_filter_x = false;
        filter_x = (input_node->get_size_x() - output_node->get_size_x()) + 1;
    } else {
        reverse_filter_x = true;
        filter_x = (output_node->get_size_x() - input_node->get_size_x()) + 1;
    }

    if (output_node->get_size_y() <= input_node->get_size_y()) {
        reverse_filter_y = false;
        filter_y = (input_node->get_size_y() - output_node->get_size_y()) + 1;
    } else {
        reverse_filter_y = true;
        filter_y = (output_node->get_size_y() - input_node->get_size_y()) + 1;
    }
    filter_size = filter_y * filter_x;

    if (weights != NULL) delete [] weights;
    if (weight_updates != NULL) delete [] weight_updates;
    if (best_weights != NULL) delete [] best_weights;
    if (previous_velocity != NULL) delete [] previous_velocity;
    if (best_velocity != NULL) delete [] best_velocity;

    weights = new float[filter_size]();
    weight_updates = new float[filter_size]();
    best_weights = new float[filter_size]();

    previous_velocity = new float[filter_size]();
    best_velocity = new float[filter_size]();

    initialize_pools(y_pools, y_pool_offset, input_node->get_size_y(), output_node->get_size_y());
    initialize_pools(x_pools, x_pool_offset, input_node->get_size_x(), output_node->get_size_x());

    needs_initialization = true;
}

void CNN_Edge::save_best_weights() {
    int current = 0;
    for (int32_t y = 0; y < filter_y; y++) {
        for (int32_t x = 0; x < filter_x; x++) {
            best_weights[current] = weights[current];
            best_velocity[current] = previous_velocity[current];
            current++;
        }
    }

    best_scale = scale;
    best_velocity_scale = previous_velocity_scale;
}

void CNN_Edge::set_weights_to_best() {
    int current = 0;
    for (int32_t y = 0; y < filter_y; y++) {
        for (int32_t x = 0; x < filter_x; x++) {
            weights[current] = best_weights[current];
            //previous_velocity[y][x] = best_velocity[y][x];
            previous_velocity[current] = 0.0;
            current++;
        }
    }

    scale = best_scale;
    previous_velocity_scale = best_velocity_scale;
}


CNN_Edge* CNN_Edge::copy() const {
    CNN_Edge* copy = new CNN_Edge();

    copy->edge_id = -1;
    copy->genome_id = genome_id;

    copy->fixed = fixed;
    copy->innovation_number = innovation_number;

    copy->type = type;

    copy->disabled = disabled;
    copy->forward_visited = forward_visited;
    copy->reverse_visited = reverse_visited;

    copy->input_node = input_node;
    copy->output_node = output_node;

    copy->input_node_innovation_number = input_node->get_innovation_number();
    copy->output_node_innovation_number = output_node->get_innovation_number();

    copy->batch_size = batch_size;
    copy->filter_x = filter_x;
    copy->filter_y = filter_y;
    copy->filter_size = copy->filter_y * copy->filter_x;

    //cout << "set copy->filter size to: " << copy->filter_size << endl;

    copy->reverse_filter_x = reverse_filter_x;
    copy->reverse_filter_y = reverse_filter_y;
    copy->needs_initialization = needs_initialization;

    copy->scale = scale;
    copy->best_scale = best_scale;
    copy->previous_velocity_scale = previous_velocity_scale;
    copy->best_velocity_scale = best_velocity_scale;

    if (weights == NULL) throw runtime_error("ERROR! copying CNN_Edge where weights == NULL");
    if (weight_updates == NULL) throw runtime_error("ERROR! copying CNN_Edge where weight_updates == NULL");
    if (best_weights == NULL) throw runtime_error("ERROR! copying CNN_Edge where best_weights == NULL");
    if (previous_velocity == NULL) throw runtime_error("ERROR! copying CNN_Edge where previous_velocity == NULL");
    if (best_velocity == NULL) throw runtime_error("ERROR! copying CNN_Edge where best_velocity == NULL");

    copy->weights = new float[copy->filter_size]();
    copy->weight_updates = new float[copy->filter_size]();
    copy->best_weights = new float[copy->filter_size]();

    copy->previous_velocity = new float[copy->filter_size]();
    copy->best_velocity = new float[copy->filter_size]();

    for (uint32_t current = 0; current < copy->filter_size; current++) {
        copy->weights[current] = weights[current];
        copy->weight_updates[current] = weight_updates[current];
        copy->best_weights[current] = best_weights[current];
        copy->previous_velocity[current] = previous_velocity[current];
        copy->best_velocity[current] = best_velocity[current];
    }

    initialize_pools(copy->y_pools, copy->y_pool_offset, input_node->get_size_y(), output_node->get_size_y());
    initialize_pools(copy->x_pools, copy->x_pool_offset, input_node->get_size_x(), output_node->get_size_x());

    return copy;
}

bool CNN_Edge::set_nodes(const vector<CNN_Node*> nodes) {
    //cout << "nodes.size(): " << nodes.size() << endl;
    //cout << "setting input node: " << input_node_innovation_number << endl;
    //cout << "setting output node: " << output_node_innovation_number << endl;

    input_node = NULL;
    output_node = NULL;

    for (uint32_t i = 0; i < nodes.size(); i++) {
        if (nodes[i]->get_innovation_number() == input_node_innovation_number) {
            input_node = nodes[i];
        }

        if (nodes[i]->get_innovation_number() == output_node_innovation_number) {
            output_node = nodes[i];
        }
    }

    if (input_node == NULL) {
        cerr << "ERROR: Could not find node with input innovation number: " << input_node_innovation_number << endl;
        cerr << " nodes innovation numbers:" << endl;
        for (uint32_t i = 0; i < nodes.size(); i++) {
            cerr << "\t" << nodes[i]->get_innovation_number() << endl;
        }

        ostringstream error_message;
        error_message << "Could not find node with input node innovation number " << input_node_innovation_number << " -- this should never happen!" << endl;
        throw runtime_error(error_message.str());
    }

    if (output_node == NULL) {
        cerr << "ERROR: Could not find node with output innovation number: " << output_node_innovation_number << endl;
        cerr << " nodes innovation numbers:" << endl;
        for (uint32_t i = 0; i < nodes.size(); i++) {
            cerr << "\t" << nodes[i]->get_innovation_number() << endl;
        }

        ostringstream error_message;
        error_message << "Could not find node with output node innovation number " << output_node_innovation_number << " -- this should never happen!" << endl;
        throw runtime_error(error_message.str());
    }

    if (output_node == input_node) {
        ostringstream error_message;
        error_message << "Error setting nodes, output_node (" << output_node_innovation_number << ") == input_node (" << input_node_innovation_number << "). This should never happen.";
        throw runtime_error(error_message.str());
    }

    if (!is_filter_correct()) return false;

    return true;
}

void CNN_Edge::set_pools() {
    initialize_pools(y_pools, y_pool_offset, input_node->get_size_y(), output_node->get_size_y());
    initialize_pools(x_pools, x_pool_offset, input_node->get_size_x(), output_node->get_size_x());
}

bool CNN_Edge::is_filter_correct() const {
    //cout << "\t\tchecking filter correctness on edge: " << innovation_number << endl;
    //cout << "\t\t\tdisabled? " << disabled << endl;
    //cout << "\t\t\treverse_filter_x? " << reverse_filter_x << ", reverse_filter_y: " << reverse_filter_y << endl;
    //cout << "\t\t\tbetween node " << input_node_innovation_number << " and " << output_node_innovation_number << endl;

    bool is_correct = true;
    if (reverse_filter_x) {
        is_correct = is_correct && (filter_x == (output_node->get_size_x() - input_node->get_size_x()) + 1);

        if (!is_correct) {
            cerr << "\t\t\tfilter_x: " << filter_x << ", should be: " << (output_node->get_size_x() - input_node->get_size_x()) + 1 << " (output_x: " << output_node->get_size_x() << " - input_x: " << input_node->get_size_x() << " + 1) " << endl;
        }
    } else {
        is_correct = is_correct && (filter_x == (input_node->get_size_x() - output_node->get_size_x()) + 1);

        if (!is_correct) {
            cerr << "\t\t\tfilter_x: " << filter_x << ", should be: " << (input_node->get_size_x() - output_node->get_size_x()) + 1 << " (input_x: " << input_node->get_size_x() << " - output_x: " << output_node->get_size_x() << " + 1) " << endl;
        }
    }

    if (reverse_filter_y) {
        is_correct = is_correct && (filter_y == (output_node->get_size_y() - input_node->get_size_y()) + 1);

        if (!is_correct) {
            cerr << "\t\t\tfilter_y: " << filter_y << ", should be: " << (output_node->get_size_y() - input_node->get_size_y()) + 1 << " (output_y: " << output_node->get_size_y() << " - input_y: " << input_node->get_size_y() << " + 1) " << endl;
        }
    } else {

        is_correct = is_correct && (filter_y == (input_node->get_size_y() - output_node->get_size_y()) + 1);

        if (!is_correct) {
            cerr << "\t\t\tfilter_y: " << filter_y << ", should be: " << (input_node->get_size_y() - output_node->get_size_y()) + 1 << " (input_y: " << input_node->get_size_y() << " - output_y: " << output_node->get_size_y() << " + 1) " << endl;
        }
    }

    if (!is_correct) {
        cerr << "\t\tfilter incorrect on edge: " << innovation_number << endl;
        cerr << "\t\t\tdisabled? " << disabled << endl;
        cerr << "\t\t\ttype: " << type << endl;
        cerr << "\t\t\treverse_filter_x? " << reverse_filter_x << ", reverse_filter_y: " << reverse_filter_y << endl;
        cerr << "\t\t\tbetween node " << input_node_innovation_number << " and " << output_node_innovation_number << endl;
    }

    return is_correct;
}

void CNN_Edge::update_batch_size(int new_batch_size) {
    batch_size = new_batch_size;
}

void CNN_Edge::alter_edge_type() {
    if (type == CONVOLUTIONAL) {
        type = POOLING;
    } else {
        type = CONVOLUTIONAL;
    }
}

void CNN_Edge::enable() {
    if (is_reachable()) {
        disabled = false;
    }
}

void CNN_Edge::disable() {
    if (!is_reachable()) {
        disabled = true;
    }
}

bool CNN_Edge::is_enabled() const {
    return !disabled;
}

bool CNN_Edge::is_disabled() const {
    return disabled;
}

bool CNN_Edge::is_reachable() const {
    return !disabled && forward_visited && reverse_visited;
}

bool CNN_Edge::is_forward_visited() const {
    return forward_visited;
}

bool CNN_Edge::is_reverse_visited() const {
    return reverse_visited;
}

void CNN_Edge::forward_visit() {
    forward_visited = true;
}

void CNN_Edge::reverse_visit() {
    reverse_visited = true;
}

void CNN_Edge::set_unvisited() {
    forward_visited = false;
    reverse_visited = false;
}


int CNN_Edge::get_number_weights() const {
    return filter_x * filter_y;
}

int CNN_Edge::get_batch_size() const {
    return batch_size;
}


int CNN_Edge::get_innovation_number() const {
    return innovation_number;
}

int CNN_Edge::get_input_innovation_number() const {
    return input_node_innovation_number;
}

int CNN_Edge::get_output_innovation_number() const {
    return output_node_innovation_number;
}


CNN_Node* CNN_Edge::get_input_node() {
    return input_node;
}

CNN_Node* CNN_Edge::get_output_node() {
    return output_node;
}

bool CNN_Edge::connects(int n1, int n2) const {
    return (input_node_innovation_number == n1) && (output_node_innovation_number == n2);
}

bool CNN_Edge::has_zero_weight() const {
    if (is_reachable()) return false;

    float filter_sum = 0.0;
    int current = 0;
    for (int32_t fy = 0; fy < filter_y; fy++) {
        for (int32_t fx = 0; fx < filter_x; fx++) {
            filter_sum += weights[current] * weights[current];
            current++;
        }
    }

    return filter_sum == 0;
}

bool CNN_Edge::has_zero_best_weight() const {
    if (is_reachable()) return false;

    float filter_sum = 0.0;
    int current = 0;
    for (int32_t fy = 0; fy < filter_y; fy++) {
        for (int32_t fx = 0; fx < filter_x; fx++) {
            filter_sum += best_weights[current] * best_weights[current];
            current++;
        }
    }

    return filter_sum == 0;
}



void CNN_Edge::print(ostream &out) {
    out << "CNN_Edge " << innovation_number << " from node " << input_node->get_innovation_number() << " to node " << output_node->get_innovation_number() << " with filter x: " << filter_x << ", y: " << filter_y << endl;

    out << "weights:" << endl;

    int current = 0;
    for (uint32_t y = 0; y < filter_y; y++) {
        out << "    ";
        for (uint32_t x = 0; x < filter_x; x++) {
            out << setw(11) << std::fixed << setprecision(7) << weights[current];
            current++;
        }
        out << endl;
    }

    out << "previous_velocity:" << endl;
    current = 0;
    for (uint32_t y = 0; y < filter_y; y++) {
        out << "    ";
        for (uint32_t x = 0; x < filter_x; x++) {
            out << setw(11) << std::fixed << setprecision(7) << previous_velocity[current];
            current++;
        }
        out << endl;
    }

    out << "weight_updates:" << endl;
    current = 0;
    for (uint32_t y = 0; y < filter_y; y++) {
        out << "    ";
        for (uint32_t x = 0; x < filter_x; x++) {
            out << setw(11) << std::fixed << setprecision(7) << weight_updates[current];
            current++;
        }
        out << endl;
    }
}

void CNN_Edge::check_output_update(const vector< vector< vector<float> > > &output, const vector< vector< vector<float> > > &input, float value, float weight, float previous_output, int batch_number, int in_y, int in_x, int out_y, int out_x) {
    if (isnan(output[batch_number][out_y][out_x]) || isinf(output[batch_number][out_y][out_x])) {
        cerr << "ERROR in edge " << innovation_number << " propagate forward!" << endl;
        cerr << "input node innovation number: " << input_node->get_innovation_number() << " at depth: " << input_node->get_depth() << endl;
        cerr << "input node inputs fired: " << input_node->get_inputs_fired() << ", total_inputs: " << input_node->get_number_inputs() << endl;
        cerr << "output node innovation number: " << output_node->get_innovation_number() << " at depth: " << output_node->get_depth() << endl;
        cerr << "output became: " << output[batch_number][out_y][out_x] << "!" << endl;
        cerr << "output[" << batch_number << "][" << out_y << "][" << out_x << "] = " << output[batch_number][out_y][out_x] << endl;
        cerr << "input[" << batch_number << "][" << in_y << "][" << in_x << "] = " << input[batch_number][in_y][in_x] << endl;
        cerr << "weight: " << weight << endl;
        cerr << "previous output: " << previous_output << endl;
        cerr << "value added: " << value << endl;

        input_node->print(cerr);
        output_node->print(cerr);

        throw runtime_error("ERROR: output because NAN or INF in check_output_update");
    }
}

void CNN_Edge::propagate_forward(bool training, bool accumulate_test_statistics, float epsilon, float alpha, bool perform_dropout, float hidden_dropout_probability, minstd_rand0 &generator) {
    if (!is_reachable()) return;

    using namespace std::chrono;
    high_resolution_clock::time_point propagate_forward_start_time = high_resolution_clock::now();

    float *input = input_node->get_values_out();
    float *pool_gradients = input_node->get_pool_gradients();
    float *output = output_node->get_values_in();

#ifdef NAN_CHECKS
    if (!is_filter_correct()) {
        ostringstream error_message;
        error_message << "ERROR: filter_x != input_node->get_size_x: " << input_node->get_size_x() << " - output_node->get_size_x: " << output_node->get_size_x() << " + 1";
        throw runtime_error(error_message.str());
    }

    float previous_output;
#endif

    int output_size_x = output_node->get_size_x();
    int output_size_y = output_node->get_size_y();
    int input_size_x = input_node->get_size_x();
    int input_size_y = input_node->get_size_y();

    if (type == CONVOLUTIONAL) {
        if (reverse_filter_y && reverse_filter_x) {
            prop_forward_ry_rx(input, weights, output, batch_size, input_size_y, input_size_x, filter_y, filter_x, output_size_y, output_size_x);
        } else if (reverse_filter_y) {
            prop_forward_ry(input, weights, output, batch_size, input_size_y, input_size_x, filter_y, filter_x, output_size_y, output_size_x);
        } else if (reverse_filter_x) {
            prop_forward_rx(input, weights, output, batch_size, input_size_y, input_size_x, filter_y, filter_x, output_size_y, output_size_x);
        } else {
            prop_forward(input, weights, output, batch_size, input_size_y, input_size_x, filter_y, filter_x, output_size_y, output_size_x);
        }

    } else if (type == POOLING) {
#ifdef NAN_CHECKS
        if (y_pools.size() != output_size_y) {
            cerr << "ERROR: POOLING y_pools.size: " << y_pools.size() << " != output_size_y: " << output_size_y << ", input_size_y: " << input_size_y << endl;
            exit(1);
        }

        if (x_pools.size() != output_size_x) {
            cerr << "ERROR: POOLING x_pools.size: " << x_pools.size() << " != output_size_x: " << output_size_x << ", input_size_x: " << input_size_x << endl;
            exit(1);
        }
#endif

        /*
        cout << "y_pools: ";
        for (int32_t i = 0; i < y_pools.size(); i++) cout << " " << y_pools[i];
        cout << endl;

        cout << "x_pools: ";
        for (int32_t i = 0; i < x_pools.size(); i++) cout << " " << x_pools[i];
        cout << endl;
        */


        bool max_pooling = true;
        if (reverse_filter_y && reverse_filter_x) {
            if (input_size_y % output_size_y == 0) {
                fisher_yates_shuffle(generator, y_pools);
                update_offset(y_pools, y_pool_offset);
                max_pooling = false;
            }

            if (input_size_x % output_size_x == 0) {
                fisher_yates_shuffle(generator, x_pools);
                update_offset(x_pools, x_pool_offset);
                max_pooling = false;
            }

            pool_forward_ry_rx(input, scale, pool_gradients, output, batch_size, input_size_y, input_size_x, output_size_y, output_size_x, y_pools, x_pools, y_pool_offset, x_pool_offset, generator, training, max_pooling);

        } else if (reverse_filter_y) {
            if (output_size_y % input_size_y == 0) {
                fisher_yates_shuffle(generator, y_pools);
                update_offset(y_pools, y_pool_offset);
                max_pooling = false;
            }

            if (input_size_x % output_size_x == 0) {
                fisher_yates_shuffle(generator, x_pools);
                update_offset(x_pools, x_pool_offset);
                max_pooling = false;
            }

            pool_forward_ry(input, scale, pool_gradients, output, batch_size, input_size_y, input_size_x, output_size_y, output_size_x, y_pools, x_pools, y_pool_offset, x_pool_offset, generator, training, max_pooling);

        } else if (reverse_filter_x) {
            if (input_size_y % output_size_y == 0) {
                fisher_yates_shuffle(generator, y_pools);
                update_offset(y_pools, y_pool_offset);
                max_pooling = false;
            }

            if (output_size_x % input_size_x == 0) {
                fisher_yates_shuffle(generator, x_pools);
                update_offset(x_pools, x_pool_offset);
                max_pooling = false;
            }

            pool_forward_rx(input, scale, pool_gradients, output, batch_size, input_size_y, input_size_x, output_size_y, output_size_x, y_pools, x_pools, y_pool_offset, x_pool_offset, generator, training, max_pooling);

        } else {
            if (output_size_y % input_size_y == 0) {
                fisher_yates_shuffle(generator, y_pools);
                update_offset(y_pools, y_pool_offset);
                max_pooling = false;
            }

            if (output_size_x % input_size_x == 0) {
                fisher_yates_shuffle(generator, x_pools);
                update_offset(x_pools, x_pool_offset);
                max_pooling = false;
            }

            pool_forward(input, scale, pool_gradients, output, batch_size, input_size_y, input_size_x, output_size_y, output_size_x, y_pools, x_pools, y_pool_offset, x_pool_offset, generator, training, max_pooling);
        }

    } else {
        cerr << "ERROR: unknown edge type in propagate_forward: " << type << endl;
        exit(1);
    }

    high_resolution_clock::time_point propagate_forward_end_time = high_resolution_clock::now();
    duration<float, std::milli> time_span = propagate_forward_end_time - propagate_forward_start_time;

    propagate_forward_time += time_span.count() / 1000.0;

	output_node->input_fired(training, accumulate_test_statistics, epsilon, alpha, perform_dropout, hidden_dropout_probability, generator);
}

void CNN_Edge::update_weights(float mu, float learning_rate, float weight_decay) {
    if (!is_reachable()) return;
    if (type == POOLING) return;

    using namespace std::chrono;
    high_resolution_clock::time_point weight_update_start_time = high_resolution_clock::now();

    float dx, pv, velocity, weight;
#ifdef NAN_CHECKS
    float previous_weight;
#endif

    for (int32_t current = 0; current < filter_size; current++) {
        dx = weight_updates[current];

        //try clipping the weight
        //dx = dx * (0.5 / fmax(0.5, fabs(dx)));

        pv = previous_velocity[current];

        velocity = (mu * pv) - learning_rate * dx;

        weight = weights[current];
#ifdef NAN_CHECKS
        previous_weight = weight;
#endif
        weight += velocity + mu * (velocity - pv);
        //weight += (-mu * pv + (1 + mu) * velocity);
        //weight += velocity;

        weight -= (weight * weight_decay);

        weights[current] = weight;

        previous_velocity[current] = velocity;

#ifdef NAN_CHECKS
        if (isnan(weights[current]) || isinf(weights[current])) {
            cerr << "ERROR! weight became " << weights[current] << " in edge: " << innovation_number << " (" << input_node_innovation_number << " to " << output_node_innovation_number << ")" << endl;
            cerr << "\tdx: " << dx << endl;
            cerr << "\tpv: " << pv << endl;
            cerr << "\tvelocity: " << velocity << endl;
            cerr << "\tprevious_weight: " << previous_weight << endl;
            exit(1);
        }
#endif

        if (weights[current] > 50.0) {
            /*
               cout << "weight > 2!" << endl;
               cout << "updating weight from " << input_node_innovation_number << " to " << output_node_innovation_number
               << ": fy: " << fy << ", fx: " << fx 
               << ", weight: " << weights[current] 
               << ", weight_update: " << weight_updates[current] 
               << ", learning_rate * dx: " << (learning_rate * dx) << endl;

               this->print(cout);
               input_node->print(cout);
               output_node->print(cout);

               exit(1);
               */

            weights[current] = 50.0;
            previous_velocity[current] = 0.0;
        } else if (weights[current] < -50.0) {
            /*
               cout << "weight < -2!" << endl;
               cout << "updating weight from " << input_node_innovation_number << " to " << output_node_innovation_number
               << ": fy: " << fy << ", fx: " << fx 
               << ", weight: " << weights[current] 
               << ", weight_update: " << weight_updates[current] 
               << ", learning_rate * dx: " << (learning_rate * dx) << endl;
               this->print(cout);
               input_node->print(cout);
               output_node->print(cout);

               exit(1);
               */

            weights[current] = -50.0;
            previous_velocity[current] = 0.0;
        }
    }

    high_resolution_clock::time_point weight_update_end_time = high_resolution_clock::now();
    duration<float, std::milli> time_span = weight_update_end_time - weight_update_start_time;

    weight_update_time += time_span.count() / 1000.0;
}

void CNN_Edge::propagate_backward(bool training, float mu, float learning_rate, float epsilon) {
    if (!is_reachable()) return;

    using namespace std::chrono;
    high_resolution_clock::time_point propagate_backward_start_time = high_resolution_clock::now();

    float *output_errors = output_node->get_errors_in();
    float *input = input_node->get_values_out();
    float *input_errors = input_node->get_errors_out();

    int input_size_x = input_node->get_size_x();
    int input_size_y = input_node->get_size_y();

    int output_size_x = output_node->get_size_x();
    int output_size_y = output_node->get_size_y();

    for (int32_t current = 0; current < filter_size; current++) {
        weight_updates[current] = 0;
    }

    if (type == CONVOLUTIONAL) {
        if (reverse_filter_x && reverse_filter_y) {
            prop_backward_ry_rx(output_errors, input, input_errors, weight_updates, weights, batch_size, input_size_y, input_size_x, filter_y, filter_x, output_size_y, output_size_x);
        } else if (reverse_filter_y) {
            prop_backward_ry(output_errors, input, input_errors, weight_updates, weights, batch_size, input_size_y, input_size_x, filter_y, filter_x, output_size_y, output_size_x);
        } else if (reverse_filter_x) {
            prop_backward_rx(output_errors, input, input_errors, weight_updates, weights, batch_size, input_size_y, input_size_x, filter_y, filter_x, output_size_y, output_size_x);
        } else {
            prop_backward(output_errors, input, input_errors, weight_updates, weights, batch_size, input_size_y, input_size_x, filter_y, filter_x, output_size_y, output_size_x);
        }

    } else if (type == POOLING) {
        float *pool_gradients = input_node->get_pool_gradients();

        float scale_update = 0.0;
        if (reverse_filter_y && reverse_filter_x) {
            pool_backward_ry_rx(input_errors, scale_update, input, pool_gradients, output_errors, batch_size, input_size_y, input_size_x, output_size_y, output_size_x, y_pools, x_pools, y_pool_offset, x_pool_offset);
        } else if (reverse_filter_y) {
            pool_backward_ry(input_errors, scale_update, input, pool_gradients, output_errors, batch_size, input_size_y, input_size_x, output_size_y, output_size_x, y_pools, x_pools, y_pool_offset, x_pool_offset);
        } else if (reverse_filter_x) {
            pool_backward_rx(input_errors, scale_update, input, pool_gradients, output_errors, batch_size, input_size_y, input_size_x, output_size_y, output_size_x, y_pools, x_pools, y_pool_offset, x_pool_offset);
        } else {
            pool_backward(input_errors, scale_update, input, pool_gradients, output_errors, batch_size, input_size_y, input_size_x, output_size_y, output_size_x, y_pools, x_pools, y_pool_offset, x_pool_offset);
        }

        float pv_scale = previous_velocity_scale;
        float velocity_scale = (mu * pv_scale) - (learning_rate * scale_update); //scale update is divided by batch size in pool_backward
        scale += velocity_scale + mu * (velocity_scale - pv_scale);
        previous_velocity_scale = velocity_scale;

        if (scale > 50.0) {
            scale = 50.0;
            previous_velocity_scale = 0.0;
        } else if (scale < -50.0) {
            scale = -50.0;
            previous_velocity_scale = 0.0;
        }

    } else {
        cerr << "ERROR: unknown edge type in poolagate_backward: " << type << endl;
        exit(1);
    }

    high_resolution_clock::time_point propagate_backward_end_time = high_resolution_clock::now();
    duration<float, std::milli> time_span = propagate_backward_end_time - propagate_backward_start_time;

    propagate_backward_time += time_span.count() / 1000.0;

    input_node->output_fired(training, mu, learning_rate, epsilon);
}

bool CNN_Edge::has_nan() const {
    //cout << "checking to see if edge " << innovation_number << " has nan or inf, filter_size: " << filter_size << endl;
    for (int32_t current = 0; current < filter_size; current++) {
        if (isnan(weights[current]) || isinf(weights[current])) return true;
        if (isnan(weight_updates[current]) || isinf(weight_updates[current])) return true;
        if (isnan(previous_velocity[current]) || isinf(previous_velocity[current])) return true;
    }
    
    if (isnan(scale) || isnan(best_scale) || isnan(previous_velocity_scale) || isnan(best_velocity_scale)) return true;
    return false;
}

void CNN_Edge::print_statistics() {
    float weight_min = std::numeric_limits<float>::max(), weight_max = -std::numeric_limits<float>::max(), weight_avg = 0.0;
    float weight_update_min = std::numeric_limits<float>::max(), weight_update_max = -std::numeric_limits<float>::max(), weight_update_avg = 0.0;
    float velocity_min = std::numeric_limits<float>::max(), velocity_max = -std::numeric_limits<float>::max(), velocity_avg = 0.0;

    for (int32_t current = 0; current < filter_size; current++) {
        if (weights[current] < weight_min) weight_min = weights[current];
        if (weights[current] > weight_max) weight_max = weights[current];
        weight_avg += weights[current];

        if (weight_updates[current] < weight_update_min) weight_update_min = weight_updates[current];
        if (weight_updates[current] > weight_update_max) weight_update_max = weight_updates[current];
        weight_update_avg += weight_updates[current];

        if (previous_velocity[current] < velocity_min) velocity_min = previous_velocity[current];
        if (previous_velocity[current] > velocity_max) velocity_max = previous_velocity[current];
        velocity_avg += previous_velocity[current];
    }

    velocity_avg /= filter_size;
    weight_avg /= filter_size;

    cerr << "edge " << setw(4) << innovation_number << " (in: " << setw(4) << input_node_innovation_number << ", out: " << setw(4) << output_node_innovation_number << ") w_min: " << weight_min << ", w_avg: " << weight_avg << ", w_max: " << weight_max << ", wu_min: " << weight_update_min << ", wu_avg: " << weight_update_avg << ", wu_max: " << weight_update_max << ", v_min: " << velocity_min << ", v_avg: " << velocity_avg << ", v_max: " << velocity_max << endl;

}

ostream &operator<<(ostream &os, const CNN_Edge* edge) {
    os << edge->edge_id << " ";
    os << edge->exact_id << " ";
    os << edge->genome_id << " ";
    os << edge->type << " ";
    os << edge->innovation_number << " ";
    os << edge->input_node_innovation_number << " ";
    os << edge->output_node_innovation_number << " ";
    os << edge->filter_x << " ";
    os << edge->filter_y << " ";
    os << edge->fixed << " ";
    os << edge->reverse_filter_x << " ";
    os << edge->reverse_filter_y << " ";
    os << edge->disabled << " ";
    os << edge->forward_visited << " ";
    os << edge->reverse_visited << " ";
    os << edge->needs_initialization << " ";
    os << edge->batch_size << endl;

    write_hexfloat(os, edge->scale);
    os << " ";
    write_hexfloat(os, edge->best_scale);
    os << " ";
    write_hexfloat(os, edge->previous_velocity_scale);
    os << " ";
    write_hexfloat(os, edge->best_velocity_scale);
    os << endl;

    os << "POOLS" << endl;
    os << edge->y_pools.size();
    for (int32_t i = 0; i < edge->y_pools.size(); i++) {
        os << " " << edge->y_pools[i];
    }
    os << endl;

    os << edge->x_pools.size();
    for (int32_t i = 0; i < edge->x_pools.size(); i++) {
        os << " " << edge->x_pools[i];
    }
    os << endl;

    os << "WEIGHTS" << endl;
    int current = 0;
    for (int32_t y = 0; y < edge->filter_y; y++) {
        for (int32_t x = 0; x < edge->filter_x; x++) {
            if (y > 0 || x > 0) os << " ";
            write_hexfloat(os, edge->weights[current]);
            current++;
        }
    }
    os << endl;

    os << "BEST_WEIGHTS" << endl;
    current = 0;
    for (int32_t y = 0; y < edge->filter_y; y++) {
        for (int32_t x = 0; x < edge->filter_x; x++) {
            if (y > 0 || x > 0) os << " ";
            write_hexfloat(os, edge->best_weights[current]);
            current++;
        }
    }
    os << endl;

    os << "PREVIOUS_VELOCITY" << endl;
    current = 0;
    for (int32_t y = 0; y < edge->filter_y; y++) {
        for (int32_t x = 0; x < edge->filter_x; x++) {
            if (y > 0 || x > 0) os << " ";
            write_hexfloat(os, edge->previous_velocity[current]);
            current++;
        }
    }
    os << endl;

    os << "BEST_VELOCITY" << endl;
    current = 0;
    for (int32_t y = 0; y < edge->filter_y; y++) {
        for (int32_t x = 0; x < edge->filter_x; x++) {
            if (y > 0 || x > 0) os << " ";
            write_hexfloat(os, edge->best_velocity[current]);
            current++;
        }
    }

    return os;
}

istream &operator>>(istream &is, CNN_Edge* edge) {
    is >> edge->edge_id;
    is >> edge->exact_id;
    is >> edge->genome_id;
    is >> edge->type;
    is >> edge->innovation_number;
    is >> edge->input_node_innovation_number;
    is >> edge->output_node_innovation_number;
    is >> edge->filter_x;
    is >> edge->filter_y;
    is >> edge->fixed;
    is >> edge->reverse_filter_x;
    is >> edge->reverse_filter_y;
    is >> edge->disabled;
    is >> edge->forward_visited;
    is >> edge->reverse_visited;
    is >> edge->needs_initialization;
    is >> edge->batch_size;

    edge->filter_size = edge->filter_y * edge->filter_x;

    //don't need to initialize memory for unreachable edges
    edge->weights = new float[edge->filter_size]();
    edge->weight_updates = new float[edge->filter_size]();
    edge->best_weights = new float[edge->filter_size]();

    edge->previous_velocity = new float[edge->filter_size]();
    edge->best_velocity = new float[edge->filter_size]();

    edge->scale = read_hexfloat(is);
    edge->best_scale = read_hexfloat(is);
    edge->previous_velocity_scale = read_hexfloat(is);
    edge->best_velocity_scale = read_hexfloat(is);

    string line;
    getline(is, line);
    getline(is, line);
    if (line.compare("POOLS") != 0) {
        cerr << "ERROR: invalid input file, expected line to be 'POOLS' but line was '" << line << "'" << endl;
        exit(1);
    }

    int pool_size;
    int value;

    is >> pool_size;
    edge->y_pools.clear();
    for (int32_t i = 0; i < pool_size; i++) {
        is >> value;
        edge->y_pools.push_back(value);
    }

    is >> pool_size;
    edge->x_pools.clear();
    for (int32_t i = 0; i < pool_size; i++) {
        is >> value;
        edge->x_pools.push_back(value);
    }

    update_offset(edge->y_pools, edge->y_pool_offset);
    update_offset(edge->x_pools, edge->x_pool_offset);

    /*
       cerr << "edge " << edge->innovation_number << ", y_pools: ";
       for (int32_t i = 0; i < edge->y_pools.size(); i++) {
       cerr << " " << edge->y_pools[i];
       }
       cerr << ", x_pools:";
       for (int32_t i = 0; i < edge->x_pools.size(); i++) {
       cerr << " " << edge->x_pools[i];
       }
       cerr << endl;
       */


    getline(is, line);
    getline(is, line);
    if (line.compare("WEIGHTS") != 0) {
        cerr << "ERROR: invalid input file, expected line to be 'WEIGHTS' but line was '" << line << "'" << endl;
        exit(1);
    }

    int current = 0;
    for (int32_t y = 0; y < edge->filter_y; y++) {
        for (int32_t x = 0; x < edge->filter_x; x++) {
            edge->weights[current] = read_hexfloat(is);
            current++;
        }
    }

    getline(is, line);
    getline(is, line);
    if (line.compare("BEST_WEIGHTS") != 0) {
        cerr << "ERROR: invalid input file, expected line to be 'BEST_WEIGHTS' but line was '" << line << "'" << endl;
        exit(1);
    }

    current = 0;
    for (int32_t y = 0; y < edge->filter_y; y++) {
        for (int32_t x = 0; x < edge->filter_x; x++) {
            edge->best_weights[current] = read_hexfloat(is);
            current++;
        }
    }

    getline(is, line);
    getline(is, line);
    if (line.compare("PREVIOUS_VELOCITY") != 0) {
        cerr << "ERROR: invalid input file, expected line to be 'PREVIOUS_VELOCITY' but line was '" << line << "'" << endl;
        exit(1);
    }

    current = 0;
    for (int32_t y = 0; y < edge->filter_y; y++) {
        for (int32_t x = 0; x < edge->filter_x; x++) {
            edge->previous_velocity[current] = read_hexfloat(is);
            current++;
        }
    }

    getline(is, line);
    getline(is, line);
    if (line.compare("BEST_VELOCITY") != 0) {
        cerr << "ERROR: invalid input file, expected line to be 'BEST_VELOCITY' but line was '" << line << "'" << endl;
        exit(1);
    }

    current = 0;
    for (int32_t y = 0; y < edge->filter_y; y++) {
        for (int32_t x = 0; x < edge->filter_x; x++) {
            edge->best_velocity[current] = read_hexfloat(is);
            current++;
        }
    }

    //pools will be initialized after nodes are set

    return is;
}

bool CNN_Edge::is_identical(const CNN_Edge *other, bool testing_checkpoint) {
    if (are_different("edge_id", edge_id, other->edge_id)) return false;
    if (are_different("exact_id", exact_id, other->exact_id)) return false;
    if (are_different("genome_id", genome_id, other->genome_id)) return false;

    if (are_different("type", type, other->type)) return false;
    if (are_different("innovation_number", innovation_number, other->innovation_number)) return false;

    if (are_different("input_node_innovation_number", input_node_innovation_number, other->input_node_innovation_number)) return false;
    if (are_different("output_node_innovation_number", output_node_innovation_number, other->output_node_innovation_number)) return false;

    if (are_different("batch_size", batch_size, other->batch_size)) return false;
    if (are_different("filter_x", filter_x, other->filter_x)) return false;
    if (are_different("filter_y", filter_y, other->filter_y)) return false;

    if (are_different("weights", filter_x * filter_y, weights, other->weights)) return false;
    
    //weight updates are zeroed at the beginning of each epoch
    //if (are_different("weight_updates", weight_updates, other->weight_updates)) return false;
    if (are_different("best_weights", filter_x * filter_y, best_weights, other->best_weights)) return false;

    if (are_different("previous_velocity", filter_x * filter_y, previous_velocity, other->previous_velocity)) return false;
    if (are_different("best_velocity", filter_x * filter_y, best_velocity, other->best_velocity)) return false;


    if (are_different("y_pools", y_pools, other->y_pools)) return false;
    if (are_different("y_pool_offset", y_pool_offset, other->y_pool_offset)) return false;
    if (are_different("x_pools", x_pools, other->x_pools)) return false;
    if (are_different("x_pool_offset", x_pool_offset, other->x_pool_offset)) return false;

    if (are_different("fixed", fixed, other->fixed)) return false;
    if (are_different("disabled", disabled, other->disabled)) return false;
    if (are_different("forward_visited", forward_visited, other->forward_visited)) return false;
    if (are_different("reverse_visited", reverse_visited, other->reverse_visited)) return false;

    if (are_different("reverse_filter_x", reverse_filter_x, other->reverse_filter_x)) return false;
    if (are_different("reverse_filter_y", reverse_filter_y, other->reverse_filter_y)) return false;
    if (are_different("needs_initialization", needs_initialization, other->needs_initialization)) return false;

    if (are_different("scale", scale, other->scale)) return false;
    if (are_different("best_scale", best_scale, other->best_scale)) return false;
    if (are_different("previous_velocity_scale", previous_velocity_scale, other->previous_velocity_scale)) return false;
    if (are_different("best_velocity_scale", best_velocity_scale, other->best_velocity_scale)) return false;

    //these are zeroed at the beginning of each batch so don't need to be similar
    //if (are_different("propagate_backward_time", propagate_backward_time, other->propagate_backward_time)) return false;
    //if (are_different("propagate_backward_time", propagate_backward_time, other->propagate_backward_time)) return false;
    //if (are_different("weight_update_time", weight_update_time, other->weight_update_time)) return false;

    return true;
}

