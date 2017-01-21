#include <fstream>
using std::ofstream;
using std::ifstream;
using std::ios;

#include <iomanip>
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

CNN_Node::CNN_Node() {
    node_id = -1;
    exact_id = -1;
    genome_id = -1;

    innovation_number = -1;
    values = NULL;
    errors = NULL;
    gradients = NULL;
    bias = NULL;
    best_bias = NULL;
    bias_velocity = NULL;
    visited = false;

    weight_count = 0;
}

CNN_Node::CNN_Node(int _innovation_number, double _depth, int _size_x, int _size_y, int _type, minstd_rand0 &generator, NormalDistribution &normal_distribution) {
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

    values = new double*[size_y];
    errors = new double*[size_y];
    gradients = new double*[size_y];
    bias = new double*[size_y];
    best_bias = new double*[size_y];
    bias_velocity = new double*[size_y];
    for (int32_t y = 0; y < size_y; y++) {
        values[y] = new double[size_x];
        errors[y] = new double[size_x];
        gradients[y] = new double[size_x];
        bias[y] = new double[size_x];
        best_bias[y] = new double[size_x];
        bias_velocity[y] = new double[size_x];
        for (int32_t x = 0; x < size_x; x++) {
            values[y][x] = 0.0;
            errors[y][x] = 0.0;
            gradients[y][x] = 0.0;
            bias[y][x] = 0.0;
            best_bias[y][x] = 0.0;
            bias_velocity[y][x] = 0.0;
        }
    }

    initialize_bias(generator, normal_distribution);
    save_best_bias();
}

CNN_Node::~CNN_Node() {
    for (int32_t y = 0; y < size_y; y++) {
        if (values[y] == NULL) {
            cerr << "ERROR, modifying node size x but values[" << y << "] == NULL" << endl;
            exit(1);
        }

        if (errors[y] == NULL) {
            cerr << "ERROR, modifying node size x but values[" << y << "] == NULL" << endl;
            exit(1);
        }

        delete [] values[y];
        delete [] errors[y];
        delete [] gradients[y];
        delete [] bias[y];
        delete [] best_bias[y];
        delete [] bias_velocity[y];
    }
    delete [] values;
    delete [] errors;
    delete [] gradients;
    delete [] bias;
    delete [] best_bias;
    delete [] bias_velocity;
}


template <class T>
void parse_array_2d(T ***output, istringstream &iss, int size_x, int size_y) {
    (*output) = new T*[size_y];
    for (int32_t y = 0; y < size_y; y++) {
        (*output)[y] = new T[size_x];
        for (int32_t x = 0; x < size_x; x++) {
            (*output)[y][x] = 0.0;
        }
    }

    int current_x = 0, current_y = 0;

    T val;
    while(iss >> val || !iss.eof()) {
        if (iss.fail()) {
            iss.clear();
            string dummy;
            iss >> dummy;
            continue;
        }

        //cout << "output[" << current_x << "][" << current_y << "]: " << val << endl;
        (*output)[current_y][current_x] = val;

        current_x++;

        if (current_x >= size_x) {
            current_x = 0;
            current_y++;
        }
    }
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
        parse_array_2d(&bias, bias_iss, size_x, size_y);

        istringstream best_bias_iss(row[10]);
        parse_array_2d(&best_bias, best_bias_iss, size_x, size_y);

        istringstream bias_velocity_iss(row[11]);
        parse_array_2d(&bias_velocity, bias_velocity_iss, size_x, size_y);

        type = atoi(row[12]);
        total_inputs = atoi(row[13]);
        total_inputs = 0;   //need to reset this because it will be modified when
                            //edges are set
        inputs_fired = atoi(row[14]);

        visited = atoi(row[15]);

        mysql_free_result(result);
    } else {
        cerr << "ERROR! Could not find cnn_node in database with node id: " << node_id << endl;
        exit(1);
    }

    //initialize arrays not stored to database
    values = new double*[size_y];
    errors = new double*[size_y];
    gradients = new double*[size_y];
    for (int32_t y = 0; y < size_y; y++) {
        values[y] = new double[size_x];
        errors[y] = new double[size_x];
        gradients[y] = new double[size_x];
        for (int32_t x = 0; x < size_x; x++) {
            values[y][x] = 0.0;
            errors[y][x] = 0.0;
            gradients[y][x] = 0.0;
        }
    }

    //cout << "read node!" << endl;
    //cout << this << endl;
}

void CNN_Node::export_to_database(int _exact_id, int _genome_id) {
    ostringstream query;

    exact_id = _exact_id;
    genome_id = _genome_id;

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

    query << "', type = " << type
        << ", total_inputs = " << total_inputs
        << ", inputs_fired = " << inputs_fired
        << ", visited = " << visited;

    mysql_exact_query(query.str());

    if (node_id < 0) {
        node_id = mysql_exact_last_insert_id();
        //cout << "set node id to: " << node_id << endl;
    }
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

    copy->values = new double*[size_y];
    copy->errors = new double*[size_y];
    copy->gradients = new double*[size_y];
    copy->bias = new double*[size_y];
    copy->best_bias = new double*[size_y];
    copy->bias_velocity = new double*[size_y];

    for (int32_t y = 0; y < size_y; y++) {
        copy->values[y] = new double[size_x];
        copy->errors[y] = new double[size_x];
        copy->gradients[y] = new double[size_x];
        copy->bias[y] = new double[size_x];
        copy->best_bias[y] = new double[size_x];
        copy->bias_velocity[y] = new double[size_x];

        for (int32_t x = 0; x < size_x; x++) {
            copy->values[y][x] = values[y][x];
            copy->errors[y][x] = errors[y][x];
            copy->gradients[y][x] = gradients[y][x];
            copy->bias[y][x] = bias[y][x];
            copy->best_bias[y][x] = best_bias[y][x];
            copy->bias_velocity[y][x] = bias_velocity[y][x];
        }
    }

    return copy;
}

void CNN_Node::add_weight_count(int _weight_count) {
    weight_count += _weight_count;
    
    //cerr << "node " << innovation_number << " setting weight count to: " << weight_count << endl;
}

int CNN_Node::get_weight_count() const {
    return weight_count;
}

void CNN_Node::initialize_bias(minstd_rand0 &generator, NormalDistribution &normal_distribution) {
    int bias_size = size_x * size_y;
    if (bias_size == 1) bias_size = 10;

    double mu = 0.0;
    //double sigma = sqrt(2.0 / bias_size);
    double sigma = 2.0 / bias_size;

    for (int32_t i = 0; i < size_y; i++) {
        for (int32_t j = 0; j < size_x; j++) {
            bias[i][j] = normal_distribution.random(generator, mu, sigma);
            best_bias[i][j] = 0.0;
            bias_velocity[i][j] = 0.0;
            //cout << "node " << innovation_number << " bias[" << i << "][" << j <<"]: " << bias[i][j] << endl;
        }
    }
}

void CNN_Node::initialize_velocities() {
    for (int32_t i = 0; i < size_y; i++) {
        for (int32_t j = 0; j < size_x; j++) {
            bias_velocity[i][j] = 0.0;
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

double** CNN_Node::get_values() {
    return values;
}

double CNN_Node::get_value(int y, int x) {
    return values[y][x];
}

double CNN_Node::set_value(int y, int x, double value) {
    return values[y][x] = value;
}


void CNN_Node::set_error(int y, int x, double value) {
    errors[y][x] = value;
}

double CNN_Node::get_error(int y, int x) {
    return errors[y][x];
}

double** CNN_Node::get_errors() {
    return errors;
}

void CNN_Node::set_gradient(int y, int x, double gradient) {
    gradients[y][x] = gradient;
}
double** CNN_Node::get_gradients() {
    return gradients;
}


void CNN_Node::print(ostream &out) {
    out << "CNN_Node " << innovation_number << ", at depth: " << depth << " of size x: " << size_x << ", y: " << size_y << endl;

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
    //cout << "resetting node: " << innovation_number << endl;
    inputs_fired = 0;

    for (int32_t y = 0; y < size_y; y++) {
        for (int32_t x = 0; x < size_x; x++) {
            values[y][x] = 0;
            errors[y][x] = 0;
            gradients[y][x] = 0;
        }
    }
}

void CNN_Node::set_values(const Image &image, int rows, int cols) {
    if (rows != size_y) {
        cerr << "ERROR: rows of input image: " << rows << " != size_x of input node: " << size_y << endl;
        exit(1);
    }

    if (cols != size_x) {
        cerr << "ERROR: cols of input image: " << cols << " != size_x of input node: " << size_x << endl;
        exit(1);
    }

    //cout << "setting input image: " << endl;
    int current = 0;
    for (int32_t y = 0; y < size_y; y++) {
        for (int32_t x = 0; x < size_x; x++) {
            values[y][x] = image.get_pixel(x, y);
            current++;
            //cout << setw(5) << values[y][x];
        }
        //cout << endl;
    }
    //cout << endl;
}

void CNN_Node::save_best_bias() {
    for (int32_t y = 0; y < size_y; y++) {
        for (int32_t x = 0; x < size_x; x++) {
            best_bias[y][x] = bias[y][x];
        }
    }
}

void CNN_Node::set_bias_to_best() {
    for (int32_t y = 0; y < size_y; y++) {
        for (int32_t x = 0; x < size_x; x++) {
            bias[y][x] = best_bias[y][x];
        }
    }
}


void CNN_Node::resize_arrays(int previous_size_x, int previous_size_y) {
    //need to delete values and errors

    if (values == NULL) {
        cerr << "ERROR, modifying node size x but values == NULL" << endl;
        exit(1);
    }

    if (errors == NULL) {
        cerr << "ERROR, modifying node size x but errors == NULL" << endl;
        exit(1);
    }

    for (int32_t y = 0; y < previous_size_y; y++) {
        if (values[y] == NULL) {
            cerr << "ERROR, modifying node size x but values[" << y << "] == NULL" << endl;
            exit(1);
        }

        if (errors[y] == NULL) {
            cerr << "ERROR, modifying node size x but errors[" << y << "] == NULL" << endl;
            exit(1);
        }

        if (gradients[y] == NULL) {
            cerr << "ERROR, modifying node size x but gradients[" << y << "] == NULL" << endl;
            exit(1);
        }


        if (bias[y] == NULL) {
            cerr << "ERROR, modifying node size x but bias[" << y << "] == NULL" << endl;
            exit(1);
        }

        if (best_bias[y] == NULL) {
            cerr << "ERROR, modifying node size x but best_bias[" << y << "] == NULL" << endl;
            exit(1);
        }

        if (bias_velocity[y] == NULL) {
            cerr << "ERROR, modifying node size x but bias_velocity[" << y << "] == NULL" << endl;
            exit(1);
        }

        //cout << "resizing, deleting values[" << y << "]" << endl;
        delete [] values[y];
        //cout << "resizing, deleting values[" << y << "]" << endl;
        delete [] errors[y];
        //cout << "resizing, deleting values[" << y << "]" << endl;
        delete [] bias[y];
        //cout << "resizing, deleting best_bias[" << y << "]" << endl;
        delete [] best_bias[y];
        //cout << "resizing, deleting bias_velocity[" << y << "]" << endl;
        delete [] bias_velocity[y];
    }
    //cout << "resizing, deleting values" << endl;
    delete [] values;
    //cout << "resizing, deleting errors" << endl;
    delete [] errors;
    //cout << "resizing, deleting gradients" << endl;
    delete [] gradients;
    //cout << "resizing, deleting bias" << endl;
    delete [] bias;
    //cout << "resizing, deleting best_bias" << endl;
    delete [] best_bias;
    //cout << "resizing, deleting bias_velocity" << endl;
    delete [] bias_velocity;

    values = new double*[size_y];
    errors = new double*[size_y];
    gradients = new double*[size_y];
    bias = new double*[size_y];
    best_bias = new double*[size_y];
    bias_velocity = new double*[size_y];
    for (int32_t y = 0; y < size_y; y++) {
        values[y] = new double[size_x];
        errors[y] = new double[size_x];
        gradients[y] = new double[size_x];
        bias[y] = new double[size_x];
        best_bias[y] = new double[size_x];
        bias_velocity[y] = new double[size_x];
        for (int32_t x = 0; x < size_x; x++) {
            values[y][x] = 0.0;
            errors[y][x] = 0.0;
            gradients[y][x] = 0.0;
            bias[y][x] = 0.0;
            best_bias[y][x] = 0.0;
            bias_velocity[y][x] = 0.0;
        }
    }
}


bool CNN_Node::modify_size_x(int change, minstd_rand0 &generator, NormalDistribution &normal_distribution) {
    int previous_size_x = size_x;

    size_x += change;

    //make sure the size doesn't drop below 1
    if (size_x <= 0) size_x = 1;
    if (size_x == previous_size_x) return false;

    resize_arrays(previous_size_x, size_y);
    initialize_bias(generator, normal_distribution);
    save_best_bias(); //save the new random weights so they are resused by this child

    return true;
}

bool CNN_Node::modify_size_y(int change, minstd_rand0 &generator, NormalDistribution &normal_distribution) {
    int previous_size_y = size_y;

    size_y += change;

    //make sure the size doesn't drop below 1
    if (size_y <= 0) size_y = 1;
    if (size_y == previous_size_y) return false;

    resize_arrays(size_x, previous_size_y);
    initialize_bias(generator, normal_distribution);
    save_best_bias(); //save the new random weights so they are resused by this child

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



void CNN_Node::input_fired() {
    inputs_fired++;

    //cout << "input fired on node: " << innovation_number << ", inputs fired: " << inputs_fired << ", total_inputs: " << total_inputs << endl;

    if (inputs_fired == total_inputs) {
        if (type != SOFTMAX_NODE) {
            for (int32_t y = 0; y < size_y; y++) {
                for (int32_t x = 0; x < size_x; x++) {
                    //values[y][x] += bias[y][x];
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
    double dx, pv, velocity;

    for (int32_t y = 0; y < size_y; y++) {
        for (int32_t x = 0; x < size_x; x++) {
            //double dx = LEARNING_RATE * (weight_update[k][l] / (size_x * size_y) + (weights[k][l] * WEIGHT_DECAY));
            //L2 regularization

            dx = learning_rate * -errors[y][x] - (bias[y][x] * weight_decay);
            //double dx = LEARNING_RATE * (weight_update[k][l] / (size_x * size_y));

            //no momemntum
            //weights[k][l] += dx;

            //momentum
            pv = bias_velocity[y][x];
            velocity = (mu * pv) - dx;
            bias[y][x] -= -mu * pv + (1 + mu) * velocity;
            bias_velocity[y][x] = velocity;

            if (bias[y][x] < -50.0) bias[y][x] = -50.0;
            if (bias[y][x] > 50.0) bias[y][x] = 50.0;
        }
    }
}

void CNN_Node::print_statistics() {
    double value_min = std::numeric_limits<double>::max(), value_max = -std::numeric_limits<double>::max(), value_avg = 0.0;
    double error_min = std::numeric_limits<double>::max(), error_max = -std::numeric_limits<double>::max(), error_avg = 0.0;

    for (int y = 0; y < size_y; y++) {
        for (int x = 0; x < size_x; x++) {
            if (values[y][x] < value_min) value_min = values[y][x];
            if (values[y][x] > value_max) value_max = values[y][x];
            value_avg += values[y][x];

            if (errors[y][x] < error_min) error_min = errors[y][x];
            if (errors[y][x] > error_max) error_max = errors[y][x];
            error_avg += errors[y][x];
        }
    }

    error_avg /= size_y * size_x;
    value_avg /= size_y * size_x;

    cerr << "node " << setw(4) << innovation_number << ", v_min: " << value_min << ", v_avg: " << value_avg << ", v_max: " << value_max << ", e_min: " << error_min << ", e_avg: " << error_avg << ", e_max: " << error_max << endl;
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
    os << node->total_inputs << endl;

    for (int32_t y = 0; y < node->size_y; y++) {
        for (int32_t x = 0; x < node->size_x; x++) {
            if (y > 0 || x > 0) os << " ";
            os << setprecision(15) << node->bias[y][x];
        }
    }
    os << endl;

    for (int32_t y = 0; y < node->size_y; y++) {
        for (int32_t x = 0; x < node->size_x; x++) {
            if (y > 0 || x > 0) os << " ";
            os << setprecision(15) << node->best_bias[y][x];
        }
    }
    os << endl;

    for (int32_t y = 0; y < node->size_y; y++) {
        for (int32_t x = 0; x < node->size_x; x++) {
            if (y > 0 || x > 0) os << " ";
            os << setprecision(15) << node->bias_velocity[y][x];
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
    is >> node->total_inputs;

    node->total_inputs = 0;
    node->inputs_fired = 0;
    node->visited = false;

    node->values = new double*[node->size_y];
    for (int32_t y = 0; y < node->size_y; y++) {
        node->values[y] = new double[node->size_x];
        for (int32_t x = 0; x < node->size_x; x++) {
            node->values[y][x] = 0.0;
        }
    }

    node->errors = new double*[node->size_y];
    for (int32_t y = 0; y < node->size_y; y++) {
        node->errors[y] = new double[node->size_x];
        for (int32_t x = 0; x < node->size_x; x++) {
            node->errors[y][x] = 0.0;
        }
    }

    node->gradients = new double*[node->size_y];
    for (int32_t y = 0; y < node->size_y; y++) {
        node->gradients[y] = new double[node->size_x];
        for (int32_t x = 0; x < node->size_x; x++) {
            node->gradients[y][x] = 0.0;
        }
    }

    double b;
    node->bias = new double*[node->size_y];
    for (int32_t y = 0; y < node->size_y; y++) {
        node->bias[y] = new double[node->size_x];
        for (int32_t x = 0; x < node->size_x; x++) {
            is >> b;
            node->bias[y][x] = 0.0;
        }
    }

    node->best_bias = new double*[node->size_y];
    for (int32_t y = 0; y < node->size_y; y++) {
        node->best_bias[y] = new double[node->size_x];
        for (int32_t x = 0; x < node->size_x; x++) {
            is >> b;
            node->best_bias[y][x] = 0.0;
        }
    }

    node->bias_velocity = new double*[node->size_y];
    for (int32_t y = 0; y < node->size_y; y++) {
        node->bias_velocity[y] = new double[node->size_x];
        for (int32_t x = 0; x < node->size_x; x++) {
            is >> b;
            node->bias_velocity[y][x] = 0.0;
        }
    }

    return is;
}


