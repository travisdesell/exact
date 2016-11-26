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
using std::mt19937;
using std::normal_distribution;


#include <string>
using std::string;

#include <vector>
using std::vector;

#include "image_tools/image_set.hxx"
#include "cnn_edge.hxx"
#include "cnn_node.hxx"


CNN_Node::CNN_Node() {
    innovation_number = -1;
    values = NULL;
    errors = NULL;
    bias = NULL;
    bias_velocity = NULL;
    visited = false;
}

CNN_Node::CNN_Node(int _innovation_number, double _depth, int _size_x, int _size_y, int _type) {
    innovation_number = _innovation_number;
    depth = _depth;
    type = _type;
    size_x = _size_x;
    size_y = _size_y;

    total_inputs = 0;
    inputs_fired = 0;

    visited = false;

    values = new double*[size_y];
    for (uint32_t y = 0; y < size_y; y++) {
        values[y] = new double[size_x];
        for (uint32_t x = 0; x < size_x; x++) {
            values[y][x] = 0.0;
        }
    }

    errors = new double*[size_y];
    for (uint32_t y = 0; y < size_y; y++) {
        errors[y] = new double[size_x];
        for (uint32_t x = 0; x < size_x; x++) {
            errors[y][x] = 0.0;
        }
    }

    bias = new double*[size_y];
    for (uint32_t y = 0; y < size_y; y++) {
        bias[y] = new double[size_x];
        for (uint32_t x = 0; x < size_x; x++) {
            bias[y][x] = 0.0;
        }
    }

    bias_velocity = new double*[size_y];
    for (uint32_t y = 0; y < size_y; y++) {
        bias_velocity[y] = new double[size_x];
        for (uint32_t x = 0; x < size_x; x++) {
            bias_velocity[y][x] = 0.0;
        }
    }
}

CNN_Node* CNN_Node::copy() const {
    CNN_Node *copy = new CNN_Node();

    copy->innovation_number = innovation_number;
    copy->depth = depth;
    copy->size_x = size_x;
    copy->size_y = size_y;

    copy->type = type;

    copy->total_inputs = 0; //this will be updated when edges are set
    copy->inputs_fired = inputs_fired;

    copy->visited = visited;

    copy->values = new double*[size_y];
    for (uint32_t y = 0; y < size_y; y++) {
        copy->values[y] = new double[size_x];
        for (uint32_t x = 0; x < size_x; x++) {
            copy->values[y][x] = values[y][x];
        }
    }

    copy->errors = new double*[size_y];
    for (uint32_t y = 0; y < size_y; y++) {
        copy->errors[y] = new double[size_x];
        for (uint32_t x = 0; x < size_x; x++) {
            copy->errors[y][x] = errors[y][x];
        }
    }

    copy->bias = new double*[size_y];
    for (uint32_t y = 0; y < size_y; y++) {
        copy->bias[y] = new double[size_x];
        for (uint32_t x = 0; x < size_x; x++) {
            copy->bias[y][x] = bias[y][x];
        }
    }

    copy->bias_velocity = new double*[size_y];
    for (uint32_t y = 0; y < size_y; y++) {
        copy->bias_velocity[y] = new double[size_x];
        for (uint32_t x = 0; x < size_x; x++) {
            copy->bias_velocity[y][x] = bias_velocity[y][x];
        }
    }

    return copy;
}

void CNN_Node::initialize_bias(mt19937 &generator) {
    int bias_size = size_x * size_y;
    if (bias_size == 1) bias_size = 10;
    normal_distribution<double> distribution(0.0, sqrt(2.0 / bias_size) );

    for (uint32_t i = 0; i < size_y; i++) {
        for (uint32_t j = 0; j < size_x; j++) {
            bias[i][j] = distribution(generator);
            //cout << "node " << innovation_number << " bias[" << i << "][" << j <<"]: " << bias[i][j] << endl;
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

void CNN_Node::print(ostream &out) {
    out << "CNN_Node " << innovation_number << ", at depth: " << depth << " of size x: " << size_x << ", y: " << size_y << endl;

    for (uint32_t i = 0; i < size_y; i++) {
        out << "    ";
        for (uint32_t j = 0; j < size_x; j++) {
            out << setw(10) << setprecision(8) << values[i][j];
        }
        out << endl;
    }

    for (uint32_t i = 0; i < size_y; i++) {
        out << "    ";
        for (uint32_t j = 0; j < size_x; j++) {
            out << setw(10) << setprecision(8) << errors[i][j];
        }
        out << endl;
    }

    for (uint32_t i = 0; i < size_y; i++) {
        out << "    ";
        for (uint32_t j = 0; j < size_x; j++) {
            out << setw(10) << setprecision(8) << bias[i][j];
        }
        out << endl;
    }

    for (uint32_t i = 0; i < size_y; i++) {
        out << "    ";
        for (uint32_t j = 0; j < size_x; j++) {
            out << setw(10) << setprecision(8) << bias_velocity[i][j];
        }
        out << endl;
    }
}

void CNN_Node::reset() {
    //cout << "resetting node: " << innovation_number << endl;
    inputs_fired = 0;

    for (uint32_t y = 0; y < size_y; y++) {
        for (uint32_t x = 0; x < size_x; x++) {
            values[y][x] = 0;
        }
    }

    for (uint32_t y = 0; y < size_y; y++) {
        for (uint32_t x = 0; x < size_x; x++) {
            errors[y][x] = 0;
        }
    }

    for (uint32_t y = 0; y < size_y; y++) {
        for (uint32_t x = 0; x < size_x; x++) {
            bias_velocity[y][x] = 0;
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
    for (uint32_t y = 0; y < size_y; y++) {
        for (uint32_t x = 0; x < size_x; x++) {
            values[y][x] = image.get_pixel(x, y) / 255.0;
            current++;
            //cout << setw(5) << values[y][x];
        }
        //cout << endl;
    }
    //cout << endl;
}

bool CNN_Node::modify_size_x(int change, mt19937 &generator) {
    int previous_size_x = size_x;

    size_x += change;

    //make sure the size doesn't drop below 1
    if (size_x <= 0) size_x = 1;
    if (size_x == previous_size_x) return false;

    //need to delete values and errors

    if (values == NULL) {
        cerr << "ERROR, modifying node size x but values == NULL" << endl;
        exit(1);
    }

    if (errors == NULL) {
        cerr << "ERROR, modifying node size x but errors == NULL" << endl;
        exit(1);
    }

    for (uint32_t y = 0; y < size_y; y++) {
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
        delete [] bias[y];
        delete [] bias_velocity[y];
    }
    delete [] values;
    delete [] errors;
    delete [] bias;
    delete [] bias_velocity;

    values = new double*[size_y];
    for (uint32_t y = 0; y < size_y; y++) {
        values[y] = new double[size_x];
        for (uint32_t x = 0; x < size_x; x++) {
            values[y][x] = 0.0;
        }
    }

    errors = new double*[size_y];
    for (uint32_t y = 0; y < size_y; y++) {
        errors[y] = new double[size_x];
        for (uint32_t x = 0; x < size_x; x++) {
            errors[y][x] = 0.0;
        }
    }

    bias = new double*[size_y];
    for (uint32_t y = 0; y < size_y; y++) {
        bias[y] = new double[size_x];
    }
    initialize_bias(generator);

    bias_velocity = new double*[size_y];
    for (uint32_t y = 0; y < size_y; y++) {
        bias_velocity[y] = new double[size_x];
        for (uint32_t x = 0; x < size_x; x++) {
            bias_velocity[y][x] = 0.0;
        }
    }

    return true;
}

bool CNN_Node::modify_size_y(int change, mt19937 &generator) {
    int previous_size_y = size_y;

    size_y += change;

    //make sure the size doesn't drop below 1
    if (size_y <= 0) size_y = 1;
    if (size_y == previous_size_y) return false;

    //need to delete values and errors

    if (values == NULL) {
        cerr << "ERROR, modifying node size y but values == NULL" << endl;
        exit(1);
    }

    if (errors == NULL) {
        cerr << "ERROR, modifying node size y but errors == NULL" << endl;
        exit(1);
    }

    for (uint32_t y = 0; y < previous_size_y; y++) {
        if (values[y] == NULL) {
            cerr << "ERROR, modifying node size y but values[" << y << "] == NULL" << endl;
            exit(1);
        }

        if (errors[y] == NULL) {
            cerr << "ERROR, modifying node size y but values[" << y << "] == NULL" << endl;
            exit(1);
        }

        delete [] values[y];
        delete [] errors[y];
        delete [] bias[y];
        delete [] bias_velocity[y];
    }
    delete [] values;
    delete [] errors;
    delete [] bias;
    delete [] bias_velocity;

    values = new double*[size_y];
    for (uint32_t y = 0; y < size_y; y++) {
        values[y] = new double[size_x];
        for (uint32_t x = 0; x < size_x; x++) {
            values[y][x] = 0.0;
        }
    }

    errors = new double*[size_y];
    for (uint32_t y = 0; y < size_y; y++) {
        errors[y] = new double[size_x];
        for (uint32_t x = 0; x < size_x; x++) {
            errors[y][x] = 0.0;
        }
    }

    bias = new double*[size_y];
    for (uint32_t y = 0; y < size_y; y++) {
        bias[y] = new double[size_x];
    }
    initialize_bias(generator);

    bias_velocity = new double*[size_y];
    for (uint32_t y = 0; y < size_y; y++) {
        bias_velocity[y] = new double[size_x];
        for (uint32_t x = 0; x < size_x; x++) {
            bias_velocity[y][x] = 0.0;
        }
    }

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


void CNN_Node::input_fired() {
    inputs_fired++;

    //cout << "input fired on node: " << innovation_number << ", inputs fired: " << inputs_fired << ", total_inputs: " << total_inputs << endl;

    if (inputs_fired == total_inputs) {
        //cout << "applying activation function to node!" << endl;
        //print(cout);

        //if (type != SOFTMAX_NODE) {
            for (uint32_t y = 0; y < size_y; y++) {
                for (uint32_t x = 0; x < size_x; x++) {
                    values[y][x] += bias[y][x];
                    //cout << "values for node " << innovation_number << " now " << values[y][x] << " after adding bias: " << bias[y][x] << endl;

                    //apply activation function
                    if (values[y][x] <= RELU_MIN) {
                        values[y][x] *= RELU_MIN_LEAK;
                    }

                    if (values[y][x] > RELU_MAX) {
                        values[y][x] = ((values[y][x] - RELU_MAX) * RELU_MAX_LEAK) + RELU_MAX;
                    }
                }
            }
        //}
    }
}

void CNN_Node::propagate_bias(double mu) {
    double dx, pv, velocity;

    for (uint32_t y = 0; y < size_y; y++) {
        for (uint32_t x = 0; x < size_x; x++) {
            //double dx = LEARNING_RATE * (weight_update[k][l] / (size_x * size_y) + (weights[k][l] * WEIGHT_DECAY));
            //L2 regularization

            dx = LEARNING_RATE * -errors[y][x] - (bias[y][x] * WEIGHT_DECAY);
            //double dx = LEARNING_RATE * (weight_update[k][l] / (size_x * size_y));

            //no momemntum
            //weights[k][l] += dx;

            //momentum
            pv = bias_velocity[y][x];
            velocity = (mu * pv) - dx;
            bias[y][x] -= -mu * pv + (1 + mu) * velocity;
            bias_velocity[y][x] = velocity;
        }
    }
}

ostream &operator<<(ostream &os, const CNN_Node* node) {
    os << node->innovation_number << " ";
    os << node->depth << " ";
    os << node->size_x << " ";
    os << node->size_y << " ";
    os << node->type << " ";
    os << node->visited << " ";
    os << node->total_inputs << endl;

    for (uint32_t y = 0; y < node->size_y; y++) {
        for (uint32_t x = 0; x < node->size_x; x++) {
            if (y > 0 || x > 0) os << " ";
            os << setprecision(15) << node->bias[y][x];
        }
    }
    os << endl;

    for (uint32_t y = 0; y < node->size_y; y++) {
        for (uint32_t x = 0; x < node->size_x; x++) {
            if (y > 0 || x > 0) os << " ";
            os << setprecision(15) << node->bias_velocity[y][x];
        }
    }

    return os;
}

std::istream &operator>>(std::istream &is, CNN_Node* node) {
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
    for (uint32_t y = 0; y < node->size_y; y++) {
        node->values[y] = new double[node->size_x];
        for (uint32_t x = 0; x < node->size_x; x++) {
            node->values[y][x] = 0.0;
        }
    }

    node->errors = new double*[node->size_y];
    for (uint32_t y = 0; y < node->size_y; y++) {
        node->errors[y] = new double[node->size_x];
        for (uint32_t x = 0; x < node->size_x; x++) {
            node->errors[y][x] = 0.0;
        }
    }

    double b;
    node->bias = new double*[node->size_y];
    for (uint32_t y = 0; y < node->size_y; y++) {
        node->bias[y] = new double[node->size_x];
        for (uint32_t x = 0; x < node->size_x; x++) {
            is >> b;
            node->bias[y][x] = 0.0;
        }
    }

    node->bias_velocity = new double*[node->size_y];
    for (uint32_t y = 0; y < node->size_y; y++) {
        node->bias_velocity[y] = new double[node->size_x];
        for (uint32_t x = 0; x < node->size_x; x++) {
            is >> b;
            node->bias_velocity[y][x] = 0.0;
        }
    }

    return is;
}


