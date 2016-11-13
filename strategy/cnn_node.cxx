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

#include <string>
using std::string;

#include <vector>
using std::vector;

#include "image_tools/image_set.hxx"
#include "cnn_node.hxx"


CNN_Node::CNN_Node() {
    innovation_number = -1;
    values = NULL;
    errors = NULL;
}

CNN_Node::CNN_Node(int _innovation_number, int _depth, int _size_x, int _size_y, bool _input, bool _output, bool _softmax) {
    innovation_number = _innovation_number;
    depth = _depth;
    input = _input;
    output = _output;
    softmax = _softmax;
    size_x = _size_x;
    size_y = _size_y;

    total_inputs = 0;
    inputs_fired = 0;

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
}

CNN_Node* CNN_Node::copy() const {
    CNN_Node *copy = new CNN_Node();

    copy->innovation_number = innovation_number;
    copy->depth = depth;
    copy->size_x = size_x;
    copy->size_y = size_y;

    copy->input = input;
    copy->output = output;
    copy->softmax = softmax;

    copy->total_inputs = total_inputs;
    copy->inputs_fired = inputs_fired;

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

    return copy;
}

bool CNN_Node::is_fixed() const {
    return input || output || softmax;
}

bool CNN_Node::is_hidden() const {
    return !(input || output || softmax);
}


bool CNN_Node::is_input() const {
    return input;
}


bool CNN_Node::is_output() const {
    return output;
}


bool CNN_Node::is_softmax() const {
    return output;
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

int CNN_Node::get_depth() const {
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


void CNN_Node::add_input() {
    total_inputs++;
}

void CNN_Node::input_fired() {
    inputs_fired++;

    //cout << "input fired on node: " << innovation_number << ", inputs fired: " << inputs_fired << ", total_inputs: " << total_inputs << endl;

    if (inputs_fired == total_inputs) {
        //cout << "applying activation function to node!" << endl;
        //print(cout);

        if (!softmax) {
            for (uint32_t y = 0; y < size_y; y++) {
                for (uint32_t x = 0; x < size_x; x++) {
                    //apply activation function
                    if (values[y][x] <= RELU_MIN) {
                        values[y][x] *= RELU_MIN_LEAK;
                    }

                    if (values[y][x] > RELU_MAX) {
                        values[y][x] = ((values[y][x] - RELU_MAX) * RELU_MAX_LEAK) + RELU_MAX;
                    }
                }
            }
        }
    }
}


ostream &operator<<(ostream &os, const CNN_Node* node) {
    os << node->innovation_number << " ";
    os << node->depth << " ";
    os << node->size_x << " ";
    os << node->size_y << " ";
    os << node->input << " ";
    os << node->output << " ";
    os << node->softmax << " ";
    os << node->total_inputs;

    return os;
}

std::istream &operator>>(std::istream &is, CNN_Node* node) {
    is >> node->innovation_number;
    is >> node->depth;
    is >> node->size_x;
    is >> node->size_y;
    is >> node->input;
    is >> node->output;
    is >> node->softmax;
    is >> node->total_inputs;

    node->inputs_fired = 0;

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

    return is;
}


