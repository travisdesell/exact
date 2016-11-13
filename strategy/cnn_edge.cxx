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


CNN_Edge::CNN_Edge() {
    innovation_number = -1;

    input_node_innovation_number = -1;
    output_node_innovation_number = -1;

    input_node = NULL;
    output_node = NULL;
}

CNN_Edge::CNN_Edge(CNN_Node *_input_node, CNN_Node *_output_node, bool _fixed, int _innovation_number, mt19937 &generator) {
    fixed = _fixed;
    innovation_number = _innovation_number;
    disabled = false;

    input_node = _input_node;
    output_node = _output_node;

    input_node_innovation_number = input_node->get_innovation_number();
    output_node_innovation_number = output_node->get_innovation_number();

    output_node->add_input();

    filter_x = (input_node->get_size_x() - output_node->get_size_x()) + 1;
    filter_y = (input_node->get_size_y() - output_node->get_size_y()) + 1;

    weights = vector< vector<double> >(filter_y, vector<double>(filter_x, 0.0));
    weight_update = vector< vector<double> >(filter_y, vector<double>(filter_x, 0.0));
    previous_velocity = vector< vector<double> >(filter_y, vector<double>(filter_x, 0.0));

    int edge_size = filter_x * filter_y;
    if (edge_size == 1) edge_size = 10;
    normal_distribution<double> distribution(0.0, sqrt(2.0 / edge_size) );

    for (uint32_t i = 0; i < weights.size(); i++) {
        for (uint32_t j = 0; j < weights[i].size(); j++) {
            weights[i][j] = distribution(generator);
        }
    }
}

CNN_Edge* CNN_Edge::copy() const {
    CNN_Edge* copy = new CNN_Edge();

    copy->fixed = fixed;
    copy->innovation_number = innovation_number;
    copy->disabled = disabled;

    copy->input_node = input_node;
    copy->output_node = output_node;

    copy->input_node_innovation_number = input_node->get_innovation_number();
    copy->output_node_innovation_number = output_node->get_innovation_number();

    copy->filter_x = filter_x;
    copy->filter_y = filter_y;

    copy->weights = vector< vector<double> >(filter_y, vector<double>(filter_x, 0.0));
    copy->weight_update = vector< vector<double> >(filter_y, vector<double>(filter_x, 0.0));
    copy->previous_velocity = vector< vector<double> >(filter_y, vector<double>(filter_x, 0.0));

    for (uint32_t y = 0; y < weights.size(); y++) {
        for (uint32_t x = 0; x < weights[y].size(); x++) {
            copy->weights[y][x] = weights[y][x];
            copy->weight_update[y][x] = weight_update[y][x];
            copy->previous_velocity[y][x] = previous_velocity[y][x];
        }
    }

    return copy;
}

void CNN_Edge::set_nodes(const vector<CNN_Node*> nodes) {
    cout << "nodes.size(): " << nodes.size() << endl;
    cout << "setting input node: " << input_node_innovation_number << endl;
    cout << "setting output node: " << output_node_innovation_number << endl;

    input_node = nodes[input_node_innovation_number];
    output_node = nodes[output_node_innovation_number];
}

int CNN_Edge::get_number_weights() const {
    return filter_x * filter_y;
}

int CNN_Edge::get_innovation_number() const {
    return innovation_number;
}

const CNN_Node* CNN_Edge::get_input_node() const {
    return input_node;
}

const CNN_Node* CNN_Edge::get_output_node() const {
    return output_node;
}

void CNN_Edge::print(ostream &out) {
    out << "CNN_Edge " << innovation_number << " of from node " << input_node->get_innovation_number() << " to node " << output_node->get_innovation_number() << " with filter x: " << filter_x << ", y: " << filter_y << endl;

    for (uint32_t i = 0; i < weights.size(); i++) {
        out << "    ";
        for (uint32_t j = 0; j < weights[i].size(); j++) {
            out << setw(9) << setprecision(3) << weights[i][j];
        }
        out << endl;
    }
}

void CNN_Edge::propagate_forward() {
    double **input = input_node->get_values();
    double **output = output_node->get_values();

    if (filter_x != (input_node->get_size_x() - output_node->get_size_x()) + 1) {
        cerr << "ERRROR: filter_x != input_node->get_size_x: " << input_node->get_size_x() << " - output_node->get_size_x: " << output_node->get_size_x() << " + 1" << endl;
        exit(1);
    }

    if (filter_y != (input_node->get_size_y() - output_node->get_size_y()) + 1) {
        cerr << "ERRROR: filter_y != input_node->get_size_y: " << input_node->get_size_y() << " - output_node->get_size_y: " << output_node->get_size_y() << " + 1" << endl;
        exit(1);
    }

    for (uint32_t k = 0; k < filter_y; k++) {
        for (uint32_t l = 0; l < filter_x; l++) {
            weight_update[k][l] = 0.0;
        }
    }

    for (uint32_t i = 0; i + filter_y - 1 < input_node->get_size_y(); i++) {
        for (uint32_t j = 0; j + filter_x - 1 < input_node->get_size_x(); j++) {
            for (uint32_t k = 0; k < filter_y; k++) {
                for (uint32_t l = 0; l < filter_x; l++) {
                    double value = weights[k][l] * input[i + k][j + l];
                    output[i][j] += value;

                    //                            cout << "weights[" << k << "][" << l <<"]: " << weights[k][l] << " * input[" << i << " + " << k << "][" << j << " + " << l << "] = " << value << endl;
                    //                            cout << "output[" << i << "][" << j << "]: " << output[i][j] << endl;
                }
            }
        }
    }

    output_node->input_fired();
}

void CNN_Edge::propagate_backward(double mu) {
    double **input = input_node->get_values();
    double **output_errors = output_node->get_errors();
    double **input_errors = input_node->get_errors();

    for (uint32_t i = 0; i + filter_y - 1 < input_node->get_size_y(); i++) {
        for (uint32_t j = 0; j + filter_x - 1 < input_node->get_size_x(); j++) {
            for (uint32_t k = 0; k < filter_y; k++) {
                for (uint32_t l = 0; l < filter_x; l++) {
                    double error = output_errors[i][j];

                    /*
                       int error_sign = 1;
                       if (error < 0) error_sign = -1;

                       double gradient = 1;
                       if (input[i][j] * weights[k][l] < RELU_MIN) gradient = RELU_MIN_LEAK;
                    // else if (input[i][j] * weights[k][l] > RELU_MAX) gradient = RELU_MAX_LEAK;
                    */

                    double update = input[i + k][j + l] * error;
                    weight_update[k][l] -= update;
                    //cout << input_node->get_innovation_number() << " to " << output_node->get_innovation_number() << " -- in: " << input << ", w: " << weights[0][0] << ", err: " << error << ", update: " << weight_update << endl;

                    input_errors[i + k][j + l] += error * weights[k][l];
                }
            }
        }
    }

    double learning_rate = 0.0005;
    double weight_decay = 0.0005;
    for (uint32_t k = 0; k < filter_y; k++) {
        for (uint32_t l = 0; l < filter_x; l++) {
            //double dx = learning_rate * (weight_update[k][l] / (filter_x * filter_y) + (weights[k][l] * weight_decay));
            //L2 regularization
            double dx = learning_rate * (weight_update[k][l] / (filter_x * filter_y) - (weights[k][l] * weight_decay));
            //double dx = learning_rate * (weight_update[k][l] / (filter_x * filter_y));

            //no momemntum
            //weights[k][l] += dx;

            //momentum
            double velocity = (mu * previous_velocity[k][l]) - dx;
            weights[k][l] -= -mu * previous_velocity[k][l] + (1 + mu) * velocity;
            previous_velocity[k][l] = velocity;

            /*
               if (weights[k][l] < -1) weights[k][l] = -1;
               if (weights[k][l] > 1) weights[k][l] = 1;
               */
        }
    }
}

ostream &operator<<(ostream &os, const CNN_Edge* edge) {
    os << edge->innovation_number << " ";
    os << edge->input_node_innovation_number << " " << edge->output_node_innovation_number << " ";
    os << edge->filter_x << " " << edge->filter_y << " ";
    os << edge->fixed << " ";
    os << edge->disabled << endl;

    for (uint32_t y = 0; y < edge->filter_y; y++) {
        for (uint32_t x = 0; x < edge->filter_x; x++) {
            if (y > 0 || x > 0) os << " ";
            os << setprecision(15) << edge->weights[y][x];
        }
    }
    os << endl;

    for (uint32_t y = 0; y < edge->filter_y; y++) {
        for (uint32_t x = 0; x < edge->filter_x; x++) {
            if (y > 0 || x > 0) os << " ";
            os << setprecision(15) << edge->previous_velocity[y][x];
        }
    }

    return os;
}

istream &operator>>(istream &is, CNN_Edge* edge) {
    is >> edge->innovation_number;
    is >> edge->input_node_innovation_number;
    is >> edge->output_node_innovation_number;
    is >> edge->filter_x;
    is >> edge->filter_y;
    is >> edge->fixed;
    is >> edge->disabled;

    edge->weights = vector< vector<double> >(edge->filter_y, vector<double>(edge->filter_x, 0.0));
    edge->weight_update = vector< vector<double> >(edge->filter_y, vector<double>(edge->filter_x, 0.0));
    edge->previous_velocity = vector< vector<double> >(edge->filter_y, vector<double>(edge->filter_x, 0.0));

    for (uint32_t y = 0; y < edge->filter_y; y++) {
        for (uint32_t x = 0; x < edge->filter_x; x++) {
            is >> edge->weights[y][x];
        }
    }

    for (uint32_t y = 0; y < edge->filter_y; y++) {
        for (uint32_t x = 0; x < edge->filter_x; x++) {
            is >> edge->previous_velocity[y][x];
        }
    }

    return is;
}



