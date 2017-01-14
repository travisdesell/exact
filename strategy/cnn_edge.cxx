#include <cmath>
//for isnan
using std::isnan;

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

#include <string>
using std::string;

#include <vector>
using std::vector;

#include "image_tools/image_set.hxx"
#include "cnn_edge.hxx"
#include "cnn_node.hxx"

#include "stdint.h"
CNN_Edge::CNN_Edge() {
    edge_id = -1;
    exact_id = -1;
    genome_id = -1;

    innovation_number = -1;

    input_node_innovation_number = -1;
    output_node_innovation_number = -1;

    input_node = NULL;
    output_node = NULL;
}

CNN_Edge::CNN_Edge(CNN_Node *_input_node, CNN_Node *_output_node, bool _fixed, int _innovation_number) {
    edge_id = -1;
    exact_id = -1;
    genome_id = -1;

    fixed = _fixed;
    innovation_number = _innovation_number;
    disabled = false;
    reverse_filter_x = false;
    reverse_filter_y = false;

    input_node = _input_node;
    output_node = _output_node;

    input_node_innovation_number = input_node->get_innovation_number();
    output_node_innovation_number = output_node->get_innovation_number();

    if (!disabled) output_node->add_input();

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

    //cout << "\t\tcreated edge " << innovation_number << " (node " << input_node_innovation_number << " to " << output_node_innovation_number << ") with filter_x: " << filter_x << " (input: " << input_node->get_size_x() << ", output: " << output_node->get_size_x() << ") and filter_y: " << filter_y << " (input: " << input_node->get_size_y() << ", output: " << output_node->get_size_y() << "), reverse filter: " << reverse_filter_x << ", reverse_filter_y: " << reverse_filter_y << endl;

    weights = vector< vector<double> >(filter_y, vector<double>(filter_x, 0.0));
    best_weights = vector< vector<double> >(filter_y, vector<double>(filter_x, 0.0));
    previous_velocity = vector< vector<double> >(filter_y, vector<double>(filter_x, 0.0));
}

CNN_Edge::~CNN_Edge() {
    input_node = NULL;
    output_node = NULL;
}

template <class T>
void parse_vector_2d(vector< vector<T> > &output, istringstream &iss, int filter_x, int filter_y) {
    output.clear();
    output = vector< vector<T> >(filter_y, vector<T>(filter_x));

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
        output[current_y][current_x] = val;

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

        exact_id = atoi(row[1]);
        genome_id = atoi(row[2]);
        innovation_number = atoi(row[3]);

        input_node_innovation_number = atoi(row[4]);
        output_node_innovation_number = atoi(row[5]);

        filter_x = atoi(row[6]);
        filter_y = atoi(row[7]);

        istringstream weights_iss(row[8]);
        parse_vector_2d(weights, weights_iss, filter_x, filter_y);

        istringstream best_weights_iss(row[9]);
        parse_vector_2d(best_weights, best_weights_iss, filter_x, filter_y);

        istringstream previous_velocity_iss(row[10]);
        parse_vector_2d(previous_velocity, previous_velocity_iss, filter_x, filter_y);

        fixed = atoi(row[11]);
        disabled = atoi(row[12]);
        reverse_filter_x = atoi(row[13]);
        reverse_filter_y = atoi(row[14]);

        mysql_free_result(result);
    } else {
        cerr << "ERROR! Could not find cnn_edge in database with edge id: " << edge_id << endl;
        exit(1);
    }

    //cout << "read edge!" << endl;
    //cout << this << endl;
}

void CNN_Edge::export_to_database(int _exact_id, int _genome_id) {
    ostringstream query;

    genome_id = _genome_id;
    exact_id = _exact_id;

    if (edge_id >= 0) {
        query << "REPLACE INTO cnn_edge SET id = " << edge_id << ",";
    } else {
        query << "INSERT INTO cnn_edge SET";
    }

    query << " exact_id = " << exact_id
        << ", genome_id = " << genome_id
        << ", innovation_number = " << innovation_number
        << ", input_node_innovation_number = " << input_node_innovation_number
        << ", output_node_innovation_number = " << output_node_innovation_number
        << ", filter_x = " << filter_x
        << ", filter_y = " << filter_y
        << ", fixed = " << fixed
        << ", disabled = " << disabled
        << ", reverse_filter_x = " << reverse_filter_x
        << ", reverse_filter_y = " << reverse_filter_y
        << ", weights = '";

    for (int32_t y = 0; y < filter_y; y++) {
        for (int32_t x = 0; x < filter_x; x++) {
            if (x != 0) query << " ";
            query << setprecision(15) << weights[y][x];
        }
        if (y != filter_y - 1) query << "\n";
    }

    query << "', best_weights = '";
    for (int32_t y = 0; y < filter_y; y++) {
        for (int32_t x = 0; x < filter_x; x++) {
            if (x != 0) query << " ";
            query << setprecision(15) << best_weights[y][x];
        }
        if (y != filter_y - 1) query << "\n";
    }

    query << "', previous_velocity = '";
    for (int32_t y = 0; y < filter_y; y++) {
        for (int32_t x = 0; x < filter_x; x++) {
            if (x != 0) query << " ";
            query << setprecision(15) << previous_velocity[y][x];
        }
        if (y != filter_y - 1) query << "\n";
    }
    query << "'";

    mysql_exact_query(query.str());

    if (edge_id < 0) {
        edge_id = mysql_exact_last_insert_id();
        //cout << "set edge id to " << edge_id << endl;
    }
}
#endif

bool CNN_Edge::equals(CNN_Edge *other) const {
    return filter_x == other->filter_x && filter_y == other->filter_y && disabled == other->disabled && reverse_filter_x == other->reverse_filter_x && reverse_filter_y == other->reverse_filter_y;
}

int CNN_Edge::get_filter_x() const {
    return filter_x;
}

int CNN_Edge::get_filter_y() const {
    return filter_y;
}


void CNN_Edge::initialize_weights(minstd_rand0 &generator, NormalDistribution &normal_distribution) {
    if (disabled) return;

    int edge_size = filter_x * filter_y;
    if (edge_size == 1) edge_size = 10;

    double mu = 0.0;
    //double sigma = sqrt(2.0 / edge_size);
    double sigma = 2.0 / edge_size;

    for (uint32_t i = 0; i < weights.size(); i++) {
        for (uint32_t j = 0; j < weights[i].size(); j++) {
            weights[i][j] = normal_distribution.random(generator, mu, sigma);
            best_weights[i][j] = 0.0;
            previous_velocity[i][j] = 0.0;
        }
    }
    //cout << "initialized weights for edge " << innovation_number << ", weights[0][0]: " << weights[0][0] << endl;
}

void CNN_Edge::initialize_velocities() {
    for (uint32_t i = 0; i < weights.size(); i++) {
        for (uint32_t j = 0; j < weights[i].size(); j++) {
            previous_velocity[i][j] = 0.0;
        }
    }
}


void CNN_Edge::reinitialize(minstd_rand0 &generator, NormalDistribution &normal_distribution) {
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

    weights = vector< vector<double> >(filter_y, vector<double>(filter_x, 0.0));
    best_weights = vector< vector<double> >(filter_y, vector<double>(filter_x, 0.0));
    previous_velocity = vector< vector<double> >(filter_y, vector<double>(filter_x, 0.0));

    initialize_weights(generator, normal_distribution);
}

void CNN_Edge::save_best_weights() {
    for (uint32_t y = 0; y < weights.size(); y++) {
        for (uint32_t x = 0; x < weights[y].size(); x++) {
            best_weights[y][x] = weights[y][x];
        }
    }
}

void CNN_Edge::set_weights_to_best() {
    for (uint32_t y = 0; y < weights.size(); y++) {
        for (uint32_t x = 0; x < weights[y].size(); x++) {
            weights[y][x] = best_weights[y][x];
        }
    }
}


CNN_Edge* CNN_Edge::copy() const {
    CNN_Edge* copy = new CNN_Edge();

    copy->edge_id = -1;
    copy->genome_id = genome_id;

    copy->fixed = fixed;
    copy->innovation_number = innovation_number;
    copy->disabled = disabled;

    copy->input_node = input_node;
    copy->output_node = output_node;

    copy->input_node_innovation_number = input_node->get_innovation_number();
    copy->output_node_innovation_number = output_node->get_innovation_number();

    copy->filter_x = filter_x;
    copy->filter_y = filter_y;

    copy->reverse_filter_x = reverse_filter_x;
    copy->reverse_filter_y = reverse_filter_y;

    copy->weights = vector< vector<double> >(filter_y, vector<double>(filter_x, 0.0));
    copy->best_weights = vector< vector<double> >(filter_y, vector<double>(filter_x, 0.0));
    copy->previous_velocity = vector< vector<double> >(filter_y, vector<double>(filter_x, 0.0));

    for (uint32_t y = 0; y < weights.size(); y++) {
        for (uint32_t x = 0; x < weights[y].size(); x++) {
            copy->weights[y][x] = weights[y][x];
            copy->best_weights[y][x] = best_weights[y][x];
            copy->previous_velocity[y][x] = previous_velocity[y][x];
        }
    }

    return copy;
}

bool CNN_Edge::set_nodes(const vector<CNN_Node*> nodes) {
    //cout << "nodes.size(): " << nodes.size() << endl;
    //cout << "setting input node: " << input_node_innovation_number << endl;
    //cout << "setting output node: " << output_node_innovation_number << endl;

    for (uint32_t i = 0; i < nodes.size(); i++) {
        if (nodes[i]->get_innovation_number() == input_node_innovation_number) {
            input_node = nodes[i];
        }

        if (nodes[i]->get_innovation_number() == output_node_innovation_number) {
            output_node = nodes[i];
            if (!disabled) output_node->add_input();
        }
    }

    if (input_node == NULL) {
        cerr << "ERROR! Could not find node with input node innovation number " << input_node_innovation_number << endl;
        cerr << "This should never happen!" << endl;
        exit(1);
    }

    if (output_node == NULL) {
        cerr << "ERROR! Could not find node with output node innovation number " << output_node_innovation_number << endl;
        cerr << "This should never happen!" << endl;
        exit(1);
    }

    if (output_node == input_node) {
        cerr << "ERROR! Setting nodes and output_node == input_node!" << endl;
        cerr << "input node innovation number: " << input_node_innovation_number << endl;
        cerr << "output node innovation number: " << output_node_innovation_number << endl;
        cerr << "This should never happen!" << endl;
        exit(1);
    }

    if (!is_filter_correct()) {
        return false;
    }

    return true;
}

bool CNN_Edge::is_filter_correct() const {
    //cout << "\t\tchecking filter correctness on edge: " << innovation_number << endl;
    //cout << "\t\t\tdisabled? " << disabled << endl;
    //cout << "\t\t\treverse_filter_x? " << reverse_filter_x << ", reverse_filter_y: " << reverse_filter_y << endl;
    //cout << "\t\t\tbetween node " << input_node_innovation_number << " and " << output_node_innovation_number << endl;

    bool is_correct = true;
    if (reverse_filter_x) {
        //cout << "\t\t\tfilter_x: " << filter_x << ", should be: " << (output_node->get_size_x() - input_node->get_size_x()) + 1 << " (output_x: " << output_node->get_size_x() << " - input_x: " << input_node->get_size_x() << " + 1) " << endl;

        is_correct = is_correct && (filter_x == (output_node->get_size_x() - input_node->get_size_x()) + 1);
    } else {
        //cout << "\t\t\tfilter_x: " << filter_x << ", should be: " << (input_node->get_size_x() - output_node->get_size_x()) + 1 << " (input_x: " << input_node->get_size_x() << " - output_x: " << output_node->get_size_x() << " + 1) " << endl;

        is_correct = is_correct && (filter_x == (input_node->get_size_x() - output_node->get_size_x()) + 1);
    }

    if (reverse_filter_y) {
        //cout << "\t\t\tfilter_y: " << filter_y << ", should be: " << (output_node->get_size_y() - input_node->get_size_y()) + 1 << " (output_y: " << output_node->get_size_y() << " - input_y: " << input_node->get_size_y() << " + 1) " << endl;

        is_correct = is_correct && (filter_y == (output_node->get_size_y() - input_node->get_size_y()) + 1);
    } else {
        //cout << "\t\t\tfilter_y: " << filter_y << ", should be: " << (input_node->get_size_y() - output_node->get_size_y()) + 1 << " (input_y: " << input_node->get_size_y() << " - output_y: " << output_node->get_size_y() << " + 1) " << endl;

        is_correct = is_correct && (filter_y == (input_node->get_size_y() - output_node->get_size_y()) + 1);
    }

    return is_correct;
}

void CNN_Edge::enable() {
    if (disabled) {
        disabled = false;
        output_node->add_input();
    }
}

void CNN_Edge::disable() {
    if (!disabled) {
        disabled = true;
        output_node->disable_input();
    }
}

bool CNN_Edge::is_disabled() const {
    return disabled;
}

int CNN_Edge::get_number_weights() const {
    return filter_x * filter_y;
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
    if (disabled) return false;

    double filter_sum = 0.0;
    for (int32_t fy = 0; fy < filter_y; fy++) {
        for (int32_t fx = 0; fx < filter_x; fx++) {
            filter_sum += (weights[fy][fx] * weights[fy][fx]);
        }
    }

    return filter_sum == 0;
}

bool CNN_Edge::has_zero_best_weight() const {
    if (disabled) return false;

    double filter_sum = 0.0;
    for (int32_t fy = 0; fy < filter_y; fy++) {
        for (int32_t fx = 0; fx < filter_x; fx++) {
            filter_sum += (best_weights[fy][fx] * best_weights[fy][fx]);
        }
    }

    return filter_sum == 0;
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

    for (uint32_t i = 0; i < previous_velocity.size(); i++) {
        out << "    ";
        for (uint32_t j = 0; j < previous_velocity[i].size(); j++) {
            out << setw(9) << setprecision(3) << previous_velocity[i][j];
        }
        out << endl;
    }
}

void CNN_Edge::propagate_forward() {
    if (disabled) return;

    double **input = input_node->get_values();
    double **output = output_node->get_values();

    /*
    if (!is_filter_correct()) {
        cerr << "ERROR: filter_x != input_node->get_size_x: " << input_node->get_size_x() << " - output_node->get_size_x: " << output_node->get_size_x() << " + 1" << endl;
        exit(1);
    }
    */

    /*
    cout << "propagating forward!" << endl;
    cout << "\tinput_x: " << input_node->get_size_x() << endl;
    cout << "\tinput_y: " << input_node->get_size_y() << endl;
    cout << "\toutput_x: " << output_node->get_size_x() << endl;
    cout << "\toutput_y: " << output_node->get_size_y() << endl;
    cout << "\tfilter_x: " << filter_x << endl;
    cout << "\tfilter_y: " << filter_y << endl;
    */

    for (int32_t fy = 0; fy < filter_y; fy++) {
        for (int32_t fx = 0; fx < filter_x; fx++) {
            if (isnan(weights[fy][fx])) {
                cerr << "ERROR in edge " << innovation_number << " propagate forward!" << endl;
                cerr << "input node innovation number: " << input_node->get_innovation_number() << " at depth: " << input_node->get_depth() << endl;
                cerr << "output node innovation number: " << output_node->get_innovation_number() << " at depth: " << output_node->get_depth() << endl;
                cerr << "weights[" << fy << "][" << fx << "] was NAN!" << endl;
            }
        }
    }

    for (int32_t y = 0; y < input_node->get_size_y(); y++) {
        for (int32_t x = 0; x < input_node->get_size_x(); x++) {
            if (isnan(input[y][x])) {
                cerr << "ERROR in edge " << innovation_number << " propagate forward!" << endl;
                cerr << "input[" << y << "][" << x << "] was NAN!" << endl;
                cerr << "input node innovation number: " << input_node->get_innovation_number() << " at depth: " << input_node->get_depth() << endl;
                cerr << "output node innovation number: " << output_node->get_innovation_number() << " at depth: " << output_node->get_depth() << endl;
                input_node->print(cerr);
                exit(1);
            }
        }
    }

    double previous_output;

    if (reverse_filter_x && reverse_filter_y) {
        for (int32_t fy = 0; fy < filter_y; fy++) {
            for (int32_t fx = 0; fx < filter_x; fx++) {
                double weight = weights[fy][fx];

                for (int32_t y = 0; y < input_node->get_size_y(); y++) {
                    for (int32_t x = 0; x < input_node->get_size_x(); x++) {
                        double value = weight * input[y][x];

                        previous_output = output[y + fy][x + fx];
                        output[y + fy][x + fx] += value;

                        if (isnan(output[y + fy][x + fx])) {
                            cerr << "ERROR in edge " << innovation_number << " propagate forward!" << endl;
                            cerr << "input node innovation number: " << input_node->get_innovation_number() << " at depth: " << input_node->get_depth() << endl;
                            cerr << "input node inputs fired: " << input_node->get_inputs_fired() << ", total_inputs: " << input_node->get_number_inputs() << endl;
                            cerr << "output node innovation number: " << output_node->get_innovation_number() << " at depth: " << output_node->get_depth() << endl;
                            cerr << "output became NAN!" << endl;
                            cerr << "output[" << y + fy << "][" << x + fx << "]" << endl;
                            cerr << "input[" << y << "][" << x << "]" << endl;
                            cerr << "weight: " << weight << endl;
                            cerr << "previous output: " << previous_output << endl;
                            cerr << "value added: " << value << endl;
                        }
                    }
                }
            }
        }

    } else if (reverse_filter_x) {
        for (int32_t fy = 0; fy < filter_y; fy++) {
            for (int32_t fx = 0; fx < filter_x; fx++) {
                double weight = weights[fy][fx];

                for (int32_t y = 0; y < output_node->get_size_y(); y++) {
                    for (int32_t x = 0; x < input_node->get_size_x(); x++) {
                        double value = weight * input[y + fy][x];

                        previous_output = output[y][x + fx];
                        output[y][x + fx] += value;

                        if (isnan(output[y][x + fx])) {
                            cerr << "ERROR in edge " << innovation_number << " propagate forward!" << endl;
                            cerr << "input node innovation number: " << input_node->get_innovation_number() << " at depth: " << input_node->get_depth() << endl;
                            cerr << "input node inputs fired: " << input_node->get_inputs_fired() << ", total_inputs: " << input_node->get_number_inputs() << endl;
                            cerr << "output node innovation number: " << output_node->get_innovation_number() << " at depth: " << output_node->get_depth() << endl;
                            cerr << "output became NAN!" << endl;
                            cerr << "output[" << y << "][" << x + fx << "]" << endl;
                            cerr << "input[" << y + fy << "][" << x << "]" << endl;
                            cerr << "weight: " << weight << endl;
                            cerr << "previous output: " << previous_output << endl;
                            cerr << "value added: " << value << endl;
                            exit(1);
                        }
                    }
                }
            }
        }

    } else if (reverse_filter_y) {
        for (int32_t fy = 0; fy < filter_y; fy++) {
            for (int32_t fx = 0; fx < filter_x; fx++) {
                double weight = weights[fy][fx];

                for (int32_t y = 0; y < input_node->get_size_y(); y++) {
                    for (int32_t x = 0; x < output_node->get_size_x(); x++) {
                        double value = weight * input[y][x + fx];

                        previous_output = output[y + fy][x];
                        output[y + fy][x] += value;

                        if (isnan(output[y + fy][x])) {
                            cerr << "ERROR in edge " << innovation_number << " propagate forward!" << endl;
                            cerr << "input node innovation number: " << input_node->get_innovation_number() << " at depth: " << input_node->get_depth() << endl;
                            cerr << "input node inputs fired: " << input_node->get_inputs_fired() << ", total_inputs: " << input_node->get_number_inputs() << endl;
                            cerr << "output node innovation number: " << output_node->get_innovation_number() << " at depth: " << output_node->get_depth() << endl;
                            cerr << "output became NAN!" << endl;
                            cerr << "output[" << y + fy << "][" << x << "]" << endl;
                            cerr << "input[" << y << "][" << x + fx << "]" << endl;
                            cerr << "weight: " << weight << endl;
                            cerr << "previous output: " << previous_output << endl;
                            cerr << "value added: " << value << endl;
                            exit(1);
                        }
                    }
                }
            }
        }

    } else {
        for (int32_t fy = 0; fy < filter_y; fy++) {
            for (int32_t fx = 0; fx < filter_x; fx++) {
                double weight = weights[fy][fx];

                for (int32_t y = 0; y < output_node->get_size_y(); y++) {
                    for (int32_t x = 0; x < output_node->get_size_x(); x++) {
                        double value = weight * input[y + fy][x + fx];

                        previous_output = output[y][x];
                        output[y][x] += value;

                        if (isnan(output[y][x])) {
                            cerr << "ERROR in edge " << innovation_number << " propagate forward!" << endl;
                            cerr << "input node innovation number: " << input_node->get_innovation_number() << " at depth: " << input_node->get_depth() << endl;
                            cerr << "input node inputs fired: " << input_node->get_inputs_fired() << ", total_inputs: " << input_node->get_number_inputs() << endl;
                            cerr << "output node innovation number: " << output_node->get_innovation_number() << " at depth: " << output_node->get_depth() << endl;
                            cerr << "output became NAN!" << endl;
                            cerr << "output[" << y << "][" << x << "]" << endl;
                            cerr << "input[" << y + fy << "][" << x + fx << "]" << endl;
                            cerr << "weight: " << weight << endl;
                            cerr << "previous output: " << previous_output << endl;
                            cerr << "value added: " << value << endl;
                            exit(1);
                        }
                    }
                }
            }
        }
    }

    for (int32_t y = 0; y < output_node->get_size_y(); y++) {
        for (int32_t x = 0; x < output_node->get_size_x(); x++) {
            if (isnan(output[y][x])) {
                cerr << "ERROR in edge " << innovation_number << " propagate forward!" << endl;
                cerr << "output[" << y << "][" << x << "] was NAN!" << endl;
                cerr << "input node innovation number: " << input_node->get_innovation_number() << " at depth: " << input_node->get_depth() << endl;
                cerr << "output node innovation number: " << output_node->get_innovation_number() << " at depth: " << output_node->get_depth() << endl;
            }
        }
    }

    output_node->input_fired();
}

inline void CNN_Edge::backprop_weight_update(int fy, int fx, double weight_update, double weight, double mu, double learning_rate, double weight_decay) {
    double dx, pv, velocity;

    /* ADAM:
     * m = beta1*m + (1-beta1)*dx
     * v = beta2*v + (1-beta2)*(dx**2)
     * x += - learning_rate * m / (np.sqrt(v) + eps)
     */



    //L2 regularization
    dx = learning_rate * (weight_update / (filter_x * filter_y) - (weight * weight_decay));

    if (isnan(dx)) {
        cerr << "ERROR! dx became NAN in backprop weight update" << endl;
        cerr << "learning rate: " << learning_rate << endl;
        cerr << "weight_update: " << weight_update << endl;
        cerr << "filter_x: " << filter_x << endl;
        cerr << "filter_y: " << filter_y << endl;
        cerr << "weight: " << weight << endl;
        cerr << "weight decay: " << weight_decay << endl;
        cerr << "mu: " << mu << endl;
        exit(1);
    }

    //no momemntum
    //weights[fy][fx] += dx;

    //momentum
    pv = previous_velocity[fy][fx];
    velocity = (mu * pv) - dx;
    weights[fy][fx] -= -mu * pv + (1 + mu) * velocity;
    previous_velocity[fy][fx] = velocity;

    if (weights[fy][fx] > 50.0) {
        weights[fy][fx] = 50.0;
        previous_velocity[fy][fx] = 0.0;
    } else if (weights[fy][fx] < -50.0) {
        weights[fy][fx] = -50.0;
        previous_velocity[fy][fx] = 0.0;
    }
}

void CNN_Edge::propagate_backward(double mu, double learning_rate, double weight_decay) {
    if (disabled) return;

    double **input = input_node->get_values();
    double **output_errors = output_node->get_errors();
    double **input_errors = input_node->get_errors();

    double weight, weight_update, error, update;
    double previous_weight_update;

    if (reverse_filter_x && reverse_filter_y) {
        //cout << "reverse filter x and y!" << endl;

        for (int32_t fy = 0; fy < filter_y; fy++) {
            for (int32_t fx = 0; fx < filter_x; fx++) {
                weight_update = 0;
                weight = weights[fy][fx];

                for (int32_t y = 0; y < input_node->get_size_y(); y++) {
                    for (int32_t x = 0; x < input_node->get_size_x(); x++) {
                        error = output_errors[y + fy][x + fx];

                        update = input[y][x] * error;
                        previous_weight_update = weight_update;
                        weight_update -= update;

                        if (isnan(weight_update)) {
                            cerr << "ERROR in edge " << innovation_number << " propagate backward!" << endl;
                            cerr << "input node innovation number: " << input_node->get_innovation_number() << " at depth: " << input_node->get_depth() << endl;
                            cerr << "input node inputs fired: " << input_node->get_inputs_fired() << ", total_inputs: " << input_node->get_number_inputs() << endl;
                            cerr << "output node innovation number: " << output_node->get_innovation_number() << " at depth: " << output_node->get_depth() << endl;
                            cerr << "ERROR! weight update became NAN!" << endl;
                            cerr << "update: " << update << endl;
                            cerr << "error: " << error << endl;
                            cerr << "input: " << input[y][x] << endl;
                            cerr << "previous weight update: " << previous_weight_update << endl;
                            cerr << "weight update: " << weight_update << endl;
                            exit(1);
                        }

                        input_errors[y][x] += error * weight;
                    }
                }
                backprop_weight_update(fy, fx, weight_update, weight, mu, learning_rate, weight_decay);
            }
        }

    } else if (reverse_filter_x) {
        //cout << "reverse filter x!" << endl;

        for (int32_t fy = 0; fy < filter_y; fy++) {
            for (int32_t fx = 0; fx < filter_x; fx++) {
                weight_update = 0;
                weight = weights[fy][fx];

                for (int32_t y = 0; y < output_node->get_size_y(); y++) {
                    for (int32_t x = 0; x < input_node->get_size_x(); x++) {
                        error = output_errors[y][x + fx];

                        update = input[y + fy][x] * error;

                        previous_weight_update = weight_update;
                        weight_update -= update;

                        if (isnan(weight_update)) {
                            cerr << "ERROR in edge " << innovation_number << " propagate backward!" << endl;
                            cerr << "input node innovation number: " << input_node->get_innovation_number() << " at depth: " << input_node->get_depth() << endl;
                            cerr << "input node inputs fired: " << input_node->get_inputs_fired() << ", total_inputs: " << input_node->get_number_inputs() << endl;
                            cerr << "output node innovation number: " << output_node->get_innovation_number() << " at depth: " << output_node->get_depth() << endl;
                            cerr << "ERROR! weight update became NAN!" << endl;
                            cerr << "update: " << update << endl;
                            cerr << "error: " << error << endl;
                            cerr << "input: " << input[y + fy][x] << endl;
                            cerr << "previous weight update: " << previous_weight_update << endl;
                            cerr << "weight update: " << weight_update << endl;
                            exit(1);
                        }


                        input_errors[y + fy][x] += error * weight;
                    }
                }
                backprop_weight_update(fy, fx, weight_update, weight, mu, learning_rate, weight_decay);
            }
        }

    } else if (reverse_filter_y) {
        //cout << "reverse filter y!" << endl;

        for (int32_t fy = 0; fy < filter_y; fy++) {
            for (int32_t fx = 0; fx < filter_x; fx++) {
                weight_update = 0;
                weight = weights[fy][fx];

                for (int32_t y = 0; y < input_node->get_size_y(); y++) {
                    for (int32_t x = 0; x < output_node->get_size_x(); x++) {
                        error = output_errors[y + fy][x];

                        update = input[y][x + fx] * error;

                        previous_weight_update = weight_update;
                        weight_update -= update;

                        if (isnan(weight_update)) {
                            cerr << "ERROR in edge " << innovation_number << " propagate backward!" << endl;
                            cerr << "input node innovation number: " << input_node->get_innovation_number() << " at depth: " << input_node->get_depth() << endl;
                            cerr << "input node inputs fired: " << input_node->get_inputs_fired() << ", total_inputs: " << input_node->get_number_inputs() << endl;
                            cerr << "output node innovation number: " << output_node->get_innovation_number() << " at depth: " << output_node->get_depth() << endl;
                            cerr << "ERROR! weight update became NAN!" << endl;
                            cerr << "update: " << update << endl;
                            cerr << "error: " << error << endl;
                            cerr << "input: " << input[y][x + fx] << endl;
                            cerr << "previous weight update: " << previous_weight_update << endl;
                            cerr << "weight update: " << weight_update << endl;
                            exit(1);
                        }


                        input_errors[y][x + fx] += error * weight;
                    }
                }
                backprop_weight_update(fy, fx, weight_update, weight, mu, learning_rate, weight_decay);
            }
        }

    } else {
        //cout << "no reverse filter!" << endl;

        for (int32_t fy = 0; fy < filter_y; fy++) {
            for (int32_t fx = 0; fx < filter_x; fx++) {
                weight_update = 0;
                weight = weights[fy][fx];

                for (int32_t y = 0; y < output_node->get_size_y(); y++) {
                    for (int32_t x = 0; x < output_node->get_size_x(); x++) {
                        error = output_errors[y][x];

                        update = input[y + fy][x + fx] * error;

                        previous_weight_update = weight_update;
                        weight_update -= update;

                        if (isnan(weight_update)) {
                            cerr << "ERROR in edge " << innovation_number << " propagate backward!" << endl;
                            cerr << "input node innovation number: " << input_node->get_innovation_number() << " at depth: " << input_node->get_depth() << endl;
                            cerr << "input node inputs fired: " << input_node->get_inputs_fired() << ", total_inputs: " << input_node->get_number_inputs() << endl;
                            cerr << "output node innovation number: " << output_node->get_innovation_number() << " at depth: " << output_node->get_depth() << endl;
                            cerr << "ERROR! weight update became NAN!" << endl;
                            cerr << "update: " << update << endl;
                            cerr << "error: " << error << endl;
                            cerr << "input: " << input[y + fy][x + fx] << endl;
                            cerr << "previous weight update: " << previous_weight_update << endl;
                            cerr << "weight update: " << weight_update << endl;
                            exit(1);
                        }


                        input_errors[y + fy][x + fx] += error * weight;
                    }
                }
                backprop_weight_update(fy, fx, weight_update, weight, mu, learning_rate, weight_decay);
            }
        }
    }
}

ostream &operator<<(ostream &os, const CNN_Edge* edge) {
    os << edge->edge_id << " ";
    os << edge->exact_id << " ";
    os << edge->genome_id << " ";
    os << edge->innovation_number << " ";
    os << edge->input_node_innovation_number << " ";
    os << edge->output_node_innovation_number << " ";
    os << edge->filter_x << " ";
    os << edge->filter_y << " ";
    os << edge->fixed << " ";
    os << edge->reverse_filter_x << " ";
    os << edge->reverse_filter_y << " ";
    os << edge->disabled << endl;

    for (int32_t y = 0; y < edge->filter_y; y++) {
        for (int32_t x = 0; x < edge->filter_x; x++) {
            if (y > 0 || x > 0) os << " ";
            os << setprecision(15) << edge->weights[y][x];
        }
    }
    os << endl;

    for (int32_t y = 0; y < edge->filter_y; y++) {
        for (int32_t x = 0; x < edge->filter_x; x++) {
            if (y > 0 || x > 0) os << " ";
            os << setprecision(15) << edge->best_weights[y][x];
        }
    }
    os << endl;

    for (int32_t y = 0; y < edge->filter_y; y++) {
        for (int32_t x = 0; x < edge->filter_x; x++) {
            if (y > 0 || x > 0) os << " ";
            os << setprecision(15) << edge->previous_velocity[y][x];
        }
    }

    return os;
}

istream &operator>>(istream &is, CNN_Edge* edge) {
    is >> edge->edge_id;
    is >> edge->exact_id;
    is >> edge->genome_id;
    is >> edge->innovation_number;
    is >> edge->input_node_innovation_number;
    is >> edge->output_node_innovation_number;
    is >> edge->filter_x;
    is >> edge->filter_y;
    is >> edge->fixed;
    is >> edge->reverse_filter_x;
    is >> edge->reverse_filter_y;
    is >> edge->disabled;

    edge->weights = vector< vector<double> >(edge->filter_y, vector<double>(edge->filter_x, 0.0));
    edge->best_weights = vector< vector<double> >(edge->filter_y, vector<double>(edge->filter_x, 0.0));
    edge->previous_velocity = vector< vector<double> >(edge->filter_y, vector<double>(edge->filter_x, 0.0));

    for (int32_t y = 0; y < edge->filter_y; y++) {
        for (int32_t x = 0; x < edge->filter_x; x++) {
            is >> edge->weights[y][x];
        }
    }

    for (int32_t y = 0; y < edge->filter_y; y++) {
        for (int32_t x = 0; x < edge->filter_x; x++) {
            is >> edge->best_weights[y][x];
        }
    }

    for (int32_t y = 0; y < edge->filter_y; y++) {
        for (int32_t x = 0; x < edge->filter_x; x++) {
            is >> edge->previous_velocity[y][x];
        }
    }

    return is;
}



