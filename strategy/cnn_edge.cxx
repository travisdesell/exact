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

    needs_initialization = true;
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
    needs_initialization = true;

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
    weight_updates = vector< vector<double> >(filter_y, vector<double>(filter_x, 0.0));
    best_weights = vector< vector<double> >(filter_y, vector<double>(filter_x, 0.0));

    previous_velocity = vector< vector<double> >(filter_y, vector<double>(filter_x, 0.0));
    best_velocity = vector< vector<double> >(filter_y, vector<double>(filter_x, 0.0));
}

CNN_Edge::~CNN_Edge() {
    input_node = NULL;
    output_node = NULL;
}

void parse_vector_2d(vector< vector<double> > &output, istringstream &iss, int filter_x, int filter_y) {
    output.clear();
    output = vector< vector<double> >(filter_y, vector<double>(filter_x));

    int current_x = 0, current_y = 0;

    double val;
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

        istringstream best_velocity_iss(row[11]);
        parse_vector_2d(best_velocity, best_velocity_iss, filter_x, filter_y);

        fixed = atoi(row[12]);
        disabled = atoi(row[13]);
        reverse_filter_x = atoi(row[14]);
        reverse_filter_y = atoi(row[15]);
        needs_initialization = atoi(row[16]);

        mysql_free_result(result);
    } else {
        cerr << "ERROR! Could not find cnn_edge in database with edge id: " << edge_id << endl;
        exit(1);
    }

    weight_updates = vector< vector<double> >(filter_y, vector<double>(filter_x, 0.0));

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
        << ", innovation_number = " << innovation_number
        << ", input_node_innovation_number = " << input_node_innovation_number
        << ", output_node_innovation_number = " << output_node_innovation_number
        << ", filter_x = " << filter_x
        << ", filter_y = " << filter_y
        << ", fixed = " << fixed
        << ", disabled = " << disabled
        << ", reverse_filter_x = " << reverse_filter_x
        << ", reverse_filter_y = " << reverse_filter_y
        << ", needs_initialization = " << needs_initialization
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

    query << "', best_velocity = '";
    for (int32_t y = 0; y < filter_y; y++) {
        for (int32_t x = 0; x < filter_x; x++) {
            if (x != 0) query << " ";
            query << setprecision(15) << best_velocity[y][x];
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

int CNN_Edge::get_edge_id() const {
    return edge_id;
}
#endif

bool CNN_Edge::equals(CNN_Edge *other) const {
    return filter_x == other->filter_x && filter_y == other->filter_y && disabled == other->disabled && reverse_filter_x == other->reverse_filter_x && reverse_filter_y == other->reverse_filter_y;
}

bool CNN_Edge::needs_init() const {
    return needs_initialization;
}

void CNN_Edge::set_needs_init() {
    needs_initialization = true;
}

int CNN_Edge::get_filter_x() const {
    return filter_x;
}

int CNN_Edge::get_filter_y() const {
    return filter_y;
}

void CNN_Edge::propagate_weight_count() {
    output_node->add_weight_count(filter_x * filter_y);
}

void CNN_Edge::initialize_weights(minstd_rand0 &generator, NormalDistribution &normal_distribution) {
    int edge_size = output_node->get_weight_count();
    if (edge_size == 0) {
        cerr << "ERROR! Initializing weights on an edge when node weight counts have not yet been set!" << endl;
        exit(1);
    }

    double mu = 0.0;
    double sigma = sqrt(2.0 / edge_size);

    for (uint32_t i = 0; i < weights.size(); i++) {
        for (uint32_t j = 0; j < weights[i].size(); j++) {
            weights[i][j] = normal_distribution.random(generator, mu, sigma);
            best_weights[i][j] = 0.0;
            previous_velocity[i][j] = 0.0;
        }
    }
    //cout << "initialized weights for edge " << innovation_number << ", weights[0][0]: " << weights[0][0] << endl;

    needs_initialization = false;
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

    weights = vector< vector<double> >(filter_y, vector<double>(filter_x, 0.0));
    weight_updates = vector< vector<double> >(filter_y, vector<double>(filter_x, 0.0));
    best_weights = vector< vector<double> >(filter_y, vector<double>(filter_x, 0.0));

    previous_velocity = vector< vector<double> >(filter_y, vector<double>(filter_x, 0.0));
    best_velocity = vector< vector<double> >(filter_y, vector<double>(filter_x, 0.0));

    needs_initialization = true;
}

void CNN_Edge::save_best_weights() {
    for (uint32_t y = 0; y < filter_y; y++) {
        for (uint32_t x = 0; x < filter_x; x++) {
            best_weights[y][x] = weights[y][x];
            best_velocity[y][x] = previous_velocity[y][x];
        }
    }
}

void CNN_Edge::set_weights_to_best() {
    for (uint32_t y = 0; y < filter_y; y++) {
        for (uint32_t x = 0; x < filter_x; x++) {
            weights[y][x] = best_weights[y][x];
            //previous_velocity[y][x] = best_velocity[y][x];
            previous_velocity[y][x] = 0;
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
    copy->needs_initialization = needs_initialization;

    copy->weights = weights;
    copy->weight_updates = weight_updates;
    copy->best_weights = best_weights;
    copy->previous_velocity = previous_velocity;
    copy->best_velocity = best_velocity;

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
            if (!disabled) output_node->add_input();
        }
    }

    if (input_node == NULL) {
        cerr << "ERROR! Could not find node with input node innovation number " << input_node_innovation_number << endl;
        cerr << "This should never happen!" << endl;
        cerr << "nodes innovation numbers:" << endl;
        for (uint32_t i = 0; i < nodes.size(); i++) {
            cerr << "\t" << nodes[i]->get_innovation_number() << endl;
        }
        exit(1);
    }

    if (output_node == NULL) {
        cerr << "ERROR! Could not find node with output node innovation number " << output_node_innovation_number << endl;
        cerr << "This should never happen!" << endl;
        cerr << "nodes innovation numbers:" << endl;
        for (uint32_t i = 0; i < nodes.size(); i++) {
            cerr << "\t" << nodes[i]->get_innovation_number() << endl;
        }
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
            out << setw(9) << fixed << setprecision(3) << weights[i][j];
        }
        out << endl;
    }

    for (uint32_t i = 0; i < previous_velocity.size(); i++) {
        out << "    ";
        for (uint32_t j = 0; j < previous_velocity[i].size(); j++) {
            out << setw(9) << fixed << setprecision(3) << previous_velocity[i][j];
        }
        out << endl;
    }
}

void CNN_Edge::check_output_update(const vector< vector<double> > &output, const vector< vector<double> > &input, double value, double weight, double previous_output, int in_y, int in_x, int out_y, int out_x) {
    if (isnan(output[out_y][out_x]) || isinf(output[out_y][out_x])) {
        cerr << "ERROR in edge " << innovation_number << " propagate forward!" << endl;
        cerr << "input node innovation number: " << input_node->get_innovation_number() << " at depth: " << input_node->get_depth() << endl;
        cerr << "input node inputs fired: " << input_node->get_inputs_fired() << ", total_inputs: " << input_node->get_number_inputs() << endl;
        cerr << "output node innovation number: " << output_node->get_innovation_number() << " at depth: " << output_node->get_depth() << endl;
        cerr << "output became: " << output[out_y][out_x] << "!" << endl;
        cerr << "output[" << out_y << "][" << out_x << "] = " << output[out_y][out_x] << endl;
        cerr << "input[" << in_y << "][" << in_x << "] = " << input[in_y][in_x] << endl;
        cerr << "weight: " << weight << endl;
        cerr << "previous output: " << previous_output << endl;
        cerr << "value added: " << value << endl;

        input_node->print(cerr);
        output_node->print(cerr);

        exit(1);
    }
}

void CNN_Edge::propagate_forward() {
    if (disabled) return;

    vector< vector<double> > &input = input_node->get_values();
    vector< vector<double> > &output = output_node->get_values();

#ifdef NAN_CHECKS
    if (!is_filter_correct()) {
        cerr << "ERROR: filter_x != input_node->get_size_x: " << input_node->get_size_x() << " - output_node->get_size_x: " << output_node->get_size_x() << " + 1" << endl;
        exit(1);
    }

    double previous_output;
#endif

    int output_size_x = output_node->get_size_x();
    int output_size_y = output_node->get_size_y();
    int input_size_x = input_node->get_size_x();
    int input_size_y = input_node->get_size_y();

    if (reverse_filter_x && reverse_filter_y) {
        for (int32_t fy = 0; fy < filter_y; fy++) {
            for (int32_t fx = 0; fx < filter_x; fx++) {
                double weight = weights[fy][fx];

                for (int32_t y = 0; y < input_size_y; y++) {
                    for (int32_t x = 0; x < input_size_x; x++) {
                        double value = weight * input[y][x];
#ifdef NAN_CHECKS
                        previous_output = output[y + fy][x + fx];
#endif
                        output[y + fy][x + fx] += value;
#ifdef NAN_CHECKS
                        check_output_update(output, input, value, weight, previous_output, y, x, y + fy, x + fx);
#endif
                    }
                }
            }
        }

    } else if (reverse_filter_x) {
        for (int32_t fy = 0; fy < filter_y; fy++) {
            for (int32_t fx = 0; fx < filter_x; fx++) {
                double weight = weights[fy][fx];

                for (int32_t y = 0; y < output_size_y; y++) {
                    for (int32_t x = 0; x < input_size_x; x++) {
                        double value = weight * input[y + fy][x];
#ifdef NAN_CHECKS
                        previous_output = output[y][x + fx];
#endif
                        output[y][x + fx] += value;
#ifdef NAN_CHECKS
                        check_output_update(output, input, value, weight, previous_output, y + fy, x, y, x + fx);
#endif
                    }
                }
            }
        }

    } else if (reverse_filter_y) {
        for (int32_t fy = 0; fy < filter_y; fy++) {
            for (int32_t fx = 0; fx < filter_x; fx++) {
                double weight = weights[fy][fx];

                for (int32_t y = 0; y < input_size_y; y++) {
                    for (int32_t x = 0; x < output_size_x; x++) {
                        double value = weight * input[y][x + fx];
#ifdef NAN_CHECKS
                        previous_output = output[y + fy][x];
#endif
                        output[y + fy][x] += value;
#ifdef NAN_CHECKS
                        check_output_update(output, input, value, weight, previous_output, y, x + fx, y + fy, x);
#endif
                    }
                }
            }
        }

    } else {
        for (int32_t fy = 0; fy < filter_y; fy++) {
            for (int32_t fx = 0; fx < filter_x; fx++) {
                double weight = weights[fy][fx];

                for (int32_t y = 0; y < output_size_y; y++) {
                    for (int32_t x = 0; x < output_size_x; x++) {
                        double value = weight * input[y + fy][x + fx];
#ifdef NAN_CHECKS
                        previous_output = output[y][x];
#endif
                        output[y][x] += value;
#ifdef NAN_CHECKS
                        check_output_update(output, input, value, weight, previous_output, y + fy, x + fx, y, x);
#endif
                    }
                }
            }
        }
    }

	output_node->input_fired();
}

void CNN_Edge::update_weights(double mu, double learning_rate, double weight_decay) {
    if (disabled) return;

    double dx, pv, velocity, previous_weight, weight;

	for (int32_t fy = 0; fy < filter_y; fy++) {
        for (int32_t fx = 0; fx < filter_x; fx++) {
            dx = weight_updates[fy][fx];
            pv = previous_velocity[fy][fx];

            velocity = (mu * pv) - learning_rate * dx;

            weight = weights[fy][fx];
#ifdef NAN_CHECKS
            previous_weight = weight;
#endif
            weight += (-mu * pv + (1 + mu) * velocity);
            weight -= (weight * weight_decay);
            weights[fy][fx] = weight;

            previous_velocity[fy][fx] = velocity;

#ifdef NAN_CHECKS
            if (isnan(weights[fy][fx]) || isinf(weights[fy][fx])) {
                cerr << "ERROR! weight became " << weights[fy][fx] << " in edge: " << innovation_number << " (" << input_node_innovation_number << " to " << output_node_innovation_number << ")" << endl;
                cerr << "\tdx: " << dx << endl;
                cerr << "\tpv: " << pv << endl;
                cerr << "\tvelocity: " << velocity << endl;
                cerr << "\tprevious_weight: " << previous_weight << endl;
                exit(1);
            }
#endif

            if (weights[fy][fx] > 100.0) {
                /*
                cout << "weight > 100!" << endl;
                cout << "updating weight from " << input_node_innovation_number << " to " << output_node_innovation_number
                    << ": fy: " << fy << ", fx: " << fx 
                    << ", weight: " << weights[fy][fx] 
                    << ", weight_update: " << weight_updates[fy][fx] 
                    << ", learning_rate * dx: " << (learning_rate * dx) << endl;

                this->print(cout);
                input_node->print(cout);
                output_node->print(cout);

                exit(1);
                */

                weights[fy][fx] = 100.0;
                previous_velocity[fy][fx] = 0.0;
            } else if (weights[fy][fx] < -100.0) {
                /*
                cout << "weight < -100!" << endl;
                cout << "updating weight from " << input_node_innovation_number << " to " << output_node_innovation_number
                    << ": fy: " << fy << ", fx: " << fx 
                    << ", weight: " << weights[fy][fx] 
                    << ", weight_update: " << weight_updates[fy][fx] 
                    << ", learning_rate * dx: " << (learning_rate * dx) << endl;
                this->print(cout);
                input_node->print(cout);
                output_node->print(cout);

                exit(1);
                */

                weights[fy][fx] = -100.0;
                previous_velocity[fy][fx] = 0.0;
            }
        }
    }
}

void CNN_Edge::check_weight_update(const vector< vector<double> > &output_errors, const vector< vector<double> > &output_gradients, const vector< vector<double> > &input, double delta, double weight_update, double previous_weight_update, int out_y, int out_x, int in_y, int in_x) {
    if (isnan(weight_update) || isinf(weight_update)) {
        cerr << "ERROR weight_update became " << weight_update << " in edge " << innovation_number << " (" << input_node_innovation_number << " to " << output_node_innovation_number << ")!" << endl;
        cerr << "\tprevious_weight_udpate: " << previous_weight_update << endl;
        cerr << "\toutput_error: " << output_errors[out_y][out_x] << endl;
        cerr << "\toutput_gradient: " << output_gradients[out_y][out_x] << endl;
        cerr << "\tdelta: " << delta << endl;
        cerr << "\tinput: " << input[in_y][in_x] << endl;

        cerr << endl << "input_node: " << endl;
        input_node->print(cerr);

        cerr << endl << "output_node: " << endl;
        output_node->print(cerr);

        exit(1);
    }
}

void CNN_Edge::propagate_backward() {
    if (disabled) return;

    vector< vector<double> > &output_errors = output_node->get_errors();
    vector< vector<double> > &output_gradients = output_node->get_gradients();

    vector< vector<double> > &input = input_node->get_values();
    vector< vector<double> > &input_errors = input_node->get_errors();

    double weight, weight_update, delta;
    
    int out_x = output_node->get_size_x();
    int out_y = output_node->get_size_y();

    if (reverse_filter_x && reverse_filter_y) {
        //cout << "reverse filter x and y!" << endl;

        for (int32_t fy = 0; fy < filter_y; fy++) {
            for (int32_t fx = 0; fx < filter_x; fx++) {
                weight_update = 0;
                weight = weights[fy][fx];

                for (int32_t y = 0; y < out_y; y++) {
                    for (int32_t x = 0; x < out_x; x++) {
                        delta = output_errors[y + fy][x + fx] * output_gradients[y + fy][x + fx];
                        weight_update += input[y][x] * delta;
#ifdef NAN_CHECKS                        
                        double previous_weight_update = weight_update;
#endif
                        input_errors[y][x] += delta * weight;

#ifdef NAN_CHECKS
                        check_weight_update(output_errors, output_gradients, input, delta, weight_update, previous_weight_update, y + fy, x + fx, y, x);
#endif
                    }
                }
                weight_updates[fy][fx] = weight_update;
            }
        }

    } else if (reverse_filter_x) {
        //cout << "reverse filter x!" << endl;

        for (int32_t fy = 0; fy < filter_y; fy++) {
            for (int32_t fx = 0; fx < filter_x; fx++) {
                weight_update = 0;
                weight = weights[fy][fx];

                for (int32_t y = 0; y < out_y; y++) {
                    for (int32_t x = 0; x < out_x; x++) {
                        delta = output_errors[y][x + fx] * output_gradients[y][x + fx];
#ifdef NAN_CHECKS                        
                        double previous_weight_update = weight_update;
#endif
                        weight_update += input[y + fy][x] * delta;
                        input_errors[y + fy][x] += delta * weight;

#ifdef NAN_CHECKS
                        check_weight_update(output_errors, output_gradients, input, delta, weight_update, previous_weight_update, y, x + fx, y + fy, x);
#endif
                    }
                }
                weight_updates[fy][fx] = weight_update;
            }
        }

    } else if (reverse_filter_y) {
        //cout << "reverse filter y!" << endl;

        for (int32_t fy = 0; fy < filter_y; fy++) {
            for (int32_t fx = 0; fx < filter_x; fx++) {
                weight_update = 0;
                weight = weights[fy][fx];

                for (int32_t y = 0; y < out_y; y++) {
                    for (int32_t x = 0; x < out_x; x++) {
                        delta = output_errors[y + fy][x] * output_gradients[y + fy][x];
#ifdef NAN_CHECKS                        
                        double previous_weight_update = weight_update;
#endif
                        weight_update += input[y][x + fx] * delta;
                        input_errors[y][x + fx] += delta * weight;

#ifdef NAN_CHECKS
                        check_weight_update(output_errors, output_gradients, input, delta, weight_update, previous_weight_update, y + fy, x, y, x + fx);
#endif
                    }
                }
                weight_updates[fy][fx] = weight_update;
            }
        }

    } else {
        //cout << "no reverse filter!" << endl;

        for (int32_t fy = 0; fy < filter_y; fy++) {
            for (int32_t fx = 0; fx < filter_x; fx++) {
                weight_update = 0;
                weight = weights[fy][fx];

                for (int32_t y = 0; y < out_y; y++) {
                    for (int32_t x = 0; x < out_x; x++) {
                        delta = output_errors[y][x] * output_gradients[y][x];
#ifdef NAN_CHECKS                        
                        double previous_weight_update = weight_update;
#endif
                        weight_update += input[y + fy][x + fx] * delta;
                        input_errors[y + fy][x + fx] += delta * weight;

#ifdef NAN_CHECKS
                        check_weight_update(output_errors, output_gradients, input, delta, weight_update, previous_weight_update, y, x, y + fy, x + fx);
#endif
                    }
                }

                weight_updates[fy][fx] = weight_update;
            }
        }
    }
}

bool CNN_Edge::has_nan() const {
    for (uint32_t y = 0; y < filter_y; y++) {
        for (uint32_t x = 0; x < filter_x; x++) {
            if (isnan(weights[y][x]) || isinf(weights[y][x])) return true;
            if (isnan(weight_updates[y][x]) || isinf(weight_updates[y][x])) return true;
            if (isnan(previous_velocity[y][x]) || isinf(previous_velocity[y][x])) return true;
        }
    }
    return false;
}

void CNN_Edge::print_statistics() {
    double weight_min = std::numeric_limits<double>::max(), weight_max = -std::numeric_limits<double>::max(), weight_avg = 0.0;
    double weight_update_min = std::numeric_limits<double>::max(), weight_update_max = -std::numeric_limits<double>::max(), weight_update_avg = 0.0;
    double velocity_min = std::numeric_limits<double>::max(), velocity_max = -std::numeric_limits<double>::max(), velocity_avg = 0.0;

    for (int fy = 0; fy < filter_y; fy++) {
        for (int fx = 0; fx < filter_x; fx++) {
            if (weights[fy][fx] < weight_min) weight_min = weights[fy][fx];
            if (weights[fy][fx] > weight_max) weight_max = weights[fy][fx];
            weight_avg += weights[fy][fx];

            if (weight_updates[fy][fx] < weight_update_min) weight_update_min = weight_updates[fy][fx];
            if (weight_updates[fy][fx] > weight_update_max) weight_update_max = weight_updates[fy][fx];
            weight_update_avg += weight_updates[fy][fx];


            if (previous_velocity[fy][fx] < velocity_min) velocity_min = previous_velocity[fy][fx];
            if (previous_velocity[fy][fx] > velocity_max) velocity_max = previous_velocity[fy][fx];
            velocity_avg += previous_velocity[fy][fx];
        }
    }

    velocity_avg /= filter_y * filter_x;
    weight_avg /= filter_y * filter_x;

    cerr << "edge " << setw(4) << innovation_number << " (in: " << setw(4) << input_node_innovation_number << ", out: " << setw(4) << output_node_innovation_number << ") w_min: " << weight_min << ", w_avg: " << weight_avg << ", w_max: " << weight_max << ", wu_min: " << weight_update_min << ", wu_avg: " << weight_update_avg << ", wu_max: " << weight_update_max << ", v_min: " << velocity_min << ", v_avg: " << velocity_avg << ", v_max: " << velocity_max << endl;

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
    os << edge->disabled << " ";
    os << edge->needs_initialization << endl;

    for (int32_t y = 0; y < edge->filter_y; y++) {
        for (int32_t x = 0; x < edge->filter_x; x++) {
            if (y > 0 || x > 0) os << " ";
            os << setprecision(17) << edge->weights[y][x];
        }
    }
    os << endl;

    for (int32_t y = 0; y < edge->filter_y; y++) {
        for (int32_t x = 0; x < edge->filter_x; x++) {
            if (y > 0 || x > 0) os << " ";
            os << setprecision(17) << edge->best_weights[y][x];
        }
    }
    os << endl;

    for (int32_t y = 0; y < edge->filter_y; y++) {
        for (int32_t x = 0; x < edge->filter_x; x++) {
            if (y > 0 || x > 0) os << " ";
            os << setprecision(17) << edge->previous_velocity[y][x];
        }
    }
    os << endl;

    for (int32_t y = 0; y < edge->filter_y; y++) {
        for (int32_t x = 0; x < edge->filter_x; x++) {
            if (y > 0 || x > 0) os << " ";
            os << setprecision(17) << edge->best_velocity[y][x];
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
    is >> edge->needs_initialization;

    edge->weights = vector< vector<double> >(edge->filter_y, vector<double>(edge->filter_x, 0.0));
    edge->weight_updates = vector< vector<double> >(edge->filter_y, vector<double>(edge->filter_x, 0.0));
    edge->best_weights = vector< vector<double> >(edge->filter_y, vector<double>(edge->filter_x, 0.0));

    edge->previous_velocity = vector< vector<double> >(edge->filter_y, vector<double>(edge->filter_x, 0.0));
    edge->best_velocity = vector< vector<double> >(edge->filter_y, vector<double>(edge->filter_x, 0.0));

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

    for (int32_t y = 0; y < edge->filter_y; y++) {
        for (int32_t x = 0; x < edge->filter_x; x++) {
            is >> edge->best_velocity[y][x];
        }
    }

    return is;
}
