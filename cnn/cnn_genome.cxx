#include <algorithm>
using std::sort;
using std::upper_bound;

#include <cmath>
using std::isnan;
using std::isinf;

#include <chrono>

#include <fstream>
using std::ofstream;
using std::ifstream;
using std::ios;

#include <limits>
using std::numeric_limits;

#include <iomanip>
using std::setw;
using std::setprecision;
using std::fixed;
using std::left;
using std::right;

#include <iostream>
using std::cout;
using std::cerr;
using std::endl;
using std::ostream;
using std::istream;
using std::hexfloat;
using std::defaultfloat;

#include <map>
using std::map;

#include <random>
using std::minstd_rand0;

#include <sstream>
using std::istringstream;
using std::ostringstream;

#include <string>
using std::string;
using std::to_string;

#include <vector>
using std::vector;


#ifdef _MYSQL_
#include "common/db_conn.hxx"
#endif

#include "comparison.hxx"
#include "common/exp.hxx"
#include "common/random.hxx"
#include "common/version.hxx"
#include "common/files.hxx"

#include "image_tools/image_set.hxx"
#include "image_tools/large_image_set.hxx"
#include "cnn_node.hxx"
#include "cnn_edge.hxx"
#include "cnn_genome.hxx"

#include "stdint.h"


void write_map(ostream &out, map<string, int> &m) {
    out << m.size();
    for (auto iterator = m.begin(); iterator != m.end(); iterator++) {

        out << " "<< iterator->first;
        out << " "<< iterator->second;
    }
}

void read_map(istream &in, map<string, int> &m) {
    int map_size;
    in >> map_size;
    for (int i = 0; i < map_size; i++) {
        string key;
        in >> key;
        int value;
        in >> value;

        m[key] = value;
    }
}


/**
 *  Initialize a genome from a file
 */
CNN_Genome::CNN_Genome(string filename, bool is_checkpoint) {
    exact_id = -1;
    genome_id = -1;
    started_from_checkpoint = is_checkpoint;

    string file_contents;

    //cout << "getting file as string: '" << filename << "'" << endl;
    file_contents = get_file_as_string(filename);
    //cout << "got file as string, erasing carraige returns" << endl;

    file_contents.erase(std::remove(file_contents.begin(), file_contents.end(), '\r'), file_contents.end());
    //cout << "erased carraige returns" << endl;

    istringstream infile_iss(file_contents);
    read(infile_iss);
}

CNN_Genome::CNN_Genome(istream &in, bool is_checkpoint) {
    exact_id = -1;
    genome_id = -1;
    started_from_checkpoint = is_checkpoint;
    read(in);
}


void CNN_Genome::set_progress_function(int (*_progress_function)(float)) {
    progress_function = _progress_function;
}

int CNN_Genome::get_genome_id() const {
    return genome_id;
}

int CNN_Genome::get_exact_id() const {
    return exact_id;
}

float CNN_Genome::get_initial_mu() const {
    return initial_mu;
}

float CNN_Genome::get_mu() const {
    return mu;
}

float CNN_Genome::get_mu_delta() const {
    return mu_delta;
}

float CNN_Genome::get_initial_learning_rate() const {
    return initial_learning_rate;
}

float CNN_Genome::get_learning_rate() const {
    return learning_rate;
}

float CNN_Genome::get_learning_rate_delta() const {
    return learning_rate_delta;
}

float CNN_Genome::get_initial_weight_decay() const {
    return initial_weight_decay;
}

float CNN_Genome::get_weight_decay() const {
    return weight_decay;
}

float CNN_Genome::get_weight_decay_delta() const {
    return weight_decay_delta;
}

float CNN_Genome::get_alpha() const {
    return alpha;
}

int CNN_Genome::get_velocity_reset() const {
    return velocity_reset;
}

float CNN_Genome::get_input_dropout_probability() const {
    return input_dropout_probability;
}

float CNN_Genome::get_hidden_dropout_probability() const {
    return hidden_dropout_probability;
}

int CNN_Genome::get_batch_size() const {
    return batch_size;
}


template <class T>
void parse_array(vector<T> &output, istringstream &iss) {
    output.clear();

    T val;
    while(iss >> val || !iss.eof()) {
        if (iss.fail()) {
            iss.clear();
            string dummy;
            iss >> dummy;
            continue;
        }
        output.push_back(val);
        //cout << val << endl;
    }
}

#ifdef _MYSQL_
CNN_Genome::CNN_Genome(int _genome_id) {
    progress_function = NULL;
    version_str = EXACT_VERSION_STR;

    ostringstream query;

    query << "SELECT * FROM cnn_genome WHERE id = " << _genome_id;

    //cout << query.str() << endl;

    mysql_exact_query(query.str());

    MYSQL_RES *result = mysql_store_result(exact_db_conn);

    if (result != NULL) {
        MYSQL_ROW row = mysql_fetch_row(result);

        genome_id = _genome_id; //this is also row[0]
        int column = 0;

        exact_id = atoi(row[++column]);

        vector<int> input_node_innovation_numbers;
        istringstream input_node_innovation_numbers_iss(row[++column]);
        //cout << "parsing input node innovation numbers" << endl;
        parse_array(input_node_innovation_numbers, input_node_innovation_numbers_iss);

        vector<int> softmax_node_innovation_numbers;
        istringstream softmax_node_innovation_numbers_iss(row[++column]);
        //cout << "parsing softmax node innovation numbers" << endl;
        parse_array(softmax_node_innovation_numbers, softmax_node_innovation_numbers_iss);

        istringstream generator_iss(row[++column]);
        generator_iss >> generator;

        istringstream normal_distribution_iss(row[++column]);
        normal_distribution_iss >> normal_distribution;

        istringstream hyperparameters_iss(row[++column]);
        //cout << "generator: " << generator << endl;

        epsilon = read_hexfloat(hyperparameters_iss);
        alpha = read_hexfloat(hyperparameters_iss);

        input_dropout_probability = read_hexfloat(hyperparameters_iss);
        hidden_dropout_probability = read_hexfloat(hyperparameters_iss);

        initial_mu = read_hexfloat(hyperparameters_iss);
        mu = read_hexfloat(hyperparameters_iss);
        mu_delta = read_hexfloat(hyperparameters_iss);

        initial_learning_rate = read_hexfloat(hyperparameters_iss);
        learning_rate = read_hexfloat(hyperparameters_iss);
        learning_rate_delta = read_hexfloat(hyperparameters_iss);

        initial_weight_decay = read_hexfloat(hyperparameters_iss);
        weight_decay = read_hexfloat(hyperparameters_iss);
        weight_decay_delta = read_hexfloat(hyperparameters_iss);


        velocity_reset = atoi(row[++column]);
        batch_size = atoi(row[++column]);

        epoch = atoi(row[++column]);
        max_epochs = atoi(row[++column]);
        reset_weights = atoi(row[++column]);

        padding = atoi(row[++column]);

        best_epoch = atoi(row[++column]);
        number_validation_images = atoi(row[++column]);
        best_validation_error = atof(row[++column]);
        best_validation_predictions = atoi(row[++column]);

        number_training_images = atoi(row[++column]);
        training_error = atof(row[++column]);
        training_predictions = atoi(row[++column]);

        number_test_images = atoi(row[++column]);
        test_error = atof(row[++column]);
        test_predictions = atoi(row[++column]);

        started_from_checkpoint = atoi(row[++column]);

        backprop_order.clear();

        generation_id = atoi(row[++column]);
        name = row[++column];
        checkpoint_filename = row[++column];
        output_filename = row[++column];

        istringstream generated_by_map_iss(row[++column]);
        read_map(generated_by_map_iss, generated_by_map);

        ostringstream node_query;
        node_query << "SELECT id FROM cnn_node WHERE genome_id = " << genome_id;
        //cout << node_query.str() << endl;

        mysql_exact_query(node_query.str());
        //cout << "node query was successful!" << endl;

        MYSQL_RES *node_result = mysql_store_result(exact_db_conn);
        //cout << "got result!" << endl;

        MYSQL_ROW node_row;
        while ((node_row = mysql_fetch_row(node_result)) != NULL) {
            int node_id = atoi(node_row[0]);
            //cout << "got node with id: " << node_id << endl;

            CNN_Node *node = new CNN_Node(node_id);
            nodes.push_back(node);

            if (find(input_node_innovation_numbers.begin(), input_node_innovation_numbers.end(), node->get_innovation_number()) != input_node_innovation_numbers.end()) {
                input_nodes.push_back(node);
            }

            if (find(softmax_node_innovation_numbers.begin(), softmax_node_innovation_numbers.end(), node->get_innovation_number()) != softmax_node_innovation_numbers.end()) {
                softmax_nodes.push_back(node);
            }
        }

        //cout << "got all nodes!" << endl;
        mysql_free_result(node_result);

        ostringstream edge_query;
        edge_query << "SELECT id FROM cnn_edge WHERE genome_id = " << genome_id;

        mysql_exact_query(edge_query.str());
        //cout << "edge query was successful!" << endl;

        MYSQL_RES *edge_result = mysql_store_result(exact_db_conn);
        //cout << "got result!" << endl;

        MYSQL_ROW edge_row;
        while ((edge_row = mysql_fetch_row(edge_result)) != NULL) {
            int edge_id = atoi(edge_row[0]);
            //cout << "got edge with id: " << edge_id << endl;

            CNN_Edge *edge = new CNN_Edge(edge_id);
            edges.push_back(edge);

            edge->set_nodes(nodes);
            edge->set_pools();
        }

        //cout << "got all edges!" << endl;
        mysql_free_result(edge_result);

        mysql_free_result(result);

    } else {
        cout << "Could not find genome with id: " << genome_id << "!" << endl;
        exit(1);
    }

    visit_nodes();

    if (epoch > 0) {
        //if this was saved at an epoch > 0, it has already been initialized
        started_from_checkpoint = true;
    }
}

void CNN_Genome::export_to_database(int _exact_id) {
    exact_id = _exact_id;

    ostringstream query;

    if (genome_id >= 0) {
        query << "REPLACE INTO cnn_genome SET id = " << genome_id << ",";
    } else {
        query << "INSERT INTO cnn_genome SET";
    }

    query << " exact_id = " << exact_id
        << ", input_node_innovation_numbers = '";

    for (uint32_t i = 0; i < input_nodes.size(); i++) {
        if (i != 0) query << " ";
        query << input_nodes[i]->get_innovation_number();
    }

    query << "', softmax_node_innovation_numbers = '";

    for (uint32_t i = 0; i < softmax_nodes.size(); i++) {
        if (i != 0) query << " ";
        query << softmax_nodes[i]->get_innovation_number();
    }


    query << "', generator = '" << generator << "'"
        << ", normal_distribution = '" << normal_distribution << "'"
        << ", hyperparameters = '";

    write_hexfloat(query, epsilon);
    query << " ";
    write_hexfloat(query, alpha);
    query << " ";
    write_hexfloat(query, input_dropout_probability);
    query << " ";
    write_hexfloat(query, hidden_dropout_probability);
    query << " ";
    write_hexfloat(query, initial_mu);
    query << " ";
    write_hexfloat(query, mu);
    query << " ";
    write_hexfloat(query, mu_delta);
    query << " ";
    write_hexfloat(query, initial_learning_rate);
    query << " ";
    write_hexfloat(query, learning_rate);
    query << " ";
    write_hexfloat(query, learning_rate_delta);
    query << " ";
    write_hexfloat(query, initial_weight_decay);
    query << " ";
    write_hexfloat(query, weight_decay);
    query << " ";
    write_hexfloat(query, weight_decay_delta);

    query << "'"
        << ", velocity_reset = '" << velocity_reset << "'"
        << ", batch_size = " << batch_size
        << ", epoch = " << epoch
        << ", max_epochs = " << max_epochs
        << ", reset_weights = " << reset_weights
        << ", padding = " << padding
        << ", best_epoch = " << best_epoch
        << ", number_validation_images = " << number_validation_images
        << ", best_validation_error = " << setprecision(15) << fixed << best_validation_error
        << ", best_validation_predictions = " << best_validation_predictions

        << ", number_training_images = " << number_training_images
        << ", training_error = " << setprecision(15) << fixed << training_error 
        << ", training_predictions = " << training_predictions

        << ", number_test_images = " << number_test_images
        << ", test_error = " << setprecision(15) << fixed << test_error 
        << ", test_predictions = " << test_predictions

        << ", started_from_checkpoint = " << started_from_checkpoint;

    //too much overhead for saving this and no use for it
    //query << ", backprop_order = ''";
    /*
    query << ", backprop_order = '";
    for (uint32_t i = 0; i < backprop_order.size(); i++) {
        if (i != 0) query << " ";
        query << setprecision(15) << backprop_order[i];
    }
    query << "'";
    */

    query << ", generation_id = " << generation_id
        << ", name = '" << name << "'"
        << ", checkpoint_filename = '" << checkpoint_filename << "'"
        << ", output_filename = '" << output_filename << "'"
        << ", generated_by_map = '";

    write_map(query, generated_by_map);

    query << "'";
    //cout << "query:\n" << query.str() << endl;

    mysql_exact_query(query.str());

    if (genome_id < 0) {
        genome_id = mysql_exact_last_insert_id(); //get last insert id from database
        cout << "setting genome id to: " << genome_id << endl;
    }

    for (uint32_t i = 0; i < nodes.size(); i++) {
        nodes[i]->export_to_database(exact_id, genome_id);
    }

    for (uint32_t i = 0; i < edges.size(); i++) {
        edges[i]->export_to_database(exact_id, genome_id);
    }
}

#endif

/**
 *  Iniitalize a genome from a set of nodes and edges
 */
CNN_Genome::CNN_Genome(int _generation_id, int _padding, int _number_training_images, int _number_validation_images, int _number_test_images, int seed, int _max_epochs, bool _reset_weights, int _velocity_reset, float _mu, float _mu_delta, float _learning_rate, float _learning_rate_delta, float _weight_decay, float _weight_decay_delta, int _batch_size, float _epsilon, float _alpha, float _input_dropout_probability, float _hidden_dropout_probability, const vector<CNN_Node*> &_nodes, const vector<CNN_Edge*> &_edges) {
    exact_id = -1;
    genome_id = -1;
    started_from_checkpoint = false;
    generator = minstd_rand0(seed);

    padding = _padding;
    number_training_images = _number_training_images;
    number_validation_images = _number_validation_images;
    number_test_images = _number_test_images;

    progress_function = NULL;

    velocity_reset = _velocity_reset;
 
    batch_size = _batch_size;
    epsilon = _epsilon;
    alpha = _alpha;

    input_dropout_probability = _input_dropout_probability;
    hidden_dropout_probability = _hidden_dropout_probability;

    mu = _mu;
    initial_mu = mu;
    mu_delta = _mu_delta;

    learning_rate = _learning_rate;
    initial_learning_rate = learning_rate;
    learning_rate_delta = _learning_rate_delta;

    weight_decay = _weight_decay;
    initial_weight_decay = weight_decay;
    weight_decay_delta = _weight_decay_delta;

    epoch = 0;
    max_epochs = _max_epochs;
    reset_weights = _reset_weights;

    best_epoch = 0;
    best_validation_predictions = 0;
    best_validation_error = EXACT_MAX_FLOAT;

    training_predictions = 0;
    training_error = EXACT_MAX_FLOAT;

    test_predictions = 0;
    test_error = EXACT_MAX_FLOAT;


    generation_id = _generation_id;

    name = "";
    output_filename = "";
    checkpoint_filename = "";

    nodes = _nodes;
    edges = _edges;

    cout << "creating genome with nodes.size(): " << nodes.size() << " and edges.size(): " << edges.size() << endl;

    input_nodes.clear();
    softmax_nodes.clear();

    for (uint32_t i = 0; i < nodes.size(); i++) {
        if (nodes[i]->is_input()) {
            //cout << "node was input!" << endl;
            input_nodes.push_back(nodes[i]);
        }

        if (nodes[i]->is_softmax()) {
            //cout << "node was softmax!" << endl;
            softmax_nodes.push_back(nodes[i]);
        }

        //cout << "resizing node " << node_copy->get_innovation_number() << " to " << batch_size << endl;
        nodes[i]->update_batch_size(batch_size);
    }

    for (uint32_t i = 0; i < edges.size(); i++) {
        if (!edges[i]->set_nodes(nodes)) {
            ostringstream error_message;
            error_message << "Error setting nodes, filter size was not correct. This should never happen.";
            throw runtime_error(error_message.str());
        }

        edges[i]->set_pools();

        //cout << "resizing edge " << edge_copy->get_innovation_number() << " to " << batch_size << endl;
        edges[i]->update_batch_size(batch_size);
    }
}


CNN_Genome::~CNN_Genome() {
    while (nodes.size() > 0) {
        CNN_Node *node = nodes.back();
        nodes.pop_back();

        delete node;
    }

    while (edges.size() > 0) {
        CNN_Edge *edge = edges.back();
        edges.pop_back();

        delete edge;
    }
    
    while (input_nodes.size() > 0) {
        input_nodes.pop_back();
    }

    while (softmax_nodes.size() > 0) {
        softmax_nodes.pop_back();
    }
    softmax_nodes.clear();
}

bool CNN_Genome::equals(CNN_Genome *other) const {
    for (int32_t i = 0; i < (int32_t)edges.size(); i++) {
        CNN_Edge *edge = edges[i];

        if (edge->is_disabled()) continue;

        bool found = false;

        for (int32_t j = 0; j < other->get_number_edges(); j++) {
            CNN_Edge *other_edge = other->get_edge(j);

            if (other_edge->is_disabled()) continue;

            if (edge->get_innovation_number() == other_edge->get_innovation_number()) {
                found = true;

                if (!edge->equals(other_edge)) return false;
            }
        }

        if (!found) return false;
    }

    //other may have edges not in this genome, need to check this as well

    for (int32_t i = 0; i < other->get_number_edges(); i++) {
        CNN_Edge *other_edge = other->get_edge(i);

        if (other_edge->is_disabled()) continue;

        bool found = false;
        
        for (int32_t j = 0; j < (int32_t)edges.size(); j++) {
            CNN_Edge* edge = edges[j];

            if (edge->is_disabled()) continue;

            if (edge->get_innovation_number() == other_edge->get_innovation_number()) {
                found = true;
            }
        }

        if (!found) return false;
    }

    return true;
}

int CNN_Genome::get_padding() const {
    return padding;
}

int CNN_Genome::get_number_training_images() const {
    return number_training_images;
}

int CNN_Genome::get_number_validation_images() const {
    return number_validation_images;
}

int CNN_Genome::get_number_test_images() const {
    return number_test_images;
}

int CNN_Genome::get_number_weights() const {
    int number_weights = 0;

    for (uint32_t i = 0; i < edges.size(); i++) {
        if (edges[i]->get_type() == CONVOLUTIONAL && edges[i]->is_reachable()) {
            number_weights += edges[i]->get_filter_x() * edges[i]->get_filter_y();
        }
    }

    return number_weights;
}

int CNN_Genome::get_operations_estimate() const {
    int operations_estimate = 0;

    float random_cost = 100.0;
    float if_cost = 15.0;
    float multiply_cost = 7.0;
    float add_cost = 1.0;

    for (uint32_t i = 0; i < nodes.size(); i++) {
        //propagate forward has 1 RELU and 1 DropOut per value in node
        //RELU is 2 ifs, 1 multiply
        //dropout is 1 if, 1 random
        if (!nodes[i]->is_reachable()) continue;

        operations_estimate += nodes[i]->get_size_x() * nodes[i]->get_size_y() * (3.0 * if_cost + multiply_cost + random_cost);
    }

    for (uint32_t i = 0; i < edges.size(); i++) {
        //propagate forward has 1 multiply 1 add per input to output
        //propagate backward has 3 multiplies and 2 adds per output to input
        if (!edges[i]->is_reachable()) continue;

        bool reverse_filter_x = edges[i]->is_reverse_filter_x();
        bool reverse_filter_y = edges[i]->is_reverse_filter_y();


        //TODO: calculate differently for POOLING nodes
        if (edges[i]->get_type() == CONVOLUTIONAL) {
            float propagate_count;

            if (reverse_filter_x && reverse_filter_y) {
                propagate_count = edges[i]->get_filter_x() * edges[i]->get_filter_y() * edges[i]->get_input_node()->get_size_x() * edges[i]->get_input_node()->get_size_y();
            } else if (reverse_filter_x) {
                propagate_count = edges[i]->get_filter_x() * edges[i]->get_filter_y() * edges[i]->get_input_node()->get_size_x() * edges[i]->get_output_node()->get_size_y();
            } else if (reverse_filter_y) {
                propagate_count = edges[i]->get_filter_x() * edges[i]->get_filter_y() * edges[i]->get_output_node()->get_size_x() * edges[i]->get_input_node()->get_size_y();
            } else {
                propagate_count = edges[i]->get_filter_x() * edges[i]->get_filter_y() * edges[i]->get_output_node()->get_size_x() * edges[i]->get_output_node()->get_size_y();
            }

            operations_estimate += propagate_count * ((4.0 * multiply_cost) + (3.0 * add_cost));

        } else {
            float propagate_count;

            if (reverse_filter_x && reverse_filter_y) {
                propagate_count = edges[i]->get_input_node()->get_size_x() * edges[i]->get_input_node()->get_size_y();
            } else if (reverse_filter_y) {
                propagate_count = edges[i]->get_input_node()->get_size_x() * edges[i]->get_output_node()->get_size_y();
            } else if (reverse_filter_x) {
                propagate_count = edges[i]->get_output_node()->get_size_x() * edges[i]->get_input_node()->get_size_y();
            } else {
                propagate_count = edges[i]->get_output_node()->get_size_x() * edges[i]->get_output_node()->get_size_y();
            }

            operations_estimate += propagate_count * 2.0 * (16.0 * add_cost) + (4.0 * multiply_cost);
        }

        //update weights has 4 multiplies 4 adds and 2 ifs per weight 
        operations_estimate += edges[i]->get_filter_x() * edges[i]->get_filter_y() * (4.0 * multiply_cost + 4.0 * add_cost + 2.0 * if_cost);
    }

    return operations_estimate;
}


int CNN_Genome::get_generation_id() const {
    return generation_id;
}

float CNN_Genome::get_best_validation_error() const {
    return best_validation_error;
}

int CNN_Genome::get_max_epochs() const {
    return max_epochs;
}

int CNN_Genome::get_epoch() const {
    return epoch;
}

int CNN_Genome::get_best_epoch() const {
    return best_epoch;
}

float CNN_Genome::get_best_validation_rate() const {
    if (best_validation_error == EXACT_MAX_FLOAT) return 0.0;

    return 100.0 * (float)best_validation_predictions / (float)number_validation_images;
}

int CNN_Genome::get_best_validation_predictions() const {
    return best_validation_predictions;
}


float CNN_Genome::get_training_error() const {
    return training_error;
}

float CNN_Genome::get_training_rate() const {
    if (training_error == EXACT_MAX_FLOAT) return 0.0;

    return 100.0 * (float)training_predictions / (float)number_training_images;
}

int CNN_Genome::get_training_predictions() const {
    return training_predictions;
}


float CNN_Genome::get_test_error() const {
    return test_error;
}

float CNN_Genome::get_test_rate() const {
    if (test_error == EXACT_MAX_FLOAT) return 0.0;

    return 100.0 * (float)test_predictions / (float)number_test_images;
}

int CNN_Genome::get_test_predictions() const {
    return test_predictions;
}

const vector<CNN_Node*> CNN_Genome::get_nodes() const {
    return nodes;
}

const vector<CNN_Edge*> CNN_Genome::get_edges() const {
    return edges;
}

void CNN_Genome::get_node_copies(vector<CNN_Node*> &node_copies) const {
    node_copies.clear();

    for (uint32_t i = 0; i < nodes.size(); i++) {
        node_copies.push_back(nodes[i]->copy());
    }
}

void CNN_Genome::get_edge_copies(vector<CNN_Edge*> &edge_copies) const {
    edge_copies.clear();

    for (uint32_t i = 0; i < edges.size(); i++) {
        edge_copies.push_back(edges[i]->copy());
    }
}

CNN_Node* CNN_Genome::get_node(int node_position) {
    return nodes.at(node_position);
}

CNN_Edge* CNN_Genome::get_edge(int edge_position) {
    return edges.at(edge_position);
}

int CNN_Genome::get_number_enabled_pooling_edges() const {
    int n_enabled_pooling_edges = 0;
    for (uint32_t i = 0; i < edges.size(); i++) {
        if (edges[i]->is_reachable() && edges[i]->get_type() == POOLING) n_enabled_pooling_edges++;
    }
    return n_enabled_pooling_edges;
}

int CNN_Genome::get_number_enabled_convolutional_edges() const {
    int n_enabled_convolutional_edges = 0;
    for (uint32_t i = 0; i < edges.size(); i++) {
        if (edges[i]->is_reachable() && edges[i]->get_type() == CONVOLUTIONAL) n_enabled_convolutional_edges++;
    }
    return n_enabled_convolutional_edges;
}


int CNN_Genome::get_number_enabled_edges() const {
    int n_enabled_edges = 0;
    for (uint32_t i = 0; i < edges.size(); i++) {
        if (edges[i]->is_reachable()) n_enabled_edges++;
    }
    return n_enabled_edges;
}

int CNN_Genome::get_number_enabled_nodes() const {
    int n_enabled_nodes = 0;
    for (uint32_t i = 0; i < nodes.size(); i++) {
        if (nodes[i]->is_reachable()) n_enabled_nodes++;
    }
    return n_enabled_nodes;
}

int CNN_Genome::get_number_edges() const {
    return edges.size();
}

int CNN_Genome::get_number_nodes() const {
    return nodes.size();
}

int CNN_Genome::get_number_softmax_nodes() const {
    return softmax_nodes.size();
}

int CNN_Genome::get_number_input_nodes() const {
    return input_nodes.size();
}

void CNN_Genome::add_node(CNN_Node* node) {
    node->update_batch_size(batch_size);
    nodes.insert( upper_bound(nodes.begin(), nodes.end(), node, sort_CNN_Nodes_by_depth()), node );

}

void CNN_Genome::add_edge(CNN_Edge* edge) {
    edge->update_batch_size(batch_size);
    edges.insert( upper_bound(edges.begin(), edges.end(), edge, sort_CNN_Edges_by_depth()), edge );
}

bool CNN_Genome::disable_edge(int edge_position) {
    CNN_Edge *edge = edges.at(edge_position);
    
    /*
    int number_inputs = edge->get_output_node()->get_number_inputs();

    if (number_inputs == 1) {
        return false;
    } else if (number_inputs == 0) {
        cerr << "ERROR: disabling an edge where the target had 0 inputs." << endl;
        cerr << "\tThis should never happen!" << endl;
        exit(1);
    }
    */

    if (edge->is_disabled()) {
        cout << "\t\tcould not disable edge " << edge_position << " because it was already disabled!" << endl;
        return true;
    } else {
        cout << "\t\tdisabling edge: " << edge_position << endl;
        edge->disable();
        return true;
    }
}

void CNN_Genome::resize_edges_around_node(int node_innovation_number) {
    for (uint32_t i = 0; i < edges.size(); i++) {
        CNN_Edge *edge = edges[i];

        if (edge->get_input_node()->get_innovation_number() == node_innovation_number) {
            cout << "\tresizing edge with innovation number " << edge->get_innovation_number() << " as input to node with innovation number " << node_innovation_number << endl;
            edge->resize();
        }

        if (edge->get_output_node()->get_innovation_number() == node_innovation_number) {
            cout << "\tresizing edge with innovation number " << edge->get_innovation_number() << " as output from node with innovation number " << node_innovation_number << endl;
            edge->resize();
        }
    }
}

vector<CNN_Node*> CNN_Genome::get_reachable_nodes() {
    vector<CNN_Node*> reachable_nodes;

    for (uint32_t i = 0; i < nodes.size(); i++) {
        if (nodes[i]->is_reachable()) {
            reachable_nodes.push_back(nodes[i]);
        }
    }

    return reachable_nodes;
}

vector<CNN_Node*> CNN_Genome::get_disabled_nodes() {
    vector<CNN_Node*> disabled_nodes;

    for (uint32_t i = 0; i < nodes.size(); i++) {
        if (nodes[i]->is_disabled()) {
            disabled_nodes.push_back(nodes[i]);
        }
    }

    return disabled_nodes;
}


vector<CNN_Node*> CNN_Genome::get_reachable_hidden_nodes() {
    vector<CNN_Node*> reachable_nodes;

    for (uint32_t i = 0; i < nodes.size(); i++) {
        if (nodes[i]->is_reachable() && nodes[i]->is_hidden()) {
            reachable_nodes.push_back(nodes[i]);
        }
    }

    return reachable_nodes;
}

vector<CNN_Edge*> CNN_Genome::get_reachable_edges() {
    vector<CNN_Edge*> reachable_edges;

    for (uint32_t i = 0; i < edges.size(); i++) {
        if (edges[i]->is_reachable()) {
            reachable_edges.push_back(edges[i]);
        }
    }

    return reachable_edges;
}

vector<CNN_Edge*> CNN_Genome::get_disabled_edges() {
    vector<CNN_Edge*> disabled_edges;

    for (uint32_t i = 0; i < edges.size(); i++) {
        if (edges[i]->is_disabled()) {
            disabled_edges.push_back(edges[i]);
        }
    }

    return disabled_edges;
}



bool CNN_Genome::sanity_check(int type) {
    //check to see if all edge filters are the correct size
    for (uint32_t i = 0; i < edges.size(); i++) {
        if (!edges[i]->is_filter_correct()) {
            cerr << "SANITY CHECK FAILED! edges[" << i << "] had incorrect filter size!" << endl;
            return false;
        }

        if (edges[i]->get_batch_size() != batch_size) {
            cerr << "SANITY CHECK FAILED! edges[" << i << "] had batch size: " << edges[i]->get_batch_size() << " != genome batch size: " << batch_size << endl;
            cerr << edges[i] << endl;
            return false;
        }
    }

    for (uint32_t i = 0; i < nodes.size(); i++) {
        if (nodes[i]->get_batch_size() != batch_size) {
            cerr << "SANITY CHECK FAILED! nodes[" << i << "] had batch size: " << nodes[i]->get_batch_size() << " != genome batch size: " << batch_size << endl;
            cerr << nodes[i] << endl;
            return false;
        }

        if (!nodes[i]->vectors_correct()) {
            cerr << "SANITY CHECK FAILED! nodes[" << i << "] had incorrectly sized vectors!" << endl;
            cerr << nodes[i] << endl;
            return false;
        }
    }


    //check for duplicate edges, make sure edge size is sane
    for (uint32_t i = 0; i < edges.size(); i++) {
        if (edges[i]->get_filter_x() <= 0 || edges[i]->get_filter_x() > 100) {
            cerr << "ERROR: edge failed sanity check, reached impossible filter_x (<= 0 or > 100)" << endl;
            cerr << "edge in position " << i << " with innovation number: " << edges[i]->get_innovation_number() << endl;
            cerr << "filter_x: " << edges[i]->get_filter_x() << ", filter_y: " << edges[i]->get_filter_y() << endl;
            return false;
        }

        if (edges[i]->get_filter_y() <= 0 || edges[i]->get_filter_y() > 100) {
            cerr << "ERROR: edge failed sanity check, reached impossible filter_y (<= 0 or > 100)" << endl;
            cerr << "edge in position " << i << " with innovation number: " << edges[i]->get_innovation_number() << endl;
            cerr << "filter_x: " << edges[i]->get_filter_x() << ", filter_y: " << edges[i]->get_filter_y() << endl;
            return false;
        }

        for (uint32_t j = i + 1; j < edges.size(); j++) {
            if (edges[i]->get_innovation_number() == edges[j]->get_innovation_number()) {
                cerr << "SANITY CHECK FAILED! edges[" << i << "] and edges[" << j << "] have the same innovation number: " << edges[i]->get_innovation_number() << endl;
                return false;
            }
        }
    }

    //check for duplicate nodes, make sure node size is sane
    for (uint32_t i = 0; i < nodes.size(); i++) {
        if (nodes[i]->get_size_x() <= 0 || nodes[i]->get_size_x() > 100) {
            cerr << "ERROR: node failed sanity check, reached impossible size_x (<= 0 or > 100)" << endl;
            cerr << "node in position " << i << " with innovation number: " << nodes[i]->get_innovation_number() << endl;
            cerr << "size_x: " << nodes[i]->get_size_x() << ", size_y: " << nodes[i]->get_size_y() << endl;
            return false;
        }

        if (nodes[i]->get_size_y() <= 0 || nodes[i]->get_size_y() > 100) {
            cerr << "ERROR: node failed sanity check, reached impossible size_y (<= 0 or > 100)" << endl;
            cerr << "node in position " << i << " with innovation number: " << nodes[i]->get_innovation_number() << endl;
            cerr << "size_x: " << nodes[i]->get_size_x() << ", size_y: " << nodes[i]->get_size_y() << endl;
            return false;
        }

        for (uint32_t j = i + 1; j < nodes.size(); j++) {
            if (nodes[i]->get_innovation_number() == nodes[j]->get_innovation_number()) {
                cerr << "SANITY CHECK FAILED! nodes[" << i << "] and nodes[" << j << "] have the same innovation number: " << nodes[i]->get_innovation_number() << endl;
                return false;
            }
        }
    }

    if (type == SANITY_CHECK_AFTER_GENERATION) {
        for (uint32_t i = 0; i < edges.size(); i++) {
            if (edges[i]->has_zero_weight() && edges[i]->get_type() == CONVOLUTIONAL) {
                cerr << "WARNING before after_generation!" << endl;
                cerr << "edge type: " << edges[i]->get_type() << endl;
                cerr << "edge in position " << i << " with innovation number: " << edges[i]->get_innovation_number() << endl;
                cerr << "filter_x: " << edges[i]->get_filter_x() << ", filter_y: " << edges[i]->get_filter_y() << endl;
                cerr << "sum of weights was 0" << endl;
                edges[i]->initialize_weights(generator, normal_distribution);
                edges[i]->save_best_weights();
                //return false;
            }
        }
        //cout << "passed checking zero best weights" << endl;

    } else if (type == SANITY_CHECK_BEFORE_INSERT) {
        //seems bias can go to 0
        /*
        for (uint32_t i = 0; i < nodes.size(); i++) {
            if (nodes[i]->has_zero_best_bias()) {
                cerr << "ERROR before insert!" << endl;
                cerr << "node in position " << i << " with innovation number: " << nodes[i]->get_innovation_number() << endl;
                cerr << "sum of best bias was 0" << endl;
                cerr << "size_x: " << nodes[i]->get_size_x() << ", size_y: " << nodes[i]->get_size_y() << endl;
                return false;
            }
        }
        */
        //cout << "passed checking zero best weights" << endl;

        /*
        for (uint32_t i = 0; i < edges.size(); i++) {
            if (edges[i]->has_zero_best_weight()) {
                cerr << "ERROR before insert!" << endl;
                cerr << "ERROR: edge in position " << i << " with innovation number: " << edges[i]->get_innovation_number() << endl;
                cerr << "sum of best weights was 0" << endl;
                cerr << "filter_x: " << edges[i]->get_filter_x() << ", filter_y: " << edges[i]->get_filter_y() << endl;
                return false;
            }
        }
        */
        //cout << "passed checking zero best weights" << endl;
    }


    //check to see if total_inputs on each node are correct (equal to non-disabled input edges)
    for (uint32_t i = 0; i < nodes.size(); i++) {
        int number_inputs = nodes[i]->get_number_inputs();

        //cout << "\t\tcounting inputs for node " << i << " (innovation number: " << nodes[i]->get_innovation_number() << ") -- should find " << number_inputs << endl;

        int counted_inputs = 0;
        for (uint32_t j = 0; j < edges.size(); j++) {
            if (!edges[j]->is_reachable()) {
                //cout << "\t\t\tedge " << j << " is disabled (" << edges[j]->get_input_innovation_number() << " to " << edges[j]->get_output_innovation_number() << ")" << endl;
                continue;
            }

            if (edges[j]->get_output_node()->get_innovation_number() == nodes[i]->get_innovation_number()) {
                //cout << "\t\t\tedge " << j << " (" << edges[j]->get_input_innovation_number() << " to " << edges[j]->get_output_innovation_number() << ") output matches node innovation number" << endl;

                if (edges[j]->get_output_node() != nodes[i]) {
                    //these should be equal
                    cerr << "SANITY CHECK FAILED! edges[" << j << "]->output_node had the same innovation number as nodes[" << j << "] but the pointers were not the same!" << endl;
                    cerr << "EDGE[" << j << "]: " << endl;
                    cerr << edges[j] << endl << endl;
                    cerr << "NODE[" << i << "]: " << endl;
                    cerr << nodes[i] << endl << endl;
                    return false;
                }
                counted_inputs++;
            } else {
                //cout << "\t\t\tedge " << j << " (" << edges[j]->get_input_innovation_number() << " to " << edges[j]->get_output_innovation_number() << ") output does not match node innovation number" << endl;
            }
        }

        if (counted_inputs != number_inputs) {
            cerr << "SANITY CHECK FAILED! nodes[" << i << "] had total inputs: " << number_inputs << " but " << counted_inputs << " inputs were counted. " << endl;
            cerr << "node innovation number: " << nodes[i]->get_innovation_number() << endl;
            for (uint32_t j = 0; j < edges.size(); j++) {
                if (edges[j]->get_output_node()->get_innovation_number() == nodes[i]->get_innovation_number()) {
                    cerr << "\tedge with innovation number " << edges[j]->get_innovation_number() << " had node as output, edge disabled? " << edges[j]->is_disabled() << endl;
                }
            }
            return false;
        }
    }

    return true;
}

bool CNN_Genome::visit_nodes() {
    //cout << "visiting nodes!" << endl;

    //check to see there is a path to from the input to each output
    //check to see if there is a path from output to inputs
    //if a node and edge is not forward and reverse visitable, then we can ignore it

    //sort nodes and edges
    sort(edges.begin(), edges.end(), sort_CNN_Edges_by_depth());


    for (uint32_t i = 0; i < nodes.size(); i++) {
        nodes[i]->set_unvisited();
    }

    for (uint32_t i = 0; i < edges.size(); i++) {
        edges[i]->set_unvisited();
    }

    for (uint32_t i = 0; i < input_nodes.size(); i++) {
        //cerr << "reverse visiting input node: " << i << endl;
        input_nodes[i]->forward_visit();
    }

    for (uint32_t i = 0; i < softmax_nodes.size(); i++) {
        //cerr << "reverse visiting softmax node: " << i << endl;
        softmax_nodes[i]->reverse_visit();
    }

    for (uint32_t i = 0; i < edges.size(); i++) {
        if (!edges[i]->is_disabled()) {
            if (edges[i]->get_input_node()->is_forward_visited() && edges[i]->get_input_node()->is_enabled()) {
                //cerr << "visiting edge: " << i << endl;
                edges[i]->forward_visit();
                edges[i]->get_output_node()->forward_visit();
            }
        }
    }

    sort(edges.begin(), edges.end(), sort_CNN_Edges_by_output_depth());
    for (int32_t i = edges.size() - 1; i >= 0; i--) {
        if (!edges[i]->is_disabled()) {

            if (edges[i]->get_output_node()->is_reverse_visited() && edges[i]->get_output_node()->is_enabled()) {
                edges[i]->reverse_visit();
                edges[i]->get_input_node()->reverse_visit();
            }
        }
    }

    sort(edges.begin(), edges.end(), sort_CNN_Edges_by_depth());

    for (uint32_t i = 0; i < edges.size(); i++) {
        //cout << "\t\tedge " << edges[i]->get_innovation_number() << " is forward visited: " << edges[i]->is_forward_visited() << ", is reverse visited: " << edges[i]->is_reverse_visited() << ", is reachable: " << edges[i]->is_reachable() << endl;

        if (edges[i]->is_reachable()) {
            edges[i]->get_input_node()->add_output();
            edges[i]->get_output_node()->add_input();
        }
    }

    /*
    for (uint32_t i = 0; i < nodes.size(); i++) {
        cout << "\t\tnode " << nodes[i]->get_innovation_number() << " is forward visited: " << nodes[i]->is_forward_visited() << ", is reverse visited: " << nodes[i]->is_reverse_visited() << ", is reachable: " << nodes[i]->is_reachable() << ", number inputs: " << nodes[i]->get_number_inputs() << ", number outputs: " << nodes[i]->get_number_outputs() << endl;
    }
    */

    for (uint32_t i = 0; i < softmax_nodes.size(); i++) {
        if (!softmax_nodes[i]->is_reachable()) {
            return false;
        }
    }

    return true;
}

void CNN_Genome::evaluate_images(const ImagesInterface &images, const vector<int> &batch, vector< vector<float> > &predictions, int offset) {
    bool training = false;
    bool accumulate_test_statistics = false;

    for (uint32_t i = 0; i < nodes.size(); i++) {
        nodes[i]->reset();
    }

    for (uint32_t channel = 0; channel < input_nodes.size(); channel++) {
        input_nodes[channel]->set_values(images, batch, channel, training, accumulate_test_statistics, input_dropout_probability, generator);
    }

    for (uint32_t i = 0; i < edges.size(); i++) {
        edges[i]->propagate_forward(training, accumulate_test_statistics, epsilon, alpha, training, hidden_dropout_probability, generator);
    }

    //may be less images than in a batch if the total number of images is not divisible by the batch size
    for (int32_t batch_number = 0; batch_number < batch.size(); batch_number++) {
        int expected_class = images.get_classification(batch[batch_number]);

        //cout << "before softmax max, batch number: " << batch_number << " -- ";
        float softmax_max = softmax_nodes[0]->get_value_in(batch_number, 0, 0);
        //cout << " " << setw(15) << fixed << setprecision(6) << softmax_nodes[0]->get_value_in(batch_number, 0,0);


        for (uint32_t i = 1; i < softmax_nodes.size(); i++) {
            //cout << " " << setw(15) << fixed << setprecision(6) << softmax_nodes[i]->get_value_in(batch_number, 0,0);
            if (softmax_nodes[i]->get_value_in(batch_number, 0, 0) > softmax_max) {
                softmax_max = softmax_nodes[i]->get_value_in(batch_number, 0, 0);
            }
        }
        //cout << endl;

        //cout << "after softmax max:" << endl;
        float softmax_sum = 0.0;
        for (uint32_t i = 0; i < softmax_nodes.size(); i++) {
            float value = softmax_nodes[i]->get_value_in(batch_number, 0, 0);
            float previous = value;

            if (isnan(value)) {
                cerr << "ERROR: value was NAN before exp!" << endl;
                exit(1);
            }
            //cout << " value - softmax_max: " << value - softmax_max << endl;

            value = exact_exp(value - softmax_max);

            //cout << " " << setw(15) << fixed << setprecision(6) << value;
            if (isnan(value)) {
                cerr << "ERROR: value was NAN AFTER exp! previously: " << previous << endl;
                exit(1);
            }

            softmax_nodes[i]->set_value_in(batch_number, 0, 0, value);
            //cout << "\tvalue " << softmax_nodes[i]->get_innovation_number() << ": " << softmax_nodes[i]->get_value_in(batch_number, 0,0) << endl;
            softmax_sum += value;

            if (isnan(softmax_sum)) {
                cerr << "ERROR: softmax_sum was NAN AFTER add!" << endl;
                exit(1);
            }
        }
        //cout << endl;

        if (softmax_sum == 0) {
            cout << "ERROR! softmax sum == 0" << endl;
            exit(1);
        }

        //cout << "softmax sum: " << softmax_sum << endl;

        float max_value = -numeric_limits<float>::max();
        int predicted_class = -1;

        //cout << "error:          ";
        for (int32_t i = 0; i < (int32_t)softmax_nodes.size(); i++) {
            float value = softmax_nodes[i]->get_value_in(batch_number, 0,0) / softmax_sum;
            //cout << "\tvalue " << softmax_nodes[i]->get_innovation_number() << ": " << softmax_nodes[i]->get_value_in(0,0) << endl;

            if (isnan(value)) {
                cerr << "ERROR: value was NAN AFTER divide by softmax_sum, previously: " << softmax_nodes[i]->get_value_in(batch_number, 0,0) << endl;
                cerr << "softmax_sum: " << softmax_sum << endl;
                exit(1);
            }

            softmax_nodes[i]->set_value_in(batch_number, 0, 0,  value);
        }

        for (int32_t i = 0; i < (int32_t)softmax_nodes.size(); i++) {
            float value = softmax_nodes[i]->get_value_in(batch_number, 0,0);
             //softmax_nodes[i]->print(cout);

            int target = 0.0;
            if (i == expected_class) {
                target = 1.0;
            }
            float error = value - target;
            float gradient = value * (1 - value);

            //if (training) cout << "\t" << softmax_nodes[i]->get_innovation_number() << " -- batch number: " << batch_number << ", value: " << value << ", error: " << error << ", gradient: " << gradient << endl;

            softmax_nodes[i]->set_error_in(batch_number, 0, 0, error * gradient);
            //softmax_nodes[i]->set_gradient_in(batch_number, 0, 0, gradient);

            if (value > max_value) {
                predicted_class = i;
                max_value = value;
            }

            predictions[batch[batch_number] - offset][i] = value;
        }
    }
}

void calculate_softmax(const vector<float> &values_in, vector<float> &values_out, vector<float> &gradient, int expected_class, int &predicted_class, float &entropy) {
    float softmax_max = values_in[0];
    predicted_class = 0;

    for (uint32_t i = 1; i < values_in.size(); i++) {
        if (values_in[i] > softmax_max) {
            softmax_max = values_in[i];
            predicted_class = i;
        }
    }

    float softmax_sum = 0.0;
    for (uint32_t i = 0; i < values_in.size(); i++) {
        values_out[i] = exact_exp(values_in[i] - softmax_max);
        //values_out[i] = exact_exp(values_in[i]);
        softmax_sum += values_out[i];
    }

    if (softmax_sum == 0 || isinf(softmax_sum) || isnan(softmax_sum)) {
        cerr << "ERROR! softmax sum was " << softmax_sum << endl;
        cerr << "values_in/values_out:" << endl;
        for (uint32_t i = 0; i < values_in.size(); i++) {
            cerr << "\tvalues_in[" << i << "]: " << values_in[i] << ", values_out[" << i << "]: " << values_out[i] << endl;
        }
        exit(1);
    }

    entropy = 0.0;
    for (uint32_t i = 0; i < values_in.size(); i++) {
        values_out[i] /= softmax_sum;

//        entropy -= values_out[i] * log(values_out[i]);
    }

    for (uint32_t i = 0; i < values_in.size(); i++) {
        if (i == expected_class) {
            gradient[i] = values_out[i] - 1;

        } else {
            gradient[i] = values_out[i];
        }

        //float beta = 1.0;
        //apply confidence penalty
        //cout << "entropy: " << entropy << ", expected class probability: " << values_out[i] << ", regular gradient: " << gradient[i];
        //gradient[i] += beta * values_out[i] * (-log(values_out[i]) - entropy);
        //cout << ", after penalty: " << gradient[i] << endl;

        //cout << "values_out[" << i << "]: " << values_out[i] << ", gradient before entropy[" << i << "]: " << gradient[i];


        //cout << ", gradient after entropy[" << i << "]: " << gradient[i] << endl;

        /*
        if (modified_softmax) {
            gradient[i] *= values_out[i] * (1.0 - values_out[i]);
        }
        */
    }

    /*
    cerr << "values_in:  ";
    for (uint32_t i = 0; i < values_in.size(); i++) {
        cerr << fixed << setw(11) << setprecision(7) << values_in[i];
    }
    cerr << endl;

    cerr << "values_out: ";
    for (uint32_t i = 0; i < values_out.size(); i++) {
        cerr << fixed << setw(11) << setprecision(7) << values_out[i];
    }
    cerr << endl;
    */
}


void CNN_Genome::check_gradients(const ImagesInterface &images) {
    vector<int> batch;
    for (uint32_t i = 0; i < batch_size; i++) {
        batch.push_back(i);
    }

    float analytic_error = 0.0;
    int analytic_predictions = 0;
    evaluate_images(images, batch, false, analytic_error, analytic_predictions, true);

    cout << "after initial evaluate, analytic_error: " << analytic_error << ", analytic_predictions: " << analytic_predictions << endl;

    for (int32_t i = edges.size() - 1; i >= 0; i--) {
        edges[i]->propagate_backward(false, mu, learning_rate, epsilon);
    }

    analytic_error = 0.0;
    analytic_predictions = 0;
    evaluate_images(images, batch, false, analytic_error, analytic_predictions, true);
    cout << "after backpropagate, analytic_error: " << analytic_error << ", analytic_predictions: " << analytic_predictions << endl;

    //test the softmax layer
    vector<float> values_in(softmax_nodes.size());
    vector<float> values_out(softmax_nodes.size());
    vector<float> analytic_softmax_gradient(softmax_nodes.size());
    vector<float> numeric_softmax_gradient1(softmax_nodes.size());
    vector<float> numeric_softmax_gradient2(softmax_nodes.size());
    vector<float> temp_gradients(softmax_nodes.size());


    int numeric_predictions;
    float diff = 1e-4;

    //may be less images than in a batch if the total number of images is not divisible by the batch size
    for (int32_t batch_number = 0; batch_number < batch.size(); batch_number++) {
        int expected_class = images.get_classification(batch[batch_number]);

        for (uint32_t i = 0; i < softmax_nodes.size(); i++) {
            values_in[i] = softmax_nodes[i]->get_value_in(batch_number, 0, 0);
        }

        int predicted_class = 0;
        float entropy = 0.0;
        calculate_softmax(values_in, values_out, analytic_softmax_gradient, expected_class, predicted_class, entropy);

        /*
        cerr << "values_in/values_out:" << endl;
        for (uint32_t i = 0; i < values_in.size(); i++) {
            cerr << "\tvalues_in[" << i << "]: " << values_in[i] << ", values_out[" << i << "]: " << values_out[i] << endl;
        }
        */

        cerr << "expected class: " << expected_class << endl;
        for (uint32_t j = 0; j < softmax_nodes.size(); j++) {
            values_in[j] += diff;
            calculate_softmax(values_in, values_out, temp_gradients, expected_class, predicted_class, entropy);

            float error1 = -log(values_out[expected_class]);

            values_in[j] -= 2.0 * diff;
            calculate_softmax(values_in, values_out, temp_gradients, expected_class, predicted_class, entropy);

            float error2 = -log(values_out[expected_class]);

            values_in[j] += diff;

            float numeric_softmax_gradient = (error1 - error2) / (2.0 * diff);
            float relative_error = (fabs(analytic_softmax_gradient[j]) - fabs(numeric_softmax_gradient)) / fmax(fabs(analytic_softmax_gradient[j]), fabs(numeric_softmax_gradient));

            cerr << "ag[" << j << "]: " << fixed << setw(11) << setprecision(7) << analytic_softmax_gradient[j] << ", ng[" << j << "]: " << fixed << setw(11) << setprecision(7) << numeric_softmax_gradient << ", re: " << fixed << setw(11) << setprecision(7) << relative_error << endl;
        }
    }


    for (int32_t i = 0; i < edges.size(); i++) {
        if (edges[i]->is_reachable() && edges[i]->get_type() == CONVOLUTIONAL) {
            //cout << "CHECKING GRADIENTS ON EDGE: " << endl;
            edges[i]->print(cout);
            //edges[i]->get_output_node()->print(cout);
            //edges[i]->get_input_node()->print(cout);

            for (int32_t j = 0; j < edges[i]->get_filter_size(); j++) {
                //cout << "adding " << diff << " to weight: " << edges[i]->get_weight(j) << endl;
                edges[i]->update_weight(j, diff);
                float numeric_error1 = 0.0;
                evaluate_images(images, batch, false, numeric_error1, numeric_predictions, true);
                //cout << "finished first evaluation!" << endl;
                //cout << "numeric_error1: " << numeric_error1 << endl;

                edges[i]->update_weight(j, -1.0 * diff);
                float check_error = 0.0;
                evaluate_images(images, batch, false, check_error, numeric_predictions, true);

                //cout << "adding " << -2.0 * diff << " to weight: " << edges[i]->get_weight(j) << endl;
                edges[i]->update_weight(j, -1.0 * diff);
                float numeric_error2 = 0.0;
                evaluate_images(images, batch, false, numeric_error2, numeric_predictions, true);
                //cout << "finished second evaluation!" << endl;
                //cout << "numeric_error2: " << numeric_error2 << endl;

                edges[i]->update_weight(j, diff);

                float analytic_gradient = edges[i]->get_weight_update(j);

                float numeric_gradient = (numeric_error1 - numeric_error2) / (2.0 * diff);

                float relative_error = fabs(analytic_gradient - numeric_gradient) / fmax(fabs(analytic_gradient), fabs(numeric_gradient));

                cerr << "edge[" << i << "], weight[" << j << "]: ae: " << fixed << setw(11) << setprecision(7) << analytic_error << ", ce: " << fixed << setw(11) << setprecision(7) << check_error << ", ne1: " << fixed << setw(11) << setprecision(7) << numeric_error1 << ", ne2: " << numeric_error2 << ", ng: " << fixed << setw(11) << setprecision(7) << numeric_gradient << ", ag: " << fixed << setw(11) << setprecision(7) << analytic_gradient << ", relative_error: " << fixed << setw(11) << setprecision(7) << relative_error << endl;
            }
        }
    }
}

void CNN_Genome::evaluate_images(const ImagesInterface &images, const vector<int> &batch, bool training, float &total_error, int &correct_predictions, bool accumulate_test_statistics) {
    for (uint32_t i = 0; i < nodes.size(); i++) {
        nodes[i]->reset();
    }

    for (uint32_t channel = 0; channel < input_nodes.size(); channel++) {
        input_nodes[channel]->set_values(images, batch, channel, training, accumulate_test_statistics, input_dropout_probability, generator);
    }

    for (uint32_t i = 0; i < edges.size(); i++) {
        edges[i]->propagate_forward(training, accumulate_test_statistics, epsilon, alpha, training, hidden_dropout_probability, generator);
    }

    vector<float> values_in(softmax_nodes.size());
    vector<float> values_out(softmax_nodes.size());
    vector<float> gradients(softmax_nodes.size());

    //may be less images than in a batch if the total number of images is not divisible by the batch size
    for (int32_t batch_number = 0; batch_number < batch.size(); batch_number++) {
        int expected_class = images.get_classification(batch[batch_number]);

        for (uint32_t i = 0; i < softmax_nodes.size(); i++) {
            values_in[i] = softmax_nodes[i]->get_value_in(batch_number, 0, 0);
        }

        int predicted_class = 0;
        float entropy = 0.0;
        calculate_softmax(values_in, values_out, gradients, expected_class, predicted_class, entropy);

        /*
        cerr << "values_in/values_out:" << endl;
        for (uint32_t i = 0; i < values_in.size(); i++) {
            cerr << "\tvalues_in[" << i << "]: " << values_in[i] << ", values_out[" << i << "]: " << values_out[i] << endl;
        }
        */

        for (int32_t i = 0; i < (int32_t)softmax_nodes.size(); i++) {
            softmax_nodes[i]->set_value_in(batch_number, 0, 0, values_out[i]);
            softmax_nodes[i]->set_error_in(batch_number, 0, 0, gradients[i]);
        }

        double previous_error = total_error;

        //float beta = 0.0; //controls the strength of the confidence penalty

        float error = values_out[expected_class];
        if (error == 0) error = 1.0 / EXACT_MAX_FLOAT;
        total_error -= log(error);
        //total_error += beta * entropy;  //entropy is negative

        if (isnan(total_error) || isinf(total_error)) {
            cerr << "ERROR! total_error became NAN or INF!" << endl;
            cerr << "previous total_error: " << previous_error << endl;
            cerr << "error: " << error << endl;
            cerr << "log(error): " << log(error) << endl;
            cerr << "predicted_class: " << predicted_class << endl;
            cerr << "expected_class: " << expected_class << endl;

            cerr << "softmax node values: " << endl;
            for (uint32_t i = 0; i < softmax_nodes.size(); i++) {
                cerr << "    " << softmax_nodes[i]->get_value_in(batch_number, 0, 0) << endl;
            }

            cerr << "values_in/values_out:" << endl;
            for (uint32_t i = 0; i < values_in.size(); i++) {
                cerr << "\tvalues_in[" << i << "]: " << values_in[i] << ", values_out[" << i << "]: " << values_out[i] << endl;
            }

            exit(1);
        }

        if (predicted_class == expected_class) {
            correct_predictions++;
        }
    }

    if (training) {
        for (int32_t i = edges.size() - 1; i >= 0; i--) {
            edges[i]->propagate_backward(training, mu, learning_rate, epsilon);
        }

        for (int32_t i = 0; i < edges.size(); i++) {
            edges[i]->update_weights(mu, learning_rate, weight_decay);
        }
    }
}

void CNN_Genome::save_to_best() {
    for (uint32_t i = 0; i < edges.size(); i++) {
        edges[i]->save_best_weights();
    }

    for (uint32_t i = 0; i < nodes.size(); i++) {
        nodes[i]->save_best_weights();
    }
}

void CNN_Genome::set_to_best() {
    for (uint32_t i = 0; i < edges.size(); i++) {
        if (edges[i]->is_reachable()) {
            edges[i]->set_weights_to_best();
        }
    }

    for (uint32_t i = 0; i < nodes.size(); i++) {
        if (nodes[i]->is_reachable()) {
            nodes[i]->set_weights_to_best();
        }
    }
}

void CNN_Genome::reset(bool _reset_weights) {
    reset_weights = _reset_weights;
    epoch = 0; 

    mu = initial_mu;
    learning_rate = initial_learning_rate;
    weight_decay = initial_weight_decay;

    best_epoch = 0;
    best_validation_error = EXACT_MAX_FLOAT;
    best_validation_predictions = 0; 

    unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
    generator = minstd_rand0(seed);
}


void CNN_Genome::initialize() {
    cout << "visiting nodes!" << endl;
    visit_nodes();

    //cout << "initializing genome!" << endl;
    for (uint32_t i = 0; i < nodes.size(); i++) {
        nodes[i]->reset_weight_count();
    }

    for (uint32_t i = 0; i < edges.size(); i++) {
        edges[i]->propagate_weight_count();
    }
    //cout << "calculated weight counts" << endl;

    if (reset_weights) {
        for (uint32_t i = 0; i < edges.size(); i++) {
            if (edges[i]->is_reachable()) {
                edges[i]->initialize_weights(generator, normal_distribution);
                edges[i]->save_best_weights();
            } else {
                //cerr << "edges[" << i << "] is unreachable, input innovation number: " << edges[i]->get_input_innovation_number() << ", output innovation_number: " << edges[i]->get_output_innovation_number()
                //    << ", forward visited: " << edges[i]->is_forward_visited() << ", reverse_visited: " << edges[i]->is_reverse_visited() << endl;
            }
        }
        //cout << "initialized weights!" << endl;

        for (uint32_t i = 0; i < nodes.size(); i++) {
            if (nodes[i]->is_reachable()) {
                nodes[i]->initialize();
                nodes[i]->save_best_weights();
            } else {
                //cerr << "nodes[" << i << "] is unreachable!" << endl;
            }
        }
        //cout << "initialized node gamma/beta!" << endl;


    } else {
        for (uint32_t i = 0; i < edges.size(); i++) {
            if (edges[i]->is_reachable() && edges[i]->needs_init()) {
                edges[i]->initialize_weights(generator, normal_distribution);
                edges[i]->save_best_weights();
            }
        }
        //cout << "reinitialized weights!" << endl;

        for (uint32_t i = 0; i < nodes.size(); i++) {
            if (nodes[i]->is_reachable() && nodes[i]->needs_init()) {
                //cout << "initializing nodes[" << i << "]" << endl;
                nodes[i]->initialize();
                nodes[i]->save_best_weights();
            } else {
                //cout << "not initializing nodes[" << i << "], reachable: " << nodes[i]->is_reachable() << ", needs init: " << nodes[i]->needs_init() << endl;
            }
        }
        //cout << "initialized node gamma/beta!" << endl;

        set_to_best();

        /*
        for (uint32_t i = 0; i < edges.size(); i++) {
            if (edges[i]->has_zero_weight()) {
                cerr << "ERROR: edge in position " << i << " with innovation number: " << edges[i]->get_innovation_number() << endl;
                cerr << "sum of weights was 0" << endl;
                cerr << "filter_x: " << edges[i]->get_filter_x() << ", filter_y: " << edges[i]->get_filter_y() << endl;
                exit(1);
            }
        }
        */
    }
}

void CNN_Genome::print_progress(ostream &out, string progress_name, float total_error, int correct_predictions, int number_images) const {
    out << setw(10) << progress_name << "[" << setw(10) << name << ", genome " << setw(5) << generation_id << "] predictions: " << setw(7) << correct_predictions << "/" << setw(7) << number_images << " (" << setw(5) << fixed << setprecision(2) << (100.0 * (float)correct_predictions/(float)number_images) << "%), best: " << setw(7) << best_validation_predictions << "/" << number_validation_images << " (" << setw(5) << fixed << setprecision(2) << (100 * (float)best_validation_predictions/(float)number_validation_images) << "%), error: " << setw(15) << setprecision(5) << fixed << total_error << ", best error: " << setw(15) << best_validation_error << " on epoch: " << setw(5) << best_epoch << ", epoch: " << setw(4) << epoch << "/" << max_epochs << ", mu: " << setw(12) << fixed << setprecision(10) << mu << ", learning_rate: " << setw(12) << fixed << setprecision(10) << learning_rate << ", weight_decay: " << setw(12) << fixed << setprecision(10) << weight_decay << endl;
}

void CNN_Genome::evaluate_large_images(const LargeImages &images, string output_directory) {
    int current_subimage = 0;

    vector< vector<int> > bins(images.get_number_classes(), vector<int>(10, 0));

    //cout << "number classes: " << images.get_number_classes() << endl;

    for (int image_number = 0; image_number < images.get_number_large_images(); image_number++) {
        int number_subimages = images.get_number_subimages(image_number);
        //cout << "image " << image_number << ", number subimages: " << number_subimages << endl;

        int number_correct = 0;

        vector< vector<float> > predictions(number_subimages, vector<float>(images.get_number_classes(), 0.0));
        //cout << "created vector!" << endl;

        for (uint32_t j = 0; j < number_subimages; j += batch_size) {

            //cout << "image " << image_number << ", number subimages: " << number_subimages << ", batch is: ";
            vector<int> batch;
            for (uint32_t k = 0; k < batch_size && (j + k) < number_subimages; k++) {
                batch.push_back( current_subimage + j + k );
                //cout << " " << current_subimage + j + k;
            }
            //cout << endl;

            //cout << "evaluating images for batch starting at " << j << " with batch size " << batch_size << endl;
            //float batch_total_error = 0.0;
            //int batch_correct_predictions = 0;
            evaluate_images(images, batch, predictions, current_subimage);
            //cout << "evaluated images!" << endl;
        }
        current_subimage += number_subimages;

        //cout << "checking predictions!" << endl;
        int classification = images.get_image_classification(image_number);
        for (int j = 0; j < number_subimages; j++) {
            int max_class = 0;
            float max_value = 0.0;

            for (int32_t k = 0; k < images.get_number_classes(); k++) {
                if (predictions[j][k] > max_value) {
                    max_value = predictions[j][k];
                    max_class = k;
                }
            }

            if (max_class == classification) number_correct++;
        }

        float percentage_correct = (float)number_correct / (float)images.get_number_subimages(image_number);

        cout << "large image " << setw(7) << image_number << " of " << setw(7) << images.get_number_large_images() << ", " << setw(7) << number_correct << " correct of " << setw(7) << number_subimages << " ("     << setw(7) << setprecision(3) << fixed << percentage_correct << "%)" << endl;

        float cutoff = 0.1;
        for (uint32_t j = 0; j < 10; j++) {
            if (percentage_correct < cutoff) {
                bins[classification][j]++;
                break;
            }
            cutoff += 0.1;
        }
    }

    for (uint32_t i = 0; i < images.get_number_classes(); i++) {
        cout << "bins for class " << i << endl;

        float cutoff = 0.1;
        for (uint32_t j = 0; j < 10; j++) {
            cout << "\tbin[" << cutoff - 0.1 << " .. " << cutoff << "]: " << bins[i][j] << endl;
            cutoff += 0.1;
        }
    }
}

void CNN_Genome::get_prediction_matrix(const MultiImagesInterface &images, int image_number, int stride, vector< vector< vector<float> > > &prediction_matrix) {
    int number_subimages = images.get_number_subimages(image_number);

    //TODO: fix, number classes should be equal to number of softmax nodes of genome
    int number_classes = images.get_number_classes();
    number_classes = 2;

    vector< vector<float> > predictions(number_subimages, vector<float>(number_classes, 0.0));
    cout << "created predictions vector for image: " << image_number << ", number subimages: " << number_subimages << ", number_classes: " << number_classes << endl;

    int initial_offset = 0;
    for (uint32_t i = 0; i < image_number; i++) {
        initial_offset += images.get_number_subimages(i);
    }
    int current_subimage = initial_offset;

    for (uint32_t j = 0; j < number_subimages; j += batch_size) {
        if (j % 10000 == 0) cout << "subimage: " << j << "/" << number_subimages << endl;

        vector<int> batch;
        for (uint32_t k = 0; k < batch_size && (j + k) < number_subimages; k++) {
            batch.push_back( current_subimage + j + k );
        }

        evaluate_images(images, batch, predictions, initial_offset);
    }

    //now create the prediction matrix to put these predictions into
    int matrix_height = images.get_large_image_height(image_number) - (images.get_image_height() - (images.get_padding() * 2)) + 1;
    int matrix_width = images.get_large_image_width(image_number) - (images.get_image_width() - (images.get_padding() * 2)) + 1;

    cout << "created prediction matrix, height: " << matrix_height << ", matrix_width: " << matrix_width << ", number classes: " << number_classes << endl;

    prediction_matrix.assign(matrix_height, vector< vector<float> >(matrix_width, vector<float>(number_classes, 0)));


    int current_prediction = 0;

    for (uint32_t y = 0; y < matrix_height; y++) {
        for (uint32_t x = 0; x < matrix_width; x++) {
            for (uint32_t c = 0; c < number_classes; c++) {
                prediction_matrix[y][x][c] = predictions[current_prediction][c];

            }
            current_prediction++;
        }
    }
}

void CNN_Genome::get_expanded_prediction_matrix(const MultiImagesInterface &images, int image_number, int stride, int prediction_class, vector< vector<float> > &extended_prediction_matrix) {

    //TODO: this could potentially be sped up by only getting the prediction matrix for the class we're interested in
    vector< vector< vector<float> > > prediction_matrix;
    get_prediction_matrix(images, image_number, stride, prediction_matrix);

    int image_height = images.get_large_image_height(image_number);
    int image_width = images.get_large_image_width(image_number);

    extended_prediction_matrix.assign(image_height, vector<float>(image_width, 0.0));

    int subimage_height = images.get_image_height() - (images.get_padding() * 2);
    int subimage_width = images.get_image_width() - (images.get_padding() * 2);

    for (uint32_t y = 0; y < prediction_matrix.size(); y += stride) {
        for (uint32_t x = 0; x < prediction_matrix[y].size(); x += stride) {

            for (uint32_t oy = 0; oy < subimage_height; oy++) {
                for (uint32_t ox = 0; ox < subimage_height; ox++) {
                    extended_prediction_matrix[y + oy][x + ox] += prediction_matrix[y][x][prediction_class];
                }
            }
        }
    }

    int32_t count = 0;
    float max_prediction = 0.0;
    float avg_prediction = 0.0;
    float min_prediction = 1.0;

    for (uint32_t y = 0; y < extended_prediction_matrix.size(); y++) {
        for (uint32_t x = 0; x < extended_prediction_matrix[y].size(); x++) {
            if (extended_prediction_matrix[y][x] > max_prediction) {
                max_prediction = extended_prediction_matrix[y][x];
            }

            if (extended_prediction_matrix[y][x] < min_prediction) {
                min_prediction = extended_prediction_matrix[y][x];
            }

            avg_prediction += extended_prediction_matrix[y][x];

            count++;
        }
    }
    avg_prediction /= count;

    cout << "min_prediction: " << min_prediction << endl;
    cout << "avg_prediction: " << avg_prediction << endl;
    cout << "max_prediction: " << max_prediction << endl;

    int max_count = fmin(subimage_height, image_height - subimage_height + 1) * fmin(subimage_width, image_width - subimage_width + 1);
    cout << "max_count: " << max_count << endl;

    //rescale the values between 0 and 1
    for (uint32_t y = 0; y < extended_prediction_matrix.size(); y++) {
        for (uint32_t x = 0; x < extended_prediction_matrix[y].size(); x++) {
            extended_prediction_matrix[y][x] /= max_count;
            //cerr << " " << extended_prediction_matrix[y][x];
        }
        //cerr << endl;
    }
}


void CNN_Genome::evaluate(const ImagesInterface &images, vector< vector<float> > &predictions) {
    predictions.clear();
    predictions.assign(images.get_number_images(), vector<float>(images.get_number_classes(), 0));

    for (uint32_t j = 0; j < images.get_number_images(); j += batch_size) {

        vector<int> batch;
        for (uint32_t k = 0; k < batch_size && (j + k) < images.get_number_images(); k++) {
            batch.push_back( j + k );
        }

        //cout << "evaluating images for batch starting at " << j << " with batch size " << batch_size << endl;
        //float batch_total_error = 0.0;
        //int batch_correct_predictions = 0;
        evaluate_images(images, batch, predictions, 0);
    }
}

void CNN_Genome::evaluate(const ImagesInterface &images, const vector<long> &order, float &total_error, int &correct_predictions, bool perform_backprop, bool accumulate_test_statistics) {
    bool training;
    if (perform_backprop) {
        training = true;
    } else {
        training = false;
    }

    total_error = 0.0;
    correct_predictions = 0;

    int required_for_reset = velocity_reset;


    using namespace std::chrono;

    high_resolution_clock::time_point epoch_start_time = high_resolution_clock::now();

    for (uint32_t i = 0; i < nodes.size(); i++) {
        nodes[i]->reset_times();
    }

    for (uint32_t i = 0; i < edges.size(); i++) {
        edges[i]->reset_times();
    }

    for (uint32_t j = 0; j < order.size(); j += batch_size) {
        vector<int> batch;
        for (uint32_t k = 0; k < batch_size && (j + k) < order.size(); k++) {
            batch.push_back( order[j + k] );
        }

        float batch_total_error = 0.0;
        int batch_correct_predictions = 0;
        evaluate_images(images, batch, training, batch_total_error, batch_correct_predictions, accumulate_test_statistics);

        /*
        cerr << "[" << setw(10) << name << ", genome " << setw(5) << generation_id << "] ";
        if (training) {
            cerr << "training batch: ";
        } else {
            cerr << "test batch: ";
        }
        cerr << setw(5) << (j / batch_size) << "/" << setw(5) << (order.size() / batch_size) << ", batch total error: " << setw(15) << fixed << setprecision(5) << batch_total_error << ", batch_correct_predictions: " << batch_correct_predictions << endl;
        */

        /*
        for (uint32_t k = 0; k < nodes.size(); k++) {
            if (nodes[k]->get_size_y() > 1 && nodes[k]->get_size_x() > 1) {
                nodes[k]->print_batch_statistics();
            }
        }
        */

        total_error += batch_total_error;
        correct_predictions += batch_correct_predictions;

        required_for_reset -= batch_size;

        if (perform_backprop && velocity_reset > 0 && required_for_reset <= 0) {
            required_for_reset += velocity_reset;
            //cout << "resetting velocities on image " << (j + 100) << ", total_error: " << total_error << ", correct_predictions: " << correct_predictions << endl;
            for (uint32_t i = 0; i < edges.size(); i++) {
                edges[i]->reset_velocities();
            }

            for (uint32_t i = 0; i < nodes.size(); i++) {
                nodes[i]->reset_velocities();
            }
        }
    }

    high_resolution_clock::time_point epoch_end_time = high_resolution_clock::now();
    duration<float, std::milli> time_span = epoch_end_time - epoch_start_time;

    float epoch_time = time_span.count() / 1000.0;
    float input_fired_time = 0.0;
    float output_fired_time = 0.0;

    float propagate_forward_time = 0.0;
    float propagate_backward_time = 0.0;
    float weight_update_time = 0.0;

    for (uint32_t i = 0; i < nodes.size(); i++) {
        nodes[i]->accumulate_times(input_fired_time, output_fired_time);
    }

    for (uint32_t i = 0; i < edges.size(); i++) {
        edges[i]->accumulate_times(propagate_forward_time, propagate_backward_time, weight_update_time);
    }

    float other_time = epoch_time - input_fired_time - output_fired_time - propagate_forward_time - propagate_backward_time;

    cerr << "epoch time: " << epoch_time << "s"
         << ", input_fired_time: " << input_fired_time
         << ", output_fired_time: " << output_fired_time
         << ", propagate_forward_time: " << propagate_forward_time
         << ", propagate_backward_time: " << propagate_backward_time
         << ", weight_update_time: " << weight_update_time
         << ", other_time: " << other_time
         << endl;
}

void CNN_Genome::evaluate(string progress_name, const ImagesInterface &images, float &total_error, int &correct_predictions) {
    backprop_order.clear();
    for (int32_t i = 0; i < images.get_number_images(); i++) {
        backprop_order.push_back(i);
    }

    evaluate(images, backprop_order, total_error, correct_predictions, false, false);

    print_progress(cerr, progress_name, total_error, correct_predictions, images.get_number_images());
}

void CNN_Genome::stochastic_backpropagation(const ImagesInterface &training_images, const ImagesInterface &validation_images) {
        stochastic_backpropagation(training_images, training_images.get_number_images(), validation_images);
}

void CNN_Genome::stochastic_backpropagation(const ImagesInterface &training_images, int training_resize, const ImagesInterface &validation_images) {
    number_training_images = training_resize;
    number_validation_images = validation_images.get_number_images();

    for (uint32_t i = 0; i < nodes.size(); i++) {
        if (!nodes[i]->is_reachable()) continue;

        if (nodes[i]->needs_init()) {
            cerr << "ERROR! nodes[" << i << "] needs init!" << endl;
            exit(1);
        }

        if (nodes[i]->has_nan()) {
            cerr << "ERROR! nodes[" << i << "] has nan or inf!" << endl;
            nodes[i]->print(cerr);
            exit(1);
        }
    }

    for (uint32_t i = 0; i < edges.size(); i++) {
        if (!edges[i]->is_reachable()) continue;

        if (edges[i]->needs_init()) {
            cerr << "ERROR! edges[" << i << "] needs init!" << endl;
            exit(1);
        }

        if (edges[i]->has_nan()) {
            cerr << "ERROR! edges[" << i << "] has nan or inf!" << endl;
            edges[i]->print(cerr);
            exit(1);
        }
    }

    if (!started_from_checkpoint) {
        backprop_order.clear();
        for (int32_t i = 0; i < training_images.get_number_images(); i++) {
            backprop_order.push_back(i);
        }

        cerr << "generator min: " << generator.min() << ", generator max: " << generator.max() << endl;

        cerr << "pre shuffle 1: " << generator() << endl;

        //shuffle the array (thanks C++ not being the same across operating systems)
        fisher_yates_shuffle(generator, backprop_order);

        cerr << "post shuffle 1: " << generator() << endl;

        best_validation_error = EXACT_MAX_FLOAT;
    }
    backprop_order.resize(training_resize);

    //sort edges by depth of input node
    sort(edges.begin(), edges.end(), sort_CNN_Edges_by_depth());

    vector<long> validation_order;
    for (uint32_t i = 0; i < number_validation_images; i++) {
        validation_order.push_back(i);
    }

    float current_training_error = 0.0;
    int current_training_predictions = 0;
    float current_validation_error = 0.0;
    int current_validation_predictions = 0;

    do {
        //shuffle the array (thanks C++ not being the same across operating systems)
        fisher_yates_shuffle(generator, backprop_order);

        evaluate(training_images, backprop_order, current_training_error, current_training_predictions, true, false);
        evaluate(validation_images, validation_order, current_validation_error, current_validation_predictions, false, false);

        bool found_improvement = false;

        if (current_validation_predictions > best_validation_predictions) {
            //cout << "current validation error: " << current_validation_error << " < best validation error: " << best_validation_error << endl;
            best_validation_error = current_validation_error;
            best_validation_predictions = current_validation_predictions;

            best_epoch = epoch;

            save_to_best();
            found_improvement = true;
            /*
        } else if (epoch < 5) {
            cout << "epoch < 5" << endl;
            best_epoch = epoch;

            save_to_best();
            found_improvement = true;
            */
        } else if (current_validation_error < 2.5 * best_validation_error) {
            //cout << "current_validation_error < 2.5 * best_validation_error" << endl;
            found_improvement = true;
        }

        print_progress(cerr, "validation", current_validation_error, current_validation_predictions, number_validation_images);
        //cerr << "best validation error: " << best_validation_error << ", best validation predictions: " << best_validation_predictions << endl;
        cerr << endl;

        if (!found_improvement) {
            cerr << "resetting weights to those on epoch: " << best_epoch << endl;
            set_to_best();
        }

        //decay mu with a max of 0.99
        mu = 0.99 - ((0.99 - mu) * mu_delta);

        learning_rate *= learning_rate_delta;
        //if (learning_rate < 0.00001) learning_rate = 0.00001;

        weight_decay *= weight_decay_delta;
        //if (weight_decay < 0.00001) weight_decay = 0.00001;

        epoch++;

        if (checkpoint_filename.compare("") != 0) {
            write_to_file(checkpoint_filename);
        }

        if (progress_function != NULL) {
            float progress = (float)epoch / (float)(max_epochs + 1.0);
            progress_function(progress);
        }

        /*
        for (uint32_t i = 0; i < edges.size(); i++) {
            if (edges[i]->get_type() == POOLING) {
                cout << "edge " << edges[i]->get_innovation_number() << ", scale: " << edges[i]->get_scale() << endl;
            }
        }
        */

        if (epoch > max_epochs) {
            break;
        }
    } while (true);

    cerr << "evaluating best weights on full training data." << endl;
    cerr << "evaluting training set with running mean/variance:" << endl;
    set_to_best();
    evaluate(training_images, backprop_order, training_error, training_predictions, false, false);
    print_progress(cerr, "best training", training_error, training_predictions, number_training_images);
}

void CNN_Genome::print_results(ostream &out) const {
    out << setw(10) << fixed << setprecision(5) << training_error;
    out << setw(15) << fixed << setprecision(5) << best_validation_error;
    out << setw(15) << fixed << setprecision(5) << test_error;
    out << setw(10) << training_predictions << setw(10) << number_training_images;
    out << setw(10) << best_validation_predictions << setw(10) << number_validation_images;
    out << setw(10) << test_predictions << setw(10) << number_test_images << endl;
} 


void CNN_Genome::evaluate_test(const ImagesInterface &test_images) {
    set_to_best();

    cerr << "evaluating best weights on test data." << endl;
    cerr << "evaluting test set with running mean/variance:" << endl;

    number_test_images = test_images.get_number_images();
    evaluate("testing", test_images, test_error, test_predictions);
}


void CNN_Genome::set_name(string _name) {
    name = _name;
}

void CNN_Genome::set_output_filename(string _output_filename) {
    output_filename = _output_filename;
}

void CNN_Genome::set_checkpoint_filename(string _checkpoint_filename) {
    checkpoint_filename = _checkpoint_filename;
}

string CNN_Genome::get_version_str() const {
    return version_str;
}

void CNN_Genome::write(ostream &outfile) {
    outfile << EXACT_VERSION_STR << endl;
    outfile << exact_id << endl;
    outfile << genome_id << endl;

    write_hexfloat(outfile, initial_mu);
    outfile << endl;
    write_hexfloat(outfile, mu);
    outfile << endl;
    write_hexfloat(outfile, mu_delta);
    outfile << endl;

    write_hexfloat(outfile, initial_learning_rate);
    outfile << endl;
    write_hexfloat(outfile, learning_rate);
    outfile << endl;
    write_hexfloat(outfile, learning_rate_delta);
    outfile << endl;

    write_hexfloat(outfile, initial_weight_decay);
    outfile << endl;
    write_hexfloat(outfile, weight_decay);
    outfile << endl;
    write_hexfloat(outfile, weight_decay_delta);
    outfile << endl;

    outfile << batch_size << endl;

    write_hexfloat(outfile, epsilon);
    outfile << endl;
    write_hexfloat(outfile, alpha);
    outfile << endl;

    write_hexfloat(outfile, input_dropout_probability);
    outfile << endl;
    write_hexfloat(outfile, hidden_dropout_probability);
    outfile << endl;

    outfile << velocity_reset << endl;

    outfile << epoch << endl;
    outfile << max_epochs << endl;
    outfile << reset_weights << endl;

    outfile << padding << endl;

    outfile << best_epoch << endl;
    outfile << number_validation_images << endl;
    outfile << best_validation_predictions << endl;
    write_hexfloat(outfile, best_validation_error);
    outfile << endl;

    outfile << number_training_images << endl;
    outfile << training_predictions << endl;
    write_hexfloat(outfile, training_error);
    outfile << endl;

    outfile << number_test_images << endl;
    outfile << test_predictions << endl;
    write_hexfloat(outfile, test_error);
    outfile << endl;

    outfile << generation_id << endl;
    outfile << normal_distribution << endl;
    //outfile << name << endl;
    //outfile << checkpoint_filename << endl;
    //outfile << output_filename << endl;

    outfile << generator << endl;

    outfile << "GENERATED_BY" << endl;
    write_map(outfile, generated_by_map);
    outfile << endl;

    outfile << "NODES" << endl;
    outfile << nodes.size() << endl;
    for (uint32_t i = 0; i < nodes.size(); i++) {
        outfile << nodes[i] << endl;
    }

    outfile << "EDGES" << endl;
    outfile << edges.size() << endl;
    for (uint32_t i = 0; i < edges.size(); i++) {
        outfile << edges[i] << endl;
    }

    outfile << "INNOVATION_NUMBERS" << endl;
    outfile << input_nodes.size() << endl;
    for (uint32_t i = 0; i < input_nodes.size(); i++) {
        outfile << input_nodes[i]->get_innovation_number() << endl;
    }

    outfile << softmax_nodes.size() << endl;
    for (uint32_t i = 0; i < softmax_nodes.size(); i++) {
        outfile << softmax_nodes[i]->get_innovation_number() << endl;
    }

    outfile << "BACKPROP_ORDER" << endl;
    outfile << backprop_order.size() << endl;
    for (uint32_t i = 0; i < backprop_order.size(); i++) {
        if (i > 0) outfile << " ";
        outfile << backprop_order[i];
    }
    outfile << endl;
}

void CNN_Genome::read(istream &infile) {
    progress_function = NULL;

    bool verbose = true;

    getline(infile, version_str);

    cerr << "read CNN_Genome file with version string: '" << version_str << "'" << endl;

    if (version_str.compare(EXACT_VERSION_STR) != 0) {
        cerr << "breaking because version_str '" << version_str << "' did not match EXACT_VERSION_STR '" << EXACT_VERSION_STR << "': " << version_str.compare(EXACT_VERSION_STR) << endl;
        return;
    }

    infile >> exact_id;
    if (verbose) cerr << "read exact_id: " << exact_id << endl;
    infile >> genome_id;
    if (verbose) cerr << "read genome_id: " << genome_id << endl;

    initial_mu = read_hexfloat(infile);
    if (verbose) cerr << "read initial_mu: " << initial_mu << endl;
    mu = read_hexfloat(infile);
    if (verbose) cerr << "read mu: " << mu << endl;
    mu_delta = read_hexfloat(infile);
    if (verbose) cerr << "read mu_delta: " << mu_delta << endl;

    initial_learning_rate = read_hexfloat(infile);
    if (verbose) cerr << "read initial_learning_rate: " << initial_learning_rate << endl;
    learning_rate = read_hexfloat(infile);
    if (verbose) cerr << "read learning_rate: " << learning_rate << endl;
    learning_rate_delta = read_hexfloat(infile);
    if (verbose) cerr << "read learning_rate_delta: " << learning_rate_delta << endl;

    initial_weight_decay = read_hexfloat(infile);
    if (verbose) cerr << "read initial_weight_decay: " << initial_weight_decay << endl;
    weight_decay = read_hexfloat(infile);
    if (verbose) cerr << "read weight_decay: " << weight_decay << endl;
    weight_decay_delta = read_hexfloat(infile);
    if (verbose) cerr << "read weight_decay_delta: " << weight_decay_delta << endl;


    infile >> batch_size;
    if (verbose) cerr << "read batch_size: " << batch_size << endl;

    epsilon = read_hexfloat(infile);
    if (verbose) cerr << "read epsilon: " << epsilon << endl;
    alpha = read_hexfloat(infile);
    if (verbose) cerr << "read alpha: " << alpha << endl;

    input_dropout_probability = read_hexfloat(infile);
    if (verbose) cerr << "read input_dropout_probability: " << input_dropout_probability << endl;
    hidden_dropout_probability = read_hexfloat(infile);
    if (verbose) cerr << "read hidden_dropout_probability: " << hidden_dropout_probability << endl;


    infile >> velocity_reset;
    if (verbose) cerr << "read velocity_reset: " << velocity_reset << endl;

    infile >> epoch;
    if (verbose) cerr << "read epoch: " << epoch << endl;
    infile >> max_epochs;
    if (verbose) cerr << "read max_epochs: " << max_epochs << endl;
    infile >> reset_weights;
    if (verbose) cerr << "read reset_weights: " << reset_weights << endl;

    infile >> padding;
    if (verbose) cerr << "read padding: " << padding << endl;

    infile >> best_epoch;
    if (verbose) cerr << "read best_epoch: " << best_epoch << endl;
    infile >> number_validation_images;
    if (verbose) cerr << "read number_validation_images: " << number_validation_images << endl;
    infile >> best_validation_predictions;
    if (verbose) cerr << "read best_validation_predictions: " << best_validation_predictions << endl;
    best_validation_error = read_hexfloat(infile);
    if (verbose) cerr << "read best_validation_error: " << best_validation_error << endl;

    infile >> number_training_images;
    if (verbose) cerr << "read number_training_images: " << number_training_images << endl;
    infile >> training_predictions;
    if (verbose) cerr << "read training_predictions: " << training_predictions << endl;
    training_error = read_hexfloat(infile);
    if (verbose) cerr << "read training_error: " << training_error << endl;

    infile >> number_test_images;
    if (verbose) cerr << "read number_test_images: " << number_test_images << endl;
    infile >> test_predictions;
    if (verbose) cerr << "read test_predictions: " << test_predictions << endl;
    test_error = read_hexfloat(infile);
    if (verbose) cerr << "read test_error: " << test_error << endl;


    infile >> generation_id;
    if (verbose) cerr << "read generation_id: " << generation_id << endl;
    //infile >> name;
    //infile >> checkpoint_filename;
    //infile >> output_filename;

    infile >> normal_distribution;
    if (verbose) cerr << "read normal distribution: '" << normal_distribution << "'" << endl;

    //for some reason linux doesn't read the generator correcly because of
    //the first newline
    string generator_str;
    getline(infile, generator_str);
    getline(infile, generator_str);
    if (verbose) cerr << "generator_str: '" << generator_str << "'" << endl;
    istringstream generator_iss(generator_str);
    generator_iss >> generator;
    //infile >> generator;

    if (verbose) {
        cerr << "read generator: " << generator << endl;
        //cerr << "rand 1: " << generator() << endl;
        //cerr << "rand 2: " << generator() << endl;
        //cerr << "rand 3: " << generator() << endl;
    }

    string line;
    getline(infile, line);

    if (line.compare("GENERATED_BY") != 0) {
        cerr << "ERROR: invalid input file, expected line to be 'GENERATED_BY' but line was '" << line << "'" << endl;
        version_str = "INALID";
        return;
    }

    read_map(infile, generated_by_map);
    if (verbose) {
        cerr << "read generated_by_map:" << endl;
        write_map(cerr, generated_by_map);
        cerr << endl;
    }
    //cerr << "reading nodes!" << endl;
    
    getline(infile, line);
    getline(infile, line);

    if (line.compare("NODES") != 0) {
        cerr << "ERROR: invalid input file, expected line to be 'NODES' but line was '" << line << "'" << endl;
        version_str = "INALID";
        return;
    }

    nodes.clear();
    int number_nodes;
    infile >> number_nodes;
    if (verbose) cerr << "reading " << number_nodes << " nodes." << endl;
    for (int32_t i = 0; i < number_nodes; i++) {
        CNN_Node *node = new CNN_Node();
        infile >> node;

        cerr << "read node: " << node->get_innovation_number() << endl;
        nodes.push_back(node);
    }

    //cerr << "reading edges!" << endl;

    getline(infile, line);
    getline(infile, line);
    if (line.compare("EDGES") != 0) {
        cerr << "ERROR: invalid input file, expected line to be 'EDGES' but line was '" << line << "'" << endl;
        version_str = "INALID";
        return;
    }

    edges.clear();
    int number_edges;
    infile >> number_edges;
    if (verbose) cerr << "reading " << number_edges << " edges." << endl;
    for (int32_t i = 0; i < number_edges; i++) {
        CNN_Edge *edge = new CNN_Edge();
        infile >> edge;

        cerr << "read edge: " << edge->get_innovation_number() << " from node " << edge->get_input_innovation_number() << " to node " << edge->get_output_innovation_number() << endl;
        if (!edge->set_nodes(nodes)) {
            cerr << "ERROR: filter size didn't match when reading genome from input file!" << endl;
            cerr << "This should never happen!" << endl;
            exit(1);
        }

        edges.push_back(edge);
    }

    getline(infile, line);
    getline(infile, line);
    if (line.compare("INNOVATION_NUMBERS") != 0) {
        cerr << "ERROR: invalid input file, expected line to be 'INNOVATION_NUMBERS' but line was '" << line << "'" << endl;
        version_str = "INALID";
        return;
    }

    input_nodes.clear();
    int number_input_nodes;
    infile >> number_input_nodes;
    if (verbose) cerr << "number input nodes: " << number_input_nodes << endl;

    for (int32_t i = 0; i < number_input_nodes; i++) {
        int input_node_innovation_number;
        infile >> input_node_innovation_number;
        //cerr << "\tinput node: " << input_node_innovation_number << endl;

        for (uint32_t i = 0; i < nodes.size(); i++) {
            if (nodes[i]->get_innovation_number() == input_node_innovation_number) {
                input_nodes.push_back(nodes[i]);
                //cerr << "input node " << input_node_innovation_number << " was in position: " << i << endl;
                break;
            }
        }
    }

    softmax_nodes.clear();
    int number_softmax_nodes;
    infile >> number_softmax_nodes;
    if (verbose) cerr << "number softmax nodes: " << number_softmax_nodes << endl;

    for (int32_t i = 0; i < number_softmax_nodes; i++) {
        int softmax_node_innovation_number;
        infile >> softmax_node_innovation_number;
        //cerr << "\tsoftmax node: " << softmax_node_innovation_number << endl;

        for (uint32_t i = 0; i < nodes.size(); i++) {
            if (nodes[i]->get_innovation_number() == softmax_node_innovation_number) {
                softmax_nodes.push_back(nodes[i]);
                //cerr << "softmax node " << softmax_node_innovation_number << " was in position: " << i << endl;
                break;
            }
        }
    }

    //cerr << "reading backprop order" << endl;

    getline(infile, line);
    getline(infile, line);
    if (line.compare("BACKPROP_ORDER") != 0) {
        cerr << "ERROR: invalid input file, expected line to be 'BACKPROP_ORDER' but line was '" << line << "'" << endl;
        version_str = "INALID";
        return;
    }

    backprop_order.clear();
    long order_size;
    infile >> order_size;
    if (verbose) cerr << "order_size: " << order_size << endl;
    for (uint32_t i = 0; i < order_size; i++) {
        long order;
        infile >> order;
        backprop_order.push_back(order);
        //cerr << "backprop order[" << i << "]: " << order << endl;
    }

    visit_nodes();
}

void CNN_Genome::write_to_file(string filename) {
    ofstream outfile(filename.c_str());
    write(outfile);
    outfile.close();
}


void CNN_Genome::print_graphviz(ostream &out) const {
    out << "digraph CNN {" << endl;

    //this will draw graph left to right instead of top to bottom
    //out << "\trankdir=LR;" << endl;

    //print the source nodes, i.e. the input
    out << "\t{" << endl;
    out << "\t\trank = source;" << endl;
    for (uint32_t i = 0; i < nodes.size(); i++) {
        if (!nodes[i]->is_input()) continue;
        out << "\t\tnode" << nodes[i]->get_innovation_number() << " [shape=box,color=green,label=\"input " << nodes[i]->get_innovation_number() << "\\n" << nodes[i]->get_size_x() << " x " << nodes[i]->get_size_y() << "\"];" << endl;
    }
    out << "\t}" << endl;
    out << endl;

    out << "\t{" << endl;
    out << "\t\trank = sink;" << endl;
    for (uint32_t i = 0; i < nodes.size(); i++) {
        if (!nodes[i]->is_softmax()) continue;
        out << "\t\tnode" << nodes[i]->get_innovation_number() << " [shape=box,color=blue,label=\"output " << (nodes[i]->get_innovation_number() - 1) << "\\n" << nodes[i]->get_size_x() << " x " << nodes[i]->get_size_y() << "\"];" << endl;
    }
    out << "\t}" << endl;
    out << endl;

    //connect the softmax nodes in order with invisible edges so they display in order

    bool printed_first = false;

    for (uint32_t i = 0; i < nodes.size(); i++) {
        if (!nodes[i]->is_softmax()) continue;

        if (!printed_first) {
            printed_first = true;
            out << "\tnode" << nodes[i]->get_innovation_number();
        } else {
            out << " -> node" << nodes[i]->get_innovation_number();
        }
    }
    out << " [style=invis];" << endl << endl;

    out << endl;

    //draw the hidden nodes
    for (uint32_t i = 0; i < nodes.size(); i++) {
        if (nodes[i]->is_input() || nodes[i]->is_softmax()) continue;
        if (!nodes[i]->is_reachable()) continue;

        out << "\t\tnode" << nodes[i]->get_innovation_number() << " [shape=box,color=black,label=\"input " << nodes[i]->get_innovation_number() << "\\n" << nodes[i]->get_size_x() << " x " << nodes[i]->get_size_y() << "\"];" << endl;
    }
    
    out << endl;

    //draw the enabled edges
    for (uint32_t i = 0; i < edges.size(); i++) {
        if (!edges[i]->is_reachable()) continue;

        if (edges[i]->get_type() == CONVOLUTIONAL) {
            out << "\tnode" << edges[i]->get_input_node()->get_innovation_number() << " -> node" << edges[i]->get_output_node()->get_innovation_number() << " [color=blue];" << endl;
        } else {
            out << "\tnode" << edges[i]->get_input_node()->get_innovation_number() << " -> node" << edges[i]->get_output_node()->get_innovation_number() << " [color=green];" << endl;
        }
    }

    out << endl;

    /*
    //draw the disabled edges and nodes in red
    for (uint32_t i = 0; i < nodes.size(); i++) {
        if (nodes[i]->is_reachable()) continue;

        out << "\t\tnode" << nodes[i]->get_innovation_number() << " [shape=box,color=red,label=\"input " << nodes[i]->get_innovation_number() << "\\n" << nodes[i]->get_size_x() << " x " << nodes[i]->get_size_y() << "\"];" << endl;
    }

    for (uint32_t i = 0; i < edges.size(); i++) {
        if (edges[i]->is_reachable()) continue;

        out << "\tnode" << edges[i]->get_input_node()->get_innovation_number() << " -> node" << edges[i]->get_output_node()->get_innovation_number() << " [color=red];" << endl;
    }
    */

    out << "}" << endl;
}

void CNN_Genome::set_generated_by(string type) {
    generated_by_map[type]++;
}

int CNN_Genome::get_generated_by(string type) {
    return generated_by_map[type];
}

bool CNN_Genome::is_identical(CNN_Genome *other, bool testing_checkpoint) {
    //if (are_different("version_str", version_str, other->version_str)) return false;

    if (are_different("normal_distribution", normal_distribution, other->normal_distribution)) return false;
    if (are_different("generator", generator, other->generator)) return false;

    if (are_different("velocity_reset", velocity_reset, other->velocity_reset)) return false;
    if (are_different("batch_size", batch_size, other->batch_size)) return false;

    if (are_different("epsilon", epsilon, other->epsilon)) return false;
    if (are_different("alpha", alpha, other->alpha)) return false;

    if (are_different("input_dropout_probability", input_dropout_probability, other->input_dropout_probability)) return false;
    if (are_different("hidden_dropout_probability", hidden_dropout_probability, other->hidden_dropout_probability)) return false;

    if (are_different("initial_mu", initial_mu, other->initial_mu)) return false;
    if (are_different("mu", mu, other->mu)) return false;
    if (are_different("mu_delta", mu_delta, other->mu_delta)) return false;

    if (are_different("initial_learning_rate", initial_learning_rate, other->initial_learning_rate)) return false;
    if (are_different("learning_rate", learning_rate, other->learning_rate)) return false;
    if (are_different("learning_rate_delta", learning_rate_delta, other->learning_rate_delta)) return false;

    if (are_different("initial_weight_decay", initial_weight_decay, other->initial_weight_decay)) return false;
    if (are_different("weight_decay", weight_decay, other->weight_decay)) return false;
    if (are_different("weight_decay_delta", weight_decay_delta, other->weight_decay_delta)) return false;

    if (are_different("epoch", epoch, other->epoch)) return false;
    if (are_different("max_epochs", max_epochs, other->max_epochs)) return false;
    if (are_different("reset_weights", reset_weights, other->reset_weights)) return false;

    if (are_different("padding", padding, other->padding)) return false;

    if (are_different("best_epoch", best_epoch, other->best_epoch)) return false;
    if (are_different("number_validation_images", number_validation_images, other->number_validation_images)) return false;
    //if (are_different("best_error", best_error, other->best_error)) return false;
    if (are_different("best_validation_predictions", best_validation_predictions, other->best_validation_predictions)) return false;
    if (are_different("best_validation_error", best_validation_error, other->best_validation_error)) return false;

    if (are_different("number_training_images", number_training_images, other->number_training_images)) return false;
    if (are_different("training_error", training_error, other->training_error)) return false;
    if (are_different("training_predictions", training_predictions, other->training_predictions)) return false;

    if (are_different("number_test_images", number_test_images, other->number_test_images)) return false;
    if (are_different("test_error", test_error, other->test_error)) return false;
    if (are_different("test_predictions", test_predictions, other->test_predictions)) return false;

    //if (are_different("started_from_checkpoint", started_from_checkpoint, other->started_from_checkpoint)) return false;
    if (are_different("backprop_order", backprop_order, other->backprop_order)) {
        if (testing_checkpoint) {
            //backprop order when read from file has to be identical
            //to make sure checkpoints are working
            return false;
        } else {
            cerr << "backprop_order is different, however from CNNs stored in a database they are not required to be the same." << endl;
        }
    }

    if (are_different("generation_id", generation_id, other->generation_id)) return false;
    if (are_different("name", name, other->name)) return false;
    if (are_different("checkpoint_filename", checkpoint_filename, other->checkpoint_filename)) return false;
    if (are_different("output_filename", output_filename, other->output_filename)) return false;

    //if (are_different("generated_by", generated_by, other->generated_by)) return false;

    for (uint32_t i = 0; i < nodes.size(); i++) {
        if (!nodes[i]->is_identical(other->nodes[i], testing_checkpoint)) {
            cerr << "IDENTICAL ERROR: nodes[" << i << "] are not the same!" << endl;
            return false;
        }
    }

    for (uint32_t i = 0; i < edges.size(); i++) {
        if (!edges[i]->is_identical(other->edges[i], testing_checkpoint)) {
            cerr << "IDENTICAL ERROR: edges[" << i << "] are not the same!" << endl;
            return false;
        }
    }

    for (uint32_t i = 0; i < input_nodes.size(); i++) {
        if (!input_nodes[i]->is_identical(other->input_nodes[i], testing_checkpoint)) {
            cerr << "IDENTICAL ERROR: input_nodes[" << i << "] are not the same!" << endl;
            return false;
        }
    }

    for (uint32_t i = 0; i < softmax_nodes.size(); i++) {
        if (!softmax_nodes[i]->is_identical(other->softmax_nodes[i], testing_checkpoint)) {
            cerr << "IDENTICAL ERROR: softmax_nodes[" << i << "] are not the same!" << endl;
            return false;
        }
    }

    return true;
}
