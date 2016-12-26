#include <algorithm>
using std::sort;
using std::upper_bound;

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

#include <random>
using std::mt19937;
using std::normal_distribution;
using std::uniform_int_distribution;
using std::uniform_real_distribution;

#include <string>
using std::string;
using std::to_string;

#include <vector>
using std::vector;


#include "image_tools/image_set.hxx"
#include "cnn_node.hxx"
#include "cnn_edge.hxx"
#include "cnn_genome.hxx"

/**
 *  Initialize a genome from a file
 */
CNN_Genome::CNN_Genome(string filename, bool is_checkpoint) {
    started_from_checkpoint = is_checkpoint;

    ifstream infile(filename);
    read(infile);
    infile.close();
}

CNN_Genome::CNN_Genome(istream &in, bool is_checkpoint) {
    started_from_checkpoint = is_checkpoint;
    read(in);
}

/**
 *  Iniitalize a genome from a set of nodes and edges
 */
CNN_Genome::CNN_Genome(int _generation_id, int seed, int _min_epochs, int _max_epochs, int _improvement_required_epochs, bool _reset_weights, double _learning_rate, double _weight_decay, const vector<CNN_Node*> &_nodes, const vector<CNN_Edge*> &_edges) {
    started_from_checkpoint = false;
    generator = mt19937(seed);
    rng_double = uniform_real_distribution<double>(0, 1.0);
    rng_long = uniform_int_distribution<long>(-numeric_limits<long>::max(), numeric_limits<long>::max());

    mu = 0.5;
    initial_mu = mu;

    learning_rate = _learning_rate;
    initial_learning_rate = learning_rate;

    weight_decay = _weight_decay;
    initial_weight_decay = weight_decay;

    epoch = 0;
    min_epochs = _min_epochs;
    max_epochs = _max_epochs;
    improvement_required_epochs = _improvement_required_epochs;
    reset_weights = _reset_weights;

    best_predictions = 0;
    best_error = numeric_limits<double>::max();

    best_predictions_epoch = 0;
    best_error_epoch = 0;

    generation_id = _generation_id;

    generated_by_disable_edge = 0;
    generated_by_enable_edge = 0;
    generated_by_split_edge = 0;
    generated_by_add_edge = 0;
    generated_by_change_size = 0;
    generated_by_change_size_x = 0;
    generated_by_change_size_y = 0;
    generated_by_crossover = 0;
    generated_by_reset_weights = 0;

    name = "";
    output_filename = "";
    checkpoint_filename = "";

    input_node = NULL;
    nodes.clear();
    softmax_nodes.clear();

    for (uint32_t i = 0; i < _nodes.size(); i++) {
        CNN_Node *node_copy = _nodes[i]->copy();

        if (node_copy->is_input()) {
            //cout << "node was input!" << endl;

            if (input_node != NULL) {
                cerr << "ERROR: multiple input nodes in genome." << endl;
                cerr << "first: " << endl;
                input_node->print(cerr);
                cerr << "second: " << endl;
                node_copy->print(cerr);
                exit(1);
            }

            input_node = node_copy;
        }

        if (node_copy->is_softmax()) {
            //cout << "node was softmax!" << endl;

            softmax_nodes.push_back(node_copy);
        }

        nodes.push_back( node_copy );
    }

    for (uint32_t i = 0; i < _edges.size(); i++) {
        CNN_Edge *edge_copy = _edges[i]->copy();

        if (!edge_copy->set_nodes(nodes)) {
            cerr << "ERROR: filter size didn't match when creating genome!" << endl;
            cerr << "This should never happen!" << endl;
            exit(1);
        }
        edges.push_back( edge_copy );
    }
}


CNN_Genome::~CNN_Genome() {
    input_node = NULL;

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
    
    while (softmax_nodes.size() > 0) {
        softmax_nodes.pop_back();
    }
    softmax_nodes.clear();
}

bool CNN_Genome::equals(CNN_Genome *other) const {
    for (int i = 0; i < edges.size(); i++) {
        CNN_Edge *edge = edges[i];

        if (edge->is_disabled()) continue;

        bool found = false;

        for (int j = 0; j < other->get_number_edges(); j++) {
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

    for (int i = 0; i < other->get_number_edges(); i++) {
        CNN_Edge *other_edge = other->get_edge(i);

        if (other_edge->is_disabled()) continue;

        bool found = false;
        
        for (int j = 0; j < edges.size(); j++) {
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

void CNN_Genome::print_best_error(ostream &out) const {
    cout << left << setw(20) << "class error:" << right;
    for (uint32_t i = 0; i < best_class_error.size(); i++) {
        cout << setw(15) << setprecision(5) << best_class_error[i];
    }
    cout << endl;
}

void CNN_Genome::print_best_predictions(ostream &out) const {
    cout << left << setw(20) << "correct predictions:" << right;
    for (uint32_t i = 0; i < best_correct_predictions.size(); i++) {
        cout << setw(15) << setprecision(5) << best_correct_predictions[i];
    }
    cout << endl;
}


int CNN_Genome::get_generation_id() const {
    return generation_id;
}

double CNN_Genome::get_fitness() const {
    return best_error;
}

int CNN_Genome::get_max_epochs() const {
    return max_epochs;
}

int CNN_Genome::get_epoch() const {
    return epoch;
}

int CNN_Genome::get_best_error_epoch() const {
    return best_error_epoch;
}

int CNN_Genome::get_best_predictions() const {
    return best_predictions;
}

int CNN_Genome::get_number_enabled_edges() const {
    int number_enabled_edges = 0;

    for (uint32_t i = 0; i < edges.size(); i++) {
        if (!edges[i]->is_disabled()) number_enabled_edges++;
    }

    return number_enabled_edges;
}


const vector<CNN_Node*> CNN_Genome::get_nodes() const {
    return nodes;
}

const vector<CNN_Edge*> CNN_Genome::get_edges() const {
    return edges;
}

CNN_Node* CNN_Genome::get_node(int node_position) {
    return nodes.at(node_position);
}

CNN_Edge* CNN_Genome::get_edge(int edge_position) {
    return edges.at(edge_position);
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


void CNN_Genome::add_node(CNN_Node* node) {
    nodes.insert( upper_bound(nodes.begin(), nodes.end(), node, sort_CNN_Nodes_by_depth()), node );

}

void CNN_Genome::add_edge(CNN_Edge* edge) {
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
            edge->reinitialize(generator);
            edge->save_best_weights();  //save weights for reuse
        }

        if (edge->get_output_node()->get_innovation_number() == node_innovation_number) {
            cout << "\tresizing edge with innovation number " << edge->get_innovation_number() << " as output from node with innovation number " << node_innovation_number << endl;
            edge->reinitialize(generator);
            edge->save_best_weights();  //save weights for reuse
        }
    }
}

bool CNN_Genome::sanity_check(int type) const {
    //check to see if all edge filters are the correct size
    for (uint32_t i = 0; i < edges.size(); i++) {
        if (!edges[i]->is_filter_correct()) {
            cerr << "SANITY CHECK FAILED! edges[" << i << "] had incorrect filter size!" << endl;
            cerr << edges[i] << endl;
            return false;
        }
    }

    //check for duplicate edges
    for (uint32_t i = 0; i < edges.size(); i++) {
        for (uint32_t j = i + 1; j < edges.size(); j++) {
            if (edges[i]->get_innovation_number() == edges[j]->get_innovation_number()) {
                cerr << "SANITY CHECK FAILED! edges[" << i << "] and edges[" << j << "] have the same innovation number: " << edges[i]->get_innovation_number() << endl;
                return false;
            }
        }
    }

    //check for duplicate nodes
    for (uint32_t i = 0; i < nodes.size(); i++) {
        for (uint32_t j = i + 1; j < nodes.size(); j++) {
            if (nodes[i]->get_innovation_number() == nodes[j]->get_innovation_number()) {
                cerr << "SANITY CHECK FAILED! nodes[" << i << "] and nodes[" << j << "] have the same innovation number: " << nodes[i]->get_innovation_number() << endl;
                return false;
            }
        }
    }

    if (type == SANITY_CHECK_AFTER_GENERATION) {
        /*
        for (uint32_t i = 0; i < nodes.size(); i++) {
            if (nodes[i]->has_zero_bias()) {
                cerr << "ERROR after generation!" << endl;
                cerr << "node in position " << i << " with innovation number: " << nodes[i]->get_innovation_number() << endl;
                cerr << "sum of bias was 0" << endl;
                cerr << "size_x: " << nodes[i]->get_size_x() << ", size_y: " << nodes[i]->get_size_y() << endl;
                return false;
            }
        }
        */
        //cout << "passed checking zero best bias" << endl;

        for (uint32_t i = 0; i < edges.size(); i++) {
            if (edges[i]->has_zero_weight()) {
                cerr << "ERROR before after_generation!" << endl;
                cerr << "ERROR: edge in position " << i << " with innovation number: " << edges[i]->get_innovation_number() << endl;
                cerr << "sum of weights was 0" << endl;
                cerr << "filter_x: " << edges[i]->get_filter_x() << ", filter_y: " << edges[i]->get_filter_y() << endl;
                return false;
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
            if (edges[j]->is_disabled()) {
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

bool CNN_Genome::outputs_connected() const {
    //check to see there is a path to from the input to each output

    for (uint32_t i = 0; i < nodes.size(); i++) {
        nodes[i]->set_unvisited();
    }

    input_node->visit();

    for (uint32_t i = 0; i < edges.size(); i++) {
        if (!edges[i]->is_disabled()) {
            if (edges[i]->get_input_node()->is_visited()) {
                edges[i]->get_output_node()->visit();
            }
        }
    }

    for (uint32_t i = 0; i < softmax_nodes.size(); i++) {
        if (!softmax_nodes[i]->is_visited()) {
            return false;
        }
    }

    return true;
}

int CNN_Genome::evaluate_image(const Image &image, vector<double> &class_error, bool do_backprop) {
    int expected_class = image.get_classification();
    int rows = image.get_rows();
    int cols = image.get_cols();

    /*
       for (uint32_t i = 0; i < edges.size(); i++) {
       edges[i]->print(cout);
       }
       */

    for (uint32_t i = 0; i < nodes.size(); i++) {
        //cout << "resetting node: " << i << endl;
        nodes[i]->reset();
    }

    input_node->set_values(image, rows, cols);

    //input_node->print(cout);
    
    /*
    for (uint32_t i = 0; i < nodes.size(); i++) {
        nodes[i]->print(cout);
    }
    */

    for (uint32_t i = 0; i < edges.size(); i++) {
        //edges[i]->print(cout);

        edges[i]->propagate_forward();
    }

    //cout << "Before softmax applied: " << endl;

    double softmax_sum = 0.0;
    for (uint32_t i = 0; i < softmax_nodes.size(); i++) {
        //softmax_nodes[i]->print(cout);

        double value = softmax_nodes[i]->get_value(0,0);
        double previous = value;

        if (isnan(value)) {
            cerr << "ERROR: value was NAN before exp!" << endl;
            exit(1);
        }

        //cout << "\tvalue " << softmax_nodes[i]->get_innovation_number() << " before exp: " << value << endl;

        //value = 1.0 / (1.0 + exp(-value));
        //value = tanh(value);
        value = exp(value);

        if (isnan(value)) {
            cerr << "ERROR: value was NAN AFTER exp! previously: " << previous << endl;
            exit(1);
        }

        softmax_nodes[i]->set_value(0, 0, value);
        //cout << "\tvalue " << softmax_nodes[i]->get_innovation_number() << ": " << softmax_nodes[i]->get_value(0,0) << endl;
        softmax_sum += value;

        if (isnan(softmax_sum)) {
            cerr << "ERROR: softmax_sum was NAN AFTER add!" << endl;
            exit(1);
        }
    }

    if (softmax_sum == 0) {
        cout << "ERROR! softmax sum == 0" << endl;
        exit(1);
    }

    //cout << "softmax sum: " << softmax_sum << endl;

    //cout << "After softmax applied: " << endl;

    //cout << "expected class: " << expected_class << endl;
    double avg_error = 0.0;
    double max_value = -100;
    int predicted_class = -1;

    for (uint32_t i = 0; i < softmax_nodes.size(); i++) {
        double value = softmax_nodes[i]->get_value(0,0) / softmax_sum;
        //cout << "\tvalue " << softmax_nodes[i]->get_innovation_number() << ": " << softmax_nodes[i]->get_value(0,0) << endl;

        if (isnan(value)) {
            cerr << "ERROR: value was NAN AFTER divide by softmax_sum, previously: " << softmax_nodes[i]->get_value(0,0) << endl;
            cerr << "softmax_sum: " << softmax_sum << endl;
            exit(1);
        }

        softmax_nodes[i]->set_value(0, 0,  value);

        //softmax_nodes[i]->print(cout);

        double error = 0.0;
        if (i == expected_class) {
            error = value - 1;
            //cout << ", error: " << error;
        } else {
            error = value;
        }
        //cout << "\t" << softmax_nodes[i]->get_innovation_number() << " -- value: " << value << ", error: " << error << endl;

        softmax_nodes[i]->set_error(0, 0, error);

        class_error[i] += fabs(error);

        if (value > max_value) {
            predicted_class = i;
            max_value = value;
        }

        avg_error += fabs(error);
    }
    //cout << "predicted class: " << predicted_class << endl;

    if (do_backprop) {
        for (int32_t i = edges.size() - 1; i >= 0; i--) {
            edges[i]->propagate_backward(mu, learning_rate, weight_decay);
        }

        for (int32_t i = nodes.size() - 1; i >= 0; i--) {
            nodes[i]->propagate_bias(mu, learning_rate, weight_decay);
        }
    }

    return predicted_class;
}

void CNN_Genome::initialize_weights() {
    for (uint32_t i = 0; i < edges.size(); i++) {
        edges[i]->initialize_weights(generator);
    }
}

void CNN_Genome::initialize_bias() {
    for (uint32_t i = 0; i < nodes.size(); i++) {
        nodes[i]->initialize_bias(generator);
    }
}

void CNN_Genome::save_weights() {
    for (uint32_t i = 0; i < edges.size(); i++) {
        edges[i]->save_best_weights();
    }
}

void CNN_Genome::save_bias() {
    for (uint32_t i = 0; i < nodes.size(); i++) {
        nodes[i]->save_best_bias();
    }
}

void CNN_Genome::reinitialize() {
    for (uint32_t i = 0; i < edges.size(); i++) {
        edges[i]->initialize_weights(generator);
        edges[i]->initialize_velocities();
        edges[i]->save_best_weights();
    }

    for (uint32_t i = 0; i < nodes.size(); i++) {
        nodes[i]->initialize_bias(generator);
        nodes[i]->initialize_velocities();
        nodes[i]->save_best_bias();
    }
}


void CNN_Genome::set_to_best() {
    for (uint32_t i = 0; i < edges.size(); i++) {
        edges[i]->set_weights_to_best();
        edges[i]->initialize_velocities();
    }

    for (uint32_t i = 0; i < nodes.size(); i++) {
        nodes[i]->set_bias_to_best();
        nodes[i]->initialize_velocities();
    }
}

void CNN_Genome::stochastic_backpropagation(const Images &images) {
    if (!started_from_checkpoint) {
        backprop_order.clear();
        for (uint32_t i = 0; i < images.get_number_images(); i++) {
            backprop_order.push_back(i);
        }

        shuffle(backprop_order.begin(), backprop_order.end(), generator); 
        //backprop_order.resize(2000);

        //cout << "initializing weights and biases!" << endl;
        if (reset_weights) {
            for (uint32_t i = 0; i < edges.size(); i++) {
                edges[i]->initialize_weights(generator);
            }

            for (uint32_t i = 0; i < nodes.size(); i++) {
                nodes[i]->initialize_bias(generator);
            }
        } else {
            set_to_best();
            //double check to make sure none of the bias or weights are all zero
            /*
            for (uint32_t i = 0; i < nodes.size(); i++) {
                if (nodes[i]->has_zero_bias()) {
                    cerr << "ERROR: node in position " << i << " with innovation number: " << nodes[i]->get_innovation_number() << endl;
                    cerr << "sum of bias was 0" << endl;
                    cerr << "size_x: " << nodes[i]->get_size_x() << ", size_y: " << nodes[i]->get_size_y() << endl;
                    exit(1);
                }
            }
            */

            for (uint32_t i = 0; i < edges.size(); i++) {
                if (edges[i]->has_zero_weight()) {
                    cerr << "ERROR: edge in position " << i << " with innovation number: " << edges[i]->get_innovation_number() << endl;
                    cerr << "sum of weights was 0" << endl;
                    cerr << "filter_x: " << edges[i]->get_filter_x() << ", filter_y: " << edges[i]->get_filter_y() << endl;
                    exit(1);
                }
            }
        }

        best_error = numeric_limits<double>::max();
    }

    //sort edges by depth of input node
    sort(edges.begin(), edges.end(), sort_CNN_Edges_by_depth());

    vector<int> class_sizes(images.get_number_classes(), 0);
    for (uint32_t i = 0; i < backprop_order.size(); i++) {
        class_sizes[ images.get_image(backprop_order[i]).get_classification() ]++;
    }

    do {
        shuffle(backprop_order.begin(), backprop_order.end(), generator); 

        vector<double> class_error(images.get_number_classes(), 0.0);
        vector<int> correct_predictions(images.get_number_classes(), 0);

        for (uint32_t j = 0; j < backprop_order.size(); j++) {
            evaluate_image(images.get_image(backprop_order[j]), class_error, true);
        }

        if (epoch % improvement_required_epochs == 0) {
            double total_error = 0.0;
            int total_predictions = 0;
            for (uint32_t j = 0; j < backprop_order.size(); j++) {
                int predicted_class = evaluate_image(images.get_image(backprop_order[j]), class_error, false);
                int expected_class = images.get_image(backprop_order[j]).get_classification();

                if (predicted_class == expected_class) {
                    correct_predictions[expected_class]++;
                    total_predictions++;
                }
            }

            /*
               for (uint32_t j = 0; j < 10; j++) {
               edges[j]->print(cout);
               }
               */

            //cout << "epoch " << epoch << " of " << epochs << endl;
            //cout << "mu: " << mu << endl;
            //cout << "class/prediction error: " << endl;
            for (uint32_t j = 0; j < class_error.size(); j++) {
                total_error += class_error[j];
                //cout << "\tclass " << setw(4) << j << ": " << setw(12) << setprecision(5) << class_error[j] << ", correct_predictions: " << correct_predictions[j] << " of " << class_sizes[j] << endl;
            }

            if (total_error < best_error) {
                best_error = total_error;
                best_error_epoch = epoch;
                best_predictions = total_predictions;
                best_predictions_epoch = epoch;

                best_class_error = class_error;
                best_correct_predictions = correct_predictions;

                if (output_filename.compare("") != 0) {
                    write_to_file(output_filename);
                }

                save_bias();
                save_weights();
            }

            cout << "[" << setw(10) << name << ", genome " << setw(5) << generation_id << "] best predictions: " << setw(10) << best_predictions << " of " << setw(10) << backprop_order.size() << ", best error: " << setw(20) << setprecision(5) << fixed << best_error << " on epoch: " << setw(5) << best_error_epoch << ", current epoch: " << setw(4) << epoch << " of " << setw(4) << max_epochs << ", current - best: " << setw(3) << (epoch - best_error_epoch) << ", mu: " << setw(10) << mu << ", learning_rate: " << setw(10) << learning_rate << ", weight_decay: " << setw(10) << weight_decay << endl;
            //cout << "total correct predictions: " << total_predictions << " of " << backprop_order.size() << endl;
            //cout << "total error:               " << left << setw(20) << setprecision(5) << fixed << total_error << endl;
            //cout << endl;
        }

        mu *= 1.010;
        if (mu > 0.95) mu = 0.95;

        learning_rate *= .99;
        if (learning_rate < 0.0001) learning_rate = 0.0001;

        weight_decay *= .99;
        if (weight_decay < 0.0001) weight_decay = 0.0001;

        epoch++;

        if (checkpoint_filename.compare("") != 0) {
            write_to_file(checkpoint_filename);
        }

        if (epoch > max_epochs) {
            break;
        }

        if (epoch > min_epochs && (epoch - best_error_epoch) > improvement_required_epochs) {
            break;
        }

    } while (true);

    /*
    for (uint32_t j = 0; j < edges.size(); j++) {
        edges[j]->print(cout);
    }
    */
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


void CNN_Genome::write(ostream &outfile) {
    outfile << initial_mu << endl;
    outfile << mu << endl;
    outfile << initial_learning_rate << endl;
    outfile << learning_rate << endl;
    outfile << initial_weight_decay << endl;
    outfile << weight_decay << endl;

    outfile << epoch << endl;
    outfile << min_epochs << endl;
    outfile << max_epochs << endl;
    outfile << improvement_required_epochs << endl;
    outfile << reset_weights << endl;

    outfile << setprecision(15) << fixed << best_predictions << endl;
    outfile << setprecision(15) << fixed << best_error << endl;
    outfile << best_predictions_epoch << endl;;
    outfile << best_error_epoch << endl;;

    outfile << generated_by_disable_edge << endl;
    outfile << generated_by_enable_edge << endl;
    outfile << generated_by_split_edge << endl;
    outfile << generated_by_add_edge << endl;
    outfile << generated_by_change_size << endl;
    outfile << generated_by_change_size_x << endl;
    outfile << generated_by_change_size_y << endl;
    outfile << generated_by_crossover << endl;

    outfile << generation_id << endl;
    //outfile << name << endl;
    //outfile << checkpoint_filename << endl;
    //outfile << output_filename << endl;

    outfile << generator << endl;
    outfile << rng_double << endl;
    outfile << rng_long << endl;

    outfile << nodes.size() << endl;
    for (uint32_t i = 0; i < nodes.size(); i++) {
        outfile << nodes[i] << endl;
    }

    outfile << edges.size() << endl;
    for (uint32_t i = 0; i < edges.size(); i++) {
        outfile << edges[i] << endl;
    }

    outfile << input_node->get_innovation_number() << endl;

    outfile << softmax_nodes.size() << endl;
    for (uint32_t i = 0; i < softmax_nodes.size(); i++) {
        outfile << softmax_nodes[i]->get_innovation_number() << endl;
    }

    outfile << backprop_order.size() << endl;
    for (uint32_t i = 0; i < backprop_order.size(); i++) {
        if (i > 0) outfile << " ";
        outfile << backprop_order[i];
    }
    outfile << endl;

    outfile << best_class_error.size() << endl;
    for (uint32_t i = 0; i < best_class_error.size(); i++) {
        if (i > 0) outfile << " ";
        outfile << best_class_error[i];
    }
    outfile << endl;

    outfile << best_correct_predictions.size() << endl;
    for (uint32_t i = 0; i < best_correct_predictions.size(); i++) {
        if (i > 0) outfile << " ";
        outfile << best_correct_predictions[i];
    }
    outfile << endl;
}

void CNN_Genome::read(istream &infile) {
    infile >> initial_mu;
    infile >> mu;
    infile >> initial_learning_rate;
    infile >> learning_rate;
    infile >> initial_weight_decay;
    infile >> weight_decay;

    infile >> epoch;
    infile >> min_epochs;
    infile >> max_epochs;
    infile >> improvement_required_epochs;
    infile >> reset_weights;

    infile >> best_predictions;
    infile >> best_error;
    infile >> best_predictions_epoch;
    infile >> best_error_epoch;

    infile >> generated_by_disable_edge;
    infile >> generated_by_enable_edge;
    infile >> generated_by_split_edge;
    infile >> generated_by_add_edge;
    infile >> generated_by_change_size;
    infile >> generated_by_change_size_x;
    infile >> generated_by_change_size_y;
    infile >> generated_by_crossover;

    infile >> generation_id;
    //infile >> name;
    //infile >> checkpoint_filename;
    //infile >> output_filename;

    infile >> generator;
    infile >> rng_double;
    infile >> rng_long;

    //cout << "reading nodes!" << endl;
    nodes.clear();
    int number_nodes;
    infile >> number_nodes;
    for (uint32_t i = 0; i < number_nodes; i++) {
        CNN_Node *node = new CNN_Node();
        infile >> node;

        //cout << "read node: " << node->get_innovation_number() << endl;
        nodes.push_back(node);
    }

    //cout << "reading edges!" << endl;
    edges.clear();
    int number_edges;
    infile >> number_edges;
    for (uint32_t i = 0; i < number_edges; i++) {
        CNN_Edge *edge = new CNN_Edge();
        infile >> edge;

        //cout << "read edge: " << edge->get_innovation_number() << endl;
        if (!edge->set_nodes(nodes)) {
            cerr << "ERROR: filter size didn't match when reading genome from input file!" << endl;
            cerr << "This should never happen!" << endl;
            exit(1);
        }

        edges.push_back(edge);
    }

    int input_node_innovation_number;
    infile >> input_node_innovation_number;

    //cout << "input node innovation number: " << input_node_innovation_number << endl;

    for (uint32_t i = 0; i < nodes.size(); i++) {
        if (nodes[i]->get_innovation_number() == input_node_innovation_number) {
            input_node = nodes[i];
            //cout << "input node was in position: " << i << endl;
            break;
        }
    }

    softmax_nodes.clear();
    int number_softmax_nodes;
    infile >> number_softmax_nodes;
    //cout << "number softmax nodes: " << number_softmax_nodes << endl;

    for (uint32_t i = 0; i < number_softmax_nodes; i++) {
        int softmax_node_innovation_number;
        infile >> softmax_node_innovation_number;
        //cout << "\tsoftmax node: " << softmax_node_innovation_number << endl;

        for (uint32_t i = 0; i < nodes.size(); i++) {
            if (nodes[i]->get_innovation_number() == softmax_node_innovation_number) {
                softmax_nodes.push_back(nodes[i]);
                //cout << "softmax node " << softmax_node_innovation_number << " was in position: " << i << endl;
                break;
            }
        }

    }

    backprop_order.clear();
    long order_size;
    infile >> order_size;
    for (uint32_t i = 0; i < order_size; i++) {
        long order;
        infile >> order;
        backprop_order.push_back(order);
    }


    best_class_error.clear();
    int error_size;
    infile >> error_size;
    for (uint32_t i = 0; i < error_size; i++) {
        double error;
        infile >> error;
        best_class_error.push_back(error);
    }


    best_correct_predictions.clear();
    int predictions_size;
    infile >> predictions_size;
    for (uint32_t i = 0; i < predictions_size; i++) {
        double predictions;
        infile >> predictions;
        best_correct_predictions.push_back(predictions);
    }
}

void CNN_Genome::write_to_file(string filename) {
    ofstream outfile(filename);
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
    out << "\t}" << endl << endl;

    out << "\t{" << endl;
    out << "\t\trank = sink;" << endl;
    for (uint32_t i = 0; i < nodes.size(); i++) {
        if (!nodes[i]->is_softmax()) continue;
        out << "\t\tnode" << nodes[i]->get_innovation_number() << " [shape=box,color=blue,label=\"output " << (nodes[i]->get_innovation_number() - 1) << "\\n" << nodes[i]->get_size_x() << " x " << nodes[i]->get_size_y() << "\"];" << endl;
    }
    out << "\t}" << endl << endl;

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


    //draw the visible edges
    for (uint32_t i = 0; i < nodes.size(); i++) {
        if (nodes[i]->is_input() || nodes[i]->is_softmax()) continue;

        out << "\tnode" << nodes[i]->get_innovation_number() << " [shape=box,label=\"node " << nodes[i]->get_innovation_number() << "\\n" << nodes[i]->get_size_x() << " x " << nodes[i]->get_size_y() << "\"];" << endl;
    }

    for (uint32_t i = 0; i < edges.size(); i++) {
        if (edges[i]->is_disabled()) {
            out << "\tnode" << edges[i]->get_input_node()->get_innovation_number() << " -> node" << edges[i]->get_output_node()->get_innovation_number() << " [color=red];" << endl;
        } else {
            out << "\tnode" << edges[i]->get_input_node()->get_innovation_number() << " -> node" << edges[i]->get_output_node()->get_innovation_number() << ";" << endl;
        }
    }

    out << "}" << endl;
}

void CNN_Genome::set_generated_by_disable_edge() {
    generated_by_disable_edge++;
}

void CNN_Genome::set_generated_by_enable_edge() {
    generated_by_enable_edge++;
}

void CNN_Genome::set_generated_by_split_edge() {
    generated_by_split_edge++;
}

void CNN_Genome::set_generated_by_add_edge() {
    generated_by_add_edge++;
}

void CNN_Genome::set_generated_by_change_size() {
    generated_by_change_size++;
}

void CNN_Genome::set_generated_by_change_size_x() {
    generated_by_change_size_x++;
}

void CNN_Genome::set_generated_by_change_size_y() {
    generated_by_change_size_y++;
}

void CNN_Genome::set_generated_by_crossover() {
    generated_by_crossover++;
}

void CNN_Genome::set_generated_by_reset_weights() {
    generated_by_reset_weights++;
}



int CNN_Genome::get_generated_by_disable_edge() {
    return generated_by_disable_edge;
}

int CNN_Genome::get_generated_by_enable_edge() {
    return generated_by_enable_edge;
}

int CNN_Genome::get_generated_by_split_edge() {
    return generated_by_split_edge;
}

int CNN_Genome::get_generated_by_add_edge() {
    return generated_by_add_edge;
}

int CNN_Genome::get_generated_by_change_size() {
    return generated_by_change_size;
}

int CNN_Genome::get_generated_by_change_size_x() {
    return generated_by_change_size_x;
}

int CNN_Genome::get_generated_by_change_size_y() {
    return generated_by_change_size_y;
}

int CNN_Genome::get_generated_by_crossover() {
    return generated_by_crossover;
}

int CNN_Genome::get_generated_by_reset_weights() {
    return generated_by_reset_weights;
}

