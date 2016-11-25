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

CNN_Genome::CNN_Genome() {
    started_from_checkpoint = false;
}


/**
 *  Iniitalize a genome from a set of nodes and edges
 */
CNN_Genome::CNN_Genome(int seed, int _epochs, const vector<CNN_Node*> &_nodes, const vector<CNN_Edge*> &_edges) {
    started_from_checkpoint = false;
    generator = mt19937(seed);
    rng_double = uniform_real_distribution<double>(0, 1.0);
    rng_long = uniform_int_distribution<long>(-numeric_limits<long>::max(), numeric_limits<long>::max());

    mu = 0.5;
    initial_mu = mu;
    epoch = 0;
    epochs = _epochs;
    best_predictions = 0;

    input_node = NULL;
    nodes.clear();
    softmax_nodes.clear();

    for (uint32_t i = 0; i < _nodes.size(); i++) {
        //cout << "copying node: " << i << endl;
        CNN_Node *node_copy = _nodes[i]->copy();
        //cout << "copied node: " << i << endl;

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
            edge->reinitialize(generator);
        }

        if (edge->get_output_node()->get_innovation_number() == node_innovation_number) {
            edge->reinitialize(generator);
        }
    }
}

bool CNN_Genome::sanity_check() const {
    //check to see if all edge filters are the correct size
    for (uint32_t i = 0; i < edges.size(); i++) {
        if (!edges[i]->is_filter_correct()) {
            cerr << "SANITY CHECK FAILED! edges[" << i << "] had incorrect filter size!" << endl;
            cerr << edges[i] << endl;
            return false;
        }
    }

    //check to see if total_inputs on each node are correct (equal to non-disabled input edges)
    for (uint32_t i = 0; i < nodes.size(); i++) {
        int number_inputs = nodes[i]->get_number_inputs();

        cout << "\t\tcounting inputs for node " << i << " (innovation number: " << nodes[i]->get_innovation_number() << ") -- should find " << number_inputs << endl;

        int counted_inputs = 0;
        for (uint32_t j = 0; j < edges.size(); j++) {
            if (edges[j]->is_disabled()) {
                cout << "\t\t\tedge " << j << " is disabled (" << edges[j]->get_input_innovation_number() << " to " << edges[j]->get_output_innovation_number() << ")" << endl;
                continue;
            }

            if (edges[j]->get_output_node()->get_innovation_number() == nodes[i]->get_innovation_number()) {
                cout << "\t\t\tedge " << j << " (" << edges[j]->get_input_innovation_number() << " to " << edges[j]->get_output_innovation_number() << ") output matches node innovation number" << endl;

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
                cout << "\t\t\tedge " << j << " (" << edges[j]->get_input_innovation_number() << " to " << edges[j]->get_output_innovation_number() << ") output does not match node innovation number" << endl;
            }
        }

        if (counted_inputs != number_inputs) {
            cerr << "SANITY CHECK FAILED! nodes[" << i << "] had total inputs: " << number_inputs << " but " << counted_inputs << " inputs were counted. " << endl;
            cerr << nodes[i] << endl;
            return false;
        }
    }

    return true;
}

bool CNN_Genome::outputs_connected() const {
    //check to see there is a path to from the input to each output


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
            edges[i]->propagate_backward(mu);
        }

        for (int32_t i = nodes.size() - 1; i >= 0; i--) {
            nodes[i]->propagate_bias(mu);
        }
    }

    return predicted_class;
}

void CNN_Genome::stochastic_backpropagation(const Images &images, string checkpoint_filename, string output_filename) {
    if (!started_from_checkpoint) {
        backprop_order.clear();
        for (uint32_t i = 0; i < images.get_number_images(); i++) {
            backprop_order.push_back(i);
        }
        cout << "backprop_order.size(): " << backprop_order.size() << endl;

        shuffle(backprop_order.begin(), backprop_order.end(), generator); 

        //TODO: don't initialize weights if using weights from parent
        cout << "initializing weights and biases!" << endl;
        for (uint32_t i = 0; i < edges.size(); i++) {
            edges[i]->initialize_weights(generator);
        }
        
        for (uint32_t i = 0; i < nodes.size(); i++) {
            nodes[i]->initialize_bias(generator);
        }
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

        if (total_predictions > best_predictions) {
            best_predictions = total_predictions;
            write_to_file(output_filename);
        }

        /*
           for (uint32_t j = 0; j < 10; j++) {
           edges[j]->print(cout);
           }
           */

        cout << "epoch " << epoch << " of " << epochs << endl;
        cout << "mu: " << mu << endl;
        cout << "class/prediction error: " << endl;
        for (uint32_t j = 0; j < class_error.size(); j++) {
            total_error += class_error[j];
            cout << "\tclass " << setw(4) << j << ": " << setw(12) << setprecision(5) << class_error[j] << ", correct_predictions: " << correct_predictions[j] << " of " << class_sizes[j] << endl;
        }
        cout << "total correct predictions: " << total_predictions << " of " << backprop_order.size() << endl;
        cout << "best predictions: " << best_predictions << " of " << backprop_order.size() << endl;
        cout << "total error: " << setw(20) << setprecision(5) << fixed << total_error << endl;
        cout << endl;

        mu += (1.0 - initial_mu) / epochs;
        if (mu > 0.9) mu = 0.9;

        epoch++;

        write_to_file(checkpoint_filename);
    } while (epoch < epochs);

    for (uint32_t j = 0; j < edges.size(); j++) {
        edges[j]->print(cout);
    }

}

void CNN_Genome::write_to_file(string filename) {
    ofstream outfile(filename);

    outfile << initial_mu << endl;
    outfile << mu << endl;
    outfile << epoch << endl;
    outfile << epochs << endl;
    outfile << best_predictions << endl;

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
    outfile.close();
}


void CNN_Genome::read_from_file(string filename, bool is_checkpoint) {
    started_from_checkpoint = is_checkpoint;

    ifstream infile(filename);

    infile >> initial_mu;
    infile >> mu;
    infile >> epoch;
    infile >> epochs;
    infile >> best_predictions;

    infile >> generator;
    infile >> rng_double;
    infile >> rng_long;

    nodes.clear();
    int number_nodes;
    infile >> number_nodes;
    for (uint32_t i = 0; i < number_nodes; i++) {
        CNN_Node *node = new CNN_Node();
        infile >> node;
        nodes.push_back(node);
    }

    edges.clear();
    int number_edges;
    infile >> number_edges;
    for (uint32_t i = 0; i < number_edges; i++) {
        CNN_Edge *edge = new CNN_Edge();
        infile >> edge;

        cout << "read edge: " << edge->get_innovation_number() << endl;
        if (!edge->set_nodes(nodes)) {
            cerr << "ERROR: filter size didn't match when reading genome from input file!" << endl;
            cerr << "This should never happen!" << endl;
            exit(1);
        }

        edges.push_back(edge);
    }

    int input_node_innovation_number;
    infile >> input_node_innovation_number;

    cout << "input node innovation number: " << input_node_innovation_number << endl;

    //TODO: fix this -- nodes will not be sorted by innovation number!!!!!
    input_node = nodes[input_node_innovation_number];

    softmax_nodes.clear();
    int number_softmax_nodes;
    infile >> number_softmax_nodes;
    cout << "number softmax nodes: " << number_softmax_nodes << endl;

    for (uint32_t i = 0; i < number_softmax_nodes; i++) {
        int softmax_node_innovation_number;
        infile >> softmax_node_innovation_number;
        cout << "\tsoftmax node: " << softmax_node_innovation_number << endl;
        softmax_nodes.push_back(nodes[softmax_node_innovation_number]);
    }

    backprop_order.clear();
    long order_size;
    infile >> order_size;
    for (uint32_t i = 0; i < order_size; i++) {
        long order;
        infile >> order;
        backprop_order.push_back(order);
    }
    infile.close();
}
