#include <fstream>
using std::ofstream;
using std::ifstream;
using std::ios;

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

#include <string>
using std::string;
using std::to_string;

#include <vector>
using std::vector;


#include "image_tools/image_set.hxx"
#include "cnn_node.hxx"
#include "cnn_edge.hxx"
#include "cnn_genome.hxx"

CNN_NEAT_Genome::CNN_NEAT_Genome() {
    started_from_checkpoint = false;
}


/**
 *  Iniitalize a genome from a set of nodes and edges
 */
CNN_NEAT_Genome::CNN_NEAT_Genome(int seed, int _epochs, vector<CNN_Node*> _nodes, vector<CNN_Edge*> _edges) {
    started_from_checkpoint = false;
    generator = mt19937(seed);

    mu = 0.5;
    initial_mu = mu;
    epoch = 0;
    epochs = _epochs;
    best_predictions = 0;

    input_node = NULL;
    nodes.clear();
    hidden_nodes.clear();
    output_nodes.clear();
    softmax_nodes.clear();

    for (uint32_t i = 0; i < _nodes.size(); i++) {
        CNN_Node *node_copy = _nodes[i]->copy();


        if (node_copy->is_input()) {
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

        if (node_copy->is_hidden()) {
            int depth = node_copy->get_depth();

            while (hidden_nodes.size() < depth - 1) {
                hidden_nodes.push_back(vector<CNN_Node*>());
            }
            hidden_nodes[depth-1].push_back(node_copy);
        }

        if (node_copy->is_output()) {
            output_nodes.push_back(node_copy);
        }

        if (node_copy->is_softmax()) {
            softmax_nodes.push_back(node_copy);
        }

        nodes.push_back( node_copy );
    }

    for (uint32_t i = 0; i < _edges.size(); i++) {
        CNN_Edge *edge_copy = _edges[i]->copy();
        edge_copy->set_nodes(nodes);
        edges.push_back( edge_copy );
    }

}

/**
 *  Initialize the initial genotype for the CNN_NEAT algorithm from
 *  a set of training images
 */
CNN_NEAT_Genome::CNN_NEAT_Genome(int number_classes, int rows, int cols, int seed, int _epochs) {
    cout << "number classes: " << number_classes << endl;
    cout << "rows: " << rows << ", cols: " << cols << endl;

    started_from_checkpoint = false;
    generator = mt19937(seed);

    mu = 0.5;
    initial_mu = mu;
    epoch = 0;
    epochs = _epochs;
    best_predictions = 0;

    int number_weights = 0;

    int node_innovation_number = 0;
    int edge_innovation_number = 0;

    input_node = new CNN_Node(node_innovation_number, 0, rows, cols, true /*input*/, false /*output*/, false /*softmax*/);
    nodes.push_back(input_node);
    node_innovation_number++;

    //first layer
    hidden_nodes.push_back(vector<CNN_Node*>());
    for (uint32_t i = 0; i < 5; i++) {
        CNN_Node *output_node = new CNN_Node(node_innovation_number, 1, 10, 10, false /*input*/, true /*output*/, false /*softmax*/);
        node_innovation_number++;

        nodes.push_back(output_node);
        hidden_nodes[0].push_back(output_node);

        edges.push_back(new CNN_Edge(input_node, output_node, true, edge_innovation_number, generator));
        number_weights += edges.back()->get_number_weights();
        cout << "connecting input_node to hidden_node[0][" << i << "], number weights: " << edges.back()->get_number_weights() << endl;
        edge_innovation_number++;
    }

    //second layer
    hidden_nodes.push_back(vector<CNN_Node*>());
    for (uint32_t i = 0; i < 31; i++) {
        CNN_Node *hidden_node = new CNN_Node(node_innovation_number, 2, 5, 5, false /*input*/, false /*output*/, false /*softmax*/);
        node_innovation_number++;

        nodes.push_back(hidden_node);
        hidden_nodes[1].push_back(hidden_node);
    }

    vector<int> input_connections;
    input_connections.push_back(0);
    input_connections.push_back(0);
    input_connections.push_back(0);
    input_connections.push_back(0);
    input_connections.push_back(1);

    int output_node = 0;

    for (uint32_t i = 0; i < hidden_nodes[1].size(); i++) {
        for (uint32_t j = 0; j < input_connections.size(); j++) {
            if (input_connections[j] == 0) continue;


            edges.push_back(new CNN_Edge(hidden_nodes[0][input_connections[j] - 1], hidden_nodes[1][output_node], true, edge_innovation_number, generator));
            number_weights += edges.back()->get_number_weights();
            cout << "connecting hidden_node[0][" << input_connections[j] - 1 << "] to hidden_node[1][" << output_node << "], number weights: " << edges.back()->get_number_weights() << endl;
            edge_innovation_number++;

        }
        cout << "input connections: ";
        for (int32_t k = 0; k < input_connections.size(); k++) {
            cout << setw(3) << input_connections[k];
        }
        cout << endl;

        input_connections[input_connections.size() - 1]++;
        for (int32_t k = input_connections.size() - 1; k >= 0; k--) {
            if (input_connections[k] > input_connections.size()) {

                int current = k - 1;
                input_connections[current]++;
                while (current > 0 && input_connections[current] > (current + 1)) {
                    current--;

                    input_connections[current]++;
                }

                for (int32_t l = current + 1; l < input_connections.size(); l++) {
                    input_connections[l] = input_connections[l - 1] + 1;
                }
            }
        }

        cout << "input connections after increment: ";
        for (int32_t k = 0; k < input_connections.size(); k++) {
            cout << setw(3) << input_connections[k];
        }
        cout << endl;

        output_node++;
    }

    for (uint32_t i = 0; i < 10; i++) {
        CNN_Node *output_node = new CNN_Node(node_innovation_number, 3, 1, 1, false /*input*/, true /*output*/, false /*softmax*/);
        node_innovation_number++;

        output_nodes.push_back(output_node);
        nodes.push_back(output_node);

        for (uint32_t j = 0; j < hidden_nodes[1].size(); j++) {

            edges.push_back(new CNN_Edge(hidden_nodes[1][j], output_node, true, edge_innovation_number, generator));
            number_weights += edges.back()->get_number_weights();
            cout << "connecting hidden_node[1][" << j << "] to output_node[" << i << "], number_weights: " <<  edges.back()->get_number_weights() << endl;
            edge_innovation_number++;
        }
    }

    for (uint32_t i = 0; i < number_classes; i++) {
        CNN_Node *softmax_node = new CNN_Node(node_innovation_number, 4, 1, 1, false /*input*/, false /*output*/, true /*softmax*/);
        node_innovation_number++;

        softmax_nodes.push_back(softmax_node);
        nodes.push_back(softmax_node);

        for (uint32_t j = 0; j < output_nodes.size(); j++) {

            edges.push_back(new CNN_Edge(output_nodes[j], softmax_node, true, edge_innovation_number, generator));
            number_weights += edges.back()->get_number_weights();
            cout << "connecting output_node[" << j << "] to softmax_node[" << i << "], number_weights: " << edges.back()->get_number_weights() << endl;
            edge_innovation_number++;
        }
    }

    /*
       for (uint32_t i = 0; i < number_classes; i++) {
       CNN_Node *output_node = new CNN_Node(node_innovation_number, 1, 1, 1, true);
       node_innovation_number++;

       output_nodes.push_back(output_node);
       nodes.push_back(output_node);

       edges.push_back(new CNN_Edge(input_node, output_node, true, edge_innovation_number));
       number_weights += edges.back()->get_number_weights();
       edge_innovation_number++;
       }

       for (uint32_t i = 0; i < number_classes; i++) {
       CNN_Node *softmax_node = new CNN_Node(node_innovation_number, 2, 1, 1, true);
       node_innovation_number++;

       softmax_nodes.push_back(softmax_node);
       nodes.push_back(softmax_node);

       for (uint32_t j = 0; j < number_classes; j++) {
       edges.push_back(new CNN_Edge(output_nodes[j], softmax_node, true, edge_innovation_number));
       number_weights += edges.back()->get_number_weights();
       edge_innovation_number++;
       }
       }
       */

    cout << "number_weights: " << number_weights << endl;
}


int CNN_NEAT_Genome::evaluate_image(const Image &image, vector<double> &class_error, bool do_backprop) {
    int expected_class = image.get_classification();
    int rows = image.get_rows();
    int cols = image.get_cols();

    //sort edges by depth of input node
    sort(edges.begin(), edges.end(), sort_CNN_Edges_by_depth());

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

    /*
       for (uint32_t i = 0; i < output_nodes.size(); i++) {
       output_nodes[i]->print(cout);
       }
       */

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
    }

    return predicted_class;
}

void CNN_NEAT_Genome::stochastic_backpropagation(const Images &images, string checkpoint_filename, string output_filename) {
    if (!started_from_checkpoint) {
        backprop_order.clear();
        for (uint32_t i = 0; i < images.get_number_images(); i++) {
            backprop_order.push_back(i);
        }
        cout << "backprop_order.size(): " << backprop_order.size() << endl;

        shuffle(backprop_order.begin(), backprop_order.end(), generator); 
    }

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

void CNN_NEAT_Genome::write_to_file(string filename) {
    ofstream outfile(filename);

    outfile << initial_mu << endl;
    outfile << mu << endl;
    outfile << epoch << endl;
    outfile << epochs << endl;
    outfile << best_predictions << endl;

    outfile << generator << endl;

    outfile << nodes.size() << endl;
    for (uint32_t i = 0; i < nodes.size(); i++) {
        outfile << nodes[i] << endl;
    }

    outfile << edges.size() << endl;
    for (uint32_t i = 0; i < edges.size(); i++) {
        outfile << edges[i] << endl;
    }

    outfile << input_node->get_innovation_number() << endl;

    outfile << hidden_nodes.size() << endl;
    for (uint32_t i = 0; i < hidden_nodes.size(); i++) {
        outfile << hidden_nodes[i].size() << endl;
        for (uint32_t j = 0; j < hidden_nodes[i].size(); j++) {
            outfile << hidden_nodes[i][j]->get_innovation_number() << endl;
        }
    }

    outfile << output_nodes.size() << endl;
    for (uint32_t i = 0; i < output_nodes.size(); i++) {
        outfile << output_nodes[i]->get_innovation_number() << endl;
    }

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


void CNN_NEAT_Genome::read_from_file(string filename) {
    started_from_checkpoint = true;

    ifstream infile(filename);

    infile >> initial_mu;
    infile >> mu;
    infile >> epoch;
    infile >> epochs;
    infile >> best_predictions;

    infile >> generator;

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
        edge->set_nodes(nodes);

        edges.push_back(edge);
    }

    int input_node_innovation_number;
    infile >> input_node_innovation_number;

    cout << "input node innovation number: " << input_node_innovation_number << endl;

    input_node = nodes[input_node_innovation_number];

    hidden_nodes.clear();
    int number_hidden_layers;
    infile >> number_hidden_layers;
    cout << "number hidden layers: " << number_hidden_layers << endl;

    for (uint32_t i = 0; i < number_hidden_layers; i++) {
        hidden_nodes.push_back(vector<CNN_Node*>());

        int hidden_layer_size;
        infile >> hidden_layer_size;

        cout << "hidden layer " << i << " size: " << hidden_layer_size << endl;

        for (uint32_t j = 0; j < hidden_layer_size; j++) {
            int hidden_node_innovation_number;
            infile >> hidden_node_innovation_number;

            cout << "\thidden node: " << hidden_node_innovation_number << endl;

            hidden_nodes[i].push_back(nodes[hidden_node_innovation_number]);
        }
    }

    output_nodes.clear();
    int number_output_nodes;
    infile >> number_output_nodes;
    cout << "number output nodes: " << number_output_nodes << endl;

    for (uint32_t i = 0; i < number_output_nodes; i++) {
        int output_node_innovation_number;
        infile >> output_node_innovation_number;
        cout << "\toutput node: " << output_node_innovation_number << endl;
        output_nodes.push_back(nodes[output_node_innovation_number]);
    }

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
