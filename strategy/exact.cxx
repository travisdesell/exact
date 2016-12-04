#include <algorithm>
using std::sort;
using std::upper_bound;

#include <iomanip>
using std::fixed;
using std::setprecision;
using std::setw;
using std::left;
using std::right;

#include <iostream>
using std::ostream;
using std::istream;

#include <limits>
using std::numeric_limits;

#include <random>
using std::mt19937;
using std::normal_distribution;
using std::uniform_int_distribution;
using std::uniform_real_distribution;

#include <sstream>
using std::istringstream;
using std::ostringstream;

#include <string>
using std::to_string;

#include <vector>
using std::vector;

#include "image_tools/image_set.hxx"
#include "cnn_node.hxx"
#include "cnn_edge.hxx"
#include "cnn_genome.hxx"
#include "exact.hxx"

EXACT::EXACT(const Images &images, int _population_size, int _min_epochs, int _max_epochs, int _improvement_required_epochs, bool _reset_edges) {
    reset_edges = _reset_edges;
    min_epochs = _min_epochs;
    max_epochs = _max_epochs;
    improvement_required_epochs = _improvement_required_epochs;

    unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
    //unsigned seed = 10;

    generator = mt19937(seed);
    rng_long = uniform_int_distribution<long>(-numeric_limits<long>::max(), numeric_limits<long>::max());
    rng_double = uniform_real_distribution<double>(0, 1.0);

    node_innovation_count = 0;
    edge_innovation_count = 0;

    inserted_genomes = 0;

    population_size = _population_size;

    image_rows = images.get_image_rows();
    image_cols = images.get_image_cols();
    number_classes = images.get_number_classes();

    genomes_generated = 0;

    number_mutations = 3;

    learning_rate = 0.001;
    weight_decay = 0.001;

    edge_disable = 2.0;
    edge_enable = 2.0;
    edge_split = 3.0;
    edge_add = 4.0;
    edge_change_stride = 0.0;
    node_change_size = 2.0;
    node_change_size_x = 0.0;
    node_change_size_y = 0.0;
    node_change_pool_size = 0.0;

    cout << "EXACT settings: " << endl;
    cout << "\tlearning_rate: " << learning_rate << endl;
    cout << "\tweight_decay: " << weight_decay << endl;
    cout << "\tmin_epochs: " << min_epochs << endl;
    cout << "\tmax_epochs: " << max_epochs << endl;
    cout << "\timprovement_required_epochs: " << improvement_required_epochs << endl;
    cout << "\tnumber_mutations: " << number_mutations << endl;
    cout << "\tedge_disable: " << edge_disable << endl;
    cout << "\tedge_split: " << edge_split << endl;
    cout << "\tedge_add: " << edge_add << endl;
    cout << "\tedge_change_stride: " << edge_change_stride << endl;
    cout << "\tnode_change_size: " << node_change_size << endl;
    cout << "\tnode_change_size_x: " << node_change_size_x << endl;
    cout << "\tnode_change_size_y: " << node_change_size_y << endl;
    cout << "\tnode_change_pool_size: " << node_change_pool_size << endl;

    double total = edge_disable + edge_enable + edge_split + edge_add + edge_change_stride +
                   node_change_size + node_change_size_x + node_change_size_y + node_change_pool_size;

    edge_disable /= total;
    edge_enable /= total;
    edge_split /= total;
    edge_add /= total;
    edge_change_stride /= total;
    node_change_size /= total;
    node_change_size_x /= total;
    node_change_size_y /= total;
    node_change_pool_size /= total;

    cout << "mutation probabilities: " << endl;
    cout << "\tedge_disable: " << edge_disable << endl;
    cout << "\tedge_split: " << edge_split << endl;
    cout << "\tedge_add: " << edge_add << endl;
    cout << "\tedge_change_stride: " << edge_change_stride << endl;
    cout << "\tnode_change_size: " << node_change_size << endl;
    cout << "\tnode_change_size_x: " << node_change_size_x << endl;
    cout << "\tnode_change_size_y: " << node_change_size_y << endl;
    cout << "\tnode_change_pool_size: " << node_change_pool_size << endl;
}

CNN_Genome* EXACT::get_best_genome() {
    return genomes[0];
}

CNN_Genome* EXACT::generate_individual() {
    CNN_Genome *genome = NULL;
    if (genomes.size() == 0) {
        //generate the initial minimal CNN
        int total_weights = 0;

        CNN_Node *input_node = new CNN_Node(node_innovation_count, 0, image_rows, image_cols, INPUT_NODE);
        input_node->initialize_bias(generator);
        input_node->save_best_bias();
        node_innovation_count++;
        all_nodes.push_back(input_node);

        for (uint32_t i = 0; i < number_classes; i++) {
            CNN_Node *softmax_node = new CNN_Node(node_innovation_count, 1, 1, 1, SOFTMAX_NODE);
            softmax_node->initialize_bias(generator);
            softmax_node->save_best_bias();
            node_innovation_count++;
            all_nodes.push_back(softmax_node);
        }

        for (uint32_t i = 0; i < number_classes; i++) {
            CNN_Edge *edge = new CNN_Edge(input_node, all_nodes[i + 1] /*ith softmax node*/, true, edge_innovation_count);
            edge->initialize_weights(generator);
            edge->save_best_weights();

            all_edges.push_back(edge);

            total_weights += all_edges.back()->get_number_weights();
            edge_innovation_count++;
        }

        long genome_seed = rng_long(generator);
        //cout << "seeding genome with: " << genome_seed << endl;

        genome = new CNN_Genome(genomes_generated++, genome_seed, min_epochs, max_epochs, improvement_required_epochs, reset_edges, learning_rate, weight_decay, all_nodes, all_edges);
        //save the weights and bias of the initially generated genome for reuse
        genome->save_weights();
        genome->save_bias();

    } else if (genomes.size() <= population_size) {
        //generate random mutatinos until genomes.size() < population_size
        while (genome == NULL) {
            genome = create_mutation();

            if (!genome->outputs_connected()) {
                cerr << "\tAll softmax nodes were not reachable, deleting genome." << endl;
                delete genome;
                genome = NULL;
            }
        }
    } else {
        //TODO: either generate repropductions or mutations

        while (genome == NULL) {
            genome = create_mutation();

            if (!genome->outputs_connected()) {
                cerr << "\tAll softmax nodes were not reachable, deleting genome." << endl;
                delete genome;
                genome = NULL;
            }
        }
    }

    if (!genome->sanity_check(SANITY_CHECK_AFTER_GENERATION)) {
        cerr << "ERROR: genome " << genome->get_generation_id() << " failed sanity check in generate individual!" << endl;
        exit(1);
    }

    if (genomes.size() < population_size) {
        //insert a copy with a bad fitness so we have more things to generate new genomes with
        CNN_Genome *genome_copy = new CNN_Genome(genomes_generated++, /*new random seed*/ rng_long(generator), min_epochs, max_epochs, improvement_required_epochs, reset_edges, learning_rate, weight_decay, genome->get_nodes(), genome->get_edges());

        //for more variability in the initial population, re-initialize weights and bias for these unevaluated copies
        genome_copy->initialize_weights();
        genome_copy->initialize_bias();
        genome_copy->save_weights();
        genome_copy->save_bias();

        insert_genome(genome_copy);
    }

    return genome;
}

string parse_fitness(double fitness) {
    if (fitness == numeric_limits<double>::max()) {
        return "UNEVALUATED";
    } else {
        return to_string(fitness);
    }
}

void EXACT::insert_genome(CNN_Genome* genome) {
    inserted_genomes++;

    if (!genome->sanity_check(SANITY_CHECK_BEFORE_INSERT)) {
        cerr << "ERROR: genome " << genome->get_generation_id() << " failed sanity check before insert!" << endl;
        exit(1);
    }
    cout << "genome " << genome->get_generation_id() << " passed sanity check with fitness: " << parse_fitness(genome->get_fitness()) << endl;

    cout << "genomes evaluated: " << setw(10) << inserted_genomes << ", inserting: " << parse_fitness(genome->get_fitness());
    if (genomes.size() == 0 || genome->get_fitness() < genomes[0]->get_fitness()) {
        cout << " -- new best fitness!";
        genome->write_to_file("global_best_" + to_string(inserted_genomes) + ".txt");

        ofstream gv_file("global_best_" + to_string(inserted_genomes) + ".gv");
        gv_file << "#EXACT settings: " << endl;
        gv_file << "#\tlearning_rate: " << learning_rate << endl;
        gv_file << "#\tweight_decay: " << weight_decay << endl;

        gv_file << "#\tmin_epochs: " << min_epochs << endl;
        gv_file << "#\tmax_epochs: " << max_epochs << endl;
        gv_file << "#\timprovement_required_epochs: " << improvement_required_epochs << endl;

        gv_file << "#\tnumber_mutations: " << number_mutations << endl;
        gv_file << "#\tedge_disable: " << edge_disable << endl;
        gv_file << "#\tedge_split: " << edge_split << endl;
        gv_file << "#\tedge_add: " << edge_add << endl;
        gv_file << "#\tedge_change_stride: " << edge_change_stride << endl;
        gv_file << "#\tnode_change_size: " << node_change_size << endl;
        gv_file << "#\tnode_change_size_x: " << node_change_size_x << endl;
        gv_file << "#\tnode_change_size_y: " << node_change_size_y << endl;
        gv_file << "#\tnode_change_pool_size: " << node_change_pool_size << endl;

        genome->print_graphviz(gv_file);
        gv_file.close();
    }
    cout << endl;

    //inorder insert the new individual
    genomes.insert( upper_bound(genomes.begin(), genomes.end(), genome, sort_genomes_by_fitness()), genome);


    //delete the worst individual if we've reached the population size
    if (genomes.size() > population_size) {
        CNN_Genome *worst = genomes.back();
        genomes.pop_back();
        delete worst;
    }
    
    cout << "genome fitnesses:" << endl;
    for (uint32_t i = 0; i < genomes.size(); i++) {
        cout << "\t" << setw(4) << i << " -- genome: " << setw(10) << genomes[i]->get_generation_id() << ", "
             << setw(20) << left << "fitness: " << right << setw(15) << setprecision(5) << fixed << parse_fitness(genomes[i]->get_fitness())
             << " (" << genomes[i]->get_best_predictions() << " correct) on epoch: " << genomes[i]->get_best_error_epoch() 
             << ", number enabled edges: " << genomes[i]->get_number_enabled_edges()
             << ", number nodes: " << genomes[i]->get_number_nodes() << endl;
    }
    cout << "genome best error: " << endl;
    for (uint32_t i = 0; i < genomes.size(); i++) {
        cout << "\t" << setw(4) << i << " -- genome: " << setw(10) << genomes[i]->get_generation_id() << ", ";
        genomes[i]->print_best_error(cout);
    }

    cout << "genome correct predictions: " << endl;
    for (uint32_t i = 0; i < genomes.size(); i++) {
        cout << "\t" << setw(4) << i << " -- genome: " << setw(10) << genomes[i]->get_generation_id() << ", ";
        genomes[i]->print_best_predictions(cout);
    }

    cout << endl;
}

/*
CNN_Genome* EXACT::create_crossover() {
    double r1 = rng_double(generator) * genomes.size();
    double r2 = rng_double(generator) * (genomes.size() - 1);
    if (r2 == r1) r2++;

    CNN_Genome *parent1 = genomes[r1];
    CNN_Genome *parent2 = genomes[r2];

    vector< CNN_Node* > child_nodes;
    vector< CNN_Edge* > child_edges;

}
*/

CNN_Genome* EXACT::create_mutation() {
    //mutation options:
    //edges:
    //  1. disable edge (but make sure output node is still reachable)
    //  2. split edge
    //  3. add edge (make sure it does not exist already)
    //  4. increase/decrease stride (not yet)
    //nodes:
    //  1. increase/decrease size_x
    //  2. increase/decrease size_y
    //  3. increase/decrease max_pool (not yet)

    long child_seed = rng_long(generator);

    CNN_Genome *parent = genomes[rng_double(generator) * genomes.size()];

    cout << "\tgenerating child " << genomes_generated << " from parent genome: " << parent->get_generation_id() << endl;

    CNN_Genome *child = new CNN_Genome(genomes_generated++, child_seed, min_epochs, max_epochs, improvement_required_epochs, reset_edges, learning_rate, weight_decay, parent->get_nodes(), parent->get_edges());
    if (parent->get_fitness() == numeric_limits<double>::max()) {
        //This parent has not actually been evaluated (the population is still initializing)
        //we can set the best_bias and best_weights randomly so that they are used when it
        //starts up

        cout << "\tparent had not been evaluated yet, but best_bias and best_weights should have been set randomly" << endl;

        /*
        for (uint32_t i = 0; i < child->get_number_nodes(); i++) {
            child->get_node(i)->initialize_bias(generator);
            child->get_node(i)->save_best_bias();
        }

        for (uint32_t i = 0; i < child->get_number_edges(); i++) {
            child->get_edge(i)->initialize_weights(generator);
            child->get_edge(i)->save_best_weights();
        }
        */
    } else {
        cout << "\tparent had been evaluated! not setting best_bias and best_weights randomly" << endl;
        cout << "\tparent fitness: " << parent->get_fitness() << endl;
    }

    if (genomes.size() == 1) return child;

    int modifications = 0;

    while (modifications < number_mutations) {
        double r = rng_double(generator);

        cerr << "\tr: " << r << endl;

        if (r < edge_disable) {
            cout << "\tDISABLING EDGE!" << endl;

            int edge_position = rng_double(generator) * child->get_number_edges();
            if (child->disable_edge(edge_position)) {
                modifications++;
            }

            continue;
        } 
        r -= edge_disable;

        if (r < edge_enable) {
            cout << "\tENABLING EDGE!" << endl;

            vector< CNN_Edge* > disabled_edges;

            for (uint32_t i = 0; i < child->get_number_edges(); i++) {
                CNN_Edge* current = child->get_edge(i);

                if (current == NULL) {
                    cout << "ERROR! edge " << i << " became null on child!" << endl;
                    exit(1);
                }

                if (current->is_disabled()) {
                    disabled_edges.push_back(current);
                }
            }
            
            if (disabled_edges.size() > 0) {
                int edge_position = rng_double(generator) * disabled_edges.size();
                CNN_Edge* disabled_edge = disabled_edges[edge_position];

                cout << "\t\tenabling edge: " << disabled_edge->get_innovation_number() << " between input node innovation number " << disabled_edge->get_input_node()->get_innovation_number() << " and output node innovation number " << disabled_edge->get_output_node()->get_innovation_number() << endl;

                disabled_edge->enable();
                //reinitialize weights for re-enabled edge
                disabled_edge->initialize_weights(generator);
                disabled_edge->save_best_weights(); //save the random weights so they are reused
                modifications++;
            } else {
                cout << "\t\tcould not enable an edge as there were no disabled edges!" << endl;
            }

            continue;
        } 
        r -= edge_enable;


        if (r < edge_split) {
            int edge_position = rng_double(generator) * child->get_number_edges();
            cout << "\tSPLITTING EDGE IN POSITION: " << edge_position << "!" << endl;

            CNN_Edge* edge = child->get_edge(edge_position);

            CNN_Node* input_node = edge->get_input_node();
            CNN_Node* output_node = edge->get_output_node();

            double depth = (input_node->get_depth() + output_node->get_depth()) / 2.0;
            int size_x = (input_node->get_size_x() + output_node->get_size_x()) / 2.0;
            int size_y = (input_node->get_size_y() + output_node->get_size_y()) / 2.0;

            CNN_Node *child_node = new CNN_Node(node_innovation_count, depth, size_x, size_y, HIDDEN_NODE);
            child_node->initialize_bias(generator);
            child_node->save_best_bias();
            node_innovation_count++;

            //add two new edges, disable the split edge
            cout << "\t\tcreating edge " << edge_innovation_count << endl;
            CNN_Edge *edge1 = new CNN_Edge(input_node, child_node, false, edge_innovation_count);
            edge_innovation_count++;
            edge1->initialize_weights(generator);
            edge1->save_best_weights(); //save the random weights so they are reused instead of 0

            cout << "\t\tcreating edge " << edge_innovation_count << endl;
            CNN_Edge *edge2 = new CNN_Edge(child_node, output_node, false, edge_innovation_count);
            edge_innovation_count++;
            edge2->initialize_weights(generator);
            edge2->save_best_weights(); //save the random weights so they are resused instead of 0

            cout << "\t\tdisabling edge " << edge->get_innovation_number() << endl;
            edge->disable();

            child->add_node(child_node);
            child->add_edge(edge1);
            child->add_edge(edge2);

            //make sure copies are added to all_edges and all_nodes
            CNN_Node *node_copy = child_node->copy();
            CNN_Edge *edge_copy_1 = edge1->copy();
            CNN_Edge *edge_copy_2 = edge2->copy();

            //insert the new node into the population in sorted order
            all_nodes.insert( upper_bound(all_nodes.begin(), all_nodes.end(), node_copy, sort_CNN_Nodes_by_depth()), node_copy);
            edge_copy_1->set_nodes(all_nodes);
            edge_copy_2->set_nodes(all_nodes);

            all_edges.insert( upper_bound(all_edges.begin(), all_edges.end(), edge_copy_1, sort_CNN_Edges_by_depth()), edge_copy_1);
            all_edges.insert( upper_bound(all_edges.begin(), all_edges.end(), edge_copy_2, sort_CNN_Edges_by_depth()), edge_copy_2);

            modifications++;

            continue;
        }
        r -= edge_split;

        if (r < edge_add) {
            cout << "\tADDING EDGE!" << endl;

            CNN_Node *node1;
            CNN_Node *node2;

            do {
                int r1 = rng_double(generator) * child->get_number_nodes();
                int r2 = rng_double(generator) * child->get_number_nodes() - 1;

                if (r1 == r2) r2++;

                if (r1 > r2) {  //swap r1 and r2 so node2 is always deeper than node1
                    int temp = r1;
                    r1 = r2;
                    r2 = temp;
                }

                node1 = child->get_node(r1);
                node2 = child->get_node(r2);
            } while (node1->get_depth() >= node2->get_depth());
            //after this while loop, node 2 will always be deeper than node 1

            int node1_innovation_number = node1->get_innovation_number();
            int node2_innovation_number = node2->get_innovation_number();

            //check to see if the edge already exists
            bool edge_exists = false;
            int all_edges_position = -1;
            for (uint32_t i = 0; i < all_edges.size(); i++) {
                if (all_edges[i]->connects(node1_innovation_number, node2_innovation_number)) {
                    edge_exists = true;
                    all_edges_position = i;
                    break;
                }
            }

            bool edge_exists_in_child = false;
            for (uint32_t i = 0; i < child->get_number_edges(); i++) {
                if (child->get_edge(i)->connects(node1_innovation_number, node2_innovation_number)) {
                    edge_exists_in_child = true;
                    break;
                }
            }

            if (edge_exists && !edge_exists_in_child) {
                //edge exists in another genome, copy from all_edges
                //we know the child has both endpoints because we grabbed node1 and node2 from the child
                cout << "\t\tcopying edge in position " << all_edges_position << " from all_edges!" << endl;
                CNN_Edge *edge_copy = all_edges[all_edges_position]->copy();

                //enable the edge in case it was disabled
                edge_copy->enable();
                if (!edge_copy->set_nodes(child->get_nodes())) {
                    cout << "\t\treinitializing weights of copy" << endl;
                    edge_copy->reinitialize(generator);
                    edge_copy->save_best_weights();
                }

                child->add_edge( edge_copy );

                modifications++;
            } else if (!edge_exists && !edge_exists_in_child) {
                //edge does not exist at all
                cout << "\t\tadding edge between node innovation numbers " << node1_innovation_number << " and " << node2_innovation_number << endl;

                CNN_Edge *edge = new CNN_Edge(node1, node2, false, edge_innovation_count);
                edge_innovation_count++;
                //insert edge in order of depth

                //enable the edge in case it was disabled
                edge->enable();
                edge->initialize_weights(generator);
                edge->save_best_weights();
                child->add_edge(edge);

                CNN_Edge *edge_copy = edge->copy();
                edge_copy->set_nodes(all_nodes);
                all_edges.insert( upper_bound(all_edges.begin(), all_edges.end(), edge_copy, sort_CNN_Edges_by_depth()), edge_copy);

                modifications++;
            } else {
                cout << "\t\tnot adding edge between node innovation numbers " << node1_innovation_number << " and " << node2_innovation_number << " because edge already exists!" << endl;
            }

            continue;
        }
        r -= edge_add;

        if (r < edge_change_stride) {
            cout << "\tCHANGING EDGE STRIDE!" << endl;

            //child->mutate(MUTATE_EDGE_STRIDE, node_innovation_count, edge_innovation_count);

            continue;
        }
        r -= edge_change_stride;

        if (r < node_change_size) {
            cout << "\tCHANGING NODE SIZE X and Y!" << endl;

            if (child->get_number_softmax_nodes() + 1 == child->get_number_nodes()) {
                cout << "\t\tno non-input or softmax nodes so cannot change node size" << endl;
                continue;
            }

            //should have a value between -2 and 2 (inclusive)
            int change = (2 * rng_double(generator)) + 1;
            if (rng_double(generator) < 0.5) change *= -1;

            //make sure we don't change the size of the input node
            cout << "\t\tnumber nodes: " << child->get_number_nodes() << endl;
            cout << "\t\tnumber softmax nodes: " << child->get_number_softmax_nodes() << endl;
            int r = (rng_double(generator) * (child->get_number_nodes() - 1 - child->get_number_softmax_nodes())) + 1;
            cout << "\t\tr: " << r << endl;

            CNN_Node *modified_node = child->get_node(r);
            cout << "\t\tselected node: " << r << " with innovation number: " << modified_node->get_innovation_number() << endl;

            if (modified_node->is_input()) {
                cout << "\t\tmodified node was input, this should never happen!" << endl;
                exit(1);
            }

            if (modified_node->is_softmax()) {
                cout << "\t\tmodified node was softmax, this should never happen!" << endl;
                exit(1);
            }

            int previous_size_x = modified_node->get_size_x();
            int previous_size_y = modified_node->get_size_y();
            cout << "\t\tsize x before resize: " << previous_size_x << " modifying by change: " << change << endl;
            cout << "\t\tsize y before resize: " << previous_size_y << " modifying by change: " << change << endl;

            bool modified_x = modified_node->modify_size_x(change, generator);
            bool modified_y = modified_node->modify_size_y(change, generator);

            if (modified_x || modified_y) {
                //need to make sure all edges with this as it's input or output get updated
                cout << "\t\tresizing edges around node: " << modified_node->get_innovation_number() << endl;

                child->resize_edges_around_node( modified_node->get_innovation_number() );
                modifications++;

                cout << "\t\tmodified size x by " << change << " from " << previous_size_x << " to " << modified_node->get_size_x() << endl;
                cout << "\t\tmodified size y by " << change << " from " << previous_size_y << " to " << modified_node->get_size_y() << endl;
            } else {
                cout << "\t\tmodification resulted in no change" << endl;
            }

            continue;
        }
        r -= node_change_size;

        if (r < node_change_size_x) {
            cout << "\tCHANGING NODE SIZE X!" << endl;

            if (child->get_number_softmax_nodes() + 1 == child->get_number_nodes()) {
                cout << "\t\tno non-input or softmax nodes so cannot change node size" << endl;
                continue;
            }

            //should have a value between -2 and 2 (inclusive)
            int change = (2 * rng_double(generator)) + 1;
            if (rng_double(generator) < 0.5) change *= -1;

            //make sure we don't change the size of the input node
            int r = (rng_double(generator) * (child->get_number_nodes() - 1 - child->get_number_softmax_nodes())) + 1;

            CNN_Node *modified_node = child->get_node(r);
            cout << "\t\tselected node: " << r << " with innovation number: " << modified_node->get_innovation_number() << endl;

            if (modified_node->is_input()) {
                cout << "\t\tmodified node was input, this should never happen!" << endl;
                exit(1);
            }

            if (modified_node->is_softmax()) {
                cout << "\t\tmodified node was softmax, this should never happen!" << endl;
                exit(1);
            }


            int previous_size_x = modified_node->get_size_x();
            cout << "\t\tsize x before resize: " << previous_size_x << " modifying by change: " << change << endl;

            if (modified_node->modify_size_x(change, generator)) {
                //need to make sure all edges with this as it's input or output get updated
                cout << "\t\tresizing edges around node: " << modified_node->get_innovation_number() << endl;

                child->resize_edges_around_node( modified_node->get_innovation_number() );
                modifications++;

                cout << "\t\tmodified size x by " << change << " from " << previous_size_x << " to " << modified_node->get_size_x() << endl;
            } else {
                cout << "\t\tmodification resulted in no change" << endl;
            }

            continue;
        }
        r -= node_change_size_x;

        if (r < node_change_size_y) {
            cout << "\tCHANGING NODE SIZE Y!" << endl;

            if (child->get_number_softmax_nodes() + 1 == child->get_number_nodes()) {
                cout << "\t\tno non-input or softmax nodes so cannot change node size" << endl;
                continue;
            }

            //should have a value between -2 and 2 (inclusive)
            int change = (2 * rng_double(generator)) + 1;
            if (rng_double(generator) < 0.5) change *= -1;

            //make sure we don't change the size of the input node
            int r = (rng_double(generator) * (child->get_number_nodes() - 1 - child->get_number_softmax_nodes())) + 1;

            CNN_Node *modified_node = child->get_node(r);
            cout << "\t\tselected node: " << r << " with innovation number: " << modified_node->get_innovation_number() << endl;

            if (modified_node->is_input()) {
                cout << "\t\tmodified node was input, this should never happen!" << endl;
                exit(1);
            }

            if (modified_node->is_softmax()) {
                cout << "\t\tmodified node was softmax, this should never happen!" << endl;
                exit(1);
            }


            int previous_size_y = modified_node->get_size_y();
            cout << "\t\tsize y before resize: " << previous_size_y << " modifying by change: " << change << endl;

            if (modified_node->modify_size_y(change, generator)) {
                //need to make sure all edges with this as it's input or output get updated
                cout << "\t\tresizing edges around node: " << modified_node->get_innovation_number() << endl;

                child->resize_edges_around_node( modified_node->get_innovation_number() );
                modifications++;

                cout << "\t\tmodified size y by " << change << " from " << previous_size_y << " to " << modified_node->get_size_y() << endl;
            } else {
                cout << "\t\tmodification resulted in no change" << endl;
            }

            continue;
        }
        r -= node_change_size_y;

        if (r < node_change_pool_size) {
            cout << "\tCHANGING NODE POOL SIZE!" << endl;

            //child->mutate(MUTATE_NODE_POOL_SIZE, node_innovation_count, edge_innovation_count);

            continue;
        }
        r -= node_change_pool_size;

        cerr << "ERROR: problem choosing mutation type -- should never get here!" << endl;
        cerr << "\tremaining random value (for mutation selection): " << r << endl;
        exit(1);
    }

    return child;
}

CNN_Genome* EXACT::create_child() {
    return NULL;
}

void EXACT::print_statistics(ostream &out) {
    double min_fitness = numeric_limits<double>::max();
    double max_fitness = -numeric_limits<double>::max();
    double avg_fitness = 0.0;
    
    for (uint32_t i = 0; i < genomes.size(); i++) {
        double fitness = genomes[i]->get_fitness();

        avg_fitness += fitness;

        if (fitness < min_fitness) min_fitness = fitness;
        if (fitness > max_fitness) max_fitness = fitness;
    }
    avg_fitness /= genomes.size();

    if (genomes.size() == 0) avg_fitness = 0.0;

    out << parse_fitness(min_fitness) << " " << parse_fitness(avg_fitness) << " " << parse_fitness(max_fitness) << endl;
}
