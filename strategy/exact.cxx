#include <algorithm>
using std::sort;
using std::upper_bound;

#include <chrono>

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
using std::minstd_rand0;
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

#ifdef _MYSQL_
#include "common/db_conn.hxx"
#endif

#include "stdlib.h"

#ifdef _MYSQL_

bool EXACT::exists_in_database(int exact_id) {
    ostringstream query;

    query << "SELECT * FROM exact_search WHERE id = " << exact_id;

    mysql_exact_query(query.str());
    
    MYSQL_RES *result = mysql_store_result(exact_db_conn);
    if (result != NULL) {
        return false;
    } else {
        return true;
    }
}

EXACT::EXACT(int exact_id) {
    ostringstream query;

    query << "SELECT * FROM exact_search WHERE id = " << exact_id;

    mysql_exact_query(query.str());
    
    MYSQL_RES *result = mysql_store_result(exact_db_conn);
    if (result != NULL) {
        MYSQL_ROW row = mysql_fetch_row(result);

        id = exact_id;  //is row 0
        search_name = string(row[1]);
        output_directory = string(row[2]);

        number_images = atoi(row[3]);
        image_rows = atoi(row[4]);
        image_cols = atoi(row[5]);
        number_classes = atoi(row[6]);

        population_size = atoi(row[7]);
        node_innovation_count = atoi(row[8]);
        edge_innovation_count = atoi(row[9]);

        genomes_generated = atoi(row[10]);
        inserted_genomes = atoi(row[11]);

        reset_edges = atoi(row[12]);
        min_epochs = atoi(row[13]);
        max_epochs = atoi(row[14]);
        improvement_required_epochs = atoi(row[15]);
        max_individuals = atoi(row[16]);

        learning_rate = atof(row[17]);
        weight_decay = atof(row[18]);

        crossover_rate = atof(row[19]);
        more_fit_parent_crossover = atof(row[20]);
        less_fit_parent_crossover = atof(row[21]);

        number_mutations = atoi(row[22]);
        edge_disable = atof(row[23]);
        edge_enable = atof(row[24]);
        edge_split = atof(row[25]);
        edge_add = atof(row[26]);
        edge_change_stride = atof(row[27]);
        node_change_size = atof(row[28]);
        node_change_size_x = atof(row[29]);
        node_change_size_y = atof(row[30]);
        node_change_pool_size = atof(row[31]);

        inserted_from_disable_edge = atoi(row[32]);
        inserted_from_enable_edge = atoi(row[33]);
        inserted_from_split_edge = atoi(row[34]);
        inserted_from_add_edge = atoi(row[35]);
        inserted_from_change_size = atoi(row[36]);
        inserted_from_change_size_x = atoi(row[37]);
        inserted_from_change_size_y = atoi(row[38]);
        inserted_from_crossover = atoi(row[39]);

        istringstream generator_iss(row[40]);
        generator_iss >> generator;
        //cout << "read generator from database: " << generator << endl;

        istringstream normal_distribution_iss(row[41]);
        normal_distribution_iss >> normal_distribution;
        //cout << "read normal_distribution from database: " << normal_distribution << endl;

        istringstream rng_long_iss(row[42]);
        rng_long_iss >> rng_long;
        //cout << "read rng_long from database: " << rng_long << endl;

        istringstream rng_double_iss(row[43]);
        rng_double_iss >> rng_double;
        //cout << "read rng_double from database: " << rng_double << endl;

        ostringstream genome_query;
        genome_query << "SELECT id FROM cnn_genome WHERE exact_id = " << id << " ORDER BY best_error LIMIT " << population_size;
        cout << genome_query.str() << endl;

        mysql_exact_query(genome_query.str());

        MYSQL_RES *genome_result = mysql_store_result(exact_db_conn);

        //cout << "got genome result" << endl;

        MYSQL_ROW genome_row;
        while ((genome_row = mysql_fetch_row(genome_result)) != NULL) {
            int genome_id = atoi(genome_row[0]);
            //cout << "got genome with id: " << genome_id << endl;

            CNN_Genome *genome = new CNN_Genome(genome_id);
            genomes.push_back(genome);
        }

        //cout << "got all genomes!" << endl;

        ostringstream node_query;
        node_query << "SELECT id FROM cnn_node WHERE exact_id = " << id << " AND genome_id = 0";
        //cout << node_query.str() << endl;

        mysql_exact_query(node_query.str());

        MYSQL_RES *node_result = mysql_store_result(exact_db_conn);

        //cout << "got node result!" << endl;

        MYSQL_ROW node_row;
        while ((node_row = mysql_fetch_row(node_result)) != NULL) {
            int node_id = atoi(node_row[0]);
            //cout << "got node with id: " << node_id << endl;

            CNN_Node *node = new CNN_Node(node_id);
            all_nodes.push_back(node);
        }

        //cout << "got all nodes!" << endl;
        mysql_free_result(node_result);

        ostringstream edge_query;
        edge_query << "SELECT id FROM cnn_edge WHERE exact_id = " << id << " AND genome_id = 0";
        //cout << edge_query.str() << endl;

        mysql_exact_query(edge_query.str());
        //cout << "edge query was successful!" << endl;

        MYSQL_RES *edge_result = mysql_store_result(exact_db_conn);
        //cout << "got result!" << endl;

        MYSQL_ROW edge_row;
        while ((edge_row = mysql_fetch_row(edge_result)) != NULL) {
            int edge_id = atoi(edge_row[0]);
            //cout << "got edge with id: " << edge_id << endl;

            CNN_Edge *edge = new CNN_Edge(edge_id);
            all_edges.push_back(edge);

            edge->set_nodes(all_nodes);
        }

        //cout << "got all edges!" << endl;
        mysql_free_result(edge_result);

        mysql_free_result(result);
    } else {
        cerr << "ERROR! could not find exact_search in database with id: " << exact_id << endl;
        exit(1);
    }
}

void EXACT::export_to_database() {
    ostringstream query;
    if (id >= 0) {
        query << "REPLACE INTO exact_search SET id = " << id << ",";
    } else {
        query << "INSERT INTO exact_search SET";
    }

    query << " search_name = '" << search_name << "'"
        << ", output_directory = '" << output_directory << "'"
        << ", number_images = " << number_images
        << ", image_rows = " << image_rows
        << ", image_cols = " << image_cols
        << ", number_classes = " << number_classes

        << ", population_size = " << population_size
        << ", node_innovation_count = " << node_innovation_count
        << ", edge_innovation_count = " << edge_innovation_count

        << ", genomes_generated = " << genomes_generated
        << ", inserted_genomes = " << inserted_genomes

        << ", reset_edges = " << reset_edges
        << ", min_epochs = " << min_epochs
        << ", max_epochs = " << max_epochs
        << ", improvement_required_epochs = " << improvement_required_epochs
        << ", max_individuals = " << max_individuals

        << ", learning_rate = " << learning_rate
        << ", weight_decay = " << weight_decay

        << ", crossover_rate = " << crossover_rate
        << ", more_fit_parent_crossover = " << more_fit_parent_crossover
        << ", less_fit_parent_crossover = " << more_fit_parent_crossover

        << ", number_mutations = " << number_mutations
        << ", edge_disable = " << edge_disable
        << ", edge_enable = " << edge_enable
        << ", edge_split = " << edge_split
        << ", edge_add = " << edge_add
        << ", edge_change_stride = " << edge_change_stride
        << ", node_change_size = " << node_change_size
        << ", node_change_size_x = " << node_change_size_x
        << ", node_change_size_y = " << node_change_size_y
        << ", node_change_pool_size = " << node_change_pool_size

        << ", inserted_from_disable_edge = " << inserted_from_disable_edge
        << ", inserted_from_enable_edge = " << inserted_from_enable_edge
        << ", inserted_from_split_edge = " << inserted_from_split_edge
        << ", inserted_from_add_edge = " << inserted_from_add_edge
        << ", inserted_from_change_size = " << inserted_from_change_size
        << ", inserted_from_change_size_x = " << inserted_from_change_size_x
        << ", inserted_from_change_size_y = " << inserted_from_change_size_y
        << ", inserted_from_crossover = " << inserted_from_crossover

        << ", generator = '" << generator << "'"
        << ", normal_distribution = '" << normal_distribution << "'"
        << ", rng_long = '" << rng_long << "'"
        << ", rng_double = '" << rng_double << "'";

    mysql_exact_query(query.str());

    if (id < 0) {
        id = mysql_exact_last_insert_id();
        cout << "inserted EXACT search with id: " << id << endl;
    }

    //need to insert genomes
    for (uint32_t i = 0; i < genomes.size(); i++) {
        genomes[i]->export_to_database(id);
    }

    //need to insert all_nodes and all_edges
    //a genome id of 0 means that they are not assigned to
    //a particular genome
    for (uint32_t i = 0; i < all_nodes.size(); i++) {
        all_nodes[i]->export_to_database(id, 0);
    }

    for (uint32_t i = 0; i < all_edges.size(); i++) {
        all_edges[i]->export_to_database(id, 0);
    }

    if ((int32_t)genomes.size() == population_size) {
        double worst_fitness = genomes.back()->get_fitness();
        ostringstream delete_query;
        delete_query << "DELETE FROM cnn_genome WHERE exact_id = " << id << " AND best_error > " << worst_fitness;
        cout << delete_query.str() << endl;
        mysql_exact_query(delete_query.str());

        ostringstream delete_node_query;
        delete_node_query << "DELETE FROM cnn_node WHERE exact_id = " << id << " AND cnn_node.id > 0 AND NOT EXISTS(SELECT id FROM cnn_genome WHERE cnn_genome.id = cnn_node.genome_id)";
        cout <<  delete_node_query.str() << endl;
        mysql_exact_query(delete_node_query.str());

        ostringstream delete_edge_query;
        delete_edge_query << "DELETE FROM cnn_edge WHERE exact_id = " << id << " AND cnn_edge.id > 0 AND NOT EXISTS(SELECT id FROM cnn_genome WHERE cnn_genome.id = cnn_edge.genome_id)";
        cout <<  delete_edge_query.str() << endl;
        mysql_exact_query(delete_edge_query.str());
    }
}

void EXACT::update_database() {
    if (id < 0) {
        cerr << "ERROR: Cannot update an exact search in the databse if it has not already been entered, id was < 0." << endl;
        return;
    }

    ostringstream query;
    query << "UPDATE exact_search SET";

    query << " genomes_generated = " << genomes_generated
        << ", inserted_genomes = " << inserted_genomes

        << ", inserted_from_disable_edge = " << inserted_from_disable_edge
        << ", inserted_from_enable_edge = " << inserted_from_enable_edge
        << ", inserted_from_split_edge = " << inserted_from_split_edge
        << ", inserted_from_add_edge = " << inserted_from_add_edge
        << ", inserted_from_change_size = " << inserted_from_change_size
        << ", inserted_from_change_size_x = " << inserted_from_change_size_x
        << ", inserted_from_change_size_y = " << inserted_from_change_size_y
        << ", inserted_from_crossover = " << inserted_from_crossover

        << ", generator = '" << generator << "'"
        << ", normal_distribution = '" << normal_distribution << "'"
        << ", rng_long = '" << rng_long << "'"
        << ", rng_double = '" << rng_double << "'";

    mysql_exact_query(query.str());

    //genomes are inserted separately

    //need to insert all_nodes and all_edges
    //a genome id of 0 means that they are not assigned to
    //a particular genome
    for (uint32_t i = 0; i < all_nodes.size(); i++) {
        all_nodes[i]->export_to_database(id, 0);
    }

    for (uint32_t i = 0; i < all_edges.size(); i++) {
        all_edges[i]->export_to_database(id, 0);
    }


    if ((int32_t)genomes.size() == population_size) {
        double worst_fitness = genomes.back()->get_fitness();
        ostringstream delete_query;
        delete_query << "DELETE FROM cnn_genome WHERE exact_id = " << id << " AND best_error > " << worst_fitness;
        cout << delete_query.str() << endl;
        mysql_exact_query(delete_query.str());

        ostringstream delete_node_query;
        delete_node_query << "DELETE FROM cnn_node WHERE exact_id = " << id << " AND cnn_node.id > 0 AND NOT EXISTS(SELECT id FROM cnn_genome WHERE cnn_genome.id = cnn_node.genome_id)";
        cout <<  delete_node_query.str() << endl;
        mysql_exact_query(delete_node_query.str());

        ostringstream delete_edge_query;
        delete_edge_query << "DELETE FROM cnn_edge WHERE exact_id = " << id << " AND cnn_edge.id > 0 AND NOT EXISTS(SELECT id FROM cnn_genome WHERE cnn_genome.id = cnn_edge.genome_id)";
        cout <<  delete_edge_query.str() << endl;
        mysql_exact_query(delete_edge_query.str());
    }
}

#endif

EXACT::EXACT(const Images &images, int _population_size, int _min_epochs, int _max_epochs, int _improvement_required_epochs, bool _reset_edges, int _max_individuals, string _output_directory, string _search_name) {

    id = -1;

    search_name = _search_name;

    output_directory = _output_directory;
    reset_edges = _reset_edges;
    min_epochs = _min_epochs;
    max_epochs = _max_epochs;
    improvement_required_epochs = _improvement_required_epochs;
    max_individuals = _max_individuals;

    unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
    //unsigned seed = 10;

    generator = minstd_rand0(seed);
    rng_long = uniform_int_distribution<long>(-numeric_limits<long>::max(), numeric_limits<long>::max());
    rng_double = uniform_real_distribution<double>(0, 1.0);

    node_innovation_count = 0;
    edge_innovation_count = 0;

    inserted_genomes = 0;

    population_size = _population_size;

    number_images = images.get_number_images();
    image_rows = images.get_image_rows();
    image_cols = images.get_image_cols();
    number_classes = images.get_number_classes();

    inserted_from_disable_edge = 0;
    inserted_from_enable_edge = 0;
    inserted_from_split_edge = 0;
    inserted_from_add_edge = 0;
    inserted_from_change_size = 0;
    inserted_from_change_size_x = 0;
    inserted_from_change_size_y = 0;
    inserted_from_crossover = 0;

    genomes_generated = 0;

    learning_rate = 0.001;
    weight_decay = 0.001;

    crossover_rate = 0.20;
    more_fit_parent_crossover = 0.80;
    less_fit_parent_crossover = 0.50;

    number_mutations = 4;
    edge_disable = 1.0;
    edge_enable = 2.0;
    edge_split = 2.0;
    edge_add = 4.0;
    edge_change_stride = 0.0;
    node_change_size = 1.0;
    node_change_size_x = 0.5;
    node_change_size_y = 0.5;
    node_change_pool_size = 0.0;

    cout << "EXACT settings: " << endl;
    cout << "\tlearning_rate: " << learning_rate << endl;
    cout << "\tweight_decay: " << weight_decay << endl;
    cout << "\tmin_epochs: " << min_epochs << endl;
    cout << "\tmax_epochs: " << max_epochs << endl;
    cout << "\timprovement_required_epochs: " << improvement_required_epochs << endl;

    cout << "\tcrossover_settings: " << endl;
    cout << "\t\tcrossover_rate: " << crossover_rate << endl;
    cout << "\t\tmore_fit_parent_crossover: " << more_fit_parent_crossover << endl;
    cout << "\t\tless_fit_parent_crossover: " << less_fit_parent_crossover << endl;

    cout << "\tmutation_settings: " << endl;
    cout << "\t\tnumber_mutations: " << number_mutations << endl;
    cout << "\t\tedge_disable: " << edge_disable << endl;
    cout << "\t\tedge_split: " << edge_split << endl;
    cout << "\t\tedge_add: " << edge_add << endl;
    cout << "\t\tedge_change_stride: " << edge_change_stride << endl;
    cout << "\t\tnode_change_size: " << node_change_size << endl;
    cout << "\t\tnode_change_size_x: " << node_change_size_x << endl;
    cout << "\t\tnode_change_size_y: " << node_change_size_y << endl;
    cout << "\t\tnode_change_pool_size: " << node_change_pool_size << endl;

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

int EXACT::get_id() const {
    return id;
}

string EXACT::get_search_name() const {
    return search_name;
}

string EXACT::get_output_directory() const {
    return output_directory;
}

int EXACT::get_number_images() const {
    return number_images;
}

CNN_Genome* EXACT::get_best_genome() {
    return genomes[0];
}

CNN_Genome* EXACT::generate_individual() {
    if (inserted_genomes >= max_individuals) return NULL;

    CNN_Genome *genome = NULL;
    if (genomes.size() == 0) {
        //generate the initial minimal CNN
        int total_weights = 0;

        CNN_Node *input_node = new CNN_Node(node_innovation_count, 0, image_rows, image_cols, INPUT_NODE, generator, normal_distribution);
        node_innovation_count++;
        all_nodes.push_back(input_node);

        for (int32_t i = 0; i < number_classes; i++) {
            CNN_Node *softmax_node = new CNN_Node(node_innovation_count, 1, 1, 1, SOFTMAX_NODE, generator, normal_distribution);
            node_innovation_count++;
            all_nodes.push_back(softmax_node);
        }

        for (int32_t i = 0; i < number_classes; i++) {
            CNN_Edge *edge = new CNN_Edge(input_node, all_nodes[i + 1] /*ith softmax node*/, true, edge_innovation_count, generator, normal_distribution);

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

    } else if ((int32_t)genomes.size() < population_size) {
        //generate random mutatinos until genomes.size() < population_size
        while (genome == NULL) {
            genome = create_mutation();

            if (!genome->outputs_connected()) {
                cerr << "\tAll softmax nodes were not reachable, deleting genome." << endl;
                delete genome;
                genome = NULL;
            } else if (population_contains(genome)) {
                cerr << "\tPopulation already contained genome, deleting genome." << endl;
                delete genome;
                genome = NULL;
            }
         }
    } else {
        if (rng_double(generator) < crossover_rate) {
            //generate a child from crossover
            while (genome == NULL) {
                genome = create_child();

                if (!genome->outputs_connected()) {
                    cerr << "\tAll softmax nodes were not reachable, deleting genome." << endl;
                    delete genome;
                    genome = NULL;
                } else if (population_contains(genome)) {
                    cerr << "\tPopulation already contained genome, deleting genome." << endl;
                    delete genome;
                    genome = NULL;
                }
            }

        } else {
            //generate a mutation
            while (genome == NULL) {
                genome = create_mutation();

                if (!genome->outputs_connected()) {
                    cerr << "\tAll softmax nodes were not reachable, deleting genome." << endl;
                    delete genome;
                    genome = NULL;
                } else if (population_contains(genome)) {
                    cerr << "\tPopulation already contained genome, deleting genome." << endl;
                    delete genome;
                    genome = NULL;
                }
            }
        }
    }

    if (!genome->sanity_check(SANITY_CHECK_AFTER_GENERATION)) {
        cerr << "ERROR: genome " << genome->get_generation_id() << " failed sanity check in generate individual!" << endl;
        exit(1);
    }

    if ((int32_t)genomes.size() < population_size) {
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

bool EXACT::population_contains(CNN_Genome *genome) const {
    for (int32_t i = 0; i < (int32_t)genomes.size(); i++) {
        //we can overwrite genomes that were inserted in the initialization phase
        //and not evaluated
        if (genomes[i]->get_fitness() == numeric_limits<double>::max()
                || genomes[i]->get_fitness() == 10000000) continue;

        if (genomes[i]->equals(genome)) {
            cout << "\tgenome was the same as genome with generation id: " << genomes[i]->get_generation_id() << endl;
            return true;
        }
    }
    return false;
}

string parse_fitness(double fitness) {
    if (fitness == numeric_limits<double>::max()) {
        return "UNEVALUATED";
    } else {
        return to_string(fitness);
    }
}

bool EXACT::insert_genome(CNN_Genome* genome) {
    bool was_inserted = true;

    inserted_genomes++;

    cout << "genomes evaluated: " << setw(10) << inserted_genomes << ", inserting: " << parse_fitness(genome->get_fitness()) << endl;

    if (population_contains(genome)) {
        cerr << "\tpopulation already contains genome! not inserting." << endl;
        delete genome;
        return false;
    }

    if (!genome->sanity_check(SANITY_CHECK_BEFORE_INSERT)) {
        cerr << "ERROR: genome " << genome->get_generation_id() << " failed sanity check before insert!" << endl;
        exit(1);
    }
    cout << "genome " << genome->get_generation_id() << " passed sanity check with fitness: " << parse_fitness(genome->get_fitness()) << endl;

    if (genomes.size() == 0 || genome->get_fitness() < genomes[0]->get_fitness()) {
        cout << "new best fitness!" << endl;

        cout << "writing new best (data) to: " << (output_directory + "/global_best_" + to_string(inserted_genomes) + ".txt") << endl;

        genome->write_to_file(output_directory + "/global_best_" + to_string(inserted_genomes) + ".txt");

        cout << "writing new best (graphviz) to: " << (output_directory + "/global_best_" + to_string(inserted_genomes) + ".txt") << endl;

        ofstream gv_file(output_directory + "/global_best_" + to_string(inserted_genomes) + ".gv");
        gv_file << "#EXACT settings: " << endl;

        gv_file << "#EXACT settings: " << endl;
        gv_file << "#\tlearning_rate: " << learning_rate << endl;
        gv_file << "#\tweight_decay: " << weight_decay << endl;
        gv_file << "#\tmin_epochs: " << min_epochs << endl;
        gv_file << "#\tmax_epochs: " << max_epochs << endl;
        gv_file << "#\timprovement_required_epochs: " << improvement_required_epochs << endl;

        gv_file << "#\tcrossover_settings: " << endl;
        gv_file << "#\t\tcrossover_rate: " << crossover_rate << endl;
        gv_file << "#\t\tmore_fit_parent_crossover: " << more_fit_parent_crossover << endl;
        gv_file << "#\t\tless_fit_parent_crossover: " << less_fit_parent_crossover << endl;

        gv_file << "#\tmutation_settings: " << endl;
        gv_file << "#\t\tnumber_mutations: " << number_mutations << endl;
        gv_file << "#\t\tedge_disable: " << edge_disable << endl;
        gv_file << "#\t\tedge_split: " << edge_split << endl;
        gv_file << "#\t\tedge_add: " << edge_add << endl;
        gv_file << "#\t\tedge_change_stride: " << edge_change_stride << endl;
        gv_file << "#\t\tnode_change_size: " << node_change_size << endl;
        gv_file << "#\t\tnode_change_size_x: " << node_change_size_x << endl;
        gv_file << "#\t\tnode_change_size_y: " << node_change_size_y << endl;
        gv_file << "#\t\tnode_change_pool_size: " << node_change_pool_size << endl;

        genome->print_graphviz(gv_file);
        gv_file.close();
    }
    cout << endl;


    if ((int32_t)genomes.size() >= population_size && genomes.size() > 0 && genome->get_fitness() >= genomes.back()->get_fitness()) {
        //this will not be inserted into the population
        cout << "not inserting genome due to poor fitness" << endl;
        was_inserted = false;
        delete genome;
    } else {
        cout << "updating search statistics" << endl;

        inserted_from_disable_edge += genome->get_generated_by_disable_edge();
        inserted_from_enable_edge += genome->get_generated_by_enable_edge();
        inserted_from_split_edge += genome->get_generated_by_split_edge();
        inserted_from_add_edge += genome->get_generated_by_add_edge();
        inserted_from_change_size += genome->get_generated_by_change_size();
        inserted_from_change_size_x += genome->get_generated_by_change_size_x();
        inserted_from_change_size_y += genome->get_generated_by_change_size_y();
        inserted_from_crossover += genome->get_generated_by_crossover();

        cout << "updated search statistics" << endl;

        cout << "inserting new genome" << endl;
        //inorder insert the new individual
        genomes.insert( upper_bound(genomes.begin(), genomes.end(), genome, sort_genomes_by_fitness()), genome);

        cout << "inserted the new genome" << endl;

        //delete the worst individual if we've reached the population size
        if ((int32_t)genomes.size() > population_size) {
            cout << "deleting worst genome" << endl;
            CNN_Genome *worst = genomes.back();
            genomes.pop_back();
            delete worst;
        }
    }

    cout << "genome fitnesses:" << endl;
    for (int32_t i = 0; i < (int32_t)genomes.size(); i++) {
        cout << "\t" << setw(4) << i << " -- genome: " << setw(10) << genomes[i]->get_generation_id() << ", "
            << setw(20) << left << "fitness: " << right << setw(15) << setprecision(5) << fixed << parse_fitness(genomes[i]->get_fitness())
            << " (" << genomes[i]->get_best_predictions() << " correct) on epoch: " << genomes[i]->get_best_error_epoch() 
            << ", number enabled edges: " << genomes[i]->get_number_enabled_edges()
            << ", number nodes: " << genomes[i]->get_number_nodes() << endl;
    }
    cout << "genome best error: " << endl;
    for (int32_t i = 0; i < (int32_t)genomes.size(); i++) {
        cout << "\t" << setw(4) << i << " -- genome: " << setw(10) << genomes[i]->get_generation_id() << ", ";
        genomes[i]->print_best_error(cout);
    }

    cout << "genome correct predictions: " << endl;
    for (int32_t i = 0; i < (int32_t)genomes.size(); i++) {
        cout << "\t" << setw(4) << i << " -- genome: " << setw(10) << genomes[i]->get_generation_id() << ", ";
        genomes[i]->print_best_predictions(cout);
    }

    cout << endl;

    return was_inserted;
}

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

    cout << "\tchild nodes:" << endl;
    for (int32_t i = 0; i < child->get_number_nodes(); i++) {
        cout << "\t\tnode innovation number: " << child->get_node(i)->get_innovation_number() << endl;
    }

    cout << "\tchild edges:" << endl;
    for (int32_t i = 0; i < child->get_number_edges(); i++) {
        cout << "\t\tedge innovation number: " << child->get_edge(i)->get_innovation_number()
            << ", input node innovation number: " << child->get_edge(i)->get_input_innovation_number()
            << ", output node innovation number: " << child->get_edge(i)->get_output_innovation_number()
            << endl;
    }

    if (parent->get_fitness() == numeric_limits<double>::max()) {
        //This parent has not actually been evaluated (the population is still initializing)
        //we can set the best_bias and best_weights randomly so that they are used when it
        //starts up

        cout << "\tparent had not been evaluated yet, but best_bias and best_weights should have been set randomly" << endl;

        /*
        for (int32_t i = 0; i < child->get_number_nodes(); i++) {
            child->get_node(i)->initialize_bias(generator);
            child->get_node(i)->save_best_bias();
        }

        for (int32_t i = 0; i < child->get_number_edges(); i++) {
            child->get_edge(i)->initialize_weights(generator);
            child->get_edge(i)->save_best_weights();
        }
        */
    } else {
        cout << "\tparent had been evaluated! not setting best_bias and best_weights randomly" << endl;
        cout << "\tparent fitness: " << parent->get_fitness() << endl;
    }

    int modifications = 0;

    while (modifications < number_mutations) {
        double r = rng_double(generator);

        cerr << "\tr: " << r << endl;

        if (r < edge_disable) {
            cout << "\tDISABLING EDGE!" << endl;

            int edge_position = rng_double(generator) * child->get_number_edges();
            if (child->disable_edge(edge_position)) {
                child->set_generated_by_disable_edge();
                modifications++;
            }

            continue;
        } 
        r -= edge_disable;

        if (r < edge_enable) {
            cout << "\tENABLING EDGE!" << endl;

            vector< CNN_Edge* > disabled_edges;

            for (int32_t i = 0; i < child->get_number_edges(); i++) {
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
                disabled_edge->initialize_weights(generator, normal_distribution);
                disabled_edge->save_best_weights(); //save the random weights so they are reused
                child->set_generated_by_enable_edge();
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

            CNN_Node *child_node = new CNN_Node(node_innovation_count, depth, size_x, size_y, HIDDEN_NODE, generator, normal_distribution);
            node_innovation_count++;

            //add two new edges, disable the split edge
            cout << "\t\tcreating edge " << edge_innovation_count << endl;
            CNN_Edge *edge1 = new CNN_Edge(input_node, child_node, false, edge_innovation_count, generator, normal_distribution);
            edge_innovation_count++;

            cout << "\t\tcreating edge " << edge_innovation_count << endl;
            CNN_Edge *edge2 = new CNN_Edge(child_node, output_node, false, edge_innovation_count, generator, normal_distribution);
            edge_innovation_count++;

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

            child->set_generated_by_split_edge();
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
            for (int32_t i = 0; i < (int32_t)all_edges.size(); i++) {
                if (all_edges[i]->connects(node1_innovation_number, node2_innovation_number)) {
                    edge_exists = true;
                    all_edges_position = i;
                    break;
                }
            }

            bool edge_exists_in_child = false;
            for (int32_t i = 0; i < child->get_number_edges(); i++) {
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
                    edge_copy->reinitialize(generator, normal_distribution);
                    edge_copy->save_best_weights();
                }

                child->add_edge( edge_copy );

                child->set_generated_by_add_edge();
                modifications++;
            } else if (!edge_exists && !edge_exists_in_child) {
                //edge does not exist at all
                cout << "\t\tadding edge between node innovation numbers " << node1_innovation_number << " and " << node2_innovation_number << endl;

                CNN_Edge *edge = new CNN_Edge(node1, node2, false, edge_innovation_count, generator, normal_distribution);
                edge_innovation_count++;
                //insert edge in order of depth

                //enable the edge in case it was disabled
                edge->enable();
                child->add_edge(edge);

                CNN_Edge *edge_copy = edge->copy();
                edge_copy->set_nodes(all_nodes);
                all_edges.insert( upper_bound(all_edges.begin(), all_edges.end(), edge_copy, sort_CNN_Edges_by_depth()), edge_copy);

                child->set_generated_by_add_edge();
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

            bool modified_x = modified_node->modify_size_x(change, generator, normal_distribution);
            bool modified_y = modified_node->modify_size_y(change, generator, normal_distribution);

            if (modified_x || modified_y) {
                //need to make sure all edges with this as it's input or output get updated
                cout << "\t\tresizing edges around node: " << modified_node->get_innovation_number() << endl;

                child->resize_edges_around_node( modified_node->get_innovation_number() );
                child->set_generated_by_change_size();
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

            if (modified_node->modify_size_x(change, generator, normal_distribution)) {
                //need to make sure all edges with this as it's input or output get updated
                cout << "\t\tresizing edges around node: " << modified_node->get_innovation_number() << endl;

                child->resize_edges_around_node( modified_node->get_innovation_number() );
                child->set_generated_by_change_size_x();
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

            if (modified_node->modify_size_y(change, generator, normal_distribution)) {
                //need to make sure all edges with this as it's input or output get updated
                cout << "\t\tresizing edges around node: " << modified_node->get_innovation_number() << endl;

                child->resize_edges_around_node( modified_node->get_innovation_number() );
                child->set_generated_by_change_size_y();
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

void attempt_node_insert(vector<CNN_Node*> &nodes, CNN_Node *node) {
    for (int32_t i = 0; i < (int32_t)nodes.size(); i++) {
        if (nodes[i]->get_innovation_number() == node->get_innovation_number()) return;
    }

    nodes.insert( upper_bound(nodes.begin(), nodes.end(), node->copy(), sort_CNN_Nodes_by_depth()), node->copy());
}

bool edges_contains(vector< CNN_Edge* > &edges, CNN_Edge *edge) {
    for (int32_t i = 0; i < (int32_t)edges.size(); i++) {
        if (edges[i]->get_innovation_number() == edge->get_innovation_number()) return true;
    }
    return false;
}

CNN_Genome* EXACT::create_child() {
    cout << "\tCREATING CHILD THROUGH CROSSOVER!" << endl;
    int r1 = rng_double(generator) * genomes.size();
    int r2 = rng_double(generator) * (genomes.size() - 1);
    if (r1 == r2) r2++;

    //parent should have higher fitness
    if (r2 < r1) {
        int tmp = r2;
        r2 = r1;
        r1 = tmp;
    }
    cout << "\t\tparent positions: " << r1 << " and " << r2 << endl;

    CNN_Genome *parent1 = genomes[r1];
    CNN_Genome *parent2 = genomes[r2];

    cout << "\t\tgenerating child " << genomes_generated << " from parents: " << parent1->get_generation_id() << " and " << parent2->get_generation_id() << endl;

    vector< CNN_Node* > child_nodes;
    vector< CNN_Edge* > child_edges;

    int p1_position = 0;
    int p2_position = 0;

    //edges are not sorted in order of innovation number, they need to be
    vector< CNN_Edge* > p1_edges;
    for (int i = 0; i < parent1->get_number_edges(); i++) {
        p1_edges.push_back(parent1->get_edge(i));
    }

    vector< CNN_Edge* > p2_edges;
    for (int i = 0; i < parent2->get_number_edges(); i++) {
        p2_edges.push_back(parent2->get_edge(i));
    }

    sort(p1_edges.begin(), p1_edges.end(), sort_CNN_Edges_by_innovation());
    sort(p2_edges.begin(), p2_edges.end(), sort_CNN_Edges_by_innovation());

    cerr << "p1 innovation numbers AFTER SORT: " << endl;
    for (int32_t i = 0; i < (int32_t)p1_edges.size(); i++) {
        cerr << "\t" << p1_edges[i]->get_innovation_number() << endl;
    }
    cerr << "p2 innovation numbers AFTER SORT: " << endl;
    for (int32_t i = 0; i < (int32_t)p2_edges.size(); i++) {
        cerr << "\t" << p2_edges[i]->get_innovation_number() << endl;
    }


    while (p1_position < (int32_t)p1_edges.size() && p2_position < (int32_t)p2_edges.size()) {
        CNN_Edge* p1_edge = p1_edges[p1_position];
        CNN_Edge* p2_edge = p2_edges[p2_position];

        int p1_innovation = p1_edge->get_innovation_number();
        int p2_innovation = p2_edge->get_innovation_number();

        if (p1_innovation == p2_innovation) {
            CNN_Edge *edge = p1_edge->copy();

            if (edges_contains(child_edges, edge)) {
                cerr << "ERROR in crossover! trying to push an edge with innovation_number: " << edge->get_innovation_number() << " and it already exists in the vector!" << endl;
                cerr << "p1_position: " << p1_position << ", p1_size: " << p1_edges.size() << endl;
                cerr << "p2_position: " << p2_position << ", p2_size: " << p2_edges.size() << endl;
                cerr << "vector innovation numbers: " << endl;
                for (int32_t i = 0; i < (int32_t)child_edges.size(); i++) {
                    cerr << "\t" << child_edges[i]->get_innovation_number() << endl;
                }
            }

            child_edges.push_back(edge->copy());

            //push back surrounding nodes
            attempt_node_insert(child_nodes, p1_edge->get_input_node());
            attempt_node_insert(child_nodes, p1_edge->get_output_node());

            p1_position++;
            p2_position++;
        } else if (p1_innovation < p2_innovation) {
            CNN_Edge *edge = p1_edge->copy();

            if (edges_contains(child_edges, edge)) {
                cerr << "ERROR in crossover! trying to push an edge with innovation_number: " << edge->get_innovation_number() << " and it already exists in the vector!" << endl;
                cerr << "p1_position: " << p1_position << ", p1_size: " << p1_edges.size() << endl;
                cerr << "p2_position: " << p2_position << ", p2_size: " << p2_edges.size() << endl;
                cerr << "vector innovation numbers: " << endl;
                for (int32_t i = 0; i < (int32_t)child_edges.size(); i++) {
                    cerr << "\t" << child_edges[i]->get_innovation_number() << endl;
                }
            }

            child_edges.push_back(edge);

            if (rng_double(generator) >= more_fit_parent_crossover) {
                edge->disable();
            }

            //push back surrounding nodes
            attempt_node_insert(child_nodes, p1_edge->get_input_node());
            attempt_node_insert(child_nodes, p1_edge->get_output_node());

            p1_position++;
        } else {
            CNN_Edge *edge = p2_edge->copy();

            if (edges_contains(child_edges, edge)) {
                cerr << "ERROR in crossover! trying to push an edge with innovation_number: " << edge->get_innovation_number() << " and it already exists in the vector!" << endl;
                cerr << "p1_position: " << p1_position << ", p1_size: " << p1_edges.size() << endl;
                cerr << "p2_position: " << p2_position << ", p2_size: " << p2_edges.size() << endl;
                cerr << "vector innovation numbers: " << endl;
                for (int32_t i = 0; i < (int32_t)child_edges.size(); i++) {
                    cerr << "\t" << child_edges[i]->get_innovation_number() << endl;
                }
            }

            child_edges.push_back(edge);

            if (rng_double(generator) >= less_fit_parent_crossover) {
                edge->disable();
            }

            //push back surrounding nodes
            attempt_node_insert(child_nodes, p2_edge->get_input_node());
            attempt_node_insert(child_nodes, p2_edge->get_output_node());

            p2_position++;
        }
    }

    while (p1_position < (int32_t)p1_edges.size()) {
        CNN_Edge* p1_edge = p1_edges[p1_position];

        CNN_Edge *edge = p1_edge->copy();

        if (edges_contains(child_edges, edge)) {
            cerr << "ERROR in crossover! trying to push an edge with innovation_number: " << edge->get_innovation_number() << " and it already exists in the vector!" << endl;
            cerr << "p1_position: " << p1_position << ", p1_size: " << p1_edges.size() << endl;
            cerr << "p1 innovation numbers: " << endl;
            for (int32_t i = 0; i < (int32_t)p1_edges.size(); i++) {
                cerr << "\t" << p1_edges[i]->get_innovation_number() << endl;
            }
            cerr << "p2_position: " << p2_position << ", p2_size: " << p2_edges.size() << endl;
            cerr << "p2 innovation numbers: " << endl;
            for (int32_t i = 0; i < (int32_t)p2_edges.size(); i++) {
                cerr << "\t" << p2_edges[i]->get_innovation_number() << endl;
            }
            cerr << "vector innovation numbers: " << endl;
            for (int32_t i = 0; i < (int32_t)child_edges.size(); i++) {
                cerr << "\t" << child_edges[i]->get_innovation_number() << endl;
            }
        }

        child_edges.push_back(edge);

        if (rng_double(generator) >= more_fit_parent_crossover) {
            edge->disable();
        }

        //push back surrounding nodes
        attempt_node_insert(child_nodes, p1_edge->get_input_node());
        attempt_node_insert(child_nodes, p1_edge->get_output_node());

        p1_position++;
    }

    while (p2_position < (int32_t)p2_edges.size()) {
        CNN_Edge* p2_edge = p2_edges[p2_position];

        CNN_Edge *edge = p2_edge->copy();

        if (edges_contains(child_edges, edge)) {
            cerr << "ERROR in crossover! trying to push an edge with innovation_number: " << edge->get_innovation_number() << " and it already exists in the vector!" << endl;
            cerr << "p1_position: " << p1_position << ", p1_size: " << p1_edges.size() << endl;
            cerr << "p1 innovation numbers: " << endl;
            for (int32_t i = 0; i < (int32_t)p1_edges.size(); i++) {
                cerr << "\t" << p1_edges[i]->get_innovation_number() << endl;
            }
            cerr << "p2_position: " << p2_position << ", p2_size: " << p2_edges.size() << endl;
            cerr << "p2 innovation numbers: " << endl;
            for (int32_t i = 0; i < (int32_t)p2_edges.size(); i++) {
                cerr << "\t" << p2_edges[i]->get_innovation_number() << endl;
            }
            cerr << "vector innovation numbers: " << endl;
            for (int32_t i = 0; i < (int32_t)child_edges.size(); i++) {
                cerr << "\t" << child_edges[i]->get_innovation_number() << endl;
            }
        }

        child_edges.push_back(edge);

        if (rng_double(generator) >= less_fit_parent_crossover) {
            edge->disable();
        }

        //push back surrounding nodes
        attempt_node_insert(child_nodes, p2_edge->get_input_node());
        attempt_node_insert(child_nodes, p2_edge->get_output_node());

        p2_position++;
    }

    sort(child_edges.begin(), child_edges.end(), sort_CNN_Edges_by_depth());
    sort(child_nodes.begin(), child_nodes.end(), sort_CNN_Nodes_by_depth());

    for (int32_t i = 0; i < (int32_t)child_edges.size(); i++) {
        if (!child_edges[i]->set_nodes(child_nodes)) {
            cout << "\t\treinitializing weights of copy" << endl;
            child_edges[i]->reinitialize(generator, normal_distribution);
            child_edges[i]->save_best_weights();
        }
    }

    long genome_seed = rng_long(generator);
    CNN_Genome *child = new CNN_Genome(genomes_generated++, genome_seed, min_epochs, max_epochs, improvement_required_epochs, reset_edges, learning_rate, weight_decay, child_nodes, child_edges);

    child->set_generated_by_crossover();

    return child;
}


void EXACT::print_statistics(ostream &out) {
    double min_fitness = numeric_limits<double>::max();
    double max_fitness = -numeric_limits<double>::max();
    double avg_fitness = 0.0;
    int fitness_count = 0;

    double min_epochs = numeric_limits<double>::max();
    double max_epochs = -numeric_limits<double>::max();
    double avg_epochs = 0.0;
    int epochs_count = 0;

    for (int32_t i = 0; i < (int32_t)genomes.size(); i++) {
        double fitness = genomes[i]->get_fitness();

        if (fitness != numeric_limits<double>::max()) {
            avg_fitness += fitness;
            fitness_count++;
        }

        if (fitness < min_fitness) min_fitness = fitness;
        if (fitness > max_fitness) max_fitness = fitness;

        double epochs = genomes[i]->get_best_error_epoch();

        if (epochs != numeric_limits<double>::max()) {
            avg_epochs += epochs;
            epochs_count++;
        }

        if (epochs < min_epochs) min_epochs = epochs;
        if (epochs > max_epochs) max_epochs = epochs;
    }
    avg_fitness /= fitness_count;
    avg_epochs /= epochs_count;

    if (fitness_count == 0) avg_fitness = 0.0;
    if (min_fitness == numeric_limits<double>::max()) min_fitness = 0;
    if (max_fitness == numeric_limits<double>::max()) max_fitness = 0;
    if (max_fitness == -numeric_limits<double>::max()) max_fitness = 0;

    if (epochs_count == 0) avg_epochs = 0.0;
    if (min_epochs == numeric_limits<double>::max()) min_epochs = 0;
    if (max_epochs == numeric_limits<double>::max()) max_epochs = 0;
    if (max_epochs == -numeric_limits<double>::max()) max_epochs = 0;

    out << setw(16) << inserted_genomes
        << setw(16) << setprecision(5) << fixed << min_fitness
        << setw(16) << setprecision(5) << fixed << avg_fitness
        << setw(16) << setprecision(5) << fixed << max_fitness
        << setw(16) << setprecision(5) << fixed << min_epochs
        << setw(16) << setprecision(5) << fixed << avg_epochs
        << setw(16) << setprecision(5) << fixed << max_epochs 
        << setw(16) << inserted_from_disable_edge
        << setw(16) << inserted_from_enable_edge
        << setw(16) << inserted_from_split_edge
        << setw(16) << inserted_from_add_edge
        << setw(16) << inserted_from_change_size
        << setw(16) << inserted_from_change_size_x
        << setw(16) << inserted_from_change_size_y
        << setw(16) << inserted_from_crossover
        << endl;
}

void EXACT::print_statistics_header(ostream &out) {
    out << ", " << setw(14) << "inserted"
        << ", " << setw(14) << "min_fitness"
        << ", " << setw(14) << "avg_fitness"
        << ", " << setw(14) << "max_fitness"
        << ", " << setw(14) << "min_epochs"
        << ", " << setw(14) << "avg_epochs"
        << ", " << setw(14) << "max_epochs"
        << ", " << setw(14) << "disable_edge"
        << ", " << setw(14) << "enable_edge"
        << ", " << setw(14) << "split_edge"
        << ", " << setw(14) << "add_edge"
        << ", " << setw(14) << "change_size"
        << ", " << setw(14) << "change_size_x"
        << ", " << setw(14) << "change_size_y"
        << ", " << setw(14) << "crossover"
        << endl;
}
