#ifndef EXACT_H
#define EXACT_H

#include <algorithm>
using std::sort;

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

#include <string>
using std::to_string;

#include <vector>
using std::vector;

#include "image_tools/image_set.hxx"
#include "cnn_node.hxx"
#include "cnn_edge.hxx"
#include "cnn_genome.hxx"


class EXACT {
    private:
        int id;

        string output_directory;

        int number_images;
        int image_rows;
        int image_cols;
        int number_classes;

        int population_size;
        int node_innovation_count;
        int edge_innovation_count;

        uniform_int_distribution<long> rng_long;
        uniform_real_distribution<double> rng_double;

        vector <CNN_Node* > all_nodes;
        vector <CNN_Edge* > all_edges;

        vector< CNN_Genome* > genomes;

        int genomes_generated;
        int inserted_genomes;

        bool reset_edges;
        int min_epochs;
        int max_epochs;
        int improvement_required_epochs;
        int max_individuals;

        mt19937 generator;

        double learning_rate;
        double weight_decay;

        double crossover_rate;
        double more_fit_parent_crossover;
        double less_fit_parent_crossover;

        int number_mutations;
        double edge_disable;
        double edge_enable;
        double edge_split;
        double edge_add;
        double edge_change_stride;
        double node_change_size;
        double node_change_size_x;
        double node_change_size_y;
        double node_change_pool_size;

        int inserted_from_disable_edge;
        int inserted_from_enable_edge;
        int inserted_from_split_edge;
        int inserted_from_add_edge;
        int inserted_from_change_size;
        int inserted_from_change_size_x;
        int inserted_from_change_size_y;
        int inserted_from_crossover;

    public:

        EXACT(const Images &images, int _population_size, int _min_epochs, int _max_epochs, int _improvement_required_epochs, bool _reset_edges, int _max_individuals, string _output_directory);

        bool population_contains(CNN_Genome *genome) const;
        CNN_Genome* get_best_genome();

        CNN_Genome* generate_individual();
        CNN_Genome* create_mutation();
        CNN_Genome* create_child();

        void insert_genome(CNN_Genome* genome);

        void print_statistics(ostream &out);
        void print_statistics_header(ostream &out);

        int get_id() const;
        int get_number_images() const;
};

#endif
