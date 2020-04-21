#ifndef EXAMM_NEAT_STRATEGY_HXX
#define EXAMM_NEAT_STRATEGY_HXX

#include <functional>
using std::function;

#include <algorithm>
using std::sort;
using std::upper_bound;

#include <random>
using std::minstd_rand0;
using std::uniform_real_distribution;

#include <string>
using std::string;

#include <vector>

#include "rnn_genome.hxx"


class Species {
    private:
        int32_t id; /**< An integer ID for this species. */

        // int32_t latest_inserted_generation_id; /**< The latest generation id of genome being generated, including the ones doing backprop by workers */

        vector<int32_t> inserted_genome_id;
        /**
         * The genomes on this species, stored in sorted order best (front) to worst (back).
         */
        vector<RNN_Genome *> genomes;

        int32_t species_not_improving_count;

    public:
        /**
         *  Initializes a species.
         */
        Species(int32_t id);


        /**
         * Returns the fitness of the best genome in the island
         *
         * \return the best fitness of the island
         */
        double get_best_fitness();

        /**
         * Returns the fitness of the worst genome in the island
         *
         * \return the worst fitness of the island
         */
        double get_worst_fitness();

        /**
         * Returns the best genomme in the island.
         *
         * \return the best genome in the island
         */
        RNN_Genome *get_best_genome();

        /**
         * Returns the worst genomme in the island.
         *
         * \return the worst genome in the island
         */
        RNN_Genome *get_worst_genome();

        /**
         * Returns the best genomme in the island.
         *
         * \return the best genome in the island
         */
        RNN_Genome *get_random_genome(uniform_real_distribution<double> &rng_0_1, minstd_rand0 &generator);

        /**
         * Returns the size of the island
         *
         * \return the number of genomes in this island.
         */
        int32_t size();

        /**
         * Checks to see if a genome already exists in the island, using
         * the RNN_Genome::equals method (which checks to see if all edges
         * and nodes are the same, but not necessarily weights).
         *
         * \return the index in the island of the duplicate genome, -1 otherwise.
         */
        int32_t contains(RNN_Genome* genome);

        /**
         * Selects a genome from the island at random and returns a copy of it.
         *
         * \param rng_0_1 is the random number distribution that generates random numbers between 0 (inclusive) and 1 (non=inclusive).
         * \param generator is the random number generator
         * \param genome will be the copied genome, an addresss to a pointer needs to be passed.
         */
        void copy_random_genome(uniform_real_distribution<double> &rng_0_1, minstd_rand0 &generator, RNN_Genome **genome);

        /**
         * Selects two different genomes from the island at random and returns copies of them.
         *
         * \param rng_0_1 is the random number distribution that generates random numbers between 0 (inclusive) and 1 (non=inclusive).
         * \param generator is the random number generator
         * \param genome1 will be the first copied genome, an addresss to a pointer needs to be passed.
         * \param genome2 will be the second copied genome, an addresss to a pointer needs to be passed.
         */
        void copy_two_random_genomes(uniform_real_distribution<double> &rng_0_1, minstd_rand0 &generator, RNN_Genome **genome1, RNN_Genome **genome2);

        /**
         * Inserts a genome into the island.
         *
         * Genomes are inserted in best to worst order genomes[0] will have
         * the best fitness and genomes[size - 1] will have the worst.
         *
         * \param genome is the genome to be inserted. 
         * \return -1 if not inserted, otherwise the index it was inserted at
         */
        int32_t insert_genome(RNN_Genome* genome);

        /**
         * Prints out the state of this island.
         *
         * \param indent is how much to indent what is printed out
         */
        void print(string indent = "");

        vector<RNN_Genome *> get_genomes();

        RNN_Genome* get_latested_genome();

        void fitness_sharing_remove(double fitness_threshold, function<double (RNN_Genome*, RNN_Genome*)> &get_distance);

        void erase_species();

        int32_t get_species_not_improving_count();

        void set_species_not_improving_count(int32_t count);
};

#endif
