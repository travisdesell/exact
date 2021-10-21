#ifndef EXAMM_ONENET_STRATEGY_HXX
#define EXAMM_ONENET_STRATEGY_HXX

#include <algorithm>
using std::sort;
using std::upper_bound;

#include <random>
using std::minstd_rand0;
using std::uniform_real_distribution;

#include <string>
using std::string;

#include <unordered_map>
using std::unordered_map;


#include "rnn_genome.hxx"


class Population {
    private:
        int32_t id;
        int32_t max_size; /**< The maximum number of genomes in the population. */
        /**
         * The genomes on this population, stored in sorted order best (front) to worst (back).
         */
        vector<RNN_Genome*> genomes;

        unordered_map<string, vector<RNN_Genome*>> structure_map;
        // int32_t status; /**> The status of this population (either Population:INITIALIZING, Population::FILLED or  Population::REPOPULATING */

    public:
        // const static int32_t INITIALIZING = 0; /**< status flag for if the population is initializing. */
        // const static int32_t FILLED = 1; /**< status flag for if the population is filled. */

        /**
         *  Initializes an population with a given max size.
         *
         *  \param max_size is the maximum number of genomes in the population.
         */
        Population(int32_t id, int32_t max_size);

        /**
         * Returns the fitness of the best genome in the population
         *
         * \return the best fitness of the population
         */
        double get_best_fitness();

        /**
         * Returns the fitness of the worst genome in the population
         *
         * \return the worst fitness of the population
         */
        double get_worst_fitness();

        /**
         * Returns the best genomme in the population.
         *
         * \return the best genome in the population
         */
        RNN_Genome *get_best_genome();

        /**
         * Returns the worst genomme in the population.
         *
         * \return the worst genome in the population
         */
        RNN_Genome *get_worst_genome();

        /**
         * Returns the maximum number of genomes the population can hold
         *
         * \return the maximum number of genomes this population can have
         */
        int32_t get_max_size();


        /**
         * Returns the size of the population
         *
         * \return the number of genomes in this population.
         */
        int32_t size();

        /**
         * Returns true if the population has Population::max_size genomes.
         *
         * \return true if the number of genomes in the population is >= size (although 
         * it should never be > size).
         */
        bool is_full();

        bool is_empty();

        /**
         * Selects a genome from the population at random and returns a copy of it.
         *
         * \param rng_0_1 is the random number distribution that generates random numbers between 0 (inclusive) and 1 (non=inclusive).
         * \param generator is the random number generator
         * \param genome will be the copied genome, an addresss to a pointer needs to be passed.
         */
        void copy_random_genome(uniform_real_distribution<double> &rng_0_1, minstd_rand0 &generator, RNN_Genome **genome);

        /**
         * Selects two different genomes from the population at random and returns copies of them.
         *
         * \param rng_0_1 is the random number distribution that generates random numbers between 0 (inclusive) and 1 (non=inclusive).
         * \param generator is the random number generator
         * \param genome1 will be the first copied genome, an addresss to a pointer needs to be passed.
         * \param genome2 will be the second copied genome, an addresss to a pointer needs to be passed.
         */
        void copy_two_random_genomes(uniform_real_distribution<double> &rng_0_1, minstd_rand0 &generator, RNN_Genome **genome1, RNN_Genome **genome2);

        // void do_population_check(int line, int initial_size);

        /**
         * Inserts a genome into the population.
         *
         * Genomes are inserted in best to worst order genomes[0] will have
         * the best fitness and genomes[size - 1] will have the worst.
         *
         * \param genome is the genome to be inserted. 
         * \return -1 if not inserted, otherwise the index it was inserted at
         */
        int32_t insert_genome(RNN_Genome* genome);

        /**
         * Prints out the state of this population.
         *
         * \param indent is how much to indent what is printed out
         */
        void print(string indent = "");
    
        vector<RNN_Genome *> get_genomes();

        void erase_population();

        void erase_structure_map();

        void sort_population(string sort_by);

};

// struct compare_online_mse {
//     inline bool operator() ( RNN_Genome* g1,  RNN_Genome* g2)
//     {
//         return (g1->get_online_validation_mse() < g2->get_online_validation_mse());
//     }
// };

#endif
