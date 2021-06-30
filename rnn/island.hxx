#ifndef EXAMM_ISLAND_STRATEGY_HXX
#define EXAMM_ISLAND_STRATEGY_HXX

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


class Island {
    private:
        int32_t id; /**< An integer ID for this island. */
        int32_t max_size; /**< The maximum number of genomes in the island. */
        int32_t erased_generation_id = -1; /**< The latest generation id of an erased island, erased_generation_id = largest_generation_id when this island is erased,
                                                to prevent deleted genomes get inserted back */
        int32_t latest_generation_id; /**< The latest generation id of genome being generated, including the ones doing backprop by workers */

        /**
         * The genomes on this island, stored in sorted order best (front) to worst (back).
         */
        vector<RNN_Genome*> genomes;

        unordered_map<string, vector<RNN_Genome*>> structure_map;
        int32_t status; /**> The status of this island (either Island:INITIALIZING, Island::FILLED or  Island::REPOPULATING */

        int32_t erase_again; /**< a flag to track if this islands has been erased */
        bool erased; /**< a flag to track if this islands has been erased */

    public:
        const static int32_t INITIALIZING = 0; /**< status flag for if the island is initializing. */
        const static int32_t FILLED = 1; /**< status flag for if the island is filled. */
        const static int32_t REPOPULATING = 2; /**< status flag for if the island is repopulating. */

        /**
         *  Initializes an island with a given max size.
         *
         *  \param max_size is the maximum number of genomes in the island.
         */
        Island(int32_t id, int32_t max_size);

        /**
         * Initializes an island filled the supplied genomes. The size of the island will be the size
         * of the supplied genome vector. The island status is set to filled.
         */
        Island(int32_t id, vector<RNN_Genome*> genomes);

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
         * Returns the maximum number of genomes the island can hold
         *
         * \return the maximum number of genomes this island can have
         */
        int32_t get_max_size();


        /**
         * Returns the size of the island
         *
         * \return the number of genomes in this island.
         */
        int32_t size();

        /**
         * Returns true if the island has Island::max_size genomes.
         *
         * \return true if the number of genomes in the island is >= size (although 
         * it should never be > size).
         */
        bool is_full();

        /**
         * Returns true if the island is initializing, i.e., it's size is <= max_size
         * and it hasn't been cleared out for repopulating.
         *
         * \return true if island is initializing.
         */
        bool is_initializing();

        /**
         * Returns true if the island is repopulating, i.e., it's size is <= max_size
         * and it has been full before but cleared out for repopulation.
         *
         * \return true if island is repopulating.
         */
        bool is_repopulating();

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

        void do_population_check(int line, int initial_size);

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
        
        /**
         * erases the entire island and set the erased_generation_id.
         */
        void erase_island();

        void erase_structure_map();

        /**
         * returns the get_erased_generation_id.
         */
        int32_t get_erased_generation_id();

        /**
         * \return the status of the island
         */
        int32_t get_status();


        /**
         * after erasing the island, sets the island status to repopulating.
         */
        void set_status(int32_t status_to_set);

        /**
         * return if this island has been erased before.
         */
        bool been_erased();

        vector<RNN_Genome *> get_genomes();

        void set_latest_generation_id(int32_t _latest_generation_id);

        int32_t get_erase_again_num();

        void set_erase_again_num();
};

#endif
