#ifndef EXAMM_ISLAND_STRATEGY_HXX
#define EXAMM_ISLAND_STRATEGY_HXX

#include <string>
using std::string;

#include "rnn_genome.hxx"


class Island {
    private:
        int32_t id; /**< An integer ID for this island. */
        int32_t max_size; /**< The maximum number of genomes in the island. */
        int32_t status; /**> The status of this island (either Island:INITIALIZING, Island::FILLED or  Island::REPOPULATING */

        /**
         * The genomes on this island, stored in sorted order best (front) to worst (back).
         */
        vector<RNN_Genome *> genomes;

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
         * Returns if the island has Island::max_size genomes.
         *
         * \return true if the number of genomes in the island is >= size (although 
         * it should never be > size).
         */
        bool is_full();

        /**
         * Checks to see if a genome already exists in the island, using
         * the RNN_Genome::equals method (which checks to see if all edges
         * and nodes are the same, but not necessarily weights).
         *
         * \return the index in the island of the duplicate genome, -1 otherwise.
         */
        int32_t contains(RNN_Genome* genome);

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
};

#endif
