#ifndef EXAMM_ISLAND_SPECIATION_STRATEGY_HXX
#define EXAMM_ISLAND_SPECIATION_STRATEGY_HXX

#include <string>
using std::string;

#include "island.hxx"
#include "rnn_genome.hxx"
#include "speciation_strategy.hxx"

class IslandSpeciationStrategy : public SpeciationStrategy {
    private:
        /**
         * All the islands which contain the genomes for this speciation strategy.
         */ 
        vector<Island*> islands;
    
    public:
        //static void register_command_line_arguments();
        //static IslandSpeciationStrategy* generate_from_command_line();

        /**
         * Creates a new IslandSpeciationStrategy.
         *
         * \param number_of_islands specifies how many islands it will us e
         * \param max_island_size specifies the maximum number of gneomes in an island
         */
        IslandSpeciationStrategy(int32_t number_of_islands, int32_t max_island_size);

        /**
         * Gets the fitness of the best genome of all the islands
         * \return the best fitness over all islands
         */
        double get_best_fitness();

        /**
         * Gets the fitness of the worst genome of all the islands
         * \return the worst fitness over all islands
         */
        double get_worst_fitness();

        /**
         * Gets the best genome of all the islands
         * \return the best genome of all islands or NULL if no genomes have yet been inserted
         */
        RNN_Genome* get_best_genome();

        /**
         * Gets the the worst genome of all the islands
         * \return the worst genome of all islands or NULL if no genomes have yet been inserted
         */
        RNN_Genome* get_worst_genome();


        /**
         * Prints out all the island's populations
         */
        void print_population();

        /**
         * Inserts a <b>copy</b> of the genome into one of the islands handled by this
         * strategy, determined by the RNN_Genome::get_group_id() method.
         *
         * The caller of this method will need to free the memory of the genome passed
         * into this method.
         *
         * \param genome is the genome to insert.
         * \return a value < 0 if the genome was not inserted, 0 if it was a new best genome
         * for all the islands, or > 0 otherwise.
         */
        int32_t insert_genome(RNN_Genome* genome);


        /**
         * Generates a new genome.
         *
         * \return the newly generated genome.
         */
        RNN_Genome* generate_genome();
};


#endif

