#ifndef EXAMM_SPECIATION_STRATEGY_HXX
#define EXAMM_SPECIATION_STRATEGY_HXX

#include <functional>
using std::function;
#include <string>
using std::string;
#include <random>
using std::minstd_rand0;
using std::uniform_real_distribution;


class SpeciationStrategy {

    public:
        /**
         * \return the number of generated genomes.
         */
        virtual int32_t get_generated_genomes() const = 0;

        /**
         * \return the number of inserted genomes.
         */
        virtual int32_t get_inserted_genomes() const = 0;

        /**
         * Gets the fitness of the best genome of all the islands
         * \return the best fitness over all islands
         */
        virtual double get_best_fitness() = 0;

        /**
         * Gets the fitness of the worst genome of all the islands
         * \return the worst fitness over all islands
         */
        virtual double get_worst_fitness() = 0;

        /**
         * Gets the best genome of all the islands
         * \return the best genome of all islands
         */
        virtual RNN_Genome* get_best_genome() = 0;

        /**
         * Gets the the worst genome of all the islands
         * \return the worst genome of all islands
         */
        virtual RNN_Genome* get_worst_genome() = 0;

        /**
         * Inserts a <b>copy</b> of the genome into this speciation strategy.
         *
         * The caller of this method will need to free the memory of the genome passed
         * into this method.
         *
         * \param genome is the genome to insert.
         * \return a value < 0 if the genome was not inserted, 0 if it was a new best genome
         * or > 0 otherwise.
         */
        virtual int32_t insert_genome(RNN_Genome* genome) = 0;

        /**
         * Generates a new genome.
         *
         * \param rng_0_1 is the random number distribution that generates random numbers between 0 (inclusive) and 1 (non=inclusive).
         * \param generator is the random number generator
         * \param mutate is the a function which performs a mutation on a genome
         * \param crossover is the function which performs crossover between two genomes
         *
         * \return the newly generated genome.
         */
        virtual RNN_Genome* generate_genome(uniform_real_distribution<double> &rng_0_1, minstd_rand0 &generator, function<void (int32_t, RNN_Genome*)> &mutate, function<RNN_Genome* (RNN_Genome*, RNN_Genome *)> &crossover, int32_t number_stir_mutations) = 0;


        /**
         * Prints out all the island's populations
         *
         * \param indent is how much to indent what is printed out
         */
        virtual void print(string indent = "") const = 0;

        /**
         * Gets speciation strategy information headers for logs
         */
        virtual string get_strategy_information_headers() const = 0;

        /**
         * Gets speciation strategy information values for logs
         */
        virtual string get_strategy_information_values() const = 0;

        virtual RNN_Genome* get_global_best_genome() = 0;
        
        /**
         * Used to set the rates to allow for dynamic rate adaptation
         *  \param _mutation_rate is the new mutation rate
         *  \param _intra_island_crossover_rate is the new intra-island crossover rate
         *  \param _inter_island_crossover_rate is the new inter-island crossover rate
         */
        virtual void set_rates(double _mutation_rate, double _intra_island_crossover_rate, double _inter_island_crossover_rate) = 0;
};

#endif
