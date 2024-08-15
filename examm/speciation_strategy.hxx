#ifndef EXAMM_SPECIATION_STRATEGY_HXX
#define EXAMM_SPECIATION_STRATEGY_HXX

#include <functional>
using std::function;
#include <string>
using std::string;
#include <random>
using std::minstd_rand0;
using std::uniform_real_distribution;

#include "island.hxx"

class SpeciationStrategy {
   public:
    /**
     * \return the number of generated genomes.
     */
    virtual int32_t get_generated_genomes() const = 0;

    /**
     * \return the number of inserted genomes.
     */
    virtual int32_t get_evaluated_genomes() const = 0;

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
     * Gets the the minimum learning rate of all the islands
     * \return the minimum learning rate of all islands or NULL if no genomes have yet been inserted
     */
    virtual double get_min_learning_rate() = 0;

    /**
     * Gets the the maximum learning rate of all the islands
     * \return the maximum learning rate of all islands or NULL if no genomes have yet been inserted
     */
    virtual double get_max_learning_rate() = 0;

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
     * \param rng_0_1 is the random number distribution that generates random numbers between 0 (inclusive) and 1
     * (non=inclusive). \param generator is the random number generator \param mutate is the a function which performs a
     * mutation on a genome \param crossover is the function which performs crossover between two genomes
     *
     * \return the newly generated genome.
     */
    virtual RNN_Genome* generate_genome(
        uniform_real_distribution<double>& rng_0_1, minstd_rand0& generator,
        function<void(int32_t, RNN_Genome*)>& mutate, function<RNN_Genome*(RNN_Genome*, RNN_Genome*)>& crossover
    ) = 0;

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
    virtual void initialize_population(function<void(int32_t, RNN_Genome*)>& mutate) = 0;
    virtual RNN_Genome* get_seed_genome() = 0;
    virtual void save_entire_population(string output_path) = 0;

    /**
    *  \return true if all the islands are full
    */
    virtual bool islands_full() const = 0;
    
    /**
    * Get the number of islands
    *
    *  \return the number of island
    */
    virtual int32_t get_islands_size() const = 0;
    
    /**
    * Get the island at a given index
    *  \param index index of required island
    *  \return the island at given index
    */
    virtual Island* get_island_at_index(int32_t index) const = 0;

    /**
    * Get the current island index
    *
    *  \return the index of current island
    */
    virtual int32_t get_generation_island() const = 0;
};

#endif
