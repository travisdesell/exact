#ifndef EXAMM_SPECIATION_STRATEGY_HXX
#define EXAMM_SPECIATION_STRATEGY_HXX

#include <functional>
using std::function;
#include <string>
using std::string;
#include <random>
using std::minstd_rand0;
using std::uniform_real_distribution;

#include "msg.hxx"

class SpeciationStrategy {
protected:
  int32_t generated_genomes = 0;
  int32_t inserted_genomes = 0;

public:
  /**
   * \return the number of generated genomes.
   */
  int32_t get_generated_genomes();

  /**
   * \return the number of inserted genomes.
   */
  int32_t get_inserted_genomes();

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
  virtual RNN_Genome *get_best_genome() = 0;

  /**
   * Gets the the worst genome of all the islands
   * \return the worst genome of all islands
   */
  virtual RNN_Genome *get_worst_genome() = 0;

  /**
   * Inserts a <b>copy</b> of the genome into this speciation strategy.
   *
   * The caller of this method will need to free the memory of the genome passed
   * into this method.
   *
   * \param genome is the genome to insert.
   * \return a value < 0 if the genome was not inserted, 0 if it was a new best
   * genome or > 0 otherwise.
   */
  virtual int32_t insert_genome(RNN_Genome *genome) = 0;

  /**
   * Generates a unit of work. The work unit is to be sent to a worker.
   */
  virtual WorkMsg *generate_work(uniform_real_distribution<double> &rng_0_1,
                              minstd_rand0 &generator) = 0;

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

  virtual RNN_Genome *get_global_best_genome() = 0;
};

#endif
