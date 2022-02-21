#ifndef EXAMM_ISLAND_SPECIATION_STRATEGY_HXX
#define EXAMM_ISLAND_SPECIATION_STRATEGY_HXX

#include <functional>
using std::function;

#include <random>
using std::minstd_rand0;
using std::uniform_real_distribution;

#include <string>
using std::string;

#include "island.hxx"
#include "msg.hxx"
#include "rnn_genome.hxx"
#include "speciation_strategy.hxx"

// Used in IslandSpeciationStrategy::parents_repopulation

class IslandSpeciationStrategy : public SpeciationStrategy {
 private:
  typedef enum {
    BEST_PARENTS,
    RANDOM_PARENTS,
    BEST_GENOME,
    BEST_ISLAND,
    NONE
  } RepopulationMethod;

  int32_t generation_island; /**< Used to track which island to generate the
                                next genome from. */

  uint32_t number_of_islands; /**< the number of islands to have. */

  uint32_t max_island_size; /**< the maximum number of genomes in an island. */

  double mutation_rate; /**< How frequently to do mutations. Note that
                           mutation_rate + intra_island_crossover_rate +
                           inter_island_crossover_rate should equal 1, if not
                           they will be scaled down such that they do. */
  double intra_island_crossover_rate; /**< How frequently to do intra-island
                                         crossovers. Note that mutation_rate +
                                         intra_island_crossover_rate +
                                         inter_island_crossover_rate should
                                         equal 1, if not they will be scaled
                                         down such that they do. */
  double inter_island_crossover_rate; /**< How frequently to do inter-island
                                         crossovers. Note that mutation_rate +
                                         intra_island_crossover_rate +
                                         inter_island_crossover_rate should
                                         equal 1, if not they will be scaled
                                         down such that they do. */

  shared_ptr<const RNN_Genome>
      seed_genome; /**< keep a reference to the seed genome so we can re-use it
     across islands and not duplicate innovation numbers. */

  string island_ranking_method; /**< The method used to find the worst island in
                                   population */

  RepopulationMethod repopulation_method; /**< The method used to repopulate the
                                             island after being erased */
  string repopulation_method_str;

  uint32_t
      extinction_event_generation_number; /**< When EXAMM reaches this
                                             generation id, an extinction event
                                             will be triggered (i.e. islands
                                             will be killed and repopulated). */
  uint32_t repopulation_mutations; /**< When an island is erradicated, it is
                                      repopulated with copies of the best genome
                                      that have this number of mutations applied
                                      to them. */
  uint32_t islands_to_exterminate; /**< When an extinction event is triggered,
                                      this is the number of islands that will be
                                      exterminated. */
  uint32_t max_genomes;

  bool repeat_extinction;

  bool
      seed_genome_was_minimal; /**< is true if we passed in a minimal genome
                                      (i.e., are not using transfer learning) */
  // int32_t worst_island;
  /**
   * All the islands which contain the genomes for this speciation strategy.
   */
  vector<Island *> islands;
  shared_ptr<const RNN_Genome> global_best_genome;

  GenomeOperators &genome_operators;

  unique_ptr<WorkMsg> generate_work_for_filled_island(
      uniform_real_distribution<double> &rng_0_1, minstd_rand0 &generator,
      Island *island);
  unique_ptr<WorkMsg> generate_work_for_initializing_island(
      uniform_real_distribution<double> &rng_0_1, minstd_rand0 &generator,
      Island *island);
  unique_ptr<WorkMsg> generate_work_for_reinitializing_island(
      uniform_real_distribution<double> &rng_0_1, minstd_rand0 &generator,
      Island *island);

 public:
  // static void register_command_line_arguments();
  // static IslandSpeciationStrategy* generate_from_command_line();

  /**
   * Transfer learning constructor.
   * \param Modification function to be applied to every copy of the seed_genome
   */
  IslandSpeciationStrategy(
      uint32_t _number_of_islands, uint32_t _max_island_size,
      shared_ptr<const RNN_Genome> _seed_genome, string _island_ranking_method,
      string _repopulation_method, uint32_t _extinction_event_generation_number,
      uint32_t _repopulation_mutations, uint32_t _islands_to_exterminate,
      bool seed_genome_was_minimal,
      optional<function<void(RNN_Genome *)>> modify,
      GenomeOperators &genome_operators);

  /**
   * \return the number of generated genomes.
   */
  int32_t get_generated_genomes() const;

  /**
   * \return the number of inserted genomes.
   */
  int32_t get_evaluated_genomes() const;

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
   * \return the best genome of all islands or NULL if no genomes have yet been
   * inserted
   */
  shared_ptr<const RNN_Genome> &get_best_genome();

  /**
   * Gets the the worst genome of all the islands
   * \return the worst genome of all islands or NULL if no genomes have yet been
   * inserted
   */
  shared_ptr<const RNN_Genome> &get_worst_genome();

  /**
   *  \return true if all the islands are full
   */
  bool islands_full() const;

  /**
   * Inserts a <b>copy</b> of the genome into one of the islands handled by this
   * strategy, determined by the RNN_Genome::get_group_id() method.
   *
   * The caller of this method will need to free the memory of the genome passed
   * into this method.
   *
   * \param genome is the genome to insert.
   * \return a value < 0 if the genome was not inserted, 0 if it was a new best
   * genome for all the islands, or > 0 otherwise.
   */
  pair<int32_t, const RNN_Genome *> insert_genome(
      unique_ptr<RNN_Genome> genome);

  /**
   * find the worst island in the population, the worst island's best genome is
   * the worst among all the islands
   *
   *  \return the worst island id
   */
  int32_t get_worst_island_by_best_genome();

  /**
   * rank the islands by their best fitness, the fitness of ranked islands are
   * in descending order
   *
   *  \return island rank from worst to the best
   */
  vector<int32_t> rank_islands();

  /**
   * See speciation_strategy.hxx
   *
   * \param rng_0_1 is the random number distribution that generates random
   * numbers between 0 (inclusive) and 1 (non=inclusive). \param generator is
   * the random number generator
   *
   * \return the newly generated genome.
   */
  unique_ptr<WorkMsg> generate_work(uniform_real_distribution<double> &rng_0_1,
                                    minstd_rand0 &generator);

  /**
   * Prints out all the island's populations
   *
   * \param indent is how much to indent what is printed out
   */
  void print(string indent = "") const;

  /**
   * Gets speciation strategy information headers for logs
   */
  string get_strategy_information_headers() const;

  /**
   * Gets speciation strategy information values for logs
   */
  string get_strategy_information_values() const;

  /**
   * Island repopulation through two random parents from two seperate islands,
   * parents can be random genomes or best genome from the island
   */
  unique_ptr<WorkMsg> parents_repopulation(
      RepopulationMethod method, uniform_real_distribution<double> &rng_0_1,
      minstd_rand0 &generator);

  /**
   * copy src island to dsc island
   *  \param best_island is the island id of the best island
   */
  void copy_island(uint32_t src_island, uint32_t dst_island);

  shared_ptr<const RNN_Genome> &get_global_best_genome();

  void set_erased_islands_status();

  void set_repopulation_method(string method);
};

#endif
