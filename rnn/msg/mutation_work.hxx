#ifndef MUTATION_WORK_HXX
#define MUTATION_WORK_HXX 1
#include "work.hxx"

class MutationWork : public Work {
 private:
  int32_t n_mutations;
  RNN_Genome *genome;

 public:
  static constexpr int32_t class_id = 0;

  /**
   * The supplied genome must not be null!
   **/
  MutationWork(RNN_Genome *_genome, int32_t _n_mutations, int32_t _generation_id = -1, int32_t _group_id = -1);
  MutationWork(istream &bin_istream);
  ~MutationWork();

  void write_to_stream(ostream &bin_ostream);
  RNN_Genome *get_genome(GenomeOperators &operators);
  int32_t get_class_id();
};

#endif
