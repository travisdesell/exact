#ifndef INITIALIZE_WORK_HXX
#define INITIALIZE_WORK_HXX
#include "work.hxx"

class InitializeWork : public Work {
 private:
  int32_t node_innovation_count;
  int32_t edge_innovation_count;

 public:
  static constexpr int32_t class_id = 3;

  InitializeWork(int32_t node_ic, int32_t edge_ic);
  InitializeWork(istream &bin_istream);
  ~InitializeWork();

  void write_to_stream(ostream &bin_ostream);
  void update_genome_operators(int32_t rank, GenomeOperators &operators);
  RNN_Genome *get_genome(GenomeOperators &operators);
  int32_t get_class_id();
};
#endif
