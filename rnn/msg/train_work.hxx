#ifndef TRAIN_WORK_HXX
#define TRAIN_WORK_HXX 1
#include "work.hxx"

class TrainWork : public Work {
 private:
  RNN_Genome *genome;

 public:
  static constexpr int32_t class_id = 5;

  /**
   * The supplied genome must not be null!
   **/
  TrainWork(RNN_Genome *_genome, int32_t _generation_id = -1, int32_t _group_id = -1);
  TrainWork(istream &bin_istream);
  ~TrainWork();

  void write_to_stream(ostream &bin_ostream);
  RNN_Genome *get_genome(GenomeOperators &operators);
  int32_t get_class_id();
};

#endif
