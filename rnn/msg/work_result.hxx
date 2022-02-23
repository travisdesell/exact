#ifndef WORK_RESULT_HXX
#define WORK_RESULT_HXX 1
#include "work.hxx"

class WorkResult : public Work {
 private:
  RNN_Genome *genome;

 public:
  static constexpr int32_t class_id = 4;

  /**
   * The supplied genome must not be null!
   **/
  WorkResult(RNN_Genome *_genome);
  WorkResult(istream &bin_istream);
  ~WorkResult();

  void write_to_stream(ostream &bin_ostream);
  RNN_Genome *get_genome(GenomeOperators &operators);
  int32_t get_class_id();
};

#endif
