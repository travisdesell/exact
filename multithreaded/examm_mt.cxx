#include <mutex>
using std::mutex;

#include <string>
using std::string;

#include <thread>
using std::thread;

#include <vector>
using std::vector;

#include "common/arguments.hxx"
#include "common/dataset_meta.hxx"
#include "common/log.hxx"
#include "examm_mt_core.cxx"
#include "rnn/examm.hxx"
#include "rnn/genome_operators.hxx"
#include "rnn/training_parameters.hxx"
#include "time_series/time_series.hxx"
int main(int argc, char **argv) {
  arguments = vector<string>(argv, argv + argc);

  Log::initialize(arguments);
  Log::set_id("init");

#define EXAMM_MT
#include "common/examm_argparse.cxx"
  examm = make_examm();
  set_innovation_counts(examm);

  vector<thread> threads;
  for (int32_t i = 0; i < number_threads; i++) {
    threads.push_back(thread(examm_thread, number_threads, i, make_genome_operators(i), random_sequence_length,
                             sequence_length_lower_bound, sequence_length_upper_bound));
  }

  for (int32_t i = 0; i < number_threads; i++) { threads[i].join(); }

  finished = true;

  Log::info("completed!\n");
  Log::release_id("init");

  delete dataset;

  return 0;
}
