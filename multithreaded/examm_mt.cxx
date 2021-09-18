#include <mutex>
using std::mutex;

#include <string>
using std::string;

#include <thread>
using std::thread;

#include <vector>
using std::vector;

#include "common/arguments.hxx"
#include "common/log.hxx"

#include "rnn/examm.hxx"
#include "rnn/work/work.hxx"
#include "rnn/training_parameters.hxx"
#include "rnn/genome_operators.hxx"

#include "time_series/time_series.hxx"
#include "common/dataset_meta.hxx"

#include "examm_mt_core.cxx"
int main(int argc, char** argv) {
    arguments = vector<string>(argv, argv + argc);

    Log::initialize(arguments);
    Log::set_id("main");

#define EXAMM_MT
#include "common/examm_argparse.cxx"

    vector<thread> threads;
    for (int32_t i = 0; i < number_threads; i++) {
        threads.push_back( thread(examm_thread, i, make_genome_operators(i)) );
    }

    for (int32_t i = 0; i < number_threads; i++) {
        threads[i].join();
    }

    finished = true;

    Log::info("completed!\n");
    Log::release_id("main");

    return 0;
}
