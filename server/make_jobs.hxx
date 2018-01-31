#ifndef EXACT_MAKE_JOBS_HXX
#define EXACT_MAKE_JOBS_HXX

#include "cnn/exact.hxx"

#define CUSHION 100
#define WORKUNITS_TO_GENERATE 100
#define REPLICATION_FACTOR  1
#define SLEEP_TIME 10

bool low_on_workunits();
void make_jobs(EXACT *exact, int workunits_to_generate);
void init_work_generation(string app_name);

#endif
