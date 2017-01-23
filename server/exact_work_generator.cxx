// This file is part of BOINC.
// http://boinc.berkeley.edu
// Copyright (C) 2008 University of California
//
// BOINC is free software; you can redistribute it and/or modify it
// under the terms of the GNU Lesser General Public License
// as published by the Free Software Foundation,
// either version 3 of the License, or (at your option) any later version.
//
// BOINC is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
// See the GNU Lesser General Public License for more details.
//
// You should have received a copy of the GNU Lesser General Public License
// along with BOINC.  If not, see <http://www.gnu.org/licenses/>.

// sample_work_generator.cpp: an example BOINC work generator.
// This work generator has the following properties
// (you may need to change some or all of these):
//
// - Runs as a daemon, and creates an unbounded supply of work.
//   It attempts to maintain a "cushion" of 100 unsent job instances.
//   (your app may not work this way; e.g. you might create work in batches)
// - Creates work for the application "example_app".
// - Creates a new input file for each job;
//   the file (and the workunit names) contain a timestamp
//   and sequence number, so they're unique.

#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>

#include <cmath>
#include <cstdlib>
#include <cstring>

#include <fstream>

#include <string>
using std::string;
using std::to_string;

#include <sstream>
using std::ostringstream;

#include <stdio.h>
#include <stdlib.h>

#include <vector>
using std::vector;

//for boinc
#include "boinc_db.h"
#include "diagnostics.h"
#include "error_numbers.h"
#include "backend_lib.h"
#include "parse.h"
#include "util.h"
#include "svn_version.h"

#include "sched_config.h"
#include "sched_util.h"
#include "sched_msgs.h"
#include "str_util.h"

//for mysql
#include "mysql.h"

#include "common/arguments.hxx"
#include "common/db_conn.hxx"
#include "image_tools/image_set.hxx"
#include "strategy/exact.hxx"
#include "server/make_jobs.hxx"

int main(int argc, char** argv) {
    vector<string> arguments = vector<string>(argv, argv + argc);

    int debug_level;
    get_argument(arguments, "--debug_level", true, debug_level);

    string search_name;
    get_argument(arguments, "--search_name", true, search_name);

    log_messages.set_debug_level(debug_level);
    if (debug_level == 4) g_print_queries = true;

    //if at any time the retval value is greater than 0, then the program
    //has failed in some manner, and the program then exits.

    //processing project's config file.
    int retval = config.parse_file();
    if (retval) {
        log_messages.printf(MSG_CRITICAL, "Can't parse config.xml: %s\n", boincerror(retval));
        exit(1);
    }

    //opening connection to database.
    retval = boinc_db.open(config.db_name, config.db_host, config.db_user, config.db_passwd);
    if (retval) {
        log_messages.printf(MSG_CRITICAL, "can't open db\n");
        exit(1);
    }

    init_work_generation();

    //initialize the EXACT algorithm
    int population_size = 100;
    get_argument(arguments, "--population_size", true, population_size);

    int min_epochs = 50;
    get_argument(arguments, "--min_epochs", true, min_epochs);

    int max_epochs = 50;
    get_argument(arguments, "--max_epochs", true, max_epochs);

    int improvement_required_epochs = 5;
    get_argument(arguments, "--improvement_required_epochs", true, improvement_required_epochs);

    bool reset_edges = false;
    get_argument(arguments, "--reset_edges", true, reset_edges);

    double learning_rate;
    get_argument(arguments, "--learning_rate", true, learning_rate);

    double learning_rate_decay;
    get_argument(arguments, "--learning_rate_decay", true, learning_rate_decay);

    double weight_decay;
    get_argument(arguments, "--weight_decay", true, weight_decay);

    double weight_decay_decay;
    get_argument(arguments, "--weight_decay_decay", true, weight_decay_decay);

    double mu;
    get_argument(arguments, "--mu", true, mu);

    double mu_decay;
    get_argument(arguments, "--mu_decay", true, mu_decay);


    int max_individuals = 1000000;
    string output_directory = "/projects/csg/exact_data/" + search_name;

    mkdir(output_directory.c_str(), 0777);

    Images images("/home/tdesell/mnist_training_data.bin");

    EXACT *exact = new EXACT(images, population_size, min_epochs, max_epochs, improvement_required_epochs, reset_edges, mu, mu_decay, learning_rate, learning_rate_decay, weight_decay, weight_decay_decay, max_individuals, output_directory, search_name);
    exact->export_to_database();

    log_messages.printf(MSG_NORMAL, "inserted exact search into database with id: %d\n", exact->get_id());

    int start_time = time(0);

    log_messages.printf(MSG_NORMAL, "starting at %d...\n", start_time);

    make_jobs(exact, WORKUNITS_TO_GENERATE);

    //export the search and all generated genomes to the database
    exact->export_to_database();

    return 0;
}

