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
#include "cnn/exact.hxx"
#include "server/make_jobs.hxx"

int main(int argc, char** argv) {
    vector<string> arguments = vector<string>(argv, argv + argc);

    string db_file;
    get_argument(arguments, "--db_file", true, db_file);
    set_db_info_filename(db_file);                                                                                   

    string app_name;
    get_argument(arguments, "--app", true, app_name);

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

    init_work_generation(app_name);

    //initialize the EXACT algorithm
    int population_size = 100;
    get_argument(arguments, "--population_size", true, population_size);

    int max_epochs = 50;
    get_argument(arguments, "--max_epochs", true, max_epochs);

    bool use_sfmp = true;
    get_argument(arguments, "--use_sfmp", true, use_sfmp);

    bool use_node_operations = true;
    get_argument(arguments, "--use_node_operations", true, use_node_operations);


    int max_genomes = 1000000;
    get_argument(arguments, "--max_genomes", true, max_genomes);

    bool reset_edges = false;
    get_argument(arguments, "--reset_edges", true, reset_edges);


    string output_directory = "/projects/csg/exact_data/" + search_name;

    mkdir(output_directory.c_str(), 0777);

    string training_file;
    get_argument(arguments, "--training_file", true, training_file);

    string validation_file;
    get_argument(arguments, "--validation_file", true, validation_file);

    string testing_file;
    get_argument(arguments, "--testing_file", true, testing_file);

    int padding = 0;
    get_argument(arguments, "--padding", true, padding);

    Images training_images(training_file, padding);
    Images validation_images(validation_file, padding, training_images.get_average(), training_images.get_std_dev());
    Images testing_images(testing_file, padding, training_images.get_average(), training_images.get_std_dev());

    EXACT *exact = new EXACT(training_images, validation_images, testing_images, padding, population_size, max_epochs, use_sfmp, use_node_operations, max_genomes, output_directory, search_name, reset_edges);

    exact->export_to_database();

    log_messages.printf(MSG_NORMAL, "inserted exact search into database with id: %d\n", exact->get_id());

    int start_time = time(0);

    log_messages.printf(MSG_NORMAL, "starting at %d...\n", start_time);

    make_jobs(exact, WORKUNITS_TO_GENERATE);

    //export the search and all generated genomes to the database
    exact->export_to_database();

    return 0;
}

