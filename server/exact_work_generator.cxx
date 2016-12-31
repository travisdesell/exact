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

#define CUSHION 1000
#define WORKUNITS_TO_GENERATE 200
#define REPLICATION_FACTOR  1
#define SLEEP_TIME 10

const char* app_name = "exact";

//the following are the templates for the workunits (exact_in.xml) and results (exact_out.xml)
//and they can be found in /projects/csg/templates
const char* in_template_file = "exact_in.xml";
const char* out_template_file = "exact_out.xml";

char* in_template;
char* out_template;
DB_APP app;
int start_time;

EXACT* exact;

void copy_file_to_download_dir(string filename) {
    char path[256];

    string short_name = filename.substr(filename.find_last_of('/') + 1);

    if ( !std::ifstream(filename) ) { 
        log_messages.printf(MSG_CRITICAL, "input filename '%s' does not exist, cannot copy to download directory.\n", filename.c_str());
        exit(1);
    }   

    int retval = config.download_path( short_name.c_str(), path );
    if (retval) {
        log_messages.printf(MSG_CRITICAL, "can't get download path for file '%s', error: %s\n", short_name.c_str(), boincerror(retval));
        exit(1);
    }   

    if ( std::ifstream(path) ) { 
        log_messages.printf(MSG_CRITICAL, "\033[1minput file '%s' already exists in download directory hierarchy as '%s', not copying.\033[0m\n", short_name.c_str(), path);
    } else {
        log_messages.printf(MSG_CRITICAL, "input file '%s' does not exist in downlaod directory hierarchy, copying to '%s'\n", short_name.c_str(), path);

        //open the first filename and copy it to the target here
        std::ifstream src(filename.c_str());
        if (!src.is_open()) {
            log_messages.printf(MSG_CRITICAL, "could not open file for reading '%s', error: %s\n", path, boincerror(ERR_FOPEN));
            exit(1);
        }   

        std::ofstream dst(path);
        if (!dst.is_open()) {
            log_messages.printf(MSG_CRITICAL, "could not open file for writing '%s', error: %s\n", path, boincerror(ERR_FOPEN));
            exit(1);
        }   

        dst << src.rdbuf();

        src.close();
        dst.close();
    }   
}

// create one new job
int make_job(CNN_Genome *genome) {
    DB_WORKUNIT wu;

    char name[256], path[256];
    char command_line[512];
    char additional_xml[512];
    const char* infiles[2];
    int retval;

    // make a unique name (for the job and its input file)
    //sprintf(name, "exact_genome_%u_%u", exact->get_id(), genome->get_generation_id());
    sprintf(name, "exact_genome_%d_%s_%u_%u", start_time, exact->get_search_name().c_str(), exact->get_id(), genome->get_generation_id());
    log_messages.printf(MSG_DEBUG, "name: '%s'\n", name);

    string dataset_filename = "/home/tdesell/mnist_training_data.bin";
    ostringstream oss;
    oss << "/projects/csg/exact_data/" << name << ".txt";
    string genome_filename = oss.str();

    log_messages.printf(MSG_DEBUG, "dataset filename: '%s'\n", dataset_filename.c_str());
    log_messages.printf(MSG_DEBUG, "genome filename: '%s'\n", genome_filename.c_str());

    genome->write_to_file(genome_filename);

    // Create the input file.
    // Put it at the right place in the download dir hierarchy
    //
    retval = config.download_path(name, path);
    if (retval) return retval;
    log_messages.printf(MSG_DEBUG, "download path: '%s'\n", path);


    //Make sure the dataset and genome files are in the download directory
    log_messages.printf(MSG_DEBUG, "copying dataset filename to download directory: '%s'\n", dataset_filename.c_str());
    copy_file_to_download_dir(dataset_filename);

    string stripped_dataset_filename  = dataset_filename.substr(dataset_filename.find_last_of("/\\") + 1);
    log_messages.printf(MSG_DEBUG, "stripped dataset filename for infiles[0]: '%s'\n", stripped_dataset_filename.c_str());
    infiles[0] = stripped_dataset_filename.c_str();
    log_messages.printf(MSG_DEBUG, "infile[0]: '%s'\n", infiles[0]);


    log_messages.printf(MSG_DEBUG, "copying genome filename to download directory: '%s'\n", genome_filename.c_str());
    copy_file_to_download_dir(genome_filename);

    string stripped_genome_filename = genome_filename.substr(genome_filename.find_last_of("/\\") + 1);
    log_messages.printf(MSG_DEBUG, "stripped genome filename for infiles[1]: '%s'\n", stripped_genome_filename.c_str());
    infiles[1] = stripped_genome_filename.c_str();
    log_messages.printf(MSG_DEBUG, "infile[1]: '%s'\n", infiles[1]);


    double fpops_per_image = genome->get_number_weights() * 250;         //TODO: figure out an estimate of how many fpops per set calculation
    double fpops_est = exact->get_number_images() * genome->get_max_epochs() * fpops_per_image;

    double credit = fpops_est / 10e10;

    // Fill in the job parameters
    wu.clear();
    wu.appid = app.id;
    strcpy(wu.name, name);
    wu.rsc_fpops_est = fpops_est;
    wu.rsc_fpops_bound = fpops_est * 100;
    wu.rsc_memory_bound = 200 * 1024 * 1024;    //200MB
    wu.rsc_disk_bound = 200 * 1024 * 1024;      //200MB
    wu.delay_bound = 60 * 60 * 24 * 7;          //7 days
    wu.min_quorum = REPLICATION_FACTOR;
    wu.target_nresults = REPLICATION_FACTOR;
    wu.max_error_results = REPLICATION_FACTOR*4;
    wu.max_total_results = REPLICATION_FACTOR*8;
    wu.max_success_results = REPLICATION_FACTOR*4;

    // Register the job with BOINC
    sprintf(path, "templates/%s", out_template_file);

    sprintf(command_line, " --samples_file samples.bin --genome_file input_genome.txt --output_file output_genome.txt --checkpoint_file checkpoint.txt");
    
    //%u %u %lu %lu", max_set_value, set_size, starting_set, sets_to_evaluate);
    log_messages.printf(MSG_DEBUG, "command line: '%s'\n", command_line);

    sprintf(additional_xml, "<credit>%.3lf</credit>", credit);
    log_messages.printf(MSG_DEBUG, "credit: '%.3lf'\n", credit);

    log_messages.printf(MSG_DEBUG, "infiles[0]: '%s'\n", infiles[0]);
    log_messages.printf(MSG_DEBUG, "infiles[1]: '%s'\n", infiles[1]);

    return create_work(
        wu,
        in_template,
        path,
        config.project_path(path),
        infiles,
        2,
        config,
        command_line,
        additional_xml
    );
}

void make_jobs() {
    uint64_t total_generated = 0;

    //this assumes only one EXACT search is going on. 
    while (total_generated < WORKUNITS_TO_GENERATE) {
        CNN_Genome *genome = exact->generate_individual();

        make_job(genome);

        delete genome;
        total_generated++;
    }
    exact->export_to_database();

    exit(1);
}

void main_loop() {
    long number_unsent_results;

    while (1) {
        check_stop_daemons();

        int retval = count_unsent_results(number_unsent_results, app.id);

        if (retval) {
            log_messages.printf(MSG_CRITICAL,"count_unsent_jobs() failed: %s\n", boincerror(retval));
            exit(retval);
        }   
        log_messages.printf(MSG_DEBUG, "%lu results are available, with a cushion of %d\n", number_unsent_results, CUSHION);

        if (number_unsent_results > CUSHION) {
            log_messages.printf(MSG_DEBUG, "sleeping %d seconds\n", SLEEP_TIME);
            sleep(SLEEP_TIME);
        } else {
            log_messages.printf(MSG_DEBUG, "generating %d workunits.\n", WORKUNITS_TO_GENERATE);

            make_jobs();

            // Now sleep for a few seconds to let the transitioner
            // create instances for the jobs we just created.
            // Otherwise we could end up creating an excess of jobs.
            sleep(SLEEP_TIME);
        }   
    }   
}

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

    //looks for applicaiton to be run. If not found, program exits.
    char buf[256];
    sprintf(buf, "where name='%s'", app_name);
    if (app.lookup(buf)) {
        log_messages.printf(MSG_CRITICAL, "can't find app %s\n", app_name);
        exit(1);
    }

    //looks for work templates, if cannot find, or are corrupted,
    //the program exits.
    sprintf(buf, "templates/%s", in_template_file);
    if (read_file_malloc(config.project_path(buf), in_template)) {
        log_messages.printf(MSG_CRITICAL, "can't read input template '%s'\n", buf);
        exit(1);
    }

    sprintf(buf, "templates/%s", out_template_file);
    if (read_file_malloc(config.project_path(buf), out_template)) {
        log_messages.printf(MSG_CRITICAL, "can't read output template '%s'\n", buf);
        exit(1);
    }

    //initialize the EXACT algorithm
    int population_size = 200;
    int min_epochs = 100;
    int max_epochs = 100;
    int improvement_required_epochs = 5;
    bool reset_edges = true;
    int max_individuals = 10000;
    string output_directory = "/home/tdesell/exact_output/";

    Images images("/home/tdesell/mnist_training_data.bin");

    exact = new EXACT(images, population_size, min_epochs, max_epochs, improvement_required_epochs, reset_edges, max_individuals, output_directory, search_name);

    start_time = time(0);

    log_messages.printf(MSG_NORMAL, "starting at %d...\n", start_time);

    //initialize_exact_database();
    main_loop();
    
    return 0;
}

