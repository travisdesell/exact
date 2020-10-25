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

//the following are the templates for the workunits (exact_in.xml) and results (exact_out.xml)
//and they can be found in /projects/csg/templates
const char* in_template_file = "exact_bn_fmp_in.xml";
const char* out_template_file = "exact_out.xml";


DB_APP app;

char* in_template;
char* out_template;
int daemon_start_time;

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
int make_job(EXACT *exact, CNN_Genome *genome, string search_name) {
    DB_WORKUNIT wu;

    char name[256], path[256];
    char command_line[512];
    char additional_xml[512];
    const char* infiles[4];

    // make a unique name (for the job and its input file)
    sprintf(name, "exact_genome_%d_%u_%u", daemon_start_time, exact->get_id(), genome->get_generation_id());
    log_messages.printf(MSG_DEBUG, "name: '%s'\n", name);

    string training_filename = exact->get_training_filename();
    string validation_filename = exact->get_validation_filename();
    string test_filename = exact->get_test_filename();

    ostringstream oss;
    oss << "/projects/csg/exact_data/" << search_name << "/" << name << ".txt";
    string genome_filename = oss.str();

    log_messages.printf(MSG_DEBUG, "training filename: '%s'\n", training_filename.c_str());
    log_messages.printf(MSG_DEBUG, "validation filename: '%s'\n", validation_filename.c_str());
    log_messages.printf(MSG_DEBUG, "test filename: '%s'\n", test_filename.c_str());
    log_messages.printf(MSG_DEBUG, "genome filename: '%s'\n", genome_filename.c_str());


    // Create the input file.
    // Put it at the right place in the download dir hierarchy
    //
    /*
    int retval = config.download_path(name, path);
    if (retval) return retval;
    log_messages.printf(MSG_DEBUG, "download path: '%s'\n", path);
    */

    //Make sure the dataset and genome files are in the download directory
    log_messages.printf(MSG_DEBUG, "copying training filename to download directory: '%s'\n", training_filename.c_str());
    copy_file_to_download_dir(training_filename);

    string stripped_training_filename  = training_filename.substr(training_filename.find_last_of("/\\") + 1);
    log_messages.printf(MSG_DEBUG, "stripped training filename for infiles[0]: '%s'\n", stripped_training_filename.c_str());
    infiles[0] = stripped_training_filename.c_str();
    log_messages.printf(MSG_DEBUG, "infile[0]: '%s'\n", infiles[0]);

    log_messages.printf(MSG_DEBUG, "copying validation filename to download directory: '%s'\n", validation_filename.c_str());
    copy_file_to_download_dir(validation_filename);

    string stripped_validation_filename  = validation_filename.substr(validation_filename.find_last_of("/\\") + 1);
    log_messages.printf(MSG_DEBUG, "stripped validation filename for infiles[1]: '%s'\n", stripped_validation_filename.c_str());
    infiles[1] = stripped_validation_filename.c_str();
    log_messages.printf(MSG_DEBUG, "infile[1]: '%s'\n", infiles[1]);

    log_messages.printf(MSG_DEBUG, "copying test filename to download directory: '%s'\n", test_filename.c_str());
    copy_file_to_download_dir(test_filename);

    string stripped_test_filename  = test_filename.substr(test_filename.find_last_of("/\\") + 1);
    log_messages.printf(MSG_DEBUG, "stripped test filename for infiles[2]: '%s'\n", stripped_test_filename.c_str());
    infiles[2] = stripped_test_filename.c_str();
    log_messages.printf(MSG_DEBUG, "infile[2]: '%s'\n", infiles[2]);

    string stripped_genome_filename = genome_filename.substr(genome_filename.find_last_of("/\\") + 1);
    log_messages.printf(MSG_DEBUG, "stripped genome filename for infiles[3]: '%s'\n", stripped_genome_filename.c_str());
    infiles[3] = stripped_genome_filename.c_str();
    log_messages.printf(MSG_DEBUG, "infile[3]: '%s'\n", infiles[3]);

    log_messages.printf(MSG_DEBUG, "writing genome filename to download directory: '%s'\n", genome_filename.c_str());
    int retval = config.download_path( stripped_genome_filename.c_str(), path );
    if (retval) {
        log_messages.printf(MSG_CRITICAL, "can't get download path for file '%s', error: %s\n", stripped_genome_filename.c_str(), boincerror(retval));
        exit(1);
    }   

    if ( std::ifstream(path) ) { 
        log_messages.printf(MSG_CRITICAL, "\033[1minput file '%s' already exists in download directory hierarchy as '%s', not copying.\033[0m\n", stripped_genome_filename.c_str(), path);
        exit(1);
    }

    log_messages.printf(MSG_DEBUG, "destination in the download path is '%s', writing genome file\n", path);
    genome->write_to_file(path);
    //copy_file_to_download_dir(genome_filename);

    double fpops_per_image = genome->get_operations_estimate();
    double fpops_est = exact->get_number_training_images() * genome->get_max_epochs() * fpops_per_image * 3.0;

    double credit = (fpops_est / 10e10) * 0.5;

    // Fill in the job parameters
    wu.clear();
    wu.appid = app.id;
    strcpy(wu.name, name);
    wu.rsc_fpops_est = fpops_est;
    wu.rsc_fpops_bound = fpops_est * 100;

    if (training_filename.find("cifar") != std::string::npos) {
        wu.rsc_memory_bound = 1500 * 1024 * 1024;    //200MB
    } else {
        wu.rsc_memory_bound = 200 * 1024 * 1024;    //200MB
    }

    wu.rsc_disk_bound = 200 * 1024 * 1024;      //200MB
    wu.delay_bound = 60 * 60 * 24 * (credit / 250);          //7 days
    wu.min_quorum = REPLICATION_FACTOR;
    wu.target_nresults = REPLICATION_FACTOR;
    wu.max_error_results = REPLICATION_FACTOR*4;
    wu.max_total_results = REPLICATION_FACTOR*8;
    wu.max_success_results = REPLICATION_FACTOR*4;

    // Register the job with BOINC
    sprintf(path, "templates/%s", out_template_file);

    sprintf(command_line, " --training_file training_samples.bin --validation_file validation_samples.bin --testing_file testing_samples.bin --genome_file input_genome.txt --output_file output_genome.txt --checkpoint_file checkpoint.txt");
    
    //%u %u %lu %lu", max_set_value, set_size, starting_set, sets_to_evaluate);
    log_messages.printf(MSG_DEBUG, "command line: '%s'\n", command_line);

    sprintf(additional_xml, "<credit>%.3lf</credit>", credit);
    log_messages.printf(MSG_DEBUG, "credit: '%.3lf'\n", credit);

    log_messages.printf(MSG_DEBUG, "infiles[0]: '%s'\n", infiles[0]);
    log_messages.printf(MSG_DEBUG, "infiles[1]: '%s'\n", infiles[1]);
    log_messages.printf(MSG_DEBUG, "infiles[2]: '%s'\n", infiles[2]);
    log_messages.printf(MSG_DEBUG, "infiles[3]: '%s'\n", infiles[3]);

    return create_work(
        wu,
        in_template,
        path,
        config.project_path(path),
        infiles,
        4,
        config,
        command_line,
        additional_xml
    );
}

bool low_on_workunits() {
    long number_unsent_results;

    int retval = count_unsent_results(number_unsent_results, app.id);

    if (retval) {
        log_messages.printf(MSG_CRITICAL,"count_unsent_jobs() failed: %s\n", boincerror(retval));
        exit(retval);
    }   
    log_messages.printf(MSG_DEBUG, "%lu results are available, with a cushion of %d\n", number_unsent_results, CUSHION);

    if (number_unsent_results > CUSHION) {
        return false;
    } else {
        return true;
    }
}

void make_jobs(EXACT *exact, int workunits_to_generate) {
    log_messages.printf(MSG_DEBUG, "generating %d workunits for exact search '%s' with id: %d\n", workunits_to_generate, exact->get_search_name().c_str(), exact->get_id());

    int64_t total_generated = 0;

    //this assumes only one EXACT search is going on. 
    while (total_generated < workunits_to_generate) {
        CNN_Genome *genome = exact->generate_individual();

        make_job(exact, genome, exact->get_search_name());

        delete genome;
        total_generated++;
    }
    exact->update_database();
}

void init_work_generation(string app_name) {
    //looks for applicaiton to be run. If not found, program exits.
    char buf[256];
    sprintf(buf, "where name='%s'", app_name.c_str());
    if (app.lookup(buf)) {
        log_messages.printf(MSG_CRITICAL, "can't find app %s\n", app_name.c_str());
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

    daemon_start_time = time(0);
}
