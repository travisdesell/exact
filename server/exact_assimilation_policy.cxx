/*
 * Copyright 2012, 2009 Travis Desell and the University of North Dakota.
 *
 * This file is part of the Toolkit for Asynchronous Optimization (TAO).
 *
 * TAO is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * TAO is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with TAO.  If not, see <http://www.gnu.org/licenses/>.
 * */

#include <math.h>
#include <sys/stat.h>
#include <sys/param.h>

#include "backend_lib.h"
#include "config.h"
#include "util.h"
#include "sched_config.h"
#include "sched_util.h"
#include "sched_msgs.h"
#include "md5_file.h"
#include "error_numbers.h"
#include "validate_util.h"
#include "str_replace.h"
#include "str_util.h"


#include <cstdlib>
#include <string>
using std::string;

#include <fstream>

#include <sstream>
using std::istringstream;

#include <unordered_map>
using std::unordered_map;

#include <vector>
using std::vector;


#include "stdint.h"
#include "mysql.h"
#include "boinc_db.h"

#include "common/db_conn.hxx"
#include "strategy/exact.hxx"
#include "strategy/cnn_genome.hxx"
#include "server/boinc_common.hxx"

#define REPLICATION_FACTOR  2

bool in_template_initialized = false;
char *in_template;

unordered_map<int, EXACT*> exact_searches;

int assimilate_handler_init(int argc, char** argv) {
    // handle project specific arguments here
    return 0;
}

void assimilate_handler_usage() {
    // describe the project specific arguments here
    // currently no project specific arguments

    /*
    fprintf(stderr,
        "    Custom options:\n"
        "    --stderr_string X     accept task if X is present in stderr_out\n"
        "    [--reject_if_present] reject (inassimilate) the task if X is present\n"
    );
    */
}


//returns 0 on success
int assimilate_handler(WORKUNIT& wu, vector<RESULT>& results, RESULT& canonical_result) {
    cerr << "in assimilate handler!" << endl;
    vector<OUTPUT_FILE_INFO> files;

    int retval = get_output_file_infos(canonical_result, files);
    if (retval) {
        log_messages.printf(MSG_CRITICAL, "[CANONICAL RESULT#%ld %s] check_set: can't get output filenames\n", canonical_result.id, canonical_result.name);
        return retval;
    }

    if (files.size() > 1) {
        log_messages.printf(MSG_CRITICAL, "[CANONICAL RESULT#%ld %s] had more than one output file: %zu\n", canonical_result.id, canonical_result.name, files.size());
        for (uint32_t i = 0; i < files.size(); i++) {
            log_messages.printf(MSG_CRITICAL, "    %s\n", files[i].path.c_str());
        }
        exit(1);
    }

    OUTPUT_FILE_INFO& fi = files[0];
    string file_contents;

    try {
        file_contents = get_file_as_string(fi.path);
        file_contents.erase(std::remove(file_contents.begin(), file_contents.end(), '\r'), file_contents.end());
    } catch (int err) {
        log_messages.printf(MSG_CRITICAL, "[CANONICAL RESULT#%ld %s] assimilate_handler: could not open file for canonical_result\n", canonical_result.id, canonical_result.name);
        log_messages.printf(MSG_CRITICAL, "     file path: %s\n", fi.path.c_str());
        return ERR_FOPEN;
    }

    istringstream file_iss(file_contents);
    string version_line;
    file_iss >> version_line;

    cout << "version string: '" << version_line << "'" << endl;
    exit(1);

    file_iss.clear();
    file_iss.seekg(0,ios::beg);

    CNN_Genome *genome = new CNN_Genome(file_iss, false);

    exit(1);
    return 0;
}
