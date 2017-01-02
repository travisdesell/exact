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

// A sample validator that requires a majority of results to be
// bitwise identical.
// This is useful only if either
// 1) your application does no floating-point math, or
// 2) you use homogeneous redundancy

#include "config.h"
#include "util.h"
#include "sched_util.h"
#include "sched_msgs.h"
#include "validate_util.h"
#include "md5_file.h"
#include "error_numbers.h"
#include "stdint.h"

#include <algorithm>

#include <iomanip>
using std::setw;

#include <iostream>
using std::cout;
using std::endl;
using std::istreambuf_iterator;

#include <fstream>
using std::ifstream;
using std::ios;

#include <sstream>
using std::istringstream;
using std::ostringstream;

#include <string>
using std::string;

#include <vector>
using std::vector;

#include "server/boinc_common.hxx"


struct EXACT_RESULT {
    string file_contents;
};

vector<char*> stderr_strings;
bool reject_if_present = false;

int validate_handler_init(int argc, char** argv) {
    // handle project specific arguments here
    return 0;
}

void validate_handler_usage() {
    // describe the project specific arguments here
    // currently no project specific arguments

    /*
    fprintf(stderr,
        "    Custom options:\n"
        "    --stderr_string X     accept task if X is present in stderr_out\n"
        "    [--reject_if_present] reject (invalidate) the task if X is present\n"
    );
    */
}


int init_result(RESULT& result, void*& data) {
    int retval;
    vector<OUTPUT_FILE_INFO> files;

    retval = get_output_file_infos(result, files);
    if (retval) {
        log_messages.printf(MSG_CRITICAL, "[RESULT#%ld %s] check_set: can't get output filenames\n", result.id, result.name);
        return retval;
    }

    if (files.size() > 1) {
        log_messages.printf(MSG_CRITICAL, "[RESULT#%ld %s] had more than one output file: %zu\n", result.id, result.name, files.size());
        for (uint32_t i = 0; i < files.size(); i++) {
            log_messages.printf(MSG_CRITICAL, "    %s\n", files[i].path.c_str());
        }
        exit(1);
    }

    OUTPUT_FILE_INFO& fi = files[0];
    if (fi.no_validate) {
        log_messages.printf(MSG_CRITICAL, "[RESULT#%ld %s] had file set to no validate: %s\n", result.id, result.name, fi.path.c_str());
        exit(1);
        //continue;
    }

    string file_contents;

    try {
        file_contents = get_file_as_string(fi.path);
    } catch (int err) {
        log_messages.printf(MSG_CRITICAL, "[RESULT#%ld %s] get_data_from_result: could not open file for result\n", result.id, result.name);
        log_messages.printf(MSG_CRITICAL, "     file path: %s\n", fi.path.c_str());
        return ERR_FOPEN;
    }

//    cout << "Parsing: " << endl << file_contents << endl;

    EXACT_RESULT* exact_result = new EXACT_RESULT;
    try {
        file_contents.erase(std::remove(file_contents.begin(), file_contents.end(), '\r'), file_contents.end());

        exact_result->file_contents = file_contents;

//        log_messages.printf(MSG_CRITICAL, "[RESULT#%ld %s] result file contents:\n%s\n", result.id, result.name, exact_result->file_contents.c_str());

    } catch (string error_message) {
        log_messages.printf(MSG_CRITICAL, "exact_validation_policy get_data_from_result([RESULT#%ld %s]) failed with error: %s\n", result.id, result.name, error_message.c_str());
        log_messages.printf(MSG_CRITICAL, "XML:\n%s\n", file_contents.c_str());
//        result.outcome = RESULT_OUTCOME_VALIDATE_ERROR;
//        result.validate_state = VALIDATE_STATE_INVALID;
        exit(1);
        return ERR_XML_PARSE;
//        exit(1);
//        throw 0;
    }

    data = (void*) exact_result;
    return 0;
}

int compare_results(
    RESULT & r1, void* data1,
    RESULT const& r2, void* data2,
    bool& match
) {
    EXACT_RESULT* f1 = (EXACT_RESULT*) data1;
    EXACT_RESULT* f2 = (EXACT_RESULT*) data2;

    if (f1->file_contents.compare(f2->file_contents) == 0) {
        match = true;
    } else {
        match = false;
        log_messages.printf(MSG_CRITICAL, "[RESULT#%ld %s] and [RESULT#%ld %s] failed sets had different file contents.\n", r1.id, r1.name, r2.id, r2.name);

        vector<OUTPUT_FILE_INFO> files;

        int retval = get_output_file_infos(r1, files);
        log_messages.printf(MSG_CRITICAL, "[RESULT#%ld %s] files: %zu\n", r1.id, r1.name, files.size());
        for (uint32_t i = 0; i < files.size(); i++) {
            log_messages.printf(MSG_CRITICAL, "    %s\n", files[i].path.c_str());
        }

        retval = get_output_file_infos(r2, files);
        log_messages.printf(MSG_CRITICAL, "[RESULT#%ld %s] files: %zu\n", r2.id, r2.name, files.size());
        for (uint32_t i = 0; i < files.size(); i++) {
            log_messages.printf(MSG_CRITICAL, "    %s\n", files[i].path.c_str());
        }

        istringstream iss1(f1->file_contents);
        istringstream iss2(f2->file_contents);


        string version_line1, version_line2;
        getline(iss1, version_line1);
        getline(iss2, version_line2);
        cout << setw(5) << 0 << setw(30) << ("'" + version_line1 + "'") << setw(30) << ("'" + version_line2 + "'") << endl;

        if (version_line1.compare(version_line2) != 0) {
            cout << "versions are different: '" << version_line1 << "' vs. '" << version_line2 << "'" << endl;
            //exit(1);
        }

        int fitness_line_number = 12;
        if (version_line1[0] == 'v') {
            fitness_line_number = 15;
        }

        string line1, line2;
        string fitness_line1, fitness_line2;
        for (uint32_t i = 1; i < 20; i++) {
            getline(iss1, line1);
            getline(iss2, line2);

            if (i == fitness_line_number) {
                fitness_line1 = line1;
                fitness_line2 = line2;
            }

            cout << setw(5) << i << setw(30) << ("'" + line1 + "'") << setw(30) << ("'" + line2 + "'") << endl;
        }

        double fitness1 = stof(fitness_line1);
        double fitness2 = stof(fitness_line2);

        cout << "fitness 1: " << fitness1 << endl;
        cout << "fitness 2: " << fitness2 << endl;

        if (fabs(fitness1 - fitness2) < 1000.0) {
            //close enough
            match = true;
        } else {
            match = false;
            //exit(1);
        }
    }

    return 0;
}

int cleanup_result(RESULT const& /*result*/, void* data) {
    EXACT_RESULT *exact_result = (EXACT_RESULT*)data;
    delete exact_result;
    return 0;
}

const char *BOINC_RCSID_7ab2b7189c = "$Id: sample_bitwise_validator.cpp 21735 2010-06-12 22:08:15Z davea $";
