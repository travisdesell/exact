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

#include <iostream>
#include <fstream>
#include <sstream>

//from undvc_commmon
#include "parse_xml.hxx"
#include "file_io.hxx"

using std::string;
using std::vector;
using std::ifstream;

struct SSS_RESULT {
    uint32_t checksum;
    vector<uint64_t> failed_sets;
};

int init_result(RESULT& result, void*& data) {
    int retval;
    vector<OUTPUT_FILE_INFO> files;

    retval = get_output_file_infos(result, files);
    if (retval) {
        log_messages.printf(MSG_CRITICAL, "[RESULT#%d %s] check_set: can't get output filenames\n", result.id, result.name);
        return retval;
    }

    if (files.size() > 1) {
        log_messages.printf(MSG_CRITICAL, "[RESULT#%d %s] had more than one output file: %zu\n", result.id, result.name, files.size());
        for (uint32_t i = 0; i < files.size(); i++) {
            log_messages.printf(MSG_CRITICAL, "    %s\n", files[i].path.c_str());
        }
        exit(1);
    }

    OUTPUT_FILE_INFO& fi = files[0];
    if (fi.no_validate) {
        log_messages.printf(MSG_CRITICAL, "[RESULT#%d %s] had file set to no validate: %s\n", result.id, result.name, fi.path.c_str());
        exit(1);
        //continue;
    }

    string fc;

    try {
        fc = get_file_as_string(fi.path);
    } catch (int err) {
        log_messages.printf(MSG_CRITICAL, "[RESULT#%d %s] get_data_from_result: could not open file for result\n", result.id, result.name);
        log_messages.printf(MSG_CRITICAL, "     file path: %s\n", fi.path.c_str());
        return ERR_FOPEN;
    }

//    cout << "Parsing: " << endl << fc << endl;

    SSS_RESULT* sss_result = new SSS_RESULT;
    try {
        sss_result->checksum = parse_xml<uint32_t>(fc, "checksum");

//        cout << "checksum: " << sss_result->checksum << endl;

        parse_xml_vector<uint64_t>(fc, "failed_subsets", sss_result->failed_sets);

//        cout << "failed subsets size: " << sss_result->failed_sets.size() << endl;
    } catch (string error_message) {
        log_messages.printf(MSG_CRITICAL, "sss_validation_policy get_data_from_result([RESULT#%d %s]) failed with error: %s\n", result.id, result.name, error_message.c_str());
        log_messages.printf(MSG_CRITICAL, "XML:\n%s\n", fc.c_str());
//        result.outcome = RESULT_OUTCOME_VALIDATE_ERROR;
//        result.validate_state = VALIDATE_STATE_INVALID;
        return ERR_XML_PARSE;
//        exit(1);
//        throw 0;
    }

    data = (void*) sss_result;
    return 0;
}

int compare_results(
    RESULT & r1, void* data1,
    RESULT const& r2, void* data2,
    bool& match
) {
    SSS_RESULT* f1 = (SSS_RESULT*) data1;
    SSS_RESULT* f2 = (SSS_RESULT*) data2;

    if (f1->checksum == f2->checksum) {
        
        if (f1->failed_sets.size() == f2->failed_sets.size()) {
            bool all_match = true;
            for (unsigned int i = 0; i < f1->failed_sets.size(); i++) {
                if (f1->failed_sets[i] != f2->failed_sets[i]) {
                    log_messages.printf(MSG_CRITICAL, "[RESULT#%d %s] and [RESULT#%d %s] failed sets[%u] did not match %lu vs %lu\n", r1.id, r1.name, r2.id, r2.name, i, f1->failed_sets[i], f2->failed_sets[i]);
                    all_match = false;
                    //exit(1);
                    break;
                }
            }

            if (all_match) {
                match = true;
            } else {
                match = false;
                //exit(1);
            }
        } else {
            match = false;
            log_messages.printf(MSG_CRITICAL, "[RESULT#%d %s] and [RESULT#%d %s] failed sets had different sizes %zu vs %zu\n", r1.id, r1.name, r2.id, r2.name, f1->failed_sets.size(), f2->failed_sets.size());
            //exit(1);
        }
    } else {
        match = false;
        log_messages.printf(MSG_CRITICAL, "[RESULT#%d %s] and [RESULT#%d %s] failed sets had different checksums %u vs %u\n", r1.id, r1.name, r2.id, r2.name, f1->checksum, f2->checksum);
//        exit(1);
    }

    return 0;
}

int cleanup_result(RESULT const& /*result*/, void* data) {
    SSS_RESULT *sss_result = (SSS_RESULT*)data;
//    delete sss_result->failed_sets;
    delete sss_result;

    return 0;
}

const char *BOINC_RCSID_7ab2b7189c = "$Id: sample_bitwise_validator.cpp 21735 2010-06-12 22:08:15Z davea $";
