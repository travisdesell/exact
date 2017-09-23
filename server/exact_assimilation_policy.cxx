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
using std::fstream;

#include <sstream>
using std::stringstream;
using std::istringstream;
using std::ostringstream;
using std::ios;

#include <unordered_map>
using std::unordered_map;

#include <vector>
using std::vector;


#include "stdint.h"
#include "mysql.h"
#include "boinc_db.h"

#include "common/arguments.hxx"
#include "common/db_conn.hxx"
#include "common/files.hxx"
#include "common/version.hxx"
#include "cnn/exact.hxx"
#include "cnn/cnn_genome.hxx"
#include "server/make_jobs.hxx"


unordered_map<int, EXACT*> exact_searches;

EXACT* get_exact_search(int exact_id) {
    EXACT *search = NULL;
    if (exact_searches.count(exact_id) > 0) {
        search = exact_searches[exact_id];

    } else {
        search = new EXACT(exact_id);
        exact_searches[exact_id] = search;
    }

    return search;
}

int assimilate_handler_init(int argc, char** argv) {
    // handle project specific arguments here
    vector<string> arguments = vector<string>(argv, argv + argc);

    for (uint32_t i = 0; i < argc; i++) {
        cout << "argv[" << i << "]: '" << argv[i] << "'" << endl;
    }

    for (uint32_t i = 0; i < arguments.size(); i++) {
        cout << "arguments[" << i << "]: '" << arguments[i] << "'" << endl;
    }

    string db_file;
    get_argument(arguments, "--db_file", true, db_file);
    set_db_info_filename(db_file);

    string app_name;
    get_argument(arguments, "--app_name", true, app_name);

    init_work_generation(app_name);

    ostringstream running_search_query;

    running_search_query << "SELECT id FROM exact_search WHERE inserted_genomes < max_genomes";
    
    mysql_exact_query(running_search_query.str());

    MYSQL_RES *exact_result = mysql_store_result(exact_db_conn);

    //cout << "got exact result" << endl;

    MYSQL_ROW exact_row;
    while ((exact_row = mysql_fetch_row(exact_result)) != NULL) {
        int exact_id = atoi(exact_row[0]);
        cout << "got exact with id: " << exact_id << endl;
        get_exact_search(exact_id);
    }

    /*
    if (low_on_workunits()) {
        for (auto it = exact_searches.begin(); it != exact_searches.end(); ++it ) {
            make_jobs(it->second, WORKUNITS_TO_GENERATE / exact_searches.size());
            it->second->export_to_database();
        }
    }
    */

    return 0;
}

int after_assimilate_pass() {

    if (low_on_workunits()) {
        uint32_t active_searches = 0;
        for (auto it = exact_searches.begin(); it != exact_searches.end(); ++it ) {
            if (it->second->get_inserted_genomes() < it->second->get_max_genomes()) {
                active_searches++;
            }
        }

        if (active_searches > 0) {
            for (auto it = exact_searches.begin(); it != exact_searches.end(); ++it ) {
                if (it->second->get_inserted_genomes() < it->second->get_max_genomes()) {
                    make_jobs(it->second, WORKUNITS_TO_GENERATE / active_searches);

                    //this should not be needed due to updates after each workunit generation
                    //it->second->export_to_database();
                }
            }
        }

        /*
        cout << "updating all exact searches in database." << endl;
        for (auto it = exact_searches.begin(); it != exact_searches.end(); ++it ) {
            if (it->second->get_inserted_genomes() < it->second->get_max_genomes()) {
                it->second->update_database();
            }
        }
        cout << "finished." << endl;
        */
    }

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

    cout << "got output file info, files.size(): " << files.size() << endl;

    if (files.size() > 1) {
        log_messages.printf(MSG_CRITICAL, "[CANONICAL RESULT#%ld %s] had more than one output file: %zu\n", canonical_result.id, canonical_result.name, files.size());
        for (uint32_t i = 0; i < files.size(); i++) {
            log_messages.printf(MSG_CRITICAL, "    %s\n", files[i].path.c_str());
        }
        exit(1);
    } else if (files.size() == 0) {
        log_messages.printf(MSG_CRITICAL, "[CANONICAL RESULT#%ld %s] had more no output files: %zu\n", canonical_result.id, canonical_result.name, files.size());
        return 0;
    }

    cout << "checked file size" << endl;

    OUTPUT_FILE_INFO& fi = files[0];
    string file_contents;

    try {
        cout << "getting file as string: '" << fi.path << "'" << endl;
        file_contents = get_file_as_string(fi.path);
        cout << "got file as string, erasing carraige returns" << endl;

        file_contents.erase(std::remove(file_contents.begin(), file_contents.end(), '\r'), file_contents.end());
        cout << "erased carraige returns" << endl;

    } catch (int err) {
        log_messages.printf(MSG_CRITICAL, "[CANONICAL RESULT#%ld %s] assimilate_handler: could not open file for canonical_result\n", canonical_result.id, canonical_result.name);
        log_messages.printf(MSG_CRITICAL, "     file path: %s\n", fi.path.c_str());
        return 0;
        //return ERR_FOPEN;
    }

    cout << "creating istringstream" << endl;

    istringstream file_iss(file_contents);
    string version_line;
    file_iss >> version_line;

    int exact_id;
    file_iss >> exact_id;

    string genome_id;
    file_iss >> genome_id;

    if (version_line[0] != 'v') {
        log_messages.printf(MSG_CRITICAL, "[CANONICAL RESULT#%ld %s] assimilate_handler: result was from old app version without version string, ignoring.\n", canonical_result.id, canonical_result.name);
        return 0;
    }

    cout << "version string: '" << version_line << "'" << endl;
    cout << "exact_id: '" << exact_id << "'" << endl;
    cout << "genome_id: '" << genome_id << "'" << endl;

    stringstream test(canonical_result.name);
    string segment;
    vector<string> segment_list;

    while(getline(test, segment, '_')) {
        segment_list.push_back(segment);
    }

    if (segment_list.size() != 6) {
        log_messages.printf(MSG_CRITICAL, "[CANONICAL RESULT#%ld %s] assimilate_handler: result had invalid name, needed 6 substrings between _ character, but had %lu.\n", canonical_result.id, canonical_result.name, segment_list.size());
    }

    string exact_id_str = segment_list[3];
    cout << "exact_id substring: " << exact_id_str << endl;

    if (exact_id_str.find_first_not_of( "0123456789" ) != string::npos) {
        log_messages.printf(MSG_CRITICAL, "[CANONICAL RESULT#%ld %s] assimilate_handler: parsing exact_id from result name resulted in non-integer: '%s'.\n", canonical_result.id, canonical_result.name, exact_id_str.c_str());
        return 0;
    }

    if (exact_id_str.size() > 5) {
        log_messages.printf(MSG_CRITICAL, "[CANONICAL RESULT#%ld %s] assimilate_handler: exact_id was invalid: '%s'.\n", canonical_result.id, canonical_result.name, exact_id_str.c_str());
        return 0;
    }

    exact_id = stoi(exact_id_str);

    if (exact_id < 0) {
        log_messages.printf(MSG_CRITICAL, "[CANONICAL RESULT#%ld %s] assimilate_handler: result had no exact_id (exact_id: %d), ignoring.\n", canonical_result.id, canonical_result.name, exact_id);
        return 0;
    }

    if (!EXACT::exists_in_database(exact_id)) {
        log_messages.printf(MSG_CRITICAL, "[CANONICAL RESULT#%ld %s] assimilate_handler: exact_id (exact_id: %d) was not in the database, ignoring.\n", canonical_result.id, canonical_result.name, exact_id);
        return 0;
     }

    file_iss.clear();
    file_iss.seekg(0, ios::beg);

    CNN_Genome *genome = NULL;
   
    try {
        genome = new CNN_Genome(file_iss, false);
    } catch (std::invalid_argument exception) {
        log_messages.printf(MSG_CRITICAL, "[CANONICAL RESULT#%ld %s] assimilate_handler: caught invalid_argument exception while generating genome: '%s'.\n", canonical_result.id, canonical_result.name, exception.what());
        return 0;
    } catch (std::runtime_error exception) {
        log_messages.printf(MSG_CRITICAL, "[CANONICAL RESULT#%ld %s] assimilate_handler: caught runtime error exception while generating genome: '%s'.\n", canonical_result.id, canonical_result.name, exception.what());
        return 0;
    }

    if (genome->get_version_str().compare(EXACT_VERSION_STR) != 0) {
        log_messages.printf(MSG_CRITICAL, "[CANONICAL RESULT#%ld %s] assimilate_handler: result was from an old version input file: '%s', expected '%s'.\n", canonical_result.id, canonical_result.name, genome->get_version_str().c_str(), EXACT_VERSION_STR);
        //exit(1);

        delete genome;
        return 0;
    }

    //cout << "result.stderr_out:\n" << canonical_result.stderr_out << endl << endl;

    EXACT *exact = get_exact_search(exact_id);
    bool was_inserted = exact->insert_genome(genome);

    if (was_inserted) {
        cout << "exporting genome to database with exact id: " << exact->get_id() << endl;
        genome->export_to_database(exact->get_id());

        ostringstream genome_update;
        genome_update << "UPDATE cnn_genome SET stderr_out = \"" << canonical_result.stderr_out << "\" WHERE id = " << genome->get_genome_id();
        mysql_exact_query(genome_update.str());
    }

    cout << "updating exact in database" << endl;
    exact->update_database();
    cout << "updated exact" << endl;

    //exact->export_to_database();
    return 0;
}
