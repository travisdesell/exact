#include <cstdio>
using std::fprintf;
using std::printf;
using std::snprintf;
using std::vsnprintf;

//for va_list, va_start
#include <stdarg.h>

#include <iostream>
using std::ofstream;

#include <thread>
using std::thread;

#include "log.hxx"
#include "files.hxx"

using std::cerr;
using std::endl;

int32_t Log::message_level = INFO;
bool Log::write_to_file = true;
int32_t Log::max_header_length = 256;
int32_t Log::max_message_length = 1024;

string Log::output_directory = "./logs";

map<thread::id, string> Log::log_ids;
map<string, FILE*> Log::output_files;

shared_mutex Log::log_ids_mutex;

void Log::register_command_line_arguments() {
    //CommandLine::create_group("Log", "");
    //CommandLine::
}

void Log::initialize() {
    //TODO: should read these from the CommandLine (to be created)

    mkpath(output_directory.c_str(), 0777);
}

void Log::set_id(string human_readable_id) {
    thread::id id = std::this_thread::get_id();

    //check and see if this human readable id has been set to a different thread id,
    //going to allow multiple threads to access the same log (for examm_mt)
    /*
    bool id_in_use = false;
    log_ids_mutex.lock();
    for (auto kv_pair = log_ids.begin(); kv_pair != log_ids.end(); kv_pair++) {
        thread::id other_thread_id = kv_pair->first;
        string other_thread_readable_id = kv_pair->second;

        if (human_readable_id == other_thread_readable_id) {
            cerr << "ERROR: thread '" << id << "' atempting to register human readable id '" << human_readable_id << "' which was already registered by thread '" << other_thread_id << "', if this was intended you need to have the other thread release the id with Log::release_id()" << endl;
        }
    }

    if (id_in_use) exit(1);
    */

    log_ids[id] = human_readable_id;

    if (write_to_file) {
        //check and see if we've already opened a file for this human readable id
        if (output_files.count(human_readable_id) == 0) {
            string output_filename = output_directory + "/" + human_readable_id;
            FILE *outfile = fopen(output_filename.c_str(), "w");
            output_files[human_readable_id] = outfile;
        }
    }

    log_ids_mutex.unlock();
}

void Log::release_id() {
    thread::id id = std::this_thread::get_id();

    auto map_iter = log_ids.find(id);
    if (map_iter != log_ids.end()) {
        if (write_to_file) {
            //flush and close the output file for this id
            string human_readable_id = map_iter->second;

            if (output_files.count(human_readable_id) == 0) {
                cerr << "ERROR: log id '" << human_readable_id << "' was already released!" << endl;
                exit(1);
            }

            FILE *outfile = output_files[human_readable_id];
            fflush(outfile);
            fclose(outfile);
        }

        log_ids.erase(map_iter);

    } else {
        cerr << "ERROR: thread '" << id << "' attemped to release it's human readable thread id without having previously set it." << endl;
        exit(1);
    }
}


void Log::write_message(const char *message_type, const char *format, va_list arguments) {
    thread::id id = std::this_thread::get_id();

    if (log_ids.count(id) == 0) {
        cerr << "ERROR: could not write message from thread '" << id << "' because it did not have a human readable id assigned (please use the Log::set_id(string) function before writing to the Log on any thread)." << endl;
        exit(1);
    }

    string human_readable_id = log_ids[id];

    if (output_files.count(human_readable_id) == 0) {
        cerr << "ERROR: There was no log information for this human readable id '" << human_readable_id << "' from thread '" << id << "'. This should never happen." << endl;
        exit(1);
    }

    //print the message header into a string
    char header_buffer[max_header_length];
    //snprintf(header_buffer, max_header_length, "[%-8s %-20s]", message_type, human_readable_id.c_str());
    snprintf(header_buffer, max_header_length, "[%s %s]", message_type, human_readable_id.c_str());

    //print the actual message contents into a string
    char message_buffer[max_message_length];
    vsnprintf(message_buffer, max_message_length, format, arguments);

    printf("%s %s\n", header_buffer, message_buffer);

    if (write_to_file) {
        FILE *outfile = output_files[human_readable_id];
        fprintf(outfile, "%s %s\n", header_buffer, message_buffer);
        fflush(outfile);
    }
}

void Log::fatal(const char *format, ...) {
    va_list arguments;
    va_start(arguments, format);
    if (message_level >= FATAL) write_message("FATAL", format, arguments);
}

void Log::error(const char* format, ...) {
    va_list arguments;
    va_start(arguments, format);
    if (message_level >= ERROR) write_message("ERROR", format, arguments);
}

void Log::warning(const char* format, ...) {
    va_list arguments;
    va_start(arguments, format);
    if (message_level >= WARNING) write_message("WARNING", format, arguments);
}

void Log::info(const char* format, ...) {
    va_list arguments;
    va_start(arguments, format);
    if (message_level >= INFO) write_message("INFO", format, arguments);
}

void Log::debug(const char* format, ...) {
    va_list arguments;
    va_start(arguments, format);
    if (message_level >= DEBUG) write_message("DEBUG", format, arguments);
}

void Log::trace(const char* format, ...) {
    va_list arguments;
    va_start(arguments, format);
    if (message_level >= TRACE) write_message("TRACE", format, arguments);
}

