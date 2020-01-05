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

#include "arguments.hxx"
#include "files.hxx"
#include "log.hxx"


using std::cerr;
using std::endl;

int32_t Log::std_message_level = INFO;
int32_t Log::file_message_level = INFO;
bool Log::write_to_file = true;
int32_t Log::max_header_length = 256;
int32_t Log::max_message_length = 1024;

string Log::output_directory = "./logs";

map<thread::id, string> Log::log_ids;
map<string, LogFile*> Log::output_files;

shared_mutex Log::log_ids_mutex;

LogFile::LogFile(FILE* _file) {
    file = _file;
}

void Log::register_command_line_arguments() {
    //CommandLine::create_group("Log", "");
    //CommandLine::
}

int8_t Log::parse_level_from_string(string level) {
    if (level.compare("0") == 0 || level.compare("NONE") == 0|| level.compare("none") == 0) {
        return Log::NONE;
    } else if (level.compare("1") == 0 || level.compare("FATAL") == 0 || level.compare("fatal") == 0) {
        return Log::FATAL;
    } else if (level.compare("2") == 0 || level.compare("ERROR") == 0 || level.compare("error") == 0) {
        return Log::ERROR;
    } else if (level.compare("3") == 0 || level.compare("WARNING") == 0 || level.compare("warning") == 0) {
        return Log::WARNING;
    } else if (level.compare("4") == 0 || level.compare("INFO") == 0 || level.compare("info") == 0) {
        return Log::INFO;
    } else if (level.compare("5") == 0 || level.compare("DEBUG") == 0 || level.compare("debug") == 0) {
        return Log::DEBUG;
    } else if (level.compare("6") == 0 || level.compare("TRACE") == 0 || level.compare("trace") == 0) {
        return Log::TRACE;
    } else if (level.compare("7") == 0 || level.compare("ALL") == 0 || level.compare("all") == 0) {
        return Log::ALL;
    } else {
        cerr << "ERROR: specified an incorrect message level for the Log: '" << level << "'" << endl;
        cerr << "Options are:" << endl;
        cerr << "\t0 or NONE or none" << endl;
        cerr << "\t1 or FATAL or fatal" << endl;
        cerr << "\t2 or ERROR or error" << endl;
        cerr << "\t3 or WARNING or warning" << endl;
        cerr << "\t4 or INFO or info" << endl;
        cerr << "\t5 or DEBUG or debug" << endl;
        cerr << "\t6 or TRACE or trace" << endl;
        cerr << "\t7 or ALL or all" << endl;
        exit(1);
    }
}

void Log::initialize(const vector<string> &arguments) {
    //TODO: should read these from the CommandLine (to be created)

    string std_message_level_str, file_message_level_str;

    get_argument(arguments, "--std_message_level", true, std_message_level_str);
    get_argument(arguments, "--file_message_level", true, file_message_level_str);
    get_argument(arguments, "--output_directory", true, output_directory);

    std_message_level = parse_level_from_string(std_message_level_str);
    file_message_level = parse_level_from_string(file_message_level_str);

    //cerr << "std_message_level: " << std_message_level << ", file_message_level: " << file_message_level << endl;

    get_argument(arguments, "--max_header_length", false, max_header_length);
    get_argument(arguments, "--max_message_length", false, max_message_length);


    mkpath(output_directory.c_str(), 0777);
}

void Log::set_id(string human_readable_id) {
    thread::id id = std::this_thread::get_id();

    //check and see if this human readable id has been set to a different thread id,
    //going to allow multiple threads to access the same log (for examm_mt)
    /*
    bool id_in_use = false;
    for (auto kv_pair = log_ids.begin(); kv_pair != log_ids.end(); kv_pair++) {
        thread::id other_thread_id = kv_pair->first;
        string other_thread_readable_id = kv_pair->second;

        if (human_readable_id == other_thread_readable_id) {
            cerr << "ERROR: thread '" << id << "' atempting to register human readable id '" << human_readable_id << "' which was already registered by thread '" << other_thread_id << "', if this was intended you need to have the other thread release the id with Log::release_id()" << endl;
        }
    }

    if (id_in_use) exit(1);
    */

    //cerr << "setting thread id " << id << " to human readable id: '" << human_readable_id << "'" << endl;

    log_ids_mutex.lock();

    log_ids[id] = human_readable_id;

    if (write_to_file) {
        //check and see if we've already opened a file for this human readable id
        if (output_files.count(human_readable_id) == 0) {
            string output_filename = output_directory + "/" + human_readable_id;
            FILE *outfile = fopen(output_filename.c_str(), "w");
            output_files[human_readable_id] = new LogFile(outfile);
        }
    }

    log_ids_mutex.unlock();
}

void Log::release_id() {
    thread::id id = std::this_thread::get_id();

    log_ids_mutex.lock();
    auto map_iter = log_ids.find(id);
    if (map_iter != log_ids.end()) {
        if (write_to_file) {
            //flush and close the output file for this id
            string human_readable_id = map_iter->second;

            //cerr << "releasing thread id " << id << " from human readable id: '" << human_readable_id << "'" << endl;

            if (output_files.count(human_readable_id) == 0) {
                cerr << "ERROR: log id '" << human_readable_id << "' was already released!" << endl;
                exit(1);
            }

            //TODO: determine if we should flush and close on release. For examm_mt the threads
            //actually will swap between use of 'main' as a thread id so flushing and closing
            //will not work in that case

            //FILE *outfile = output_files[human_readable_id];
            //fflush(outfile);
            //fclose(outfile);
        }

        log_ids.erase(map_iter);

    } else {
        cerr << "ERROR: thread '" << id << "' attemped to release it's human readable thread id without having previously set it." << endl;
        exit(1);
    }
    log_ids_mutex.unlock();
}


void Log::write_message(bool print_header, int8_t message_level, const char *message_type, const char *format, va_list arguments) {

    thread::id id = std::this_thread::get_id();

    if (log_ids.count(id) == 0) {
        cerr << "ERROR: could not write message from thread '" << id << "' because it did not have a human readable id assigned (please use the Log::set_id(string) function before writing to the Log on any thread)." << endl;
        cerr << "message:" << endl;
        vprintf(format, arguments);
        cerr << endl;
        exit(1);
    }

    string human_readable_id = log_ids[id];

    if (output_files.count(human_readable_id) == 0) {
        cerr << "ERROR: There was no log information for this human readable id '" << human_readable_id << "' from thread '" << id << "'. This should never happen." << endl;
        exit(1);
    }

    //print the message header into a string
    char header_buffer[max_header_length];
    //we only need to print the header for some messages
    if (print_header) {
        //snprintf(header_buffer, max_header_length, "[%-8s %-20s]", message_type, human_readable_id.c_str());
        snprintf(header_buffer, max_header_length, "[%-7s %-21s]", message_type, human_readable_id.c_str());
    }

    //print the actual message contents into a string
    char message_buffer[max_message_length];
    vsnprintf(message_buffer, max_message_length, format, arguments);

    if (std_message_level >= message_level) {
        if (print_header) {
            printf("%s %s", header_buffer, message_buffer);
        } else {
            printf("%s", message_buffer);
        }
    }

    if (file_message_level >= message_level) {
        LogFile* log_file = output_files[human_readable_id];

        //lock this log_file in case multiple threads are trying to write
        //to the same file
        log_file->file_mutex.lock();
        if (print_header) {
            fprintf(log_file->file, "%s %s", header_buffer, message_buffer);
        } else {
            fprintf(log_file->file, "%s", message_buffer);
        }
        fflush(log_file->file);
        log_file->file_mutex.unlock();
    }
}

bool Log::at_level(int8_t level) {
    return level >= std_message_level || level >= file_message_level;
}

void Log::fatal(const char *format, ...) {
    //not writing this type of message to either std out or a file
    if (std_message_level < FATAL && file_message_level < FATAL) return;

    va_list arguments;
    va_start(arguments, format);
    write_message(true, FATAL, "FATAL", format, arguments);
}

void Log::error(const char* format, ...) {
    //not writing this type of message to either std out or a file
    if (std_message_level < ERROR && file_message_level < ERROR) return;

    va_list arguments;
    va_start(arguments, format);
    write_message(true, ERROR, "ERROR", format, arguments);
}

void Log::warning(const char* format, ...) {
    //not writing this type of message to either std out or a file
    if (std_message_level < WARNING && file_message_level < WARNING) return;

    va_list arguments;
    va_start(arguments, format);
    write_message(true, WARNING, "WARNING", format, arguments);
}

void Log::info(const char* format, ...) {
    //not writing this type of message to either std out or a file
    if (std_message_level < INFO && file_message_level < INFO) return;

    va_list arguments;
    va_start(arguments, format);
    write_message(true, INFO, "INFO", format, arguments);
}

void Log::debug(const char* format, ...) {
    //not writing this type of message to either std out or a file
    if (std_message_level < DEBUG && file_message_level < DEBUG) return;

    va_list arguments;
    va_start(arguments, format);
    write_message(true, DEBUG, "DEBUG", format, arguments);
}

void Log::trace(const char* format, ...) {
    //not writing this type of message to either std out or a file
    if (std_message_level < TRACE && file_message_level < TRACE) return;

    va_list arguments;
    va_start(arguments, format);
    write_message(true, TRACE, "TRACE", format, arguments);
}

void Log::fatal_no_header(const char *format, ...) {
    //not writing this type of message to either std out or a file
    if (std_message_level < FATAL && file_message_level < FATAL) return;

    va_list arguments;
    va_start(arguments, format);
    write_message(false, FATAL, "FATAL", format, arguments);
}

void Log::error_no_header(const char* format, ...) {
    //not writing this type of message to either std out or a file
    if (std_message_level < ERROR && file_message_level < ERROR) return;

    va_list arguments;
    va_start(arguments, format);
    write_message(false, ERROR, "ERROR", format, arguments);
}

void Log::warning_no_header(const char* format, ...) {
    //not writing this type of message to either std out or a file
    if (std_message_level < WARNING && file_message_level < WARNING) return;

    va_list arguments;
    va_start(arguments, format);
    write_message(false, WARNING, "WARNING", format, arguments);
}

void Log::info_no_header(const char* format, ...) {
    //not writing this type of message to either std out or a file
    if (std_message_level < INFO && file_message_level < INFO) return;

    va_list arguments;
    va_start(arguments, format);
    write_message(false, INFO, "INFO", format, arguments);
}

void Log::debug_no_header(const char* format, ...) {
    //not writing this type of message to either std out or a file
    if (std_message_level < DEBUG && file_message_level < DEBUG) return;

    va_list arguments;
    va_start(arguments, format);
    write_message(false, DEBUG, "DEBUG", format, arguments);
}

void Log::trace_no_header(const char* format, ...) {
    //not writing this type of message to either std out or a file
    if (std_message_level < TRACE && file_message_level < TRACE) return;

    va_list arguments;
    va_start(arguments, format);
    write_message(false, TRACE, "TRACE", format, arguments);
}

