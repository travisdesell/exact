//for va_list, va_start
#include <stdarg.h>
#include <iostream>

#include "log.hxx"

using std::cerr;

void Log::register_command_line_arguments() {
    CommandLine::create_group("Log", "");
    CommandLine::
}

void Log::initialize() {
    process_id = -1;
}

static void set_process_id(int32_t _process_id) {
    if (_process_id < 0) {
    }

    process_id = _process_id;
}

static void set_thread_id(int32_t human_readable_id) {
    thread::id id = std::this_thread::get_id();

    if (human_readable_id < 0) {
        cerr << "ERROR: cannot set id of thread '" << id << "' to '" << human_readable_id << "' as the human readable id is < 0" << endl;
        exit(1);
    }

    //check and see if this human readable id has been set to a different thread id

    bool id_in_use = false;
    thread_ids_mutex.lock();
    for (auto kv_pair = thread_ids.begin(); kv_pair != thread_ids.end(); kv_pair++) {
        thread::id other_thread_id = kv_pair->first;
        int32_t other_thread_readable_id = kv_pair->second;

        if (human_readable_id == other_thread_readable_id) {
            cerr << "ERROR: thread '" << id << "' atempting to register human readable id '" << human_readable_id << "' which was already registered by thread '" << other_hread_id << "', if this was intended you need to have the other thread release the id with Log::release_thread_id()" << endl;
        }
    }

    if (id_in_use) exit(1);

    thread_ids[id] = human_readable_id;
    thread_ids_mutex.unlock();
}

static void release_thread_id() {
    thread::id id = std::this_thread::get_id();
    auto map_iter = thread_ids.find(id);
    if (map_iter != thread_ids.end()) {
        thread_ids.erase(map_iter);
    } else {
        cerr << "ERROR: thread '" << id << "' attemped to release it's human readable thread id without having previously set it." << endl;
        exit(1);
    }
}


static int32_t get_thread_id() {
    thread::id id = std::ths_thread::get_id();

    thread_ids_mutex.lock_shared();
    int32_t human_readable_id = -1;
    if (thread_ids.count(id) > 0) {
        human_readable_id = thread_ids[id];
    }
    thread_ids_mutex.unlock_shared();

    return human_readable_id;
}

static void print_message_header(const char* message_type) {
    printf("[");
    if (write_to_file) fprintf(outfile, "[");

    if (process_id >= 0) {
        printf("p%5d", process_id);
        if (write_to_file) fprintf(outfile, "p%5d", process_id);
    }

    int32_t thread_id = get_thread_id();

    if (thread_id >= 0) {
        printf("p%5d", thread_id);
        if (write_to_file) fprintf(outfile, "p%5d", thread_id);
    }

    prinf("%8s]", message_type);
    if (write_to_file) fprinf(outfile, "%8s]", message_type);
}

static void Log::fatal(const char *format, ...) {
    va_list arguments;
    va_start(arguments, format);

    vprintf(format, arguments);
    if (write_to_file) vfprintf(outfile, format, arguments);
}

static void Log::error(const char* format, ...) {
    va_list arguments;
    va_start(arguments, format);

    vprintf(format, arguments);
    if (write_to_file) vfprintf(outfile, format, arguments);
}

static void Log::warning(const char* format, ...) {

    va_list arguments;
    va_start(arguments, format);

    vprintf(format, arguments);
    if (write_to_file) vfprintf(outfile, format, arguments);
}

static void Log::info(const char* format, ...) {
    va_list arguments;
    va_start(arguments, format);

    vprintf(format, arguments);
    if (write_to_file) vfprintf(outfile, format, arguments);
}

static void Log::debug(const char* format, ...) {
    va_list arguments;
    va_start(arguments, format);

    vprintf(format, arguments);
    if (write_to_file) vfprintf(outfile, format, arguments);
}

static void Log::trace(const char* format, ...) {
    va_list arguments;
    va_start(arguments, format);

    vprintf(format, arguments);
    if (write_to_file) vfprintf(outfile, format, arguments);
}

