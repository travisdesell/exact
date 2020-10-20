#ifndef EXONA_LOG
#define EXONA_LOG

#include <cstdio>  

#include <iostream>
using std::ofstream;

#include <map>
using std::map;

#include <mutex>
using std::mutex;

#include <shared_mutex>
using std::shared_mutex;

#include <string>
using std::string;

#include <thread>
using std::thread;

#include "string_format.hxx"

typedef enum log_level {
    LOG_LEVEL_NONE = 0,
    LOG_LEVEL_FATAL = 20,
    LOG_LEVEL_ERROR = 40,
    LOG_LEVEL_WARNING = 60,
    LOG_LEVEL_INFO = 80,
    LOG_LEVEL_DEBUG = 100,
    LOG_LEVEL_TRACE = 120,
    LOG_LEVEL_ALL = 140
} log_level_t;

#define fatal(format, ...) \
    log(__FILE__, sizeof(__FILE__)-1, __func__, sizeof(__func__)-1, __LINE__, \
    LOG_LEVEL_FATAL, format, ##__VA_ARGS__)
#define error(format, ...) \
    log(__FILE__, sizeof(__FILE__)-1, __func__, sizeof(__func__)-1, __LINE__, \
    LOG_LEVEL_ERROR, format, ##__VA_ARGS__)
#define warning(format, ...) \
    log(__FILE__, sizeof(__FILE__)-1, __func__, sizeof(__func__)-1, __LINE__, \
    LOG_LEVEL_WARNING, format, ##__VA_ARGS__)
#define info(format, ...) \
    log(__FILE__, sizeof(__FILE__)-1, __func__, sizeof(__func__)-1, __LINE__, \
    LOG_LEVEL_INFO, format, ##__VA_ARGS__)
#define debug(format, ...) \
    log(__FILE__, sizeof(__FILE__)-1, __func__, sizeof(__func__)-1, __LINE__, \
    LOG_LEVEL_DEBUG, format, ##__VA_ARGS__)
#define trace(format, ...) \
    log(__FILE__, sizeof(__FILE__)-1, __func__, sizeof(__func__)-1, __LINE__, \
    LOG_LEVEL_TRACE, format, ##__VA_ARGS__)

class LogFile {
    private:
        FILE* file;
        mutex file_mutex;

    public:
        LogFile(FILE* file);

    friend class Log;
};

class Log {
    private:
        /**
         * Specifies which messages to log.
         * Messages will be written to the standard output log if their type is <= message_level.
         */
        static log_level_t std_message_level;

        /**
         * Specifies which messages to log.
         * Messages will be written to the log file if their type is <= message_level.
         */

        static log_level_t file_message_level;

        /**
         * Specifies if the logs should also be written to a flie.
         */
        static bool write_to_file;

        /**
         * Specifies the maximum length for the message header.
         */
        static int32_t max_header_length;

        /**
         * Specifies the maximum message buffer length.
         */
        static int32_t max_message_length;

        /**
         * Specifies which directory to write the log files to (if the logs
         * are being writtten to a file).
         */
        static string output_directory;

        /**
         * A map of C++ thread ids (which are not human readable) to human
         * readable integer ids.
         * Thread ids are specified with the Log::set_id(string) 
         * method. 
         */
        static map<thread::id, string> log_ids;

        /**
         *  The MPI process rank for this Log instance. Set to -1 if not specified or not using MPI.
         */
        static int32_t process_rank;

        /**
         * Defaults at -1, when set to a process rank (>= 0) the Log will only print messages if its rank is the same as the restricted rank. This is useful, for allowing initialization messages to only be printed out by a single process.
         */
        static int32_t restricted_rank;

        /**
         * A map of human readable ids to output files which the log messages
         * will be written to.
         */
        static map<string, LogFile*> output_files;

        /**
         * A std::shared_mutex protecting the Log::thread_ids map.
         * The Log::set_thread_id(int32_t) mehod needs write access so it will use the std::shared_mutex::lock()x and std::shared_mutex()::unlock() mechanism, while the Log::get_thread_id() only needs read access and can used the std::shared_mutex::lock_shared() and std::shared_mutex::unlock_shared() methods.
         */
        static shared_mutex log_ids_mutex;


    public:
        static string get_level_str(log_level_t level);

        /**
         * Registers used command line arguments and instructions with the CommandLine class.
         * Needs to be called at the beginning of the main method of any programming using the Log class. Will register the following command line arguments and instructions:
         *  -# log_level : specifies the max message level of messages to be printed
         *  -# log_directory : specifies the output directory for message logs
         */
        static void register_command_line_arguments();

        /**
         * \param a string representation of the message level, either as text (e.g., "INFO") or as a number (e.g., 4)
         *
         * \return the message level as an int8_t (i.e., one of the message level constants)
         */
        static log_level_t parse_level_from_string(string level);

        /**
         *  Initializes the Log given arguments retreived from the CommandLine class.
         *  \param arguments the command line arguments
         *  Log::register_command_line_arguments() must be called before Log::initialize()
         */
        static void initialize(const vector<string> &arguments);

        /**
         * Sets the MPI process rank for this Log.
         *
         * \param _process_rank is the MPI rank of this process
         */
        static void set_rank(int32_t _process_rank);

        /**
         * Specifies which MPI process to allow messages from. A value < 0 will allow messages from any rank.
         *\param _restricted_rank is the MPI rank to only print messages from
         */
        static void restrict_to_rank(int32_t _restricted_rank);

        /**
         * Clears the MPI process rank restriction from messages allowing any process to write to the log (sets Log::restricted_rank back to -1).
         */
        static void clear_rank_restriction();

        /**
         * Sets a human readable thread id for this thread.
         * 
         * This will use std::this_thread::get_id() to get c++'s automatically
         * generated thread id, and then put it in the Log::thread_ids map, so 
         * it can be used to write cleaner logs.
         *
         * This will report an error and exit if another thread has already
         * reserved this human readable id.
         *
         * \param human_readable_id a human readable thread id
         */
        static void set_id(string human_readable_id);

        /**
         * Releases a the human readable thread id previously set
         * by by the provided human readable id;
         *
         * This will report an error and exit if this human readable id
         * has not already been set, or if it has been relased by another
         * thread.
         *
         * \param human_readable_id is a human readable thread id which has previously been set with Log::set_id(string)
         */
        static void release_id(string human_readable_id);

        /**
         * Determines if either output level (the file or standard output) level
         * is above the level passed as a parameter.
         *
         * \param level is the level to be greater than or equal to
         *
         * \return true if either the file or standard output level is greater than or equal to the passed level
         */
        static bool at_level(log_level_t level);

        /**
         * Don't use log directly. Use one of the log macros (info, debug, trace etc)
         * The log macros will automatically provide the file, function and line of the calling function
         *
         * Potentially writes the message to either standard output or the log file if the message level is high enough.
         *
         * \param file the c string of the file name
         * \param filelen the length of the file name string
         * \param func the c string of the function name
         * \param funclen the length of the function name
         * \param line the line number where log was called
         * \param level the log level to write to
         * \param format the format string, e.g. "my name is %s"
         * \param args the args for the formated string
         */
        static void log(const char *file, size_t filelen, const char *func, size_t funclen, long line, log_level_t level, const char *format, ...);

};

#endif
