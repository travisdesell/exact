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
        static int32_t std_message_level;

        /**
         * Specifies which messages to log.
         * Messages will be written to the log file if their type is <= message_level.
         */

        static int32_t file_message_level;

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


        /**
         * Potentially writes the message to either standard output or the log file if the message level is high enough.
         *
         * \param print_header specifies if the header to the message should be printed out
         * \param message_level the level of the message to potentially be printed out
         * \param message_type a string representation of this message type
         * \param message_type the format string for this message (as in printf)
         * \param arguments the arguments for the print statement
         */
        static void write_message(bool print_header, int8_t message_level, const char *message_type, const char *format, va_list arguments);

    public:
        static const int8_t NONE = 0;  /**< Specifies no messages will be logged. */
        static const int8_t FATAL = 1; /**< Specifies only fatal messages will be logged. */
        static const int8_t ERROR = 2; /**< Specifies error and above messages will be logged. */
        static const int8_t WARNING = 3; /**< Specifies warning and above messages will be logged. */
        static const int8_t INFO = 4; /**< Specifies info and above messages will be logged. */
        static const int8_t DEBUG = 5; /**< Specifies debug and above messages will be logged. */
        static const int8_t TRACE = 6; /**< Specifies trace and above messages will be logged. */
        static const int8_t ALL = 7; /**< Specifies all messages will be logged. */

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
        static int8_t parse_level_from_string(string level);

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
        static bool at_level(int8_t level);

        static void fatal(const char* format, ...); /**< Logs a fatal message. varargs are the same as in printf. */
        static void error(const char* format, ...); /**< Logs an error message. varargs are the same as in printf. */
        static void warning(const char* format, ...); /**< Logs a warning message. varargs are the same as in printf. */
        static void info(const char* format, ...); /**< Logs an info message. varargs are the same as in printf. */
        static void debug(const char* format, ...); /**< Logs a debug message. varargs are the same as in printf. */
        static void trace(const char* format, ...); /**< Logs a trace message. varargs are the same as in printf. */

        static void fatal_no_header(const char* format, ...); /**< Logs a fatal message. Does not print the message header (useful if doing multiple log prints to the same line). varargs are the same as in printf. */
        static void error_no_header(const char* format, ...); /**< Logs an error message. Does not print the message header (useful if doing multiple log prints to the same line).  varargs are the same as in printf. */
        static void warning_no_header(const char* format, ...); /**< Logs a warning message. Does not print the message header (useful if doing multiple log prints to the same line).  varargs are the same as in printf. */
        static void info_no_header(const char* format, ...); /**< Logs an info message. Does not print the message header (useful if doing multiple log prints to the same line).  varargs are the same as in printf. */
        static void debug_no_header(const char* format, ...); /**< Logs a debug message. Does not print the message header (useful if doing multiple log prints to the same line).  varargs are the same as in printf. */
        static void trace_no_header(const char* format, ...); /**< Logs a trace message. Does not print the message header (useful if doing multiple log prints to the same line).  varargs are the same as in printf. */

};


#endif
