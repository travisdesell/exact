#ifndef EXONA_LOG
#define EXONA_LOG

#include <cstdio>  

#include <iostream>
using std::ofstream;

#include <map>
using std::map;

#include <shared_mutex>
using std::shared_mutex;

#include <string>
using std::string;

#include <thread>
using std::thread;


class Log {
    private:
        static const int NONE = 0;  /**< Specifies no messages will be logged. */
        static const int FATAL = 1; /**< Specifies only fatal messages will be logged. */
        static const int ERROR = 2; /**< Specifies error and above messages will be logged. */
        static const int WARNING = 3; /**< Specifies warning and above messages will be logged. */
        static const int INFO = 4; /**< Specifies info and above messages will be logged. */
        static const int DEBUG = 5; /**< Specifies debug and above messages will be logged. */
        static const int TRACE = 6; /**< Specifies trace and above messages will be logged. */
        static const int ALL = 7; /**< Specifies all messages will be logged. */

            
        /** Specifies which messages to log.
         * Messages will be writtin to the log if their type is <= message_level.
         */
        static int32_t message_level;

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
         * A map of human readable ids to output files which the log messages
         * will be written to.
         */
        static map<string, FILE*> output_files;

        /**
         * A std::shared_mutex protecting the Log::thread_ids map.
         * The Log::set_thread_id(int32_t) mehod needs write access so it will use the std::shared_mutex::lock()x and std::shared_mutex()::unlock() mechanism, while the Log::get_thread_id() only needs read access and can used the std::shared_mutex::lock_shared() and std::shared_mutex::unlock_shared() methods.
         */
        static shared_mutex log_ids_mutex;


        /**
         *
         */
        static void write_message(const char *message_type, const char *format, va_list arguments);

    public:
        /**
         * Registers used command line arguments and instructions with the CommandLine class.
         * Needs to be called at the beginning of the main method of any programming using the Log class. Will register the following command line arguments and instructions:
         *  -# log_level : specifies the max message level of messages to be printed
         *  -# log_directory : specifies the output directory for message logs
         */
        static void register_command_line_arguments();

        /**
         *  Initializes the Log given arguments retreived from the CommandLine class.
         *  Log::register_command_line_arguments() must be called before Log::initialize()
         */
        static void initialize();


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
         * \param human_readable_id a human readable thread id, this needs to be >= 0.
         */
        static void set_id(string human_readable_id);

        /**
         * Releases a the human readable thread id previously set
         * by this thread.
         *
         * This will use std::this_thread::get_id() to get c++'s automatically
         * generated thread id to look up the human readable id to remmove from
         * the map.
         *
         * This will report an error and exit if the thread has not already
         * set a thread id.
         */
        static void release_id();



        static void fatal(const char* format, ...); /**< Logs a fatal message. varargs are the same as in printf. */
        static void error(const char* format, ...); /**< Logs an error message. varargs are the same as in printf. */
        static void warning(const char* format, ...); /**< Logs a warning message. varargs are the same as in printf. */
        static void info(const char* format, ...); /**< Logs an info message. varargs are the same as in printf. */
        static void debug(const char* format, ...); /**< Logs a debug message. varargs are the same as in printf. */
        static void trace(const char* format, ...); /**< Logs a trace message. varargs are the same as in printf. */

};


#endif
