#ifndef EXONA_LOG
#define EXONA_LOG

#include <map>
#include <shared_mutex>
#include <thread>



class Log {
    private:
        static final int NONE = 0;  /**< Specifies no messages will be logged. */
        static final int FATAL = 1; /**< Specifies only fatal messages will be logged. */
        static final int ERROR = 2; /**< Specifies error and above messages will be logged. */
        static final int WARNING = 3; /**< Specifies warning and above messages will be logged. */
        static final int INFO = 4; /**< Specifies info and above messages will be logged. */
        static final int DEBUG = 5; /**< Specifies debug and above messages will be logged. */
        static final int TRACE = 6; /**< Specifies trace and above messages will be logged. */
        static final int ALL = 7; /**< Specifies all messages will be logged. */

            
        /** Specifies which messages to log.
         * Messages will be writtin to the log if their type is <= message_level.
         */
        static int32_t message_level;

        /**
         * Specifies if the logs should also be written to a flie.
         */
        static bool write_to_file;

        /**
         *  For MPI applications, specifies the process id. This will be set to
         *  -1 in the Log::initialize() method and means there are no
         *  process ids (i.e., this is a single process binary).
         */
        static int32_t process_id;

        /**
         * A map of C++ thread ids (which are not human readable) to human
         * readable integer ids.
         * Thread ids are specified with the Log::set_thread_id(int32_t) 
         * method. 
         */
        static map<thread::id, int32_t> thread_ids;

        /**
         * A std::shared_mutex protecting the Log::thread_ids map.
         * The Log::set_thread_id(int32_t) mehod needs write access so it will use the std::shared_mutex::lock()x and std::shared_mutex()::unlock() mechanism, while the Log::get_thread_id() only needs read access and can used the std::shared_mutex::lock_shared() and std::shared_mutex::unlock_shared() methods.
         */
        static shared_mutex thread_ids_mutex;

        /**
         * 
         */
        static inline string get_message_header();

    public:
        /**
         * Registers used command line arguments and instructions with the CommandLine class.
         * Needs to be called at the beginning of the main method of any programming using the Log class. Will register the following command line arguments and instructions:
         *  -# log_level : specifies the max message level of messages to be printed
         *  -# log_directory : specifies the output directory for message logs
         */
        static void register_command_line_arguments();


        /**
         * Sets the MPI process ID for this Log.
         *
         * \param _process_id the MPI process_id, this needs to be >= 0.
         */
        static void set_process_id(int32_t _process_id);

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
        static void set_thread_id(int32_t human_readable_id);

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
        static void release_thread_id();


        /**
         *  Initializes the Log given arguments retreived from the CommandLine class.
         *  Log::register_command_line_arguments() must be called before Log::initialize()
         */
        static void initialize();

        static void fatal(const char* format, ...); /**< Logs a fatal message. varargs are the same as in printf. */
        static void error(const char* format, ...); /**< Logs an error message. varargs are the same as in printf. */
        static void warning(const char* format, ...); /**< Logs a warning message. varargs are the same as in printf. */
        static void info(const char* format, ...); /**< Logs an info message. varargs are the same as in printf. */
        static void debug(const char* format, ...); /**< Logs a debug message. varargs are the same as in printf. */
        static void trace(const char* format, ...); /**< Logs a trace message. varargs are the same as in printf. */

}


#endif
