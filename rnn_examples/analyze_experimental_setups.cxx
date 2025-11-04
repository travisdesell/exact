#include <algorithm>
using std::sort;

#include <fstream>
using std::ifstream;
using std::ofstream;

#include <iomanip>
using std::setw;
using std::fixed;
using std::setprecision;

#include <iostream>
using std::cout;
using std::endl;

#include <map>
using std::map;

#include <sstream>
using std::ostringstream;

#include <string>
using std::string;
using std::to_string;

#include <vector>
using std::vector;

#include <dirent.h>
#include <sys/stat.h>

#include "common/arguments.hxx"
#include "common/log.hxx"
#include "rnn/rnn_genome.hxx"

struct RunStats {
    string run_name;
    string genome_filename;
    double best_validation_mse;
    int32_t enabled_weights;
    
    RunStats(string run, string fn, double mse, int32_t weights) 
        : run_name(run), genome_filename(fn), best_validation_mse(mse), enabled_weights(weights) {}
};

struct ExperimentalSetupStats {
    string setup_name;
    vector<RunStats> runs;
    double min_fitness;
    double max_fitness;
    double avg_fitness;
    int32_t min_weights;
    int32_t max_weights;
    double avg_weights;
    int32_t num_runs;
};

bool is_directory(const string& path) {
    struct stat info;
    if (stat(path.c_str(), &info) != 0) {
        return false;
    }
    return S_ISDIR(info.st_mode);
}

vector<string> get_subdirectories(const string& directory) {
    vector<string> dirs;
    DIR* dir = opendir(directory.c_str());
    if (dir == nullptr) {
        return dirs;
    }
    
    struct dirent* entry;
    while ((entry = readdir(dir)) != nullptr) {
        string name = entry->d_name;
        if (name == "." || name == "..") {
            continue;
        }
        string full_path = directory + "/" + name;
        if (is_directory(full_path)) {
            dirs.push_back(name);
        }
    }
    closedir(dir);
    
    sort(dirs.begin(), dirs.end());
    return dirs;
}

string get_latest_global_best_file(const string& directory) {
    vector<string> files;
    DIR* dir = opendir(directory.c_str());
    if (dir == nullptr) {
        return "";
    }
    
    struct dirent* entry;
    while ((entry = readdir(dir)) != nullptr) {
        string filename = entry->d_name;
        if (filename.find("global_best_genome_") == 0 && filename.find(".bin") == filename.length() - 4) {
            files.push_back(directory + "/" + filename);
        }
    }
    closedir(dir);
    
    if (files.empty()) {
        return "";
    }
    
    // Sort to get the latest (highest number)
    sort(files.begin(), files.end());
    return files.back();
}

RunStats analyze_run(const string& run_directory, const string& run_name) {
    string genome_file = get_latest_global_best_file(run_directory);
    
    if (genome_file.empty()) {
        Log::error("No global best genome found in run: %s\n", run_directory.c_str());
        return RunStats(run_name, "", 0.0, 0);
    }
    
    try {
        RNN_Genome* genome = new RNN_Genome(genome_file);
        
        double mse = genome->get_best_validation_mse();
        int32_t weights = genome->get_enabled_number_weights();
        
        string basename = genome_file.substr(genome_file.find_last_of("/\\") + 1);
        RunStats stats(run_name, basename, mse, weights);
        
        Log::info("Run %s: MSE: %.6f, Enabled weights: %d\n", run_name.c_str(), mse, weights);
        
        delete genome;
        return stats;
    } catch (const std::exception& e) {
        Log::error("Failed to load genome %s: %s\n", genome_file.c_str(), e.what());
        return RunStats(run_name, "", 0.0, 0);
    }
}

ExperimentalSetupStats analyze_experimental_setup(const string& setup_directory, const string& setup_name) {
    ExperimentalSetupStats stats;
    stats.setup_name = setup_name;
    
    vector<string> run_dirs = get_subdirectories(setup_directory);
    
    for (const string& run_dir : run_dirs) {
        string run_path = setup_directory + "/" + run_dir;
        RunStats run_stats = analyze_run(run_path, run_dir);
        
        // Only add valid runs (those with valid genomes)
        if (!run_stats.genome_filename.empty() && run_stats.best_validation_mse > 0.0) {
            stats.runs.push_back(run_stats);
        }
    }
    
    if (stats.runs.empty()) {
        Log::warning("No valid runs found in experimental setup: %s\n", setup_name.c_str());
        return stats;
    }
    
    // Extract fitnesses and weights
    vector<double> fitnesses;
    vector<int32_t> weights;
    
    for (const RunStats& run : stats.runs) {
        fitnesses.push_back(run.best_validation_mse);
        weights.push_back(run.enabled_weights);
    }
    
    sort(fitnesses.begin(), fitnesses.end());
    sort(weights.begin(), weights.end());
    
    stats.min_fitness = fitnesses.front();
    stats.max_fitness = fitnesses.back();
    stats.avg_fitness = 0.0;
    for (double f : fitnesses) {
        stats.avg_fitness += f;
    }
    stats.avg_fitness /= fitnesses.size();
    
    stats.min_weights = weights.front();
    stats.max_weights = weights.back();
    stats.avg_weights = 0.0;
    for (int32_t w : weights) {
        stats.avg_weights += w;
    }
    stats.avg_weights /= weights.size();
    
    stats.num_runs = stats.runs.size();
    
    return stats;
}

int main(int argc, char** argv) {
    vector<string> arguments = vector<string>(argv, argv + argc);
    
    // Ensure log arguments are provided (required by Log::initialize)
    bool has_std_level = false, has_file_level = false, has_log_dir = false;
    for (size_t i = 0; i < arguments.size(); i++) {
        if (arguments[i] == "--std_message_level") has_std_level = true;
        if (arguments[i] == "--file_message_level") has_file_level = true;
        if (arguments[i] == "--output_directory") has_log_dir = true;
    }
    
    // Add default log arguments if not provided
    if (!has_std_level) {
        arguments.push_back("--std_message_level");
        arguments.push_back("INFO");
    }
    if (!has_file_level) {
        arguments.push_back("--file_message_level");
        arguments.push_back("NONE");
    }
    if (!has_log_dir) {
        arguments.push_back("--output_directory");
        arguments.push_back("/tmp");
    }
    
    Log::initialize(arguments);
    Log::set_id("main");
    
    string root_directory;
    get_argument(arguments, "--root_directory", true, root_directory);
    
    string output_filename;
    get_argument(arguments, "--output_file", false, output_filename);
    
    if (output_filename.empty()) {
        output_filename = root_directory + "/experimental_setups_analysis.csv";
    }
    
    Log::info("Analyzing experimental setups in: %s\n", root_directory.c_str());
    
    // Get all experimental setup directories
    vector<string> setup_dirs = get_subdirectories(root_directory);
    
    if (setup_dirs.empty()) {
        Log::fatal("No experimental setup directories found in: %s\n", root_directory.c_str());
        return 1;
    }
    
    Log::info("Found %d experimental setup directories\n", setup_dirs.size());
    
    vector<ExperimentalSetupStats> all_setups;
    
    // Analyze each experimental setup
    for (const string& setup_dir : setup_dirs) {
        string setup_path = root_directory + "/" + setup_dir;
        Log::info("Analyzing experimental setup: %s\n", setup_dir.c_str());
        
        ExperimentalSetupStats stats = analyze_experimental_setup(setup_path, setup_dir);
        if (stats.num_runs > 0) {
            all_setups.push_back(stats);
        }
    }
    
    if (all_setups.empty()) {
        Log::fatal("No valid experimental setups found\n");
        return 1;
    }
    
    // Write results to CSV
    ofstream results_file(output_filename);
    
    // Write summary header
    results_file << "Experimental Setup,Num Runs,Min Fitness,Avg Fitness,Max Fitness,Min Enabled Weights,Avg Enabled Weights,Max Enabled Weights" << endl;
    results_file << fixed << setprecision(6);
    
    for (const ExperimentalSetupStats& stats : all_setups) {
        results_file << stats.setup_name << "," 
                     << stats.num_runs << ","
                     << stats.min_fitness << ","
                     << stats.avg_fitness << ","
                     << stats.max_fitness << ","
                     << stats.min_weights << ","
                     << (int)stats.avg_weights << ","
                     << stats.max_weights << endl;
    }
    
    results_file << endl;
    
    // Write detailed per-run data
    results_file << "Experimental Setup,Run,Fitness (Validation MSE),Enabled Weights" << endl;
    results_file << setprecision(6);
    
    for (const ExperimentalSetupStats& stats : all_setups) {
        for (const RunStats& run : stats.runs) {
            results_file << stats.setup_name << ","
                         << run.run_name << ","
                         << run.best_validation_mse << ","
                         << run.enabled_weights << endl;
        }
    }
    
    results_file.close();
    
    // Print summary to console
    cout << endl << "=== Experimental Setups Analysis ===" << endl;
    cout << "Number of experimental setups analyzed: " << all_setups.size() << endl;
    cout << endl;
    cout << setw(30) << "Setup" 
         << setw(10) << "Runs" 
         << setw(15) << "Min Fitness" 
         << setw(15) << "Avg Fitness" 
         << setw(15) << "Max Fitness" 
         << setw(15) << "Min Weights" 
         << setw(15) << "Avg Weights" 
         << setw(15) << "Max Weights" << endl;
    cout << string(120, '-') << endl;
    
    for (const ExperimentalSetupStats& stats : all_setups) {
        cout << setw(30) << stats.setup_name 
             << setw(10) << stats.num_runs
             << setw(15) << setprecision(6) << stats.min_fitness 
             << setw(15) << setprecision(6) << stats.avg_fitness 
             << setw(15) << setprecision(6) << stats.max_fitness 
             << setw(15) << stats.min_weights 
             << setw(15) << (int)stats.avg_weights 
             << setw(15) << stats.max_weights << endl;
    }
    
    cout << endl << "Detailed results saved to: " << output_filename << endl;
    
    Log::release_id("main");
    return 0;
}

