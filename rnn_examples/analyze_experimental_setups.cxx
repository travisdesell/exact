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

struct GenomeStats {
    int32_t genome_number;
    double initial_fitness;
    double final_fitness;
    int32_t bp_epochs;
    long bp_time_ms;
    
    GenomeStats() : genome_number(0), initial_fitness(0.0), final_fitness(0.0), bp_epochs(0), bp_time_ms(0) {}
};

struct InsertedGenomeInfo {
    int32_t total_inserted_genomes;
    int32_t num_insertion_events;
    long total_time_ms;
    double insertion_rate_per_second;  // genomes per second
    
    InsertedGenomeInfo() : total_inserted_genomes(0), num_insertion_events(0), 
                          total_time_ms(0), insertion_rate_per_second(0.0) {}
};

struct RunStats {
    string run_name;
    string genome_filename;
    double best_validation_mse;
    int32_t enabled_weights;
    vector<GenomeStats> genome_stats;
    int32_t num_genomes_with_stats;
    double avg_initial_fitness;
    double avg_final_fitness;
    double avg_fitness_improvement;
    double avg_bp_epochs;
    double avg_bp_time_ms;
    long total_bp_time_ms;
    InsertedGenomeInfo inserted_info;
    
    RunStats(string run, string fn, double mse, int32_t weights) 
        : run_name(run), genome_filename(fn), best_validation_mse(mse), enabled_weights(weights),
          num_genomes_with_stats(0), avg_initial_fitness(0.0), avg_final_fitness(0.0),
          avg_fitness_improvement(0.0), avg_bp_epochs(0.0), avg_bp_time_ms(0.0), total_bp_time_ms(0) {}
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
    // Genome stats aggregates
    double avg_initial_fitness;
    double avg_final_fitness;
    double avg_fitness_improvement;
    double avg_bp_epochs;
    double avg_bp_time_ms;
    long total_bp_time_ms;
    int32_t total_genomes_with_stats;
    // Inserted genomes aggregates
    int32_t total_inserted_genomes;
    int32_t total_insertion_events;
    double avg_inserted_genomes_per_run;
    double avg_insertion_rate_per_second;
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

vector<GenomeStats> read_genome_stats_log(const string& directory) {
    vector<GenomeStats> stats;
    string csv_file = directory + "/genome_stats_log.csv";
    
    ifstream file(csv_file);
    if (!file.is_open()) {
        // File doesn't exist, return empty vector
        return stats;
    }
    
    string line;
    // Skip header
    if (!getline(file, line)) {
        file.close();
        return stats;
    }
    
    // Read data rows
    while (getline(file, line)) {
        if (line.empty()) {
            continue;
        }
        
        // Parse CSV line
        vector<string> fields;
        stringstream ss(line);
        string field;
        
        while (getline(ss, field, ',')) {
            fields.push_back(field);
        }
        
        if (fields.size() >= 5) {
            GenomeStats gs;
            try {
                gs.genome_number = stoi(fields[0]);
                gs.initial_fitness = stod(fields[1]);
                gs.final_fitness = stod(fields[2]);
                gs.bp_epochs = stoi(fields[3]);
                gs.bp_time_ms = stol(fields[4]);
                stats.push_back(gs);
            } catch (const std::exception& e) {
                Log::warning("Failed to parse genome stats row: %s\n", line.c_str());
            }
        }
    }
    
    file.close();
    return stats;
}

InsertedGenomeInfo read_fitness_log(const string& directory) {
    InsertedGenomeInfo info;
    string csv_file = directory + "/fitness_log.csv";
    
    ifstream file(csv_file);
    if (!file.is_open()) {
        // File doesn't exist, return empty info
        return info;
    }
    
    string line;
    // Read header to find column positions
    if (!getline(file, line)) {
        file.close();
        return info;
    }
    
    // Parse header to find "Inserted Genomes" and "Time" columns
    vector<string> header_fields;
    stringstream header_ss(line);
    string header_field;
    int inserted_genomes_col = -1;
    int time_col = -1;
    int col_index = 0;
    
    while (getline(header_ss, header_field, ',')) {
        // Trim whitespace
        header_field.erase(0, header_field.find_first_not_of(" \t"));
        header_field.erase(header_field.find_last_not_of(" \t") + 1);
        
        if (header_field == "Inserted Genomes") {
            inserted_genomes_col = col_index;
        } else if (header_field == "Time") {
            time_col = col_index;
        }
        col_index++;
    }
    
    if (inserted_genomes_col == -1) {
        Log::warning("Could not find 'Inserted Genomes' column in fitness_log.csv\n");
        file.close();
        return info;
    }
    
    // Read data rows
    long last_time_ms = 0;
    while (getline(file, line)) {
        if (line.empty()) {
            continue;
        }
        
        // Parse CSV line
        vector<string> fields;
        stringstream ss(line);
        string field;
        
        while (getline(ss, field, ',')) {
            fields.push_back(field);
        }
        
        if (fields.size() > inserted_genomes_col) {
            try {
                int32_t inserted = stoi(fields[inserted_genomes_col]);
                info.total_inserted_genomes = inserted;  // Last value is the total
                info.num_insertion_events++;
                
                // Get time if available
                if (time_col >= 0 && time_col < (int)fields.size()) {
                    last_time_ms = stol(fields[time_col]);
                }
            } catch (const std::exception& e) {
                Log::warning("Failed to parse fitness log row: %s\n", line.c_str());
            }
        }
    }
    
    info.total_time_ms = last_time_ms;
    
    // Calculate insertion rate (genomes per second)
    if (info.total_time_ms > 0) {
        info.insertion_rate_per_second = (double)info.total_inserted_genomes / (info.total_time_ms / 1000.0);
    }
    
    file.close();
    return info;
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
        
        // Read genome stats log
        stats.genome_stats = read_genome_stats_log(run_directory);
        stats.num_genomes_with_stats = stats.genome_stats.size();
        
        // Read fitness log for inserted genomes info
        stats.inserted_info = read_fitness_log(run_directory);
        
        // Calculate averages
        if (stats.num_genomes_with_stats > 0) {
            double sum_initial = 0.0;
            double sum_final = 0.0;
            double sum_improvement = 0.0;
            double sum_epochs = 0.0;
            double sum_time = 0.0;
            
            for (const GenomeStats& gs : stats.genome_stats) {
                sum_initial += gs.initial_fitness;
                sum_final += gs.final_fitness;
                sum_improvement += (gs.initial_fitness - gs.final_fitness);
                sum_epochs += gs.bp_epochs;
                sum_time += gs.bp_time_ms;
                stats.total_bp_time_ms += gs.bp_time_ms;
            }
            
            stats.avg_initial_fitness = sum_initial / stats.num_genomes_with_stats;
            stats.avg_final_fitness = sum_final / stats.num_genomes_with_stats;
            stats.avg_fitness_improvement = sum_improvement / stats.num_genomes_with_stats;
            stats.avg_bp_epochs = sum_epochs / stats.num_genomes_with_stats;
            stats.avg_bp_time_ms = sum_time / stats.num_genomes_with_stats;
        }
        
        Log::info("Run %s: MSE: %.6f, Enabled weights: %d, Genomes with stats: %d, Inserted genomes: %d\n", 
                  run_name.c_str(), mse, weights, stats.num_genomes_with_stats, stats.inserted_info.total_inserted_genomes);
        
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
    
    // Calculate genome stats aggregates
    stats.total_genomes_with_stats = 0;
    double sum_initial = 0.0;
    double sum_final = 0.0;
    double sum_improvement = 0.0;
    double sum_epochs = 0.0;
    double sum_time = 0.0;
    stats.total_bp_time_ms = 0;
    
    for (const RunStats& run : stats.runs) {
        if (run.num_genomes_with_stats > 0) {
            stats.total_genomes_with_stats += run.num_genomes_with_stats;
            sum_initial += run.avg_initial_fitness * run.num_genomes_with_stats;
            sum_final += run.avg_final_fitness * run.num_genomes_with_stats;
            sum_improvement += run.avg_fitness_improvement * run.num_genomes_with_stats;
            sum_epochs += run.avg_bp_epochs * run.num_genomes_with_stats;
            sum_time += run.avg_bp_time_ms * run.num_genomes_with_stats;
            stats.total_bp_time_ms += run.total_bp_time_ms;
        }
    }
    
    if (stats.total_genomes_with_stats > 0) {
        stats.avg_initial_fitness = sum_initial / stats.total_genomes_with_stats;
        stats.avg_final_fitness = sum_final / stats.total_genomes_with_stats;
        stats.avg_fitness_improvement = sum_improvement / stats.total_genomes_with_stats;
        stats.avg_bp_epochs = sum_epochs / stats.total_genomes_with_stats;
        stats.avg_bp_time_ms = sum_time / stats.total_genomes_with_stats;
    } else {
        stats.avg_initial_fitness = 0.0;
        stats.avg_final_fitness = 0.0;
        stats.avg_fitness_improvement = 0.0;
        stats.avg_bp_epochs = 0.0;
        stats.avg_bp_time_ms = 0.0;
    }
    
    // Calculate inserted genomes aggregates
    stats.total_inserted_genomes = 0;
    stats.total_insertion_events = 0;
    double sum_insertion_rate = 0.0;
    int runs_with_insertion_data = 0;
    
    for (const RunStats& run : stats.runs) {
        if (run.inserted_info.total_inserted_genomes > 0) {
            stats.total_inserted_genomes += run.inserted_info.total_inserted_genomes;
            stats.total_insertion_events += run.inserted_info.num_insertion_events;
            sum_insertion_rate += run.inserted_info.insertion_rate_per_second;
            runs_with_insertion_data++;
        }
    }
    
    if (stats.num_runs > 0) {
        stats.avg_inserted_genomes_per_run = (double)stats.total_inserted_genomes / stats.num_runs;
    } else {
        stats.avg_inserted_genomes_per_run = 0.0;
    }
    
    if (runs_with_insertion_data > 0) {
        stats.avg_insertion_rate_per_second = sum_insertion_rate / runs_with_insertion_data;
    } else {
        stats.avg_insertion_rate_per_second = 0.0;
    }
    
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
    results_file << "Experimental Setup,Num Runs,Min Fitness,Avg Fitness,Max Fitness,Min Enabled Weights,Avg Enabled Weights,Max Enabled Weights,"
                 << "Total Genomes with Stats,Avg Initial Fitness,Avg Final Fitness,Avg Fitness Improvement,Avg BP Epochs,Avg BP Time (ms),Total BP Time (ms),"
                 << "Total Inserted Genomes,Total Insertion Events,Avg Inserted Genomes per Run,Avg Insertion Rate (genomes/sec)" << endl;
    results_file << fixed << setprecision(6);
    
    for (const ExperimentalSetupStats& stats : all_setups) {
        results_file << stats.setup_name << "," 
                     << stats.num_runs << ","
                     << stats.min_fitness << ","
                     << stats.avg_fitness << ","
                     << stats.max_fitness << ","
                     << stats.min_weights << ","
                     << (int)stats.avg_weights << ","
                     << stats.max_weights << ","
                     << stats.total_genomes_with_stats << ","
                     << stats.avg_initial_fitness << ","
                     << stats.avg_final_fitness << ","
                     << stats.avg_fitness_improvement << ","
                     << setprecision(2) << stats.avg_bp_epochs << ","
                     << setprecision(0) << stats.avg_bp_time_ms << ","
                     << stats.total_bp_time_ms << ","
                     << stats.total_inserted_genomes << ","
                     << stats.total_insertion_events << ","
                     << setprecision(2) << stats.avg_inserted_genomes_per_run << ","
                     << setprecision(4) << stats.avg_insertion_rate_per_second << endl;
        results_file << setprecision(6);
    }
    
    results_file << endl;
    
    // Write detailed per-run data
    results_file << "Experimental Setup,Run,Fitness (Validation MSE),Enabled Weights,"
                 << "Genomes with Stats,Avg Initial Fitness,Avg Final Fitness,Avg Fitness Improvement,Avg BP Epochs,Avg BP Time (ms),Total BP Time (ms),"
                 << "Inserted Genomes,Insertion Events,Total Time (ms),Insertion Rate (genomes/sec)" << endl;
    results_file << setprecision(6);
    
    for (const ExperimentalSetupStats& stats : all_setups) {
        for (const RunStats& run : stats.runs) {
            results_file << stats.setup_name << ","
                         << run.run_name << ","
                         << run.best_validation_mse << ","
                         << run.enabled_weights << ","
                         << run.num_genomes_with_stats << ","
                         << run.avg_initial_fitness << ","
                         << run.avg_final_fitness << ","
                         << run.avg_fitness_improvement << ","
                         << setprecision(2) << run.avg_bp_epochs << ","
                         << setprecision(0) << run.avg_bp_time_ms << ","
                         << run.total_bp_time_ms << ","
                         << run.inserted_info.total_inserted_genomes << ","
                         << run.inserted_info.num_insertion_events << ","
                         << run.inserted_info.total_time_ms << ","
                         << setprecision(4) << run.inserted_info.insertion_rate_per_second << endl;
            results_file << setprecision(6);
        }
    }
    
    results_file << endl;
    
    // Write detailed per-genome data
    results_file << "Experimental Setup,Run,Genome Number,Initial Fitness,Final Fitness,BP Epochs,BP Time (ms)" << endl;
    results_file << setprecision(6);
    
    for (const ExperimentalSetupStats& stats : all_setups) {
        for (const RunStats& run : stats.runs) {
            for (const GenomeStats& gs : run.genome_stats) {
                results_file << stats.setup_name << ","
                             << run.run_name << ","
                             << gs.genome_number << ","
                             << gs.initial_fitness << ","
                             << gs.final_fitness << ","
                             << gs.bp_epochs << ","
                             << gs.bp_time_ms << endl;
            }
        }
    }
    
    results_file.close();
    
    // Print summary to console
    cout << endl << "=== Experimental Setups Analysis ===" << endl;
    cout << "Number of experimental setups analyzed: " << all_setups.size() << endl;
    cout << endl;
    
    // Fitness and weights summary
    cout << "--- Fitness and Network Size Summary ---" << endl;
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
    
    cout << endl;
    
    // Inserted genomes summary
    cout << "--- Inserted Genomes Summary ---" << endl;
    cout << setw(30) << "Setup"
         << setw(15) << "Total Inserted"
         << setw(15) << "Insert Events"
         << setw(20) << "Avg per Run"
         << setw(20) << "Insert Rate (gen/sec)" << endl;
    cout << string(100, '-') << endl;
    
    for (const ExperimentalSetupStats& stats : all_setups) {
        if (stats.total_inserted_genomes > 0) {
            cout << setw(30) << stats.setup_name
                 << setw(15) << stats.total_inserted_genomes
                 << setw(15) << stats.total_insertion_events
                 << setw(20) << setprecision(2) << stats.avg_inserted_genomes_per_run
                 << setw(20) << setprecision(4) << stats.avg_insertion_rate_per_second << endl;
        } else {
            cout << setw(30) << stats.setup_name
                 << setw(15) << "N/A"
                 << setw(15) << "N/A"
                 << setw(20) << "N/A"
                 << setw(20) << "N/A" << endl;
        }
    }
    
    cout << endl;
    
    // Genome stats summary
    cout << "--- Backpropagation Statistics Summary ---" << endl;
    cout << setw(30) << "Setup"
         << setw(12) << "Genomes"
         << setw(18) << "Avg Init Fitness"
         << setw(18) << "Avg Final Fitness"
         << setw(18) << "Avg Improvement"
         << setw(12) << "Avg Epochs"
         << setw(15) << "Avg Time (ms)"
         << setw(15) << "Total Time (ms)" << endl;
    cout << string(140, '-') << endl;
    
    for (const ExperimentalSetupStats& stats : all_setups) {
        if (stats.total_genomes_with_stats > 0) {
            cout << setw(30) << stats.setup_name
                 << setw(12) << stats.total_genomes_with_stats
                 << setw(18) << setprecision(6) << stats.avg_initial_fitness
                 << setw(18) << setprecision(6) << stats.avg_final_fitness
                 << setw(18) << setprecision(6) << stats.avg_fitness_improvement
                 << setw(12) << setprecision(1) << stats.avg_bp_epochs
                 << setw(15) << setprecision(0) << stats.avg_bp_time_ms
                 << setw(15) << stats.total_bp_time_ms << endl;
        } else {
            cout << setw(30) << stats.setup_name
                 << setw(12) << "0"
                 << setw(18) << "N/A"
                 << setw(18) << "N/A"
                 << setw(18) << "N/A"
                 << setw(12) << "N/A"
                 << setw(15) << "N/A"
                 << setw(15) << "N/A" << endl;
        }
    }
    
    cout << endl << "Detailed results saved to: " << output_filename << endl;
    
    Log::release_id("main");
    return 0;
}

