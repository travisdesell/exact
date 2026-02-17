#include <algorithm>
using std::sort;

#include <chrono>

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

#include <vector>
using std::vector;

#include <dirent.h>

#include "common/arguments.hxx"
#include "common/log.hxx"
#include "rnn/rnn_genome.hxx"

struct GenomeStats {
    string filename;
    double best_validation_mse;
    int32_t enabled_weights;
    
    GenomeStats(string fn, double mse, int32_t weights) 
        : filename(fn), best_validation_mse(mse), enabled_weights(weights) {}
};

vector<string> get_global_best_files(string directory) {
    vector<string> files;
    DIR* dir = opendir(directory.c_str());
    if (dir == nullptr) {
        cout << "Error: Could not open directory: " << directory << endl;
        return files;
    }
    
    struct dirent* entry;
    while ((entry = readdir(dir)) != nullptr) {
        string filename = entry->d_name;
        if (filename.find("global_best_genome_") == 0 && filename.find(".bin") == filename.length() - 4) {
            files.push_back(directory + "/" + filename);
        }
    }
    closedir(dir);
    
    sort(files.begin(), files.end());
    return files;
}

double calculate_mean(const vector<double>& values) {
    if (values.empty()) return 0.0;
    double sum = 0.0;
    for (double v : values) {
        sum += v;
    }
    return sum / values.size();
}

double calculate_mean_int(const vector<int32_t>& values) {
    if (values.empty()) return 0.0;
    double sum = 0.0;
    for (int32_t v : values) {
        sum += v;
    }
    return sum / values.size();
}

int main(int argc, char** argv) {
    vector<string> arguments = vector<string>(argv, argv + argc);
    
    Log::initialize(arguments);
    Log::set_id("main");
    
    string output_directory;
    get_argument(arguments, "--output_directory", true, output_directory);
    
    string results_filename = output_directory + "/global_bests_analysis.csv";
    
    vector<string> genome_files = get_global_best_files(output_directory);
    
    if (genome_files.empty()) {
        Log::fatal("No global best genome files found in: %s\n", output_directory.c_str());
        return 1;
    }
    
    Log::info("Found %d global best genome files\n", genome_files.size());
    
    vector<GenomeStats> stats;
    vector<double> fitnesses;
    vector<int32_t> enabled_weights_list;
    
    for (string filename : genome_files) {
        try {
            RNN_Genome* genome = new RNN_Genome(filename);
            
            double mse = genome->get_best_validation_mse();
            int32_t weights = genome->get_enabled_number_weights();
            
            string basename = filename.substr(filename.find_last_of("/\\") + 1);
            stats.push_back(GenomeStats(basename, mse, weights));
            fitnesses.push_back(mse);
            enabled_weights_list.push_back(weights);
            
            Log::info("Loaded: %s - MSE: %.6f, Enabled weights: %d\n", basename.c_str(), mse, weights);
            
            delete genome;
        } catch (const std::exception& e) {
            Log::error("Failed to load genome %s: %s\n", filename.c_str(), e.what());
        }
    }
    
    if (stats.empty()) {
        Log::fatal("No valid genomes loaded\n");
        return 1;
    }
    
    // Calculate statistics
    sort(fitnesses.begin(), fitnesses.end());
    sort(enabled_weights_list.begin(), enabled_weights_list.end());
    
    double min_fitness = fitnesses.front();
    double max_fitness = fitnesses.back();
    double mean_fitness = calculate_mean(fitnesses);
    
    int32_t min_weights = enabled_weights_list.front();
    int32_t max_weights = enabled_weights_list.back();
    double mean_weights = calculate_mean_int(enabled_weights_list);
    
    // Write results to CSV
    ofstream results_file(results_filename);
    results_file << "Statistic,Fitness (Validation MSE),Enabled Weights" << endl;
    results_file << fixed << setprecision(6);
    results_file << "Min," << min_fitness << "," << min_weights << endl;
    results_file << "Max," << max_fitness << "," << max_weights << endl;
    results_file << setprecision(4);
    results_file << "Average," << mean_fitness << "," << mean_weights << endl;
    results_file << "Number of Runs," << stats.size() << "," << stats.size() << endl;
    results_file << endl;
    results_file << "Run,Filename,Fitness (Validation MSE),Enabled Weights" << endl;
    for (size_t i = 0; i < stats.size(); i++) {
        results_file << (i+1) << "," << stats[i].filename << "," << setprecision(6) << stats[i].best_validation_mse 
                     << "," << stats[i].enabled_weights << endl;
    }
    results_file.close();
    
    // Print results to console
    cout << endl << "=== Global Best Genomes Analysis ===" << endl;
    cout << "Number of runs analyzed: " << stats.size() << endl;
    cout << endl << "Fitness (Validation MSE) Statistics:" << endl;
    cout << "  Min:  " << setprecision(6) << min_fitness << endl;
    cout << "  Max:  " << max_fitness << endl;
    cout << "  Avg:  " << setprecision(4) << mean_fitness << endl;
    cout << endl << "Enabled Weights Statistics:" << endl;
    cout << "  Min:  " << min_weights << endl;
    cout << "  Max:  " << max_weights << endl;
    cout << "  Avg:  " << (int)mean_weights << endl;
    cout << endl << "Detailed results saved to: " << results_filename << endl;
    
    Log::release_id("main");
    return 0;
}

