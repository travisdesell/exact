#include <algorithm>
using std::sort;
using std::find;

#include <chrono>
#include <cstring>

#include <iomanip>
using std::setw;

#include <iostream>
using std::cerr;
using std::cout;
using std::endl;

#include <mutex>
using std::mutex;

#include <string>
using std::string;

#include <thread>
using std::thread;

#include <map>
using std::map;

#include <vector>
using std::vector;

#include "common/arguments.hxx"

#include "rnn/rnn_genome.hxx"

#include "dirent.h"

#include "run_statistics.hxx"
#include "tracker.hxx"


class Run {
    public:
        string output_type;
        string search_type;
        string slice_name;

        vector<RNN_Genome*> genomes;

        Run(string _output_type, string _search_type, string _slice_name) {
            output_type = _output_type;
            search_type = _search_type;
            slice_name = _slice_name;
        }

        void add_genome(RNN_Genome *genome) {
            genomes.push_back(genome);
        }

        void print() {
            cout << "{ \"output_type\" : \"" << output_type << "\", \"search_type\" : \"" << search_type << "\", \"slice_name\" : \"" << slice_name << "\"";
            cout << ", \"fitness\" : [";

            cout << genomes[0]->get_fitness();

            for (int i = 1; i < genomes.size(); i++) {
                cout << ", " << genomes[i]->get_fitness();
            }

            cout << "] ";
            cout << "}";
        }

        ~Run() {
            while (genomes.size() > 0) {
                RNN_Genome *last = genomes.back();
                delete last;
                genomes.pop_back();
            }
        }
};

bool extension_is(string name, string extension) {
    if (name.size() < 4) return false;

    string ext = name.substr(name.length() - 4, 4);

    //cout << "comparing '" << ext << "' to '" << extension << "'" << endl;

    return ext.compare(extension) == 0;
}

int main(int argc, char** argv) {
    vector<string> arguments = vector<string>(argv, argv + argc);

    //directory structure is:
    // output_type / search_type / slice / repeat bests

    string dir_name = arguments[1];

    vector<string> output_types;
    vector<string> search_types;
    vector<string> slice_names;
    vector<Run*> runs;

    DIR *output_dir;
    if ((output_dir = opendir(dir_name.c_str())) == NULL) {
        cerr << "ERROR: could not open directory: '" << dir_name.c_str() << "'" << endl;
        exit(1);
    }

    struct dirent *output_ent;

    bool verbose = false;

    //iterate through the first directory
    while ((output_ent = readdir(output_dir)) != NULL) {
        if (output_ent->d_name[0] == '.') continue;
        if (strcmp(output_ent->d_name, "logs") == 0) continue;

        string output_type = output_ent->d_name;
        output_types.push_back(output_type);
        if (verbose) cout << "set output type to '" << output_type << "'" << endl;

        string search_dir_name = dir_name + "/" + output_ent->d_name;
        if (verbose) cout << "processing search directory " << search_dir_name << endl;

        DIR *search_dir;
        if ((search_dir = opendir(search_dir_name.c_str())) == NULL) {
            cerr << "ERROR: could not open search directory: '" << search_dir_name.c_str() << "'" << endl;
            exit(1);
        }

        struct dirent *search_ent;
        while ((search_ent = readdir(search_dir)) != NULL) {
            if (search_ent->d_name[0] == '.') continue;

            //insert search type if it doesn't exist yet
            string search_type = search_ent->d_name;
            if (find(search_types.begin(), search_types.end(), search_type) == search_types.end()) {
                search_types.push_back(search_type);
            }
            if (verbose) cout << "\tset search type to '" << search_type << "'" << endl;

            string slice_dir_name = search_dir_name + "/" + search_ent->d_name;
            if (verbose) cout << "\tprocessing slice directory " << slice_dir_name << endl;

            DIR *slice_dir;
            if ((slice_dir = opendir(slice_dir_name.c_str())) == NULL) {
                cerr << "ERROR: could not open slice directory: '" << slice_dir_name.c_str() << "'" << endl;
                exit(1);
            }

            struct dirent *slice_ent;
            while ((slice_ent = readdir(slice_dir)) != NULL) {
                if (slice_ent->d_name[0] == '.') continue;
                if (extension_is(slice_ent->d_name, ".csv")) continue;

                //insert slice type if it doesn't exist yet
                string slice_name = slice_ent->d_name;
                if (find(slice_names.begin(), slice_names.end(), slice_name) == slice_names.end()) {
                    slice_names.push_back(slice_name);
                }

                if (verbose) cout << "\t\tset slice to '" << slice_name << "'" << endl;

                string repeat_dir_name = slice_dir_name + "/" + slice_ent->d_name;
                if (verbose) cout << "\t\tprocessing repeat directory  '" << repeat_dir_name << "'" << endl;

                DIR *repeat_dir;
                if ((repeat_dir = opendir(repeat_dir_name.c_str())) == NULL) {
                    cerr << "ERROR: could not open repeat directory: '" << repeat_dir_name.c_str() << "'" << endl;
                    exit(1);
                }

                Run *run = new Run(output_type, search_type, slice_name);

                int repeat_count = 0;
                struct dirent *repeat_ent;
                while ((repeat_ent = readdir(repeat_dir)) != NULL) {
                    //skip all the non-binaries
                    if (!extension_is(repeat_ent->d_name, ".bin")) continue;

                    string repeat_name = repeat_dir_name + "/" + repeat_ent->d_name;

                    if (verbose) cout << "\t\t\tprocessing genome binary '" << repeat_name << "' for '" << output_type << "'" << " and " << search_type << " and " << slice_name;
                    RNN_Genome *genome = new RNN_Genome(repeat_name, false);
                    run->add_genome(genome);
                    if (verbose) cout << ", fitness: " << genome->get_fitness() << endl;
                    repeat_count++;
                }
                closedir(repeat_dir);
                if (verbose) cout << "\t\t\trepeats: " << repeat_count << endl;
                if (repeat_count != 10) exit(1);

                runs.push_back(run);
            }
            closedir(slice_dir);

    for (int i = 0; i < output_types.size(); i++) {
        cout << "GENERATING sorted stats for: '" << output_types[i] << "'" << endl;

        vector<RunStatistics*> current = output_sorted_statistics[output_types[i]];

        cout << "got current!" << endl;

        double avg_min = 0.0;
        double avg_avg = 0.0;
        double avg_max = 0.0;

        for (int j = 0; j < current.size(); j++) {
            avg_min += current[j]->mae.min();
            avg_avg += current[j]->mae.avg();
            avg_max += current[j]->mae.max();
        }

        avg_min /= current.size();
        avg_avg /= current.size();
        avg_max /= current.size();

        cout << "calculated averages" << endl;

        double stddev_min = 0.0;
        double stddev_avg = 0.0;
        double stddev_max = 0.0;

        for (int j = 0; j < current.size(); j++) {
            stddev_min += (current[j]->mae.min() - avg_min) * (current[j]->mae.min() - avg_min);
            stddev_avg += (current[j]->mae.avg() - avg_avg) * (current[j]->mae.avg() - avg_avg);
            stddev_max += (current[j]->mae.max() - avg_max) * (current[j]->mae.max() - avg_max);
        }
        closedir(search_dir);

        stddev_min = sqrt(stddev_min / (current.size() - 1));
        stddev_avg = sqrt(stddev_avg / (current.size() - 1));
        stddev_max = sqrt(stddev_max / (current.size() - 1));

        cout << "calculated stddevs!" << endl;

        for (int j = 0; j < current.size(); j++) {
            cout << "current[" << j << "]->run_type: " << current[j]->run_type << endl;

            current[j]->set_deviation_from_mean_min((current[j]->mae.min() - avg_min) / stddev_min);
            current[j]->set_deviation_from_mean_avg((current[j]->mae.avg() - avg_avg) / stddev_avg);
            current[j]->set_deviation_from_mean_max((current[j]->mae.max() - avg_max) / stddev_max);

            consolidated_statistics[ current[j]->run_type ]->dfm_min += current[j]->dfm_min / output_types.size();
            consolidated_statistics[ current[j]->run_type ]->dfm_avg += current[j]->dfm_avg / output_types.size();
            consolidated_statistics[ current[j]->run_type ]->dfm_max += current[j]->dfm_max / output_types.size();
        }

        cout << "updated consolidated statistics!" << endl;

        vector<RunStatistics*> sorted_by_min = current;
        vector<RunStatistics*> sorted_by_avg = current;
        vector<RunStatistics*> sorted_by_max = current;

        cout << "sorting!" << endl;

        sort(sorted_by_min.begin(), sorted_by_min.end(), less_than_dfm_min());
        sort(sorted_by_avg.begin(), sorted_by_avg.end(), less_than_dfm_avg());
        sort(sorted_by_max.begin(), sorted_by_max.end(), less_than_dfm_max());

        cout << endl << endl;
        cout << "RANKINGS FOR '" << output_types[i] << "'" << endl;

        cout << "\\begin{table}" << endl;
        cout << "\\begin{scriptsize}" << endl;
        cout << "\\centering" << endl;
        cout << "\\begin{tabular}{|r|l||r|l||r|l||}" << endl;
        cout << "\\hline" << endl;
        cout << "\\multicolumn{2}{|c||}{{\\bf Best Case}} & \\multicolumn{2}{|c||}{{\\bf Avg. Case}} & \\multicolumn{2}{|c|}{{\\bf Worst Case}} \\\\" << endl;
        cout << "\\hline" << endl;

        for (int j = 0; j < current.size(); j++) {
            cout << setw(15) << fix_run_type(sorted_by_min[j]->run_type) << " & " << setw(15) << setprecision(5) << sorted_by_min[j]->dfm_min << " & ";
            cout << setw(15) << fix_run_type(sorted_by_avg[j]->run_type) << " & " << setw(15) << setprecision(5) << sorted_by_avg[j]->dfm_avg << " & ";
            cout << setw(15) << fix_run_type(sorted_by_max[j]->run_type) << " & " << setw(15) << setprecision(5) << sorted_by_max[j]->dfm_max << "\\\\" << endl;
        }

        cout << "\\hline" << endl;
        cout << "\\end{tabular}" << endl;
        cout << "\\caption{\\label{table:consolidated_rankings} Rankings for all the EXAMM run types predicting " << output_types[i] << ".}" << endl;
        cout << "\\end{scriptsize}" << endl;
        cout << "\\end{table}" << endl;
    }

    vector<ConsolidatedStatistics*> min_stats_vector;
    vector<ConsolidatedStatistics*> avg_stats_vector;
    vector<ConsolidatedStatistics*> max_stats_vector;

    for (auto i = consolidated_statistics.begin(); i != consolidated_statistics.end(); i++) {
        min_stats_vector.push_back(i->second);
        avg_stats_vector.push_back(i->second);
        max_stats_vector.push_back(i->second);
    }
    closedir(output_dir);

    cout << "[" << endl;
    for (int i = 0; i < runs.size(); i++) {
        runs[i]->print();
        if (i < runs.size() - 1) cout << ",";
        cout << endl;
    }
    cout << "]" << endl;

    return 0;
}
