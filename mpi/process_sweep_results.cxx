#include <algorithm>
using std::sort;

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


map<string, map<string, vector<RNN_Genome*>>> genome_map;

vector<RunStatistics*> run_statistics;


bool extension_is(string name, string extension) {
    if (name.size() < 4) return false;

    string ext = name.substr(name.length() - 4, 4);

    //cout << "comparing '" << ext << "' to '" << extension << "'" << endl;

    return ext.compare(extension) == 0;
}

string current_output;
string current_run_type;

void process_dir(string dir_name, int depth) {
    DIR *dir;
    struct dirent *ent;
    if ((dir = opendir(dir_name.c_str())) != NULL) {
        /* print all the files and directories within directory */
        while ((ent = readdir(dir)) != NULL) {
            if (ent->d_name[0] == '.') continue;
            if (strcmp(ent->d_name, "logs") == 0 && depth == 0) continue;

            if (depth == 0) {
                current_output = ent->d_name;
                cout << "set current output to '" << current_output << "'" << endl;
            }

            if (depth == 1) {
                current_run_type = ent->d_name;
                cout << "set current run type to '" << current_run_type << "'" << endl;
            }

            if (depth == 2 && extension_is(ent->d_name, ".csv")) {
                continue;
            }

            string sub_dir_name = dir_name + "/" + ent->d_name;
            //cout << sub_dir_name << ", depth: " << depth << endl;

            if (depth == 3 && extension_is(sub_dir_name, ".bin")) {
                cout << "\tprocessing genome binary '" << sub_dir_name << "' for '" << current_output << "'" << " and " << current_run_type;
                RNN_Genome *genome = new RNN_Genome(sub_dir_name);
                cout << ", fitness: " << genome->get_fitness() << endl;

                genome_map[current_output][current_run_type].push_back(genome);
            } else if (depth < 3) {
                process_dir(sub_dir_name, depth + 1);
            }
        }
        closedir(dir);

    } else {
        /* could not open directory */
        //cerr << "ERROR: could not open the directory '" << dir_name << "'" << endl;
    }
}



int main(int argc, char** argv) {
    vector<string> arguments = vector<string>(argv, argv + argc);

    string path = arguments[1];
    process_dir(path, 0);

    vector<string> output_types;
    output_types.push_back("flame");
    output_types.push_back("oil");
    output_types.push_back("rpm");
    output_types.push_back("pitch");


    cout << "search_name,slice_name,best_mse,best_mae,n_edges,n_rec_edges,n_nodes,n_ff,n_lstm,n_ugrnn,n_delta,n_mgu,n_gru" << endl;

    //iterate over the map of output names
    for (auto i = genome_map.begin(); i != genome_map.end(); i++) {
        string output_name = i->first;

        //iterate over the map of run types
        for (auto j = i->second.begin(); j != i->second.end(); j++) {
            string run_type = j->first;

            RunStatistics *rs = new RunStatistics(output_name, run_type);

            //iterate over the vector of genomes
            for (auto k = j->second.begin(); k != j->second.end(); k++) {
                RNN_Genome *genome = *k;

                cout << i->first << "," << j->first << ","
                    << genome->get_best_validation_mse() << ","
                    << genome->get_best_validation_mae() << ","
                    << genome->get_enabled_edge_count() << ","
                    << genome->get_enabled_recurrent_edge_count() << ","
                    << genome->get_enabled_node_count() << ","
                    << genome->get_enabled_node_count(SIMPLE_NODE) << ","
                    << genome->get_enabled_node_count(LSTM_NODE) << ","
                    << genome->get_enabled_node_count(UGRNN_NODE) << ","
                    << genome->get_enabled_node_count(DELTA_NODE) << ","
                    << genome->get_enabled_node_count(MGU_NODE) << ","
                    << genome->get_enabled_node_count(GRU_NODE) << endl;

                rs->mse.track(genome->get_best_validation_mse());
                rs->mae.track(genome->get_best_validation_mae());
                rs->edge.track(genome->get_enabled_edge_count());
                rs->rec_edge.track(genome->get_enabled_recurrent_edge_count());

                rs->node.track(genome->get_enabled_node_count());
                rs->ff.track(genome->get_enabled_node_count(SIMPLE_NODE));
                rs->lstm.track(genome->get_enabled_node_count(LSTM_NODE));
                rs->ugrnn.track(genome->get_enabled_node_count(UGRNN_NODE));
                rs->delta.track(genome->get_enabled_node_count(DELTA_NODE));
                rs->mgu.track(genome->get_enabled_node_count(MGU_NODE));
                rs->gru.track(genome->get_enabled_node_count(GRU_NODE));
            }
            cout << endl;

            cout << rs->to_string_min() << endl;
            cout << rs->to_string_avg() << endl;
            cout << rs->to_string_max() << endl;
            cout << rs->to_string_stddev() << endl;
            cout << rs->to_string_correlate("mse", rs->mse) << endl;
            cout << rs->to_string_correlate("mae", rs->mae) << endl;

            /*
               cout << i->first << ",min," << mse.min() << "," << mae.min() << "," << edge.min() << "," << rec_edge.min() << "," << node.min() << "," << ff.min() << "," << lstm.min() << "," << ugrnn.min() << "," << delta.min() << "," << mgu.min() << "," << gru.min() << endl;
               cout << i->first << ",avg," << mse.avg() << "," << mae.avg() << "," << edge.avg() << "," << rec_edge.avg() << "," << node.avg() << "," << ff.avg() << "," << lstm.avg() << "," << ugrnn.avg() << "," << delta.avg() << "," << mgu.avg() << "," << gru.avg() << endl;
               cout << i->first << ",max," << mse.max() << "," << mae.max() << "," << edge.max() << "," << rec_edge.max() << "," << node.max() << "," << ff.max() << "," << lstm.max() << "," << ugrnn.max() << "," << delta.max() << "," << mgu.max() << "," << gru.max() << endl;
               cout << i->first << ",stddev," << mse.stddev() << "," << mae.stddev() << "," << edge.stddev() << "," << rec_edge.stddev() << "," << node.stddev() << "," << ff.stddev() << "," << lstm.stddev() << "," << ugrnn.stddev() << "," << delta.stddev() << "," << mgu.stddev() << "," << gru.stddev() << endl;
               cout << i->first << ",mse correlation," << mse.correlate(mse) << "," << mae.correlate(mse) << "," << edge.correlate(mse) << "," << rec_edge.correlate(mse) << "," << node.correlate(mse) << "," << ff.correlate(mse) << "," << lstm.correlate(mse) << "," << ugrnn.correlate(mse) << "," << delta.correlate(mse) << "," << mgu.correlate(mse) << "," << gru.correlate(mse) << endl;
               cout << i->first << ",mae correlation," << mse.correlate(mae) << "," << mae.correlate(mae) << "," << edge.correlate(mae) << "," << rec_edge.correlate(mae) << "," << node.correlate(mae) << "," << ff.correlate(mae) << "," << lstm.correlate(mae) << "," << ugrnn.correlate(mae) << "," << delta.correlate(mae) << "," << mgu.correlate(mae) << "," << gru.correlate(mae) << endl;
               */

            cout << endl;
            cout << endl;


            run_statistics.push_back(rs);

        }
    }


    //process the kfold sweep directories, these should start with "sweep"
    DIR *dir, *subdir;
    struct dirent *ent;
    if ((dir = opendir(path.c_str())) != NULL) {
        /* print all the files and directories within directory */
        while ((ent = readdir(dir)) != NULL) {
            if (ent->d_name[0] != 's') continue;

            string sweep_directory = ent->d_name;
            cout << "processing sweep directory: " << ent->d_name << endl;
            string sub_dir_name = path + "/" + ent->d_name;
            if ((subdir = opendir(sub_dir_name.c_str())) != NULL) {
                while ((ent = readdir(subdir)) != NULL) {
                    if (strlen(ent->d_name) > 4 && extension_is(ent->d_name, ".csv")) {

                        Tracker mse;
                        Tracker mae;

                        cout << "processing '" << (sub_dir_name + "/" + ent->d_name) << "'" << endl;
                        ifstream infile(sub_dir_name + "/" + ent->d_name);

                        string line;

                        while (getline(infile, line)) {
                            //cout << line << endl;

                            stringstream ss(line);
                            string s;

                            getline(ss, s, ',');
                            //int fold = stoi(s);

                            getline(ss, s, ',');
                            //int repeat = stoi(s);

                            getline(ss, s, ',');
                            //int runtime = stoi(s);

                            getline(ss, s, ',');
                            //double training_mse = stod(s);

                            getline(ss, s, ',');
                            //double training_mae = stod(s);

                            getline(ss, s, ',');
                            double test_mse = stod(s);

                            getline(ss, s, ',');
                            double test_mae = stod(s);

                            /*
                               cout << "fold: " << fold << endl;
                               cout << "repeat: " << repeat << endl;
                               cout << "runtime: " << runtime << endl;
                               cout << "training_mse: " << training_mse << endl;
                               cout << "training_mae: " << training_mae << endl;
                               cout << "test_mse: " << test_mse << endl;
                               cout << "test_mae: " << test_mae << endl;
                               */

                            mse.track(test_mse);
                            mae.track(test_mae);
                        }

                        infile.close();

                        int length = strlen(ent->d_name);
                        string search_type(ent->d_name);
                        search_type = search_type.substr(9, length - 13);

                        cout << search_type << ",min," << mse.min() << "," << mae.min() << endl;
                        cout << search_type << ",avg," << mse.avg() << "," << mae.avg() << endl;
                        cout << search_type << ",max," << mse.max() << "," << mae.max() << endl;
                        cout << search_type << ",stddev," << mse.stddev() << "," << mae.stddev() << endl;

                        cout << endl;
                        cout << endl;

                        //run_statistics.push_back(new RunStatistics(sweep_directory + "_" + search_type, mse, mae));
                    } else {
                        //cout << "skipping: '" << ent->d_name << "'" << endl;
                    }

                }

            } else {
                /* could not open directory */
                //cerr << "ERROR: could not open the directory '" << dir_name << "'" << endl;
            }

        }
        closedir(dir);

    } else {
        /* could not open directory */
        //cerr << "ERROR: could not open the directory '" << dir_name << "'" << endl;
    }




    for (int i = 0; i < output_types.size(); i++) {
        cout << endl << endl;

        cout << run_statistics[0]->overview_header();

        for (int j = 0; j < run_statistics.size(); j++) {
            if (run_statistics[j]->output_name.compare(output_types[i]) == 0) {
                string run_type = run_statistics[j]->run_type;
                if (run_type.find("simple") == string::npos || run_type.find("all") != string::npos) {
                    cout << run_statistics[j]->to_overview_string() << endl;
                }
            }
        }

        cout << run_statistics[0]->overview_footer(output_types[i]);
    }

    cout << endl << endl << endl;

    for (int i = 0; i < output_types.size(); i++) {
        cout << endl << endl;

        cout << run_statistics[0]->overview_ff_header();

        for (int j = 0; j < run_statistics.size(); j++) {
            if (run_statistics[j]->output_name.compare(output_types[i]) == 0) {
                string run_type = run_statistics[j]->run_type;
                if (run_type.find("simple") != string::npos && run_type.find("all") == string::npos) {
                    cout << run_statistics[j]->to_overview_ff_string() << endl;
                }
            }
        }

        cout << run_statistics[0]->overview_ff_footer(output_types[i]);
    }

    cout << endl << endl << endl;



    cout << "\\begin{table}" << endl;
    cout << "\\begin{scriptsize}" << endl;
    cout << "\\centering" << endl;
    cout << "\\begin{tabular}{|l|r|r|r|r|r|r|r|r|r|}" << endl;

    cout << "\\hline" << endl;
    cout << run_statistics[0]->correlate_header() << endl;
    cout << "\\hline" << endl;

    for (int i = 0; i < run_statistics.size(); i++) {
        if (run_statistics[i]->run_type.find("all") != string::npos) {
            cout << run_statistics[i]->to_string_correlate("mse", run_statistics[i]->mse) << "\\\\" << endl;
        }
    }

    cout << "\\hline" << endl;
    cout << "\\end{tabular}" << endl;
    cout << "\\caption{\\label{table:consolidated_rankings} Hidden node count correlations for EXAMM runs evolving all memory neuron types.}" << endl;
    cout << "\\end{scriptsize}" << endl;
    cout << "\\end{table}" << endl;

    cout << "\\begin{table}" << endl;
    cout << "\\begin{scriptsize}" << endl;
    cout << "\\centering" << endl;
    cout << "\\begin{tabular}{|l|r|r|r|r|r|r|r|r|r|r|r|r|r|r|}" << endl;

    cout << "\\hline" << endl;
    cout << " "
        << "& \\multicolumn{4}{|c|}{FF}"
        << "& \\multicolumn{4}{|c|}{LSTM}"
        << "& \\multicolumn{4}{|c|}{UGRNN}"
        << "& \\multicolumn{4}{|c|}{Delta}"
        << "& \\multicolumn{4}{|c|}{MGU}"
        << "& \\multicolumn{4}{|c|}{GRU}"
        << "\\\\" << endl;

    cout << "\\hline" << endl;
    cout << "Run Type"
        << " & Min & Avg & Max & Corr"
        << " & Min & Avg & Max & Corr"
        << " & Min & Avg & Max & Corr"
        << " & Min & Avg & Max & Corr"
        << " & Min & Avg & Max & Corr"
        << " & Min & Avg & Max & Corr"
        << "\\\\" << endl;

    cout << "\\hline" << endl;
    cout << "\\hline" << endl;

    for (int i = 0; i < run_statistics.size(); i++) {
        if (run_statistics[i]->run_type.find("all") != string::npos) {
            cout << run_statistics[i]->output_name
                << " & " << run_statistics[i]->ff.min() << " & " << setprecision(1) << run_statistics[i]->ff.avg() << " & " << run_statistics[i]->ff.max() << "&" << setprecision(2) << run_statistics[i]->ff.correlate(run_statistics[i]->mse)
                << " & " << run_statistics[i]->lstm.min() << " & " << setprecision(1) << run_statistics[i]->lstm.avg() << " & " << run_statistics[i]->lstm.max() << "&" << setprecision(2) << run_statistics[i]->lstm.correlate(run_statistics[i]->mse)
                << " & " << run_statistics[i]->ugrnn.min() << " & " << setprecision(1) << run_statistics[i]->ugrnn.avg() << " & " << run_statistics[i]->ugrnn.max() << "&" << setprecision(2) << run_statistics[i]->ugrnn.correlate(run_statistics[i]->mse)
                << " & " << run_statistics[i]->delta.min() << " & " << setprecision(1) << run_statistics[i]->delta.avg() << " & " << run_statistics[i]->delta.max() << "&" << setprecision(2) << run_statistics[i]->delta.correlate(run_statistics[i]->mse)
                << " & " << run_statistics[i]->mgu.min() << " & " << setprecision(1) << run_statistics[i]->mgu.avg() << " & " << run_statistics[i]->mgu.max() << "&" << setprecision(2) << run_statistics[i]->mgu.correlate(run_statistics[i]->mse)
                << " & " << run_statistics[i]->gru.min() << " & " << setprecision(1) << run_statistics[i]->gru.avg() << " & " << run_statistics[i]->gru.max() << "&" << setprecision(2) << run_statistics[i]->gru.correlate(run_statistics[i]->mse)
                << "\\\\" << endl;
        }
    }

    cout << "\\hline" << endl;
    cout << "\\end{tabular}" << endl;
    cout << "\\caption{\\label{table:consolidated_rankings} Hidden node count correlations for EXAMM runs evolving all memory neuron types.}" << endl;
    cout << "\\end{scriptsize}" << endl;
    cout << "\\end{table}" << endl;







    map<string, vector<RunStatistics*>> output_sorted_statistics;
    cout << endl << endl;

    for (int i = 0; i < output_types.size(); i++) {
        for (int j = 0; j < run_statistics.size(); j++) {
            if (run_statistics[j]->output_name.compare(output_types[i]) == 0) {
                output_sorted_statistics[output_types[i]].push_back(run_statistics[j]);
            }
        }
    }

    vector<string> run_types;
    run_types.push_back("all_norec");
    run_types.push_back("all_rec");
    run_types.push_back("ff_norec");
    run_types.push_back("ff_rec");

    run_types.push_back("delta_norec");
    run_types.push_back("lstm_norec");
    run_types.push_back("mgu_norec");
    run_types.push_back("gru_norec");
    run_types.push_back("ugrnn_norec");

    run_types.push_back("delta_rec");
    run_types.push_back("lstm_rec");
    run_types.push_back("mgu_rec");
    run_types.push_back("gru_rec");
    run_types.push_back("ugrnn_rec");

    run_types.push_back("delta_simple_norec");
    run_types.push_back("lstm_simple_norec");
    run_types.push_back("mgu_simple_norec");
    run_types.push_back("gru_simple_norec");
    run_types.push_back("ugrnn_simple_norec");

    run_types.push_back("delta_simple_rec");
    run_types.push_back("lstm_simple_rec");
    run_types.push_back("mgu_simple_rec");
    run_types.push_back("gru_simple_rec");
    run_types.push_back("ugrnn_simple_rec");

    map<string, ConsolidatedStatistics*> consolidated_statistics;

    for (int i = 0; i < run_types.size(); i++) {
        consolidated_statistics[run_types[i]] = new ConsolidatedStatistics(run_types[i]);
    }

    for (int i = 0; i < output_types.size(); i++) {
        cout << "GENERATING sorted stats for: '" << output_types[i] << "'" << endl;

        vector<RunStatistics*> current = output_sorted_statistics[output_types[i]];

        cout << "got current!" << endl;

        double avg_min = 0.0;
        double avg_avg = 0.0;
        double avg_max = 0.0;

        for (int j = 0; j < current.size(); j++) {
            avg_min += current[j]->mse.min();
            avg_avg += current[j]->mse.avg();
            avg_max += current[j]->mse.max();
        }

        avg_min /= current.size();
        avg_avg /= current.size();
        avg_max /= current.size();

        cout << "calculated averages" << endl;

        double stddev_min = 0.0;
        double stddev_avg = 0.0;
        double stddev_max = 0.0;

        for (int j = 0; j < current.size(); j++) {
            stddev_min += (current[j]->mse.min() - avg_min) * (current[j]->mse.min() - avg_min);
            stddev_avg += (current[j]->mse.avg() - avg_avg) * (current[j]->mse.avg() - avg_avg);
            stddev_max += (current[j]->mse.max() - avg_max) * (current[j]->mse.max() - avg_max);
        }

        stddev_min = sqrt(stddev_min / (current.size() - 1));
        stddev_avg = sqrt(stddev_avg / (current.size() - 1));
        stddev_max = sqrt(stddev_max / (current.size() - 1));

        cout << "calculated stddevs!" << endl;

        for (int j = 0; j < current.size(); j++) {
            cout << "current[" << j << "]->run_type: " << current[j]->run_type << endl;

            current[j]->set_deviation_from_mean_min((current[j]->mse.min() - avg_min) / stddev_min);
            current[j]->set_deviation_from_mean_avg((current[j]->mse.avg() - avg_avg) / stddev_avg);
            current[j]->set_deviation_from_mean_max((current[j]->mse.max() - avg_max) / stddev_max);

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

    sort(min_stats_vector.begin(), min_stats_vector.end(), cs_less_than_min());
    sort(avg_stats_vector.begin(), avg_stats_vector.end(), cs_less_than_avg());
    sort(max_stats_vector.begin(), max_stats_vector.end(), cs_less_than_max());

    cout << endl << endl;
    cout << "\\begin{table}" << endl;
    cout << "\\begin{scriptsize}" << endl;
    cout << "\\centering" << endl;
    cout << "\\begin{tabular}{|r|l||r|l||r|l||}" << endl;
    cout << "\\hline" << endl;
    cout << "\\multicolumn{2}{|c||}{{\\bf Best Case}} & \\multicolumn{2}{|c||}{{\\bf Avg. Case}} & \\multicolumn{2}{|c|}{{\\bf Worst Case}} \\\\" << endl;
    cout << "\\hline" << endl;

    for (int i = 0; i < min_stats_vector.size(); i++) {
        cout << setw(15) << fix_run_type(min_stats_vector[i]->run_type) << " & " << setw(15) << setprecision(5) << min_stats_vector[i]->dfm_min << " & ";
        cout << setw(15) << fix_run_type(avg_stats_vector[i]->run_type) << " & " << setw(15) << setprecision(5) << avg_stats_vector[i]->dfm_avg << " & ";
        cout << setw(15) << fix_run_type(max_stats_vector[i]->run_type) << " & " << setw(15) << setprecision(5) << max_stats_vector[i]->dfm_max << "\\\\" << endl;
    }

    cout << "\\hline" << endl;
    cout << "\\end{tabular}" << endl;
    cout << "\\caption{\\label{table:consolidated_rankings} Combined rankings for all EXAMM run types.}" << endl;
    cout << "\\end{scriptsize}" << endl;
    cout << "\\end{table}" << endl;

    return 0;
}

