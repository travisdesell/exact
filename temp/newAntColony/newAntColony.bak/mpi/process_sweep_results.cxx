#include <chrono>

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

map<string, map<string, vector<RNN_Genome*>>> genome_map;


bool extension_is(string name, string extension) {
    if (name.size() < 4) return false;

    string ext = name.substr(name.length() - 4, 4);

    //cout << "comparing '" << ext << "' to '" << extension << "'" << endl;

    return ext.compare(extension) == 0;
}

string current_search;
string current_slice;

void process_dir(string dir_name, int depth) {
    DIR *dir;
    struct dirent *ent;
    if ((dir = opendir(dir_name.c_str())) != NULL) {
        /* print all the files and directories within directory */
        while ((ent = readdir(dir)) != NULL) {
            if (ent->d_name[0] == '.') continue;

            if (depth == 0) {
                current_search = ent->d_name;
                cout << "set current search to '" << current_search << "'" << endl;
            }

            if (depth == 1 && !extension_is(ent->d_name, ".csv")) {
                current_slice = ent->d_name;
                cout << "set current slice to '" << current_slice << "'" << endl;
            }

            string sub_dir_name = dir_name + "/" + ent->d_name;
            //cout << sub_dir_name << ", depth: " << depth << endl;

            if (depth == 2 && extension_is(sub_dir_name, ".bin")) {
                //cout << "processing genome binary '" << sub_dir_name << "' for '" << current_search << "'" << endl;
                //genomes.push_back(new RepeatGenome(current_search, current_slice, sub_dir_name));

                genome_map[current_search][current_slice].push_back(new RNN_Genome(sub_dir_name, false));
            } else if (depth < 2) {
                process_dir(sub_dir_name, depth + 1);
            }
        }
        closedir(dir);

    } else {
        /* could not open directory */
        //cerr << "ERROR: could not open the directory '" << dir_name << "'" << endl;
    }

}

class Tracker {
    private:
        int32_t count;
        double _min;
        double sum;
        double _max;

        vector<double> values;

    public:
        Tracker() {
            count = 0;
            sum = 0.0;
            _min = 999999;
            _max = -999999;
        }

        void track(double value) {
            if (value < _min) _min = value;
            if (value > _max) _max = value;

            count++;
            sum += value;

            values.push_back(value);
        }

        double min() {
            return _min;
        }

        double max() {
            return _max;
        }

        double avg() {
            return sum / count;
        }

        double stddev() {
            double _avg = avg();
            double _stddev = 0;

            for (int i = 0; i < values.size(); i++) {
                double tmp = (values[i] - _avg);
                _stddev += tmp * tmp;
            }

            _stddev = sqrt(_stddev / (values.size() - 1));

            return _stddev;
        }

        double correlate(Tracker &other) {
            double avg1 = avg();
            double avg2 = other.avg();

            double stddev1 = stddev();
            double stddev2 = other.stddev();

            double correlation = 0.0;

            for (int i = 0; i < values.size(); i++) {
                correlation += (values[i] - avg1) * (other.values[i] - avg2);
            }

            correlation /= (count - 1) * stddev1 * stddev2;

            return correlation;
        }

};

int main(int argc, char** argv) {
    vector<string> arguments = vector<string>(argv, argv + argc);

    string path = arguments[1];
    process_dir(path, 0);

    cout << "search_name,slice_name,best_mse,best_mae,n_edges,n_rec_edges,n_nodes,n_ff,n_lstm,n_ugrnn,n_delta,n_mgu,n_gru" << endl;

    //iterate over the map of search names
    for (auto i = genome_map.begin(); i != genome_map.end(); i++) {

        Tracker mse;
        Tracker mae;
        Tracker edge;
        Tracker rec_edge;
        Tracker node;
        Tracker ff;
        Tracker lstm;
        Tracker ugrnn;
        Tracker delta;
        Tracker mgu;
        Tracker gru;

        //iterate over the map of slice names
        for (auto j = i->second.begin(); j != i->second.end(); j++) {

            //iterate over the vector of genomes
            for (auto k = j->second.begin(); k != j->second.end(); k++) {
                RNN_Genome *genome = *k;

                /*
                cout << i->first << "," << j->first << "," 
                    << genome->get_best_validation_mse() << "," 
                    << genome->get_best_validation_mae() << "," 
                    << genome->get_enabled_edge_count() << "," 
                    << genome->get_enabled_recurrent_edge_count() << ","
                    << genome->get_enabled_node_count() << ","
                    << genome->get_enabled_node_count(FEED_FORWARD_NODE) << ","
                    << genome->get_enabled_node_count(LSTM_NODE) << ","
                    << genome->get_enabled_node_count(UGRNN_NODE) << ","
                    << genome->get_enabled_node_count(DELTA_NODE) << ","
                    << genome->get_enabled_node_count(MGU_NODE) << ","
                    << genome->get_enabled_node_count(GRU_NODE) << endl;
                    */

                mse.track(genome->get_best_validation_mse());
                mae.track(genome->get_best_validation_mae());
                edge.track(genome->get_enabled_edge_count());
                rec_edge.track(genome->get_enabled_recurrent_edge_count());

                node.track(genome->get_enabled_node_count());
                ff.track(genome->get_enabled_node_count(FEED_FORWARD_NODE));
                lstm.track(genome->get_enabled_node_count(LSTM_NODE));
                ugrnn.track(genome->get_enabled_node_count(UGRNN_NODE));
                delta.track(genome->get_enabled_node_count(DELTA_NODE));
                mgu.track(genome->get_enabled_node_count(MGU_NODE));
                gru.track(genome->get_enabled_node_count(GRU_NODE));
            }
        }

        cout << i->first << ",min," << mse.min() << "," << mae.min() << "," << edge.min() << "," << rec_edge.min() << "," << node.min() << "," << ff.min() << "," << lstm.min() << "," << ugrnn.min() << "," << delta.min() << "," << mgu.min() << "," << gru.min() << endl;
        cout << i->first << ",avg," << mse.avg() << "," << mae.avg() << "," << edge.avg() << "," << rec_edge.avg() << "," << node.avg() << "," << ff.avg() << "," << lstm.avg() << "," << ugrnn.avg() << "," << delta.avg() << "," << mgu.avg() << "," << gru.avg() << endl;
        cout << i->first << ",max," << mse.max() << "," << mae.max() << "," << edge.max() << "," << rec_edge.max() << "," << node.max() << "," << ff.max() << "," << lstm.max() << "," << ugrnn.max() << "," << delta.max() << "," << mgu.max() << "," << gru.max() << endl;
        cout << i->first << ",stddev," << mse.stddev() << "," << mae.stddev() << "," << edge.stddev() << "," << rec_edge.stddev() << "," << node.stddev() << "," << ff.stddev() << "," << lstm.stddev() << "," << ugrnn.stddev() << "," << delta.stddev() << "," << mgu.stddev() << "," << gru.stddev() << endl;
        cout << i->first << ",mse correlation," << mse.correlate(mse) << "," << mae.correlate(mse) << "," << edge.correlate(mse) << "," << rec_edge.correlate(mse) << "," << node.correlate(mse) << "," << ff.correlate(mse) << "," << lstm.correlate(mse) << "," << ugrnn.correlate(mse) << "," << delta.correlate(mse) << "," << mgu.correlate(mse) << "," << gru.correlate(mse) << endl;
        cout << i->first << ",mae correlation," << mse.correlate(mae) << "," << mae.correlate(mae) << "," << edge.correlate(mae) << "," << rec_edge.correlate(mae) << "," << node.correlate(mae) << "," << ff.correlate(mae) << "," << lstm.correlate(mae) << "," << ugrnn.correlate(mae) << "," << delta.correlate(mae) << "," << mgu.correlate(mae) << "," << gru.correlate(mae) << endl;

        cout << endl;
    }

    /*
    string genome_filename;
    get_argument(arguments, "--genome_file", true, genome_filename);
    RNN_Genome *genome = new RNN_Genome(genome_filename, true);

    vector<string> testing_filenames;
    get_argument_vector(arguments, "--testing_filenames", true, testing_filenames);

    TimeSeriesSets *time_series_sets = TimeSeriesSets::generate_test(testing_filenames, genome->get_input_parameter_names(), genome->get_output_parameter_names());
    cout << "got time series sets" << endl;
    time_series_sets->normalize(genome->get_normalize_mins(), genome->get_normalize_maxs());
    cout << "normalized time series." << endl;

    int32_t time_offset = 1;
    get_argument(arguments, "--time_offset", true, time_offset);

    time_series_sets->export_test_series(time_offset, testing_inputs, testing_outputs);


    vector<double> best_parameters = genome->get_best_parameters();
    cout << "MSE: " << genome->get_mse(best_parameters, testing_inputs, testing_outputs) << endl;
    cout << "MAE: " << genome->get_mae(best_parameters, testing_inputs, testing_outputs) << endl;
    genome->write_predictions(testing_filenames, best_parameters, testing_inputs, testing_outputs);

    */

    return 0;
}
