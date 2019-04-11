#include <iostream>
using std::endl;


#include <iomanip>
using std::setw;
using std::setprecision;

#include <regex>
using std::regex;
using std::regex_replace;

#include <string>
using std::string;

#include <sstream>
using std::ostringstream;

#include "tracker.hxx"
#include "run_statistics.hxx"



string fix_run_type(string run_type) {
    run_type = regex_replace(run_type, regex("_norec"), "");
    run_type = regex_replace(run_type, regex("_"), "+");

    return run_type;
}

ConsolidatedStatistics::ConsolidatedStatistics(string _run_type) : run_type(_run_type) {
    dfm_min = 0.0;
    dfm_avg = 0.0;
    dfm_max = 0.0;

}

string ConsolidatedStatistics::to_string_min() {
    ostringstream oss;

    oss << fix_run_type(run_type) << "," << dfm_min;

    return oss.str();
}

string ConsolidatedStatistics::to_string_avg() {
    ostringstream oss;

    oss << fix_run_type(run_type) << "," << dfm_avg;

    return oss.str();
}

string ConsolidatedStatistics::to_string_max() {
    ostringstream oss;

    oss << fix_run_type(run_type) << "," << dfm_max;

    return oss.str();
}


RunStatistics::RunStatistics(string _output_name, string _run_type) : output_name(_output_name), run_type(_run_type) {
}

void RunStatistics::set_deviation_from_mean_min(double _dfm_min) {
    dfm_min = _dfm_min;
}

void RunStatistics::set_deviation_from_mean_avg(double _dfm_avg) {
    dfm_avg = _dfm_avg;
}

void RunStatistics::set_deviation_from_mean_max(double _dfm_max) {
    dfm_max = _dfm_max;
}

string RunStatistics::to_string_min() {
    ostringstream oss;

    oss << output_name << "," << fix_run_type(run_type) << ",min," << mse.min() << "," << mae.min() << "," << edge.min() << "," << rec_edge.min() << "," << node.min() << "," << ff.min() << "," << lstm.min() << "," << ugrnn.min() << "," << delta.min() << "," << mgu.min() << "," << gru.min();

    return oss.str();
}

string RunStatistics::to_string_avg() {
    ostringstream oss;

    oss << output_name << "," << fix_run_type(run_type) << ",avg," << mse.avg() << "," << mae.avg() << "," << edge.avg() << "," << rec_edge.avg() << "," << node.avg() << "," << ff.avg() << "," << lstm.avg() << "," << ugrnn.avg() << "," << delta.avg() << "," << mgu.avg() << "," << gru.avg();

    return oss.str();
}

string RunStatistics::to_string_max() {
    ostringstream oss;


    oss << output_name << "," << fix_run_type(run_type) << ",max," << mse.max() << "," << mae.max() << "," << edge.max() << "," << rec_edge.max() << "," << node.max() << "," << ff.max() << "," << lstm.max() << "," << ugrnn.max() << "," << delta.max() << "," << mgu.max() << "," << gru.max();

    return oss.str();
}

string RunStatistics::to_string_stddev() {
    ostringstream oss;

    oss << output_name << "," << fix_run_type(run_type) << ",stddev," << mse.stddev() << "," << mae.stddev() << "," << edge.stddev() << "," << rec_edge.stddev() << "," << node.stddev() << "," << ff.stddev() << "," << lstm.stddev() << "," << ugrnn.stddev() << "," << delta.stddev() << "," << mgu.stddev() << "," << gru.stddev();

    return oss.str();
}


string RunStatistics::correlate_header() {
    ostringstream oss;
    oss << "Prediction & Edges & Rec. Edges & Hidden Nodes & FF & LSTM & UGRNN & Delta & MGU & GRU \\\\";
    return oss.str();
}

string RunStatistics::to_string_correlate(string target_name, Tracker &target) {
    ostringstream oss;

    oss << output_name << " & " << edge.correlate(target) << " & " << rec_edge.correlate(target) << " & " << node.correlate(target) << " & " << ff.correlate(target) << " & " << lstm.correlate(target) << " & " << ugrnn.correlate(target) << " & " << delta.correlate(target) << " & " << mgu.correlate(target) << " & " << gru.correlate(target);

    return oss.str();
}



string RunStatistics::overview_header() {
    ostringstream oss;
    
    oss << "\\begin{table*}" << endl;
    oss << "\\begin{scriptsize}" << endl;
    oss << "\\centering" << endl;

    oss << "\\begin{tabular}{|l|r|r|r|r|r|r|r|r|r|r|r|r|r|r|r|}" << endl;
    oss << "\\hline" << endl;

    oss << " "
        << " & \\multicolumn{3}{|c|}{MSE}" 
        << " & \\multicolumn{4}{|c|}{Edges}" 
        << " & \\multicolumn{4}{|c|}{Rec. Edges}" 
        << " & \\multicolumn{4}{|c|}{Hidden Nodes}" 
        << "\\\\" << endl;

    oss << "\\hline" << endl;

    oss << "Run Type"
        << " & Min & Avg & Max "
        << " & Min & Avg & Max & Corr."
        << " & Min & Avg & Max & Corr."
        << " & Min & Avg & Max & Corr."
        << "\\\\" << endl;

    oss << "\\hline" << endl;
    oss << "\\hline" << endl;

    return oss.str();
}

string RunStatistics::overview_footer(string type) {
    ostringstream oss;

    oss << "\\hline" << endl;
    oss << "\\end{tabular}" << endl;
    oss << "\\caption{\\label{table:consolidated_rankings} FILL IN FOR '" << type << "'.}" << endl;
    oss << "\\end{scriptsize}" << endl;
    oss << "\\end{table*}" << endl;

    return oss.str();
}


string RunStatistics::to_overview_string() {
    ostringstream oss;

    oss << run_type 
        << " & " << mse.min() << " & " << mse.avg() << " & " << mse.max()
        << " & " << edge.min() << " & " << setprecision(2) << edge.avg() << " & " << edge.max()
        << " & " << setprecision(3) << edge.correlate(mae)
        << " & " << rec_edge.min() << " & " << setprecision(2) << rec_edge.avg() << " & " << rec_edge.max()
        << " & " << setprecision(3) << rec_edge.correlate(mae)
        << " & " << node.min() << " & " << setprecision(2) << node.avg() << " & " << node.max()
        << " & " << setprecision(3) << node.correlate(mae)
        << "\\\\";
        
    return oss.str();
}





string RunStatistics::overview_ff_header() {
    ostringstream oss;
    
    oss << "\\begin{table*}" << endl;
    oss << "\\begin{scriptsize}" << endl;
    oss << "\\centering" << endl;

    oss << "\\begin{tabular}{|l|r|r|r|r|r|r|r|r|r|r|r|r|r|r|r|r|r|r|r|}" << endl;
    oss << "\\hline" << endl;

    oss << " "
        << " & \\multicolumn{3}{|c|}{MSE}" 
        << " & \\multicolumn{4}{|c|}{Edges}" 
        << " & \\multicolumn{4}{|c|}{Rec. Edges}" 
        << " & \\multicolumn{4}{|c|}{Memory Nodes}" 
        << " & \\multicolumn{4}{|c|}{FF Nodes}" 
        << "\\\\" << endl;

    oss << "\\hline" << endl;

    oss << "Run Type"
        << " & Min & Avg & Max "
        << " & Min & Avg & Max & Corr."
        << " & Min & Avg & Max & Corr."
        << " & Min & Avg & Max & Corr."
        << " & Min & Avg & Max & Corr."
        << "\\\\" << endl;

    oss << "\\hline" << endl;
    oss << "\\hline" << endl;

    return oss.str();
}

string RunStatistics::overview_ff_footer(string type) {
    ostringstream oss;

    oss << "\\hline" << endl;
    oss << "\\end{tabular}" << endl;
    oss << "\\caption{\\label{table:consolidated_rankings} FILL IN FOR '" << type << "'.}" << endl;
    oss << "\\end{scriptsize}" << endl;
    oss << "\\end{table*}" << endl;

    return oss.str();
}


string RunStatistics::to_overview_ff_string() {
    ostringstream oss;

    oss << run_type 
        << " & " << setprecision(5) << mse.min() << " & " << setprecision(5) << mse.avg() << " & " << setprecision(5) << mse.max()
        << " & " << edge.min() << " & " << setprecision(2) << edge.avg() << " & " << edge.max()
        << " & " << edge.correlate(mae)
        << " & " << rec_edge.min() << " & " << setprecision(2) << rec_edge.avg() << " & " << rec_edge.max()
        << " & " << setprecision(3) << rec_edge.correlate(mae);

    if (lstm.max() > 0) {
        oss << " & " << lstm.min() << " & " << setprecision(2) << lstm.avg() << " & " << lstm.max()
            << " & " << setprecision(3) << lstm.correlate(mae);
    }

    if (ugrnn.max() > 0) {
        oss << " & " << ugrnn.min() << " & " << setprecision(2) << ugrnn.avg() << " & " << ugrnn.max()
            << " & " << setprecision(3) << ugrnn.correlate(mae);
    }

    if (delta.max() > 0) {
        oss << " & " << delta.min() << " & " << setprecision(2) << delta.avg() << " & " << delta.max()
            << " & " << setprecision(3) << delta.correlate(mae);
    }

    if (mgu.max() > 0) {
        oss << " & " << mgu.min() << " & " << setprecision(2) << mgu.avg() << " & " << mgu.max()
            << " & " << setprecision(3) << mgu.correlate(mae);
    }

    if (gru.max() > 0) {
        oss << " & " << gru.min() << " & " << setprecision(2) << gru.avg() << " & " << gru.max()
            << " & " << setprecision(3) << gru.correlate(mae);
    }

    oss << " & " << ff.min() << " & " << setprecision(2) << ff.avg() << " & " << ff.max()
        << " & " << setprecision(3) << ff.correlate(mae)
        << "\\\\";
        
    return oss.str();
}
