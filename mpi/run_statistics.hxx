#ifndef EXACT_RUN_STATISTICS
#define EXACT_RUN_STATISTICS

#include <string>
using std::string;

#include "tracker.hxx"

string fix_run_type(string run_type);

class ConsolidatedStatistics {
    public:
        string run_type;

        double dfm_min;
        double dfm_avg;
        double dfm_max;

        ConsolidatedStatistics(string _run_type);

        string to_string_min();
        string to_string_avg();
        string to_string_max();
};

struct cs_less_than_min {
    inline bool operator() (const ConsolidatedStatistics *s1, const ConsolidatedStatistics *s2)
    {
        return (s1->dfm_min < s2->dfm_min);
    }
};

struct cs_less_than_avg {
    inline bool operator() (const ConsolidatedStatistics *s1, const ConsolidatedStatistics *s2)
    {
        return (s1->dfm_avg < s2->dfm_avg);
    }
};

struct cs_less_than_max {
    inline bool operator() (const ConsolidatedStatistics *s1, const ConsolidatedStatistics *s2)
    {
        return (s1->dfm_max < s2->dfm_max);
    }
};


class RunStatistics {
    public:
        string output_name;
        string run_type;

        double dfm_min;
        double dfm_avg;
        double dfm_max;

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

        RunStatistics(string _output_name, string _run_type);

        void set_deviation_from_mean_min(double _dfm_min);
        void set_deviation_from_mean_avg(double _dfm_avg);
        void set_deviation_from_mean_max(double _dfm_max);

        string correlate_header();

        string to_string_min();
        string to_string_avg();
        string to_string_max();
        string to_string_stddev();
        string to_string_correlate(string target_name, Tracker &target);

        string overview_header();
        string overview_footer(string type);
        string to_overview_string();

        string overview_ff_header();
        string overview_ff_footer(string type);
        string to_overview_ff_string();
};

struct less_than_min {
    inline bool operator() (const RunStatistics *s1, const RunStatistics *s2)
    {
        return (s1->mae.min() < s2->mae.min());
    }
};

struct less_than_avg {
    inline bool operator() (const RunStatistics *s1, const RunStatistics *s2)
    {
        return (s1->mae.avg() < s2->mae.avg());
    }
};

struct less_than_max {
    inline bool operator() (const RunStatistics *s1, const RunStatistics *s2)
    {
        return (s1->mae.max() < s2->mae.max());
    }
};

struct less_than_dfm_min {
    inline bool operator() (const RunStatistics *s1, const RunStatistics *s2)
    {
        return (s1->dfm_min < s2->dfm_min);
    }
};

struct less_than_dfm_avg {
    inline bool operator() (const RunStatistics *s1, const RunStatistics *s2)
    {
        return (s1->dfm_avg < s2->dfm_avg);
    }
};

struct less_than_dfm_max {
    inline bool operator() (const RunStatistics *s1, const RunStatistics *s2)
    {
        return (s1->dfm_max < s2->dfm_max);
    }
};





#endif
