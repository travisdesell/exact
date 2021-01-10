DatasetMeta::DatasetMeta(
        const vector<string> &_input_parameter_names,
        const vector<string> &_output_parameter_names,
        const string _normalize_type,
        const map<string,double> &_normalize_mins,
        const map<string,double> &_normalize_maxs,
        const map<string,double> &_normalize_avgs,
        const map<string,double> &_normalize_std_devs) 
    :   input_parameter_names(_input_parameter_names),
        output_parameter_names(_output_parameter_names),
        normalize_type(_normalize_type),
        normalize_mins(_normalize_mins),
        normalize_maxs(_normalize_maxs),
        normalize_avgs(_normalize_avgs),
        normalize_std_devs(_normalize_std_devs) {
    // Test
    input_parameter_names.push("hi");
}
