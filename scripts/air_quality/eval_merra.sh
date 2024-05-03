INPUT_PARAMETERS="lon lat lev AIRDENS SO4 SO2 RH PS H O3 T U V"
# OUTPUT_PARAMETERS="CO"
OUTPUT_PARAMETERS="CO SO4 SO2 O3"

exp_name="/home/aidan/sandbox/DEEPSPrj/output/merra/multivar_B/evaluation"
mkdir -p $exp_name

../../build/rnn_examples/evaluate_rnn \
--testing_filenames /home/aidan/sandbox/DEEPSPrj/data/MERRA/merra_eval_1000.csv \
--time_offset 1 \
--input_parameter_names $INPUT_PARAMETERS \
--output_parameter_names $OUTPUT_PARAMETERS \
--genome_file $1 \
--output_directory $exp_name \
--std_message_level INFO \
--file_message_level ERROR
# --bp_iterations $epochs \
