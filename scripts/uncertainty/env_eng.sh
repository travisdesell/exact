INPUT_PARAMETERS="E1_CHT1 E1_CHT2 E1_CHT3 E1_CHT4 E1_EGT1 E1_EGT2 E1_EGT3 E1_EGT4 E1_FFlow E1_OilP E1_OilT E1_RPM"  
OUTPUT_PARAMETERS="E1_OilP" 

TRAIN_FILES=$EXAMM_HOME/data/ngafid_sens_rnn/train/*.csv
TEST_FILES=$EXAMM_HOME/data/ngafid_sens_rnn/test/*.csv

NORM_METHOD=min_max

exp_name="$EXAMM_HOME/test_output/ngafid_c712_917"
mkdir -p $exp_name
