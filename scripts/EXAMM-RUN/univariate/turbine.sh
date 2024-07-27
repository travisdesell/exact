#!/bin/sh
EXAMM_PATH="/Users/zimenglyu/Documents/code/git/examm_debug/exact"
DATA_PATH="/Users/zimenglyu/Downloads/EXAMM-Data"
RESULT_PATH=$EXAMM_PATH"/AAAI_EXAMM/univariate"
DATASET="turbine"

INPUT_PARAMETERS="Ba_avg  Rt_avg  DCs_avg  Cm_avg  P_avg  Q_avg  S_avg  Cosphi_avg  Ds_avg  Db1t_avg  Db2t_avg  Dst_avg  Gb1t_avg  Gb2t_avg  Git_avg  Gost_avg  Ya_avg  Yt_avg  Ws1_avg  Ws2_avg  Ws_avg  Wa_avg  Va_avg  Ot_avg  Nf_avg  Nu_avg  Rs_avg  Rbt_avg"
OUTPUT_PARAMETERS="P_avg"

for offset in 1 6 12
do
    for repeat in {0..9}
    do 
        exp_name=$RESULT_PATH/$DATASET/offset_${offset}/$repeat
        mkdir -p $exp_name
        echo "Running EXAMM code with ${DATASET} dataset   results will be saved to: "$exp_name
        echo "###----------------------------------------------------------------------------###"

        mpirun -np 4 $EXAMM_PATH/build/mpi/examm_mpi \
        --training_filenames $DATA_PATH/${DATASET}_train.csv \
        --test_filenames $DATA_PATH/${DATASET}_validation.csv \
        --time_offset $offset \
        --input_parameter_names $OUTPUT_PARAMETERS \
        --output_parameter_names $OUTPUT_PARAMETERS \
        --number_islands 10 \
        --island_size 10 \
        --max_genomes 10000 \
        --repopulation_method "bestGenome" \
        --extinction_event_generation_number 150 \
        --islands_to_exterminate 1 \
        --num_mutations 2 \
        --bp_iterations 10 \
        --normalize "avg_std_dev" \
        --train_sequence_length $((50+offset)) \
        --validation_sequence_length $((50+offset)) \
        --output_directory $exp_name \
        --possible_node_types simple UGRNN MGU GRU delta LSTM \
        --std_message_level INFO \
        --file_message_level INFO

        # --------------- Testing Best Genome ----------------
        echo "----------------- Testing Best Genome -----------------"

        GENOME=$exp_name/global_best_genome_*.bin
        echo "Found global best genome: " $GENOME

        TESTFILE=$DATA_PATH/${DATASET}_test.csv

        $EXAMM_PATH/build/rnn_examples/evaluate_rnn \
        --genome_file $GENOME \
        --testing_filenames $TESTFILE \
        --time_offset $offset \
        --output_directory $exp_name \
        --std_message_level INFO \
        --file_message_level INFO
    done
done

