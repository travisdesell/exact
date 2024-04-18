#!/usr/bin/zsh

INPUT_PARAMETERS='AltAGL AltB AltGPS AltMSL BaroA E1_CHT1 E1_CHT2 E1_CHT3 E1_CHT4 E1_EGT1 E1_EGT2 E1_EGT3 E1_EGT4 E1_FFlow E1_OilP E1_OilT E1_RPM FQtyL FQtyR GndSpd IAS LatAc NormAc OAT Pitch Roll TAS VSpd VSpdG WndDr WndSpd'
OUTPUT_PARAMETERS='E1_EGT1 E1_EGT2 E1_EGT3 E1_EGT4 E1_FFlow E1_OilP E1_OilT E1_RPM'

offset=1
bp_epoch=1000

for SIZE in 1 2 4; do
  for CELL_TYPE in dnas; do
    for fold in 0 1 2 3 4 5 6 7 8 9; do
      output_dir=ground_truth_experiments/results/$CELL_TYPE/$SIZE/$fold
      mkdir -p $output_dir
      Release/rnn_examples/train_rnn \
          --training_filenames datasets/2019_ngafid_transfer/c172_file_[1-9].csv \
          --test_filenames datasets/2019_ngafid_transfer/c172_file_1[0-2].csv \
          --time_offset $offset \
          --input_parameter_names ${=INPUT_PARAMETERS} \
          --output_parameter_names ${=OUTPUT_PARAMETERS} \
          --bp_iterations $bp_epoch \
          --stochastic \
          --rnn_type $CELL_TYPE \
          --normalize min_max \
          --num_hidden_layers $SIZE \
          --hidden_layer_size $SIZE \
          --random_sequence_length \
          --sequence_length_lower_bound 50 \
          --sequence_length_upper_bound 100 \
          --max_recurrent_depth 1 \
          --weight_update adagrad \
          --output_directory $output_dir \
          --log_filename fitness.csv \
          --learning_rate 0.01 \
          --std_message_level ERROR \
          --file_message_level INFO &
    done
  done
  wait
done

