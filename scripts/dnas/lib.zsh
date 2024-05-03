#!/bin/zsh

offset=1
MAX_GENOMES=10
N_ISLANDS=4
ISLAND_SIZE=32

run_examm() {
  output_dir=$output_dir_prefix/bp_$bp_epoch/output_$output_params/$fold
  mkdir -p $output_dir
  echo srun -n $np Release/mpi/examm_mpi \
      --training_filenames ${=training_filenames} \
      --test_filenames ${=test_filenames} \
      --time_offset $offset \
      --input_parameter_names ${=INPUT_PARAMETERS} \
      --output_parameter_names $output_params \
      --bp_iterations $bp_epoch \
      --normalize min_max \
      --max_recurrent_depth 1 \
      --output_directory $output_dir \
      --log_filename fitness.csv \
      --learning_rate 0.01 \
      --std_message_level INFO \
      --file_message_level INFO \
      --max_genomes $MAX_GENOMES \
      --island_size $ISLAND_SIZE \
      --number_islands $N_ISLANDS

  touch $output_dir/completed
}

run_group() {
  for output_params in $OUTPUTS; do
    for bp_epoch in $bp_epoch_set; do
      for fold in $(seq 1 $nfolds); do
        run_examm
      done
    done
  done
}

coal() {
    INPUT_PARAMETERS="Conditioner_Inlet_Temp Conditioner_Outlet_Temp Coal_Feeder_Rate Primary_Air_Flow Primary_Air_Split System_Secondary_Air_Flow_Total Secondary_Air_Flow Secondary_Air_Split Tertiary_Air_Split Total_Comb_Air_Flow Supp_Fuel_Flow Main_Flm_Int" 
    training_filenames=(datasets/2018_coal/burner_[0-9].csv)
    test_filenames=(datasets/2018_coal/burner_1[0-1].csv)
    OUTPUTS=("Main_Flm_Int" "Supp_Fuel_Flow")
    run_group
}

aviation() {
    INPUT_PARAMETERS='AltAGL AltB AltGPS AltMSL BaroA E1_CHT1 E1_CHT2 E1_CHT3 E1_CHT4 E1_EGT1 E1_EGT2 E1_EGT3 E1_EGT4 E1_FFlow E1_OilP E1_OilT E1_RPM FQtyL FQtyR GndSpd IAS LatAc NormAc OAT Pitch Roll TAS VSpd VSpdG WndDr WndSpd'
    OUTPUTS=("E1_CHT1" "Pitch")
    training_filenames=(datasets/2019_ngafid_transfer/c172_file_[1-9].csv)
    test_filenames=(datasets/2019_ngafid_transfer/c172_file_1[0-2].csv)
    run_group
}

wind() {
    INPUT_PARAMETERS="Ba_avg Rt_avg DCs_avg Cm_avg P_avg S_avg Cosphi_avg Db1t_avg Db2t_avg Dst_avg Gb1t_avg Gb2t_avg Git_avg Gost_avg Ya_avg Yt_avg Ws_avg Wa_avg Ot_avg Nf_avg Nu_avg Rbt_avg"
    OUTPUTS=("Cm_avg" "P_avg")
    training_filenames=(datasets/2020_wind_engine/turbine_R80711_2017-2020_[1-9].csv datasets/2020_wind_engine/turbine_R80711_2017-2020_1[0-9].csv datasets/2020_wind_engine/turbine_R80711_2017-2020_2[0-4].csv)
    test_filenames=(datasets/2020_wind_engine/turbine_R80711_2017-2020_2[5-9].csv datasets/2020_wind_engine/turbine_R80711_2017-2020_3[0-1].csv)
    run_group
}

