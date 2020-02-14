#!/bin/bash
PARAMS=""
while (( "$#" )); do
  case "$1" in
    -d|--data_dir)
      DATA_DIR=$2
      shift 2
      ;;
    -p|--examm_loc)
      EXAMM=$2
      shift 2
      ;;
    -c|--no_cpu)
      NO_CPU=$2
      shift 2
      ;;
    -o|--out_pars)
      OUT_PAR=$2
      shift 2
      ;;
    -g|--max_gen)
      MAX_GEN=$2
      shift 2
      ;;
    --) # end argument parsing
      shift
      break
      ;;
    -*|--*=) # unsupported flags
      echo "Error: Unsupported flag $1" >&2
      exit 1
      ;;
    *) # preserve positional arguments
      PARAMS="$PARAMS $1"
      shift
      ;;
  esac
done
# set positional arguments in their proper place
eval set -- "$PARAMS"

if test -z "$DATA_DIR"; then
  echo "ERROR: Data Directory is Missing.... Please Use -d or --data_dir"
  exit 1
fi
if test -z "$EXAMM"; then
  echo "ERROR: EXAMM Directory is Missing.... Please Use -p or --examm_loc"
  exit 1
fi
if test -z "$NO_CPU"; then
  echo "INFO: Using Default Number of CPU: 4"
  NO_CPU=4
fi
if test -z "$MAX_GEN"; then
  MAX_GEN=4000
fi

AC_TYPE=c172
INPUT_PARAMETERS="AltAGL AltB AltGPS AltMSL BaroA E1_CHT1 E1_CHT2 E1_CHT3 E1_CHT4 E1_EGT1 E1_EGT2 E1_EGT3 E1_EGT4 E1_FFlow E1_OilP E1_OilT E1_RPM FQtyL FQtyR GndSpd IAS LatAc NormAc OAT Pitch Roll TAS VSpd VSpdG WndDr WndSpd"
if [ $OUT_PAR = "engine_par" ]; then
  OUT_ADD=""
  OUT_PARAMETERS="E1_EGT1 E1_EGT2 E1_EGT3 E1_EGT4"
elif [ $OUT_PAR = "none_engine_par" ]; then
  OUT_ADD=""
  OUT_PARAMETERS="Pitch Roll AltMSL IAS LatAc NormAc "$OUT_ADD
else
  echo "ERROR: Invalid Output Parameters: $OUT_PAR ... Please use [engine_par, none_engine_par]"
  exit 1
fi
# INPUTS_TO_REMOVE="--inputs_to_remove "
# OUTPUTS_TO_REMOVE="--outputs_to_remove "
INPUTS_TO_REMOVE=""
OUTPUTS_TO_REMOVE=""
# EXTRA_INPUTS="--extra_inputs 8"
EXTRA_INPUTS=""
MAX_GENOMES=$MAX_GEN
ISLANDS=4
POPULATION=10
BIN_FILE="rnn_genome_"
BINARY_FILE="--genome_bin ../../basic_genomes/$OUT_PAR/c172_basic_genomes/$BIN_FILE"
# BINARY_FILE=""
TRAIN_FILES=""
for x in 01 02 03 04 05 06 07 08 09; do
    TRAIN_FILES=$TRAIN_FILES$DATA_DIR/$AC_TYPE"_file$x.csv "
done
TEST_FILES=""
for x in 10 11 12; do
    TEST_FILES=$TEST_FILES$DATA_DIR/$AC_TYPE"_file$x.csv "
done

for no_stir_mutations in 1 4 16 64 256
do
  for folder in  0 1 2 3 4 5 6 7 8 9
  do
      exp_name="../results/stirred_$AC_TYPE/$no_stir_mutations/$folder"
      mkdir -p $exp_name
      echo "\tIteration: "$exp_name
      echo "\t###-------------------###"
      time mpirun \
          --oversubscribe \
          -np $NO_CPU $EXAMM/build/mpi/examm_mpi \
          --training_filenames $TRAIN_FILES \
          --test_filenames $TEST_FILES \
          --time_offset 1 \
          --input_parameter_names $INPUT_PARAMETERS \
          --output_parameter_names $OUT_PARAMETERS \
          --population_size $POPULATION \
          --max_genomes $MAX_GENOMES \
          --bp_iterations 4 \
          --normalize \
          --output_directory $exp_name \
          --max_recurrent_depth 10 \
          --number_islands $ISLANDS \
          $BINARY_FILE$folder.bin \
          $INPUTS_TO_REMOVE \
          $OUTPUTS_TO_REMOVE \
          $EXTRA_INPUTS \
          $EXTRA_OUTPUTS  \
          --std_message_level ERROR \
          --file_message_level ERROR $VER \
          --stir_mutations $no_stir_mutations
  
    done
done
