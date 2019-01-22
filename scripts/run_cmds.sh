#/bin/bash
export OMP_NUM_THREADS=1
CKPT_DIR=checkpoints/test
MAIN_SCRIPT_PATH=../run.py
CONFIG_FILE=test_config.json

run_test () {
  echo "Testing $1"
  python $MAIN_SCRIPT_PATH -c $CONFIG_FILE ${@:1:99} >test_${1}.log 2>error_${1}.log && echo "$1 succeeded" || { cat test_${1}.log && cat error_${1}.log && echo "$1 failed" && exit 1; }
}

python $MAIN_SCRIPT_PATH record_data --help

VAE_FILE=${CKPT_DIR}/vae.hdf5

run_test record_data -n 1 ${VAE_FILE}

run_test train_vae ${VAE_FILE}

MEMORY_FILE=${CKPT_DIR}/memory.hdf5

run_test convert_data ${VAE_FILE} ${MEMORY_FILE}

run_test train_mem ${MEMORY_FILE}

run_test train_ctrl

run_test eval -n 1 -c ./checkpoints/test/controller_weights_best.ckpt

rm -rf checkpoints
