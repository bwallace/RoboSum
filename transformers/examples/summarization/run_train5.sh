export OUTPUT_DIR_NAME=pl_title
export CURRENT_DIR=${PWD}
export OUTPUT_DIR=${CURRENT_DIR}/${OUTPUT_DIR_NAME}

# Make output directory if it doesn't exist
mkdir -p $OUTPUT_DIR

# Add parent directory to python path to access lightning_base.py
export PYTHONPATH="../":"${PYTHONPATH}"

python finetune4.py \
--data_dir=/Users/byronwallace/code/RoboSum/PubMed_Summary/Data/pl_title/ \
--model_name_or_path=facebook/bart-large-cnn \
--learning_rate=3e-5 \
--train_batch_size=4 \
--max_source_length=1024 \
--eval_batch_size=4 \
--output_dir=$OUTPUT_DIR \
--num_train_epochs=3 \
--modify_tokenizer=True \
--do_eval $@ \
--gpu_n=0 \
--evaluate_during_training=True \
--do_predict $@ 


