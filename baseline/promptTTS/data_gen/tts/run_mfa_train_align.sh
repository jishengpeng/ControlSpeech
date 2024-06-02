#!/bin/bash
set -e
NUM_JOB=${NUM_JOB:-36}
echo "| Training MFA using ${NUM_JOB} cores."
BASE_DIR=data/processed/$CORPUS
MODEL_NAME=${MODEL_NAME:-"mfa_model"}
PRETRAIN_MODEL_NAME=${PRETRAIN_MODEL_NAME:-"mfa_model_pretrain"}
MFA_INPUTS=${MFA_INPUTS:-"mfa_inputs"}
MFA_OUTPUTS=${MFA_OUTPUTS:-"mfa_outputs"}
MFA_CMD=${MFA_CMD:-"train"}
echo "$CORPUS"

rm -rf $BASE_DIR/mfa_outputs_tmp $BASE_DIR/mfa_tmp
if [ "$MFA_CMD" = "train" ]; then
  mfa train $BASE_DIR/$MFA_INPUTS $BASE_DIR/mfa_dict.txt $BASE_DIR/mfa_outputs_tmp -t $BASE_DIR/mfa_tmp -o $BASE_DIR/$MODEL_NAME.zip --clean -j $NUM_JOB --config_path data_gen/tts/mfa_train_config.yaml
fi
# rm -rf $BASE_DIR/mfa_tmp $BASE_DIR/$MFA_OUTPUTS
# mkdir -p $BASE_DIR/$MFA_OUTPUTS
# find $BASE_DIR/mfa_outputs_tmp -regex ".*\.TextGrid" -print0 | xargs -0 -i mv {} $BASE_DIR/$MFA_OUTPUTS/
# if [ -e "$BASE_DIR/mfa_outputs_tmp/unaligned.txt" ]; then
#   cp $BASE_DIR/mfa_outputs_tmp/unaligned.txt $BASE_DIR/
# fi
# rm -rf $BASE_DIR/mfa_outputs_tmp


rm -rf $BASE_DIR/mfa_outputs
mkdir -p $BASE_DIR/mfa_outputs
find $BASE_DIR/mfa_tmp/mfa_inputs_train_acoustic_model/sat_2_ali/textgrids -maxdepth 1 -regex ".*/[0-9]+" -print0 | xargs -0 -i rsync -a {}/ $BASE_DIR/mfa_outputs/
if [ -e "$BASE_DIR/mfa_outputs_tmp/unaligned.txt" ]; then
  cp $BASE_DIR/mfa_outputs_tmp/unaligned.txt $BASE_DIR/
fi
rm -rf $BASE_DIR/mfa_outputs_tmp
rm -rf $BASE_DIR/mfa_tmp