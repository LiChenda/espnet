#!/bin/bash

# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

log() {
    local fname=${BASH_SOURCE[1]##*/}
    echo -e "$(date '+%Y-%m-%dT%H:%M:%S') (${fname}:${BASH_LINENO[0]}:${FUNCNAME[1]}) $*"
}
min() {
  local a b
  a=$1
  for b in "$@"; do
      if [ "${b}" -le "${a}" ]; then
          a="${b}"
      fi
  done
  echo "${a}"
}
SECONDS=0

# General configuration
stage=1          # Processes starts from the specified stage.
stop_stage=13   # Processes is stopped at the specified stage.
ngpu=1           # The number of gpus ("0" uses cpu, otherwise use gpu).
num_nodes=1      # The number of nodes
mem=10G          # Memory per CPU
nj=32            # The number of parallel jobs.
dumpdir=dump     # Directory to dump features.
inference_nj=32     # The number of parallel jobs in decoding.
gpu_inference=false # Whether to perform gpu decoding.
expdir=exp       # Directory to save experiments.
skip_eval=false      # Skip decoding and evaluation stages

# Data preparation related
local_data_opts= # The options given to local/data.sh.

# Speed perturbation related
speed_perturb_factors=  # perturbation factors, e.g. "0.9 1.0 1.1" (separated by space).

# Feature extraction related
feats_type=raw    # Feature type (raw or fbank_pitch).
audio_format=flac # Audio format (only in feats_type=raw).
fs=16k            # Sampling rate.
min_wav_duration=0.1   # Minimum duration in second
max_wav_duration=20    # Maximum duration in second

# Joint model related
joint_tag=    # Suffix to the result dir for enhancement model training.
joint_config= # Config for ehancement model training.
joint_args=   # Arguments for enhancement model training, e.g., "--max_epoch 10".
            # Note that it will overwrite args in enhancement config.
joint_exp=    # Specify the direcotry path for ASR experiment. If this option is specified, joint_tag is ignored.

# Enhancement model related
spk_num=2
noise_type_num=1

# ASR model related
feats_normalize=global_mvn  # Normalizaton layer type
num_splits_asr=1   # Number of splitting for lm corpus

# Training data related
use_dereverb_ref=false
use_noise_ref=false


# Evaluation related
scoring_protocol="STOI SDR SAR SIR"
ref_channel=0

# Tokenization related
token_type=bpe      # Tokenization type (char or bpe).
nbpe=30             # The number of BPE vocabulary.
bpemode=unigram     # Mode of BPE (unigram or bpe).
oov="<unk>"         # Out of vocabulary symbol.
blank="<blank>"     # CTC blank symbol
sos_eos="<sos/eos>" # sos and eos symbole
bpe_input_sentence_size=100000000 # Size of input sentence for BPE.
bpe_nlsyms=         # non-linguistic symbols list, separated by a comma, for BPE
bpe_char_cover=1.0 # character coverage when modeling BPE

# Language model related
use_lm=true       # Use language model for ASR decoding.
lm_tag=           # Suffix to the result dir for language model training.
lm_exp=           # Specify the direcotry path for LM experiment. If this option is specified, lm_tag is ignored.
lm_config=        # Config for language model training.
lm_args=          # Arguments for language model training, e.g., "--max_epoch 10".
                  # Note that it will overwrite args in lm config.
use_word_lm=false # Whether to use word language model.
num_splits_lm=1   # Number of splitting for lm corpus
# shellcheck disable=SC2034
word_vocab_size=10000 # Size of word vocabulary.


# Decoding related
inference_lm=valid.loss.best.pth       # Language modle path for decoding.
decode_tag=    # Suffix to the result dir for decoding.
decode_config= # Config for decoding.
decode_args="--normalize_output_wav true "   # Arguments for decoding, e.g., "--lm_weight 0.1".
                                            # Note that it will overwrite args in decode config.
# TODO(Jing): needs more clean configure choice
decode_joint_model=valid.loss.best.pth
decode_lm=valid.loss.best.pth       # Language modle path for decoding.
decode_asr_model=valid.acc.best.pth # ASR model path for decoding.
                                    # e.g.
                                    # decode_asr_model=train.loss.best.pth
                                    # decode_asr_model=3epoch.pth
                                    # decode_asr_model=valid.acc.best.pth
                                    # decode_asr_model=valid.loss.ave.pth

# [Task dependent] Set the datadir name created by local/data.sh
train_set=       # Name of training set.
valid_set=       # Name of validation set used for monitoring/tuning network training
test_sets=       # Names of test sets. Multiple items (e.g., both dev and eval sets) can be specified.
enh_speech_fold_length=800 # fold_length for speech data during enhancement training
# TODO(Jing): should add the wsj1 texts in run?
srctexts=        # Used for the training of BPE and LM and the creation of a vocabulary list.
lm_dev_text=     # Text file path of language model development set.
lm_test_text=    # Text file path of language model evaluation set.
nlsyms_txt=none  # Non-linguistic symbol list if existing.
cleaner=none     # Text cleaner.
g2p=none         # g2p method (needed if token_type=phn).
lang=noinfo      # The language type of corpus
asr_speech_fold_length=800 # fold_length for speech data during ASR training
asr_text_fold_length=150   # fold_length for text data during ASR training
lm_fold_length=150         # fold_length for LM training

help_message=$(cat << EOF
Usage: $0 --train-set <train_set_name> --valid-set <valid_set_name> --test_sets <test_set_names>

Options:
    # General configuration
    --stage         # Processes starts from the specified stage (default="${stage}").
    --stop_stage    # Processes is stopped at the specified stage (default="${stop_stage}").
    --ngpu          # The number of gpus ("0" uses cpu, otherwise use gpu, default="${ngpu}").
    --num_nodes     # The number of nodes
    --nj            # The number of parallel jobs (default="${nj}").
    --inference_nj  # The number of parallel jobs in inference (default="${inference_nj}").
    --gpu_inference # Whether to use gpu for inference (default="${gpu_inference}").
    --dumpdir       # Directory to dump features (default="${dumpdir}").
    --expdir        # Directory to save experiments (default="${expdir}").

    # Data preparation related
    --local_data_opts # The options given to local/data.sh (default="${local_data_opts}").

    # Speed perturbation related
    --speed_perturb_factors   # speed perturbation factors, e.g. "0.9 1.0 1.1" (separated by space, default="${speed_perturb_factors}").

    # Feature extraction related
    --feats_type   # Feature type (only support raw currently).
    --audio_format # Audio format (only in feats_type=raw, default="${audio_format}").
    --fs           # Sampling rate (default="${fs}").
    --min_wav_duration # Minimum duration in second (default="${min_wav_duration}").
    --max_wav_duration # Maximum duration in second (default="${max_wav_duration}").


    # Enhancemnt model related
    --joint_tag    # Suffix to the result dir for enhancement model training (default="${joint_tag}").
    --joint_config # Config for enhancement model training (default="${joint_config}").
    --joint_args   # Arguments for enhancement model training, e.g., "--max_epoch 10" (default="${joint_args}").
                 # Note that it will overwrite args in enhancement config.
    --spk_num    # Number of speakers in the input audio (default="${spk_num}")
    --noise_type_num  # Number of noise types in the input audio (default="${noise_type_num}")
    --feats_normalize # Normalizaton layer type (default="${feats_normalize}").

    # Training data related
    --use_dereverb_ref # Whether or not to use dereverberated signal as an additional reference
                         for training a dereverberation model (default="${use_dereverb_ref}")
    --use_noise_ref    # Whether or not to use noise signal as an additional reference
                         for training a denoising model (default="${use_noise_ref}")

    # Enhancement related
    --decode_args      # Arguments for enhancement in the inference stage (default="${decode_args}")
    --decode_joint_model # Enhancement model path for inference (default="${decode_joint_model}").

    # Evaluation related
    --scoring_protocol    # Metrics to be used for scoring (default="${scoring_protocol}")
    --ref_channel         # Reference channel of the reference speech will be used if the model
                            output is single-channel and reference speech is multi-channel
                            (default="${ref_channel}")

    # [Task dependent] Set the datadir name created by local/data.sh
    --train_set     # Name of training set (required).
    --valid_set       # Name of development set (required).
    --test_sets     # Names of evaluation sets (required).
    --enh_speech_fold_length # fold_length for speech data during enhancement training  (default="${enh_speech_fold_length}").
EOF
)

log "$0 $*"
. utils/parse_options.sh

if [ $# -ne 0 ]; then
    log "${help_message}"
    log "Error: No positional arguments are required."
    exit 2
fi

. ./path.sh
. ./cmd.sh


# Check required arguments
[ -z "${train_set}" ] && { log "${help_message}"; log "Error: --train_set is required"; exit 2; };
[ -z "${valid_set}" ] &&   { log "${help_message}"; log "Error: --valid_set is required"  ; exit 2; };
[ -z "${test_sets}" ] && { log "${help_message}"; log "Error: --test_sets is required"; exit 2; };

data_feats=${dumpdir}/raw


if [ ${spk_num} -le 1 ]; then
    # Use the same text as ASR for lm training if not specified.
    [ -z "${lm_dev_text}" ] && lm_dev_text="${data_feats}/${valid_set}/text"
    # Use the text of the 1st evaldir if lm_test is not specified
    [ -z "${lm_test_text}" ] && lm_test_text="${data_feats}/${test_sets%% *}/text"
fi

# Check tokenization type
token_listdir=data/token_list
bpedir="${token_listdir}/bpe_${bpemode}${nbpe}"
bpeprefix="${bpedir}"/model
bpemodel="${bpeprefix}".model
bpetoken_list="${bpedir}"/tokens.txt
chartoken_list="${token_listdir}"/char/tokens.txt
# NOTE: keep for future development.
# shellcheck disable=SC2034
wordtoken_list="${token_listdir}"/word/tokens.txt

if [ "${token_type}" = bpe ]; then
    token_list="${bpetoken_list}"
elif [ "${token_type}" = char ]; then
    token_list="${chartoken_list}"
    bpemodel=none
else
    log "Error: not supported --token_type '${token_type}'"
    exit 2
fi
if ${use_word_lm}; then
    log "Error: Word LM is not supported yet"
    exit 2

    lm_token_list="${wordtoken_list}"
    lm_token_type=word
else
    lm_token_list="${token_list}"
    lm_token_type="${token_type}"
fi


# Set tag for naming of model directory
if [ -z "${joint_tag}" ]; then
    if [ -n "${joint_config}" ]; then
        joint_tag="$(basename "${joint_config}" .yaml)_${feats_type}_${token_type}"
    else
        joint_tag="train_${feats_type}_${token_type}"
    fi
    # Add overwritten arg's info
    if [ -n "${joint_args}" ]; then
        joint_tag+="$(echo "${joint_args}" | sed -e "s/--/\_/g" -e "s/[ |=]//g")"
    fi
fi
if [ -z "${lm_tag}" ]; then
    if [ -n "${lm_config}" ]; then
        lm_tag="$(basename "${lm_config}" .yaml)_${lm_token_type}"
    else
        lm_tag="train_${lm_token_type}"
    fi
    # Add overwritten arg's info
    if [ -n "${lm_args}" ]; then
        lm_tag+="$(echo "${lm_args}" | sed -e "s/--/\_/g" -e "s/[ |=]//g")"
    fi
fi
if [ -z "${decode_tag}" ]; then
    if [ -n "${decode_config}" ]; then
        decode_tag="$(basename "${decode_config}" .yaml)"
    else
        decode_tag=decode
    fi
    # Add overwritten arg's info
    #if [ -n "${decode_args}" ]; then
    #    decode_tag+="$(echo "${decode_args}" | sed -e "s/--/\_/g" -e "s/[ |=]//g")"
    #fi
    if "${use_lm}"; then
        decode_tag+="_lm_${lm_tag}_$(echo "${decode_lm}" | sed -e "s/\//_/g" -e "s/\.[^.]*$//g")"
    fi
    decode_tag+="_asr_model_$(echo "${decode_asr_model}" | sed -e "s/\//_/g" -e "s/\.[^.]*$//g")"
fi


# The directory used for collect-stats mode
joint_stats_dir="${expdir}/joint_stats_${feats_type}_${fs}"
if [ -n "${speed_perturb_factors}" ]; then
    joint_stats_dir="${joint_stats_dir}_sp"
fi
lm_stats_dir="${expdir}/lm_stats"

# The directory used for training commands
if [ -z "${joint_exp}" ]; then
    joint_exp="${expdir}/joint_${joint_tag}"
fi
if [ -n "${speed_perturb_factors}" ]; then
    joint_exp="${joint_exp}_sp"
fi
if [ -z "${lm_exp}" ]; then
    lm_exp="${expdir}/lm_${lm_tag}"
fi

# ========================== Main stages start from here. ==========================

if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
    log "Stage 1: Data preparation for data/${train_set}, data/${valid_set}, etc."
    # [Task dependent] Need to create data.sh for new corpus
    local/data.sh ${local_data_opts}
fi

if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
    if ! $use_dereverb_ref && [ -n "${speed_perturb_factors}" ]; then
       log "Stage 2: Speed perturbation: data/${train_set} -> data/${train_set}_sp"

        _scp_list="wav.scp "
        for i in $(seq ${spk_num}); do
            _scp_list+="spk${i}.scp "
        done

       for factor in ${speed_perturb_factors}; do
           if [[ $(bc <<<"${factor} != 1.0") == 1 ]]; then
               scripts/utils/perturb_enh_data_dir_speed.sh "${factor}" "data/${train_set}" "data/${train_set}_sp${factor}" "${_scp_list}"
               _dirs+="data/${train_set}_sp${factor} "
           else
               # If speed factor is 1, same as the original
               _dirs+="data/${train_set} "
           fi
       done
       utils/combine_data.sh --extra-files "${_scp_list}" "data/${train_set}_sp" ${_dirs}
    else
       log "Skip stage 2: Speed perturbation"
    fi
fi

if [ -n "${speed_perturb_factors}" ]; then
    train_set="${train_set}_sp"
fi

if [ ${stage} -le 3 ] && [ ${stop_stage} -ge 3 ]; then

    log "Stage 3: Format wav.scp: data/ -> ${data_feats}/org/"

    # ====== Recreating "wav.scp" ======
    # Kaldi-wav.scp, which can describe the file path with unix-pipe, like "cat /some/path |",
    # shouldn't be used in training process.
    # "format_wav_scp.sh" dumps such pipe-style-wav to real audio file
    # and also it can also change the audio-format and sampling rate.
    # If nothing is need, then format_wav_scp.sh does nothing:
    # i.e. the input file format and rate is same as the output.

    for dset in "${train_set}" "${valid_set}" ${test_sets}; do
        utils/copy_data_dir.sh data/"${dset}" "${data_feats}/org/${dset}"
        # TODO(Jing): should be more precise.
        cp data/"${dset}"/text_spk* "${data_feats}/org/${dset}"
        rm -f ${data_feats}/org/${dset}/{segments,wav.scp,reco2file_and_channel}
        _opts=
        if [ -e data/"${dset}"/segments ]; then
            # "segments" is used for splitting wav files which are written in "wav".scp
            # into utterances. The file format of segments:
            #   <segment_id> <record_id> <start_time> <end_time>
            #   "e.g. call-861225-A-0050-0065 call-861225-A 5.0 6.5"
            # Where the time is written in seconds.
            _opts+="--segments data/${dset}/segments "
        fi


        _spk_list=" "
        for i in $(seq ${spk_num}); do
            _spk_list+="spk${i} "
        done
        if $use_noise_ref; then
            # reference for denoising ("noise1 noise2 ... niose${noise_type_num} ")
            _spk_list+=$(for n in $(seq $noise_type_num); do echo -n "noise$n "; done)
        fi
        if $use_dereverb_ref; then
            # reference for dereverberation
            _spk_list+="dereverb "
        fi

        for spk in ${_spk_list} "wav" ; do
            # shellcheck disable=SC2086
            scripts/audio/format_wav_scp.sh --nj "${nj}" --cmd "${train_cmd}" \
                --out-filename "${spk}.scp" \
                --audio-format "${audio_format}" --fs "${fs}" ${_opts} \
                "data/${dset}/${spk}.scp" "${data_feats}/org/${dset}" \
                "${data_feats}/org/${dset}/logs/${spk}" "${data_feats}/org/${dset}/data/${spk}"

        done
        echo "${feats_type}" > "${data_feats}/org/${dset}/feats_type"

    done
fi


if [ ${stage} -le 4 ] && [ ${stop_stage} -ge 4 ]; then
    log "Stage 4: Remove short data: ${data_feats}/org -> ${data_feats}"

    for dset in "${train_set}" "${valid_set}"; do
    # NOTE: Not applying to test_sets to keep original data

        _spk_list=" "
        for i in $(seq ${spk_num}); do
            _spk_list+="spk${i} "
        done
        if $use_noise_ref; then
            # reference for denoising ("noise1 noise2 ... niose${noise_type_num} ")
            _spk_list+=$(for n in $(seq $noise_type_num); do echo -n "noise$n "; done)
        fi
        if $use_dereverb_ref; then
            # reference for dereverberation
            _spk_list+="dereverb "
        fi

        # Copy data dir
        utils/copy_data_dir.sh "${data_feats}/org/${dset}" "${data_feats}/${dset}"
        cp "${data_feats}/org/${dset}/feats_type" "${data_feats}/${dset}/feats_type"
        for spk in ${_spk_list};do
            cp "${data_feats}/org/${dset}/${spk}.scp" "${data_feats}/${dset}/${spk}.scp"
            cp "${data_feats}/org/${dset}/text_${spk}" "${data_feats}/${dset}/text_${spk}"
            # Remove empty text
            <"${data_feats}/org/${dset}/text_${spk}" \
                awk ' { if( NF != 1 ) print $0; } ' >"${data_feats}/${dset}/text_${spk}"
        done

        _fs=$(python3 -c "import humanfriendly as h;print(h.parse_size('${fs}'))")
        _min_length=$(python3 -c "print(int(${min_wav_duration} * ${_fs}))")
        _max_length=$(python3 -c "print(int(${max_wav_duration} * ${_fs}))")

        # utt2num_samples is created by format_wav_scp.sh
        <"${data_feats}/org/${dset}/utt2num_samples" \
            awk -v min_length="${_min_length}" -v max_length="${_max_length}" \
                '{ if ($2 > min_length && $2 < max_length ) print $0; }' \
                >"${data_feats}/${dset}/utt2num_samples"
        for spk in ${_spk_list} "wav"; do
            <"${data_feats}/org/${dset}/${spk}.scp" \
                utils/filter_scp.pl "${data_feats}/${dset}/utt2num_samples"  \
                >"${data_feats}/${dset}/${spk}.scp"
        done
        for spk in ${_spk_list}; do
            <"${data_feats}/org/${dset}/text_${spk}" \
                utils/filter_scp.pl "${data_feats}/${dset}/utt2num_samples"  \
                >"${data_feats}/${dset}/text_${spk}"
        done


        # fix_data_dir.sh leaves only utts which exist in all files
        utils/fix_data_dir.sh "${data_feats}/${dset}"
    done
fi

if [ ${stage} -le 5 ] && [ ${stop_stage} -ge 5 ]; then
    if [ "${token_type}" = bpe ]; then
        log "Stage 5: Generate token_list from ${data_feats}/srctexts using BPE"

        # shellcheck disable=SC2002
        cat ${srctexts} | awk ' { if( NF != 1 ) print $0; } ' >"${data_feats}/srctexts"

        mkdir -p "${bpedir}"
        # shellcheck disable=SC2002
        <"${data_feats}/srctexts" cut -f 2- -d" "  > "${bpedir}"/train.txt

        if [ -n "${bpe_nlsyms}" ]; then
            _opts_spm="--user_defined_symbols=${bpe_nlsyms}"
        else
            _opts_spm=""
        fi

        spm_train \
            --input="${bpedir}"/train.txt \
            --vocab_size="${nbpe}" \
            --model_type="${bpemode}" \
            --model_prefix="${bpeprefix}" \
            --character_coverage=${bpe_char_cover} \
            --input_sentence_size="${bpe_input_sentence_size}" \
            ${_opts_spm}

        _opts="--bpemodel ${bpemodel}"

    elif [ "${token_type}" = char ]; then
        log "Stage 5: Generate character level token_list from ${data_feats}/srctexts"
        # shellcheck disable=SC2002
        cat ${srctexts} | awk ' { if( NF != 1 ) print $0; } ' >"${data_feats}/srctexts"
        # TODO: the valid should also be cat together.
        _opts="--non_linguistic_symbols ${nlsyms_txt}"

    else
        log "Error: not supported --token_type '${token_type}'"
        exit 2
    fi

    # The first symbol in token_list must be "<blank>" and the last must be also sos/eos:
    # 0 is reserved for CTC-blank for ASR and also used as ignore-index in the other task

    python3 -m espnet2.bin.tokenize_text  \
        --token_type "${token_type}" \
        --input "${data_feats}/srctexts" --output "${token_list}" ${_opts} \
        --field 2- \
        --cleaner "${cleaner}" \
        --g2p "${g2p}" \
        --write_vocabulary true \
        --add_symbol "${blank}:0" \
        --add_symbol "${oov}:1" \
        --add_symbol "${sos_eos}:-1"
    pwd

    # Create word-list for word-LM training
    if ${use_word_lm}; then
        log "Generate word level token_list from ${data_feats}/srctexts"
        python3 -m espnet2.bin.tokenize_text \
            --token_type word \
            --input "${data_feats}/srctexts" --output "${lm_token_list}" \
            --field 2- \
            --cleaner "${cleaner}" \
            --g2p "${g2p}" \
            --write_vocabulary true \
            --vocabulary_size "${word_vocab_size}" \
            --add_symbol "${blank}:0" \
            --add_symbol "${oov}:1" \
            --add_symbol "${sos_eos}:-1"
    fi

fi


# ========================== Data preparation is done here. ==========================

# ========================== LM Begins ==========================
if "${use_lm}"; then
    if [ ${stage} -le 6 ] && [ ${stop_stage} -ge 6 ]; then
        log "Stage 6: LM collect stats: train_set=${data_feats}/srctexts, dev_set=${lm_dev_text}"

        # the srctexts should gets another one to aovid the duplicated keys
        if [ ${spk_num} -ge 2 ]; then
            _spk_list=" "
            for i in $(seq ${spk_num}); do
                _spk_list+="spk${i} "
            done
            for spk in ${_spk_list}; do
                awk '{print "'$spk'_"$0}' "${data_feats}/${train_set}/text_${spk}" > "${data_feats}/srctexts_${spk}"
                awk '{print "'$spk'_"$0}' "${data_feats}/${valid_set}/text_${spk}" > "${data_feats}/valid_srctexts_${spk}"
                awk '{print "'$spk'_"$0}' "${data_feats}/org/${test_sets}/text_${spk}" > "${data_feats}/test_srctexts_${spk}"
            done
            cat ${data_feats}/srctexts_spk* | awk ' { if( NF != 1 ) print $0; } '  > "${data_feats}/srctexts_with_spk"
            cat ${data_feats}/valid_srctexts_spk* | awk ' { if( NF != 1 ) print $0; } '  > "${data_feats}/valid_srctexts_with_spk"
            cat ${data_feats}/test_srctexts_spk* | awk ' { if( NF != 1 ) print $0; } '  > "${data_feats}/test_srctexts_with_spk"
            lm_dev_text="${data_feats}/valid_srctexts_with_spk"      # Text file path of language model development set.
            lm_test_text="${data_feats}/test_srctexts_with_spk"    # Text file path of language model evaluation set.
        fi

        _opts=
        if [ -n "${lm_config}" ]; then
            # To generate the config file: e.g.
            #   % python3 -m espnet2.bin.lm_train --print_config --optim adam
            _opts+="--config ${lm_config} "
        fi

        # 1. Split the key file
        _logdir="${lm_stats_dir}/logdir"
        mkdir -p "${_logdir}"
        # Get the minimum number among ${nj} and the number lines of input files
        _nj=$(min "${nj}" "$(<${data_feats}/srctexts_with_spk wc -l)" "$(<${lm_dev_text} wc -l)")

        key_file="${data_feats}/srctexts_with_spk"
        split_scps=""
        for n in $(seq ${_nj}); do
            split_scps+=" ${_logdir}/train.${n}.scp"
        done
        # shellcheck disable=SC2086
        utils/split_scp.pl "${key_file}" ${split_scps}

        key_file="${lm_dev_text}"
        split_scps=""
        for n in $(seq ${_nj}); do
            split_scps+=" ${_logdir}/dev.${n}.scp"
        done
        # shellcheck disable=SC2086
        utils/split_scp.pl "${key_file}" ${split_scps}

        # 2. Submit jobs
        log "LM collect-stats started... log: '${_logdir}/stats.*.log'"
        # NOTE: --*_shape_file doesn't require length information if --batch_type=unsorted,
        #       but it's used only for deciding the sample ids.

        # shellcheck disable=SC2086
        ${train_cmd} JOB=1:"${_nj}" "${_logdir}"/stats.JOB.log \
            python3 -m espnet2.bin.lm_train \
                --collect_stats true \
                --use_preprocessor true \
                --bpemodel "${bpemodel}" \
                --token_type "${lm_token_type}"\
                --token_list "${lm_token_list}" \
                --non_linguistic_symbols "${nlsyms_txt}" \
                --cleaner "${cleaner}" \
                --g2p "${g2p}" \
                --train_data_path_and_name_and_type "${data_feats}/srctexts_with_spk,text,text" \
                --valid_data_path_and_name_and_type "${lm_dev_text},text,text" \
                --train_shape_file "${_logdir}/train.JOB.scp" \
                --valid_shape_file "${_logdir}/dev.JOB.scp" \
                --output_dir "${_logdir}/stats.JOB" \
                ${_opts} ${lm_args}

        # 3. Aggregate shape files
        _opts=
        for i in $(seq "${_nj}"); do
            _opts+="--input_dir ${_logdir}/stats.${i} "
        done
        # shellcheck disable=SC2086
        python3 -m espnet2.bin.aggregate_stats_dirs ${_opts} --output_dir "${lm_stats_dir}"

        # Append the num-tokens at the last dimensions. This is used for batch-bins count
        <"${lm_stats_dir}/train/text_shape" \
            awk -v N="$(<${lm_token_list} wc -l)" '{ print $0 "," N }' \
            >"${lm_stats_dir}/train/text_shape.${lm_token_type}"

        <"${lm_stats_dir}/valid/text_shape" \
            awk -v N="$(<${lm_token_list} wc -l)" '{ print $0 "," N }' \
            >"${lm_stats_dir}/valid/text_shape.${lm_token_type}"
    fi


    if [ ${stage} -le 7 ] && [ ${stop_stage} -ge 7 ]; then
        log "Stage 7: LM Training: train_set=${data_feats}/srctexts, dev_set=${lm_dev_text}"

        _opts=
        if [ -n "${lm_config}" ]; then
            # To generate the config file: e.g.
            #   % python3 -m espnet2.bin.lm_train --print_config --optim adam
            _opts+="--config ${lm_config} "
        fi

        if [ "${num_splits_lm}" -gt 1 ]; then
            # If you met a memory error when parsing text files, this option may help you.
            # The corpus is split into subsets and each subset is used for training one by one in order,
            # so the memory footprint can be limited to the memory required for each dataset.

            _split_dir="${lm_stats_dir}/splits${num_splits_lm}"
            if [ ! -f "${_split_dir}/.done" ]; then
                rm -f "${_split_dir}/.done"
                python3 -m espnet2.bin.split_scps \
                  --scps "${data_feats}/srctexts_with_spk" "${lm_stats_dir}/train/text_shape.${lm_token_type}" \
                  --num_splits "${num_splits_lm}" \
                  --output_dir "${_split_dir}"
                touch "${_split_dir}/.done"
            else
                log "${_split_dir}/.done exists. Spliting is skipped"
            fi

            _opts+="--train_data_path_and_name_and_type ${_split_dir}/srctexts_with_spk,text,text "
            _opts+="--train_shape_file ${_split_dir}/text_shape.${lm_token_type} "
            _opts+="--multiple_iterator true "

        else
            pwd
            _opts+="--train_data_path_and_name_and_type ${data_feats}/srctexts_with_spk,text,text "
            _opts+="--train_shape_file ${lm_stats_dir}/train/text_shape.${lm_token_type} "
        fi

        # NOTE(kamo): --fold_length is used only if --batch_type=folded and it's ignored in the other case

        log "LM training started... log: '${lm_exp}/train.log'"
        # shellcheck disable=SC2086
        python3 -m espnet2.bin.launch \
            --cmd "${cuda_cmd} --name ${lm_exp}/train.log" \
            --log "${lm_exp}"/train.log \
            --ngpu "${ngpu}" \
            --num_nodes "${num_nodes}" \
            --init_file_prefix "${joint_exp}"/.dist_init_ \
            --multiprocessing_distributed true -- \
            python3 -m espnet2.bin.lm_train \
                --ngpu "${ngpu}" \
                --use_preprocessor true \
                --bpemodel "${bpemodel}" \
                --token_type "${lm_token_type}"\
                --token_list "${lm_token_list}" \
                --non_linguistic_symbols "${nlsyms_txt}" \
                --cleaner "${cleaner}" \
                --g2p "${g2p}" \
                --valid_data_path_and_name_and_type "${lm_dev_text},text,text" \
                --valid_shape_file "${lm_stats_dir}/valid/text_shape.${lm_token_type}" \
                --fold_length "${lm_fold_length}" \
                --resume true \
                --output_dir "${lm_exp}" \
                ${_opts} ${lm_args}

    fi


    if [ ${stage} -le 8 ] && [ ${stop_stage} -ge 8 ]; then
        log "Stage 8: Calc perplexity: ${lm_test_text}"
        _opts=
        # TODO(kamo): Parallelize?
        log "Perplexity calculation started... log: '${lm_exp}/perplexity_test/lm_calc_perplexity.log'"
        # shellcheck disable=SC2086
        ${cuda_cmd} --gpu "${ngpu}" "${lm_exp}"/perplexity_test/lm_calc_perplexity.log \
            python3 -m espnet2.bin.lm_calc_perplexity \
                --ngpu "${ngpu}" \
                --data_path_and_name_and_type "${lm_test_text},text,text" \
                --train_config "${lm_exp}"/config.yaml \
                --model_file "${lm_exp}/${decode_lm}" \
                --output_dir "${lm_exp}/perplexity_test" \
                ${_opts}
        log "PPL: ${lm_test_text}: $(cat ${lm_exp}/perplexity_test/ppl)"

    fi

else
    log "Stage 6-8: Skip lm-related stages: use_lm=${use_lm}"
fi
# ========================== LM Done here.==========================


if [ ${stage} -le 9 ] && [ ${stop_stage} -ge 9 ]; then
    _joint_train_dir="${data_feats}/${train_set}"
    _joint_valid_dir="${data_feats}/${valid_set}"
    log "Stage 9: Joint collect stats: train_set=${_joint_train_dir}, valid_set=${_joint_valid_dir}"

    _opts=
    if [ -n "${joint_config}" ]; then
        # To generate the config file: e.g.
        #   % python3 -m espnet2.bin.enh_train --print_config --optim adam
        _opts+="--config ${joint_config} "
    fi

    _scp=wav.scp
    # "sound" supports "wav", "flac", etc.
    _type=sound

    # 1. Split the key file
    _logdir="${joint_stats_dir}/logdir"
    mkdir -p "${_logdir}"

    # Get the minimum number among ${nj} and the number lines of input files
    _nj=$(min "${nj}" "$(<${_joint_train_dir}/${_scp} wc -l)" "$(<${_joint_valid_dir}/${_scp} wc -l)")

    key_file="${_joint_train_dir}/${_scp}"
    split_scps=""
    for n in $(seq "${_nj}"); do
        split_scps+=" ${_logdir}/train.${n}.scp"
    done
    # shellcheck disable=SC2086
    utils/split_scp.pl "${key_file}" ${split_scps}

    key_file="${_joint_valid_dir}/${_scp}"
    split_scps=""
    for n in $(seq "${_nj}"); do
        split_scps+=" ${_logdir}/valid.${n}.scp"
    done
    # shellcheck disable=SC2086
    utils/split_scp.pl "${key_file}" ${split_scps}

    # 2. Submit jobs
    log "Joint collect-stats started... log: '${_logdir}/stats.*.log'"

    # prepare train and valid data parameters
    _train_data_param="--train_data_path_and_name_and_type ${_joint_train_dir}/wav.scp,speech_mix,sound "
    _valid_data_param="--valid_data_path_and_name_and_type ${_joint_valid_dir}/wav.scp,speech_mix,sound "
    for spk in $(seq "${spk_num}"); do
        _train_data_param+="--train_data_path_and_name_and_type ${_joint_train_dir}/spk${spk}.scp,speech_ref${spk},sound "
        _valid_data_param+="--valid_data_path_and_name_and_type ${_joint_valid_dir}/spk${spk}.scp,speech_ref${spk},sound "
        _train_data_param+="--train_data_path_and_name_and_type ${_joint_train_dir}/text_spk${spk},text_ref${spk},text "
        _valid_data_param+="--valid_data_path_and_name_and_type ${_joint_valid_dir}/text_spk${spk},text_ref${spk},text "
    done

    if $use_dereverb_ref; then
        # reference for dereverberation
        _train_data_param+="--train_data_path_and_name_and_type ${_joint_train_dir}/dereverb.scp,dereverb_ref,sound "
        _valid_data_param+="--valid_data_path_and_name_and_type ${_joint_valid_dir}/dereverb.scp,dereverb_ref,sound "
    fi

    if $use_noise_ref; then
        # reference for denoising
        _train_data_param+=$(for n in $(seq $noise_type_num); do echo -n \
            "--train_data_path_and_name_and_type ${_joint_train_dir}/noise${n}.scp,noise_ref${n},sound "; done)
        _valid_data_param+=$(for n in $(seq $noise_type_num); do echo -n \
            "--valid_data_path_and_name_and_type ${_joint_valid_dir}/noise${n}.scp,noise_ref${n},sound "; done)
    fi

    # NOTE: --*_shape_file doesn't require length information if --batch_type=unsorted,
    #       but it's used only for deciding the sample ids.


    # shellcheck disable=SC2086
    ${train_cmd} JOB=1:"${_nj}" "${_logdir}"/stats.JOB.log \
        python3 -m espnet2.bin.enh_asr_train \
            --collect_stats true \
            --use_preprocessor true \
            --bpemodel "${bpemodel}" \
            --token_type "${token_type}" \
            --token_list "${token_list}" \
            --non_linguistic_symbols "${nlsyms_txt}" \
            --cleaner "${cleaner}" \
            --g2p "${g2p}" \
            ${_train_data_param} \
            ${_valid_data_param} \
            --train_shape_file "${_logdir}/train.JOB.scp" \
            --valid_shape_file "${_logdir}/valid.JOB.scp" \
            --output_dir "${_logdir}/stats.JOB" \
            ${_opts} ${joint_args}

    # 3. Aggregate shape files
    _opts=
    for i in $(seq "${_nj}"); do
        _opts+="--input_dir ${_logdir}/stats.${i} "
    done
    # shellcheck disable=SC2086
    python3 -m espnet2.bin.aggregate_stats_dirs ${_opts} --output_dir "${joint_stats_dir}"

fi


if [ ${stage} -le 10 ] && [ ${stop_stage} -ge 10 ]; then
    _joint_train_dir="${data_feats}/${train_set}"
    _joint_valid_dir="${data_feats}/${valid_set}"
    log "Stage 10: Joint model Training: train_set=${_joint_train_dir}, valid_set=${_joint_valid_dir}"

    _opts=
    if [ -n "${joint_config}" ]; then
        # To generate the config file: e.g.
        #   % python3 -m espnet2.bin.enh_train --print_config --optim adam
        _opts+="--config ${joint_config} "
    fi

    _scp=wav.scp
    # "sound" supports "wav", "flac", etc.
    _type=sound
    _fold_length="$((enh_speech_fold_length * 100))"
    # _opts+="--frontend_conf fs=${fs} "

    # prepare train and valid data parameters
    _train_data_param="--train_data_path_and_name_and_type ${_joint_train_dir}/wav.scp,speech_mix,sound "
    _train_shape_param="--train_shape_file ${joint_stats_dir}/train/speech_mix_shape "
    _valid_data_param="--valid_data_path_and_name_and_type ${_joint_valid_dir}/wav.scp,speech_mix,sound "
    _valid_shape_param="--valid_shape_file ${joint_stats_dir}/valid/speech_mix_shape "
    _fold_length_param="--fold_length ${_fold_length} "
    for spk in $(seq "${spk_num}"); do
        _train_data_param+="--train_data_path_and_name_and_type ${_joint_train_dir}/spk${spk}.scp,speech_ref${spk},sound "
        _train_data_param+="--train_data_path_and_name_and_type ${_joint_train_dir}/text_spk${spk},text_ref${spk},text "
        _train_shape_param+="--train_shape_file ${joint_stats_dir}/train/speech_ref${spk}_shape "
        _train_shape_param+="--train_shape_file ${joint_stats_dir}/train/text_ref${spk}_shape "
        _valid_data_param+="--valid_data_path_and_name_and_type ${_joint_valid_dir}/spk${spk}.scp,speech_ref${spk},sound "
        _valid_data_param+="--valid_data_path_and_name_and_type ${_joint_valid_dir}/text_spk${spk},text_ref${spk},text "
        _valid_shape_param+="--valid_shape_file ${joint_stats_dir}/valid/speech_ref${spk}_shape "
        _valid_shape_param+="--valid_shape_file ${joint_stats_dir}/valid/text_ref${spk}_shape "
        _fold_length_param+="--fold_length ${_fold_length} "
        _fold_length_param+="--fold_length ${_fold_length} "
    done

    if $use_dereverb_ref; then
        # reference for dereverberation
        _train_data_param+="--train_data_path_and_name_and_type ${_joint_train_dir}/dereverb.scp,dereverb_ref,sound "
        _train_shape_param+="--train_shape_file ${joint_stats_dir}/train/dereverb_ref_shape "
        _valid_data_param+="--valid_data_path_and_name_and_type ${_joint_valid_dir}/dereverb.scp,dereverb_ref,sound "
        _valid_shape_param+="--valid_shape_file ${joint_stats_dir}/valid/dereverb_ref_shape "
        _fold_length_param+="--fold_length ${_fold_length} "
    fi

    if $use_noise_ref; then
        # reference for denoising
        for n in $(seq "${noise_type_num}"); do
            _train_data_param+="--train_data_path_and_name_and_type ${_joint_train_dir}/noise${n}.scp,noise_ref${n},sound "
            _train_shape_param+="--train_shape_file ${joint_stats_dir}/train/noise_ref${n}_shape "
            _valid_data_param+="--valid_data_path_and_name_and_type ${_joint_valid_dir}/noise${n}.scp,noise_ref${n},sound "
            _valid_shape_param+="--valid_shape_file ${joint_stats_dir}/valid/noise_ref${n}_shape "
            _fold_length_param+="--fold_length ${_fold_length} "
        done
    fi


    log "enh training started... log: '${joint_exp}/train.log'"
    # shellcheck disable=SC2086
    python3 -m espnet2.bin.launch \
        --cmd "${cuda_cmd} --name ${joint_exp}/train.log --mem ${mem}" \
        --log "${joint_exp}"/train.log \
        --ngpu "${ngpu}" \
        --num_nodes "${num_nodes}" \
        --init_file_prefix "${joint_exp}"/.dist_init_ \
        --multiprocessing_distributed true -- \
        python3 -m espnet2.bin.enh_asr_train \
            --use_preprocessor true \
            --bpemodel "${bpemodel}" \
            --token_type "${token_type}" \
            --token_list "${token_list}" \
            --non_linguistic_symbols "${nlsyms_txt}" \
            --cleaner "${cleaner}" \
            --g2p "${g2p}" \
            ${_train_data_param} \
            ${_valid_data_param} \
            ${_train_shape_param} \
            ${_valid_shape_param} \
            ${_fold_length_param} \
            --resume true \
            --output_dir "${joint_exp}" \
            ${_opts} ${joint_args}

fi


if ! "${skip_eval}"; then
    if [ ${stage} -le 11 ] && [ ${stop_stage} -ge 11 ]; then
        log "Stage 11: Decoding: training_dir=${joint_exp}"

        if ${gpu_inference}; then
            _cmd="${cuda_cmd}"
            _ngpu=1
        else
            _cmd="${decode_cmd}"
            _ngpu=0
        fi

        _opts=
        if [ -n "${decode_config}" ]; then
            _opts+="--config ${decode_config} "
        fi
        if "${use_lm}"; then
            if "${use_word_lm}"; then
                _opts+="--word_lm_train_config ${lm_exp}/config.yaml "
                _opts+="--word_lm_file ${lm_exp}/${inference_lm} "
            else
                _opts+="--lm_train_config ${lm_exp}/config.yaml "
                _opts+="--lm_file ${lm_exp}/${inference_lm} "
            fi
        fi

        for dset in ${test_sets}; do
            _data="${data_feats}/${dset}"
            _dir="${joint_exp}/decode_${dset}_${decode_tag}"
            _logdir="${_dir}/logdir"
            mkdir -p "${_logdir}"

            _feats_type="$(<${_data}/feats_type)"
            if [ "${_feats_type}" = raw ]; then
                _scp=wav.scp
                _type=sound
            else
                _scp=feats.scp
                _type=kaldi_ark
            fi

            _decode_data_param="--data_path_and_name_and_type ${_data}/${_scp},speech_mix,${_type} "
            for spk in $(seq "${spk_num}"); do
                _decode_data_param+="--data_path_and_name_and_type ${_data}/spk${spk}.scp,speech_ref${spk},${_type} "
            done

            # 1. Split the key file
            key_file=${_data}/'wav.scp'
            split_scps=""
            _nj=$(min "${inference_nj}" "$(<${key_file} wc -l)")
            for n in $(seq "${_nj}"); do
                split_scps+=" ${_logdir}/keys.${n}.scp"
            done
            # shellcheck disable=SC2086
            utils/split_scp.pl "${key_file}" ${split_scps}

            # 2. Submit decoding jobs
            log "Decoding started... log: '${_logdir}/joint_inference.*.log'"
            # shellcheck disable=SC2086
            ${_cmd} --gpu "${_ngpu}" JOB=1:"${_nj}" "${_logdir}"/joint_inference.JOB.log \
                python3 -m espnet2.bin.enh_asr_inference \
                    --ngpu "${_ngpu}" \
                    --fs "${fs}" \
                    ${_decode_data_param} \
                    --key_file "${_logdir}"/keys.JOB.scp \
                    --joint_train_config "${joint_exp}"/config.yaml \
                    --joint_model_file "${joint_exp}"/"${decode_joint_model}" \
                    --output_dir "${_logdir}"/output.JOB \
                    ${_opts} ${decode_args}

            # 3. Concatenates the output files from each jobs
            for spk in $(seq "${spk_num}"); do
                for f in token token_int score text; do
                    for i in $(seq "${_nj}"); do
                        cat "${_logdir}/output.${i}/1best_recog_spk${spk}/${f}"
                    done | LC_ALL=C sort -k1 >"${_dir}/${f}_spk${spk}"
                done
                for i in $(seq "${_nj}"); do
                    cat "${_logdir}/output.${i}/SDR_spk${spk}"
                done | LC_ALL=C sort -k1 >"${_dir}/SDR_spk${spk}"
            done

            for protocol in "SDR"; do
                # shellcheck disable=SC2046
                paste $(for j in $(seq ${spk_num}); do echo "${_dir}"/"${protocol}"_spk"${j}" ; done)  |
                awk 'BEIGN{sum=0}
                    {n=0;score=0;for (i=2; i<=NF; i+=2){n+=1;score+=$i}; sum+=score/n}
                    END{print sum/NR}' > "${_dir}/enh_result_${protocol,,}.txt"
            done
        done
    fi


    if [ ${stage} -le 12 ] && [ ${stop_stage} -ge 12 ]; then
        log "Stage 12: Scoring"
        if [ "${token_type}" = pnh ]; then
            log "Error: Not implemented for token_type=phn"
            exit 1
        fi

        for dset in ${test_sets}; do
            _data="${data_feats}/${dset}"
            _dir="${joint_exp}/decode_${dset}_${decode_tag}"

            for _type in cer wer ter; do
                [ "${_type}" = ter ] && [ ! -f "${bpemodel}" ] && continue

                _scoredir="${_dir}/score_${_type}"
                mkdir -p "${_scoredir}"

                if [ "${_type}" = wer ]; then
                    # Tokenize text to word level
                    for spk in $(seq "${spk_num}"); do
                        paste \
                            <(<"${_data}/text_spk${spk}" \
                                  python3 -m espnet2.bin.tokenize_text  \
                                      -f 2- --input - --output - \
                                      --token_type word \
                                      --non_linguistic_symbols "${nlsyms_txt}" \
                                      --remove_non_linguistic_symbols true \
                                      --cleaner "${cleaner}" \
                                      ) \
                            <(<"${_data}/text_spk${spk}" awk '{ print "(" $1 ")" }') \
                                >"${_scoredir}/ref_spk${spk}.trn"

                        # NOTE(kamo): Don't use cleaner for hyp
                        paste \
                            <(<"${_dir}/text_spk${spk}"  \
                                  python3 -m espnet2.bin.tokenize_text  \
                                      -f 2- --input - --output - \
                                      --token_type word \
                                      --non_linguistic_symbols "${nlsyms_txt}" \
                                      --remove_non_linguistic_symbols true \
                                      ) \
                            <(<"${_data}/text_spk${spk}" awk '{ print "(" $1 ")" }') \
                                >"${_scoredir}/hyp_spk${spk}.trn"
                    done


                elif [ "${_type}" = cer ]; then
                    # Tokenize text to char level
                    for spk in $(seq "${spk_num}"); do
                        paste \
                            <(<"${_data}/text_spk${spk}" \
                                  python3 -m espnet2.bin.tokenize_text  \
                                      -f 2- --input - --output - \
                                      --token_type char \
                                      --non_linguistic_symbols "${nlsyms_txt}" \
                                      --remove_non_linguistic_symbols true \
                                      --cleaner "${cleaner}" \
                                      ) \
                            <(<"${_data}/text_spk${spk}" awk '{ print "(" $1 ")" }') \
                                >"${_scoredir}/ref_spk${spk}.trn"

                        # NOTE(kamo): Don't use cleaner for hyp
                        paste \
                            <(<"${_dir}/text_spk${spk}"  \
                                  python3 -m espnet2.bin.tokenize_text  \
                                      -f 2- --input - --output - \
                                      --token_type char \
                                      --non_linguistic_symbols "${nlsyms_txt}" \
                                      --remove_non_linguistic_symbols true \
                                      ) \
                            <(<"${_data}/text_spk${spk}" awk '{ print "(" $1 ")" }') \
                                >"${_scoredir}/hyp_spk${spk}.trn"
                    done

                elif [ "${_type}" = ter ]; then
                    # Tokenize text using BPE
                    for spk in $(seq "${spk_num}"); do
                        paste \
                            <(<"${_data}/text_spk${spk}" \
                                  python3 -m espnet2.bin.tokenize_text  \
                                      -f 2- --input - --output - \
                                      --token_type bpe \
                                      --bpemodel "${bpemodel}" \
                                      --cleaner "${cleaner}" \
                                    ) \
                            <(<"${_data}/text_spk${spk}" awk '{ print "(" $1 ")" }') \
                                >"${_scoredir}/ref_spk${spk}.trn"

                        # NOTE(kamo): Don't use cleaner for hyp
                        paste \
                            <(<"${_dir}/text_spk${spk}" \
                                  python3 -m espnet2.bin.tokenize_text  \
                                      -f 2- --input - --output - \
                                      --token_type bpe \
                                      --bpemodel "${bpemodel}" \
                                      --cleaner "${cleaner}" \
                                      ) \
                            <(<"${_data}/text_spk${spk}" awk '{ print "(" $1 ")" }') \
                                >"${_scoredir}/hyp_spk${spk}.trn"
                        done
                fi

                # FIXME(Jing): cat the files of each spk
                sclite \
                    -r "${_scoredir}/ref.trn" trn \
                    -h "${_scoredir}/hyp.trn" trn \
                    -i rm -o all stdout > "${_scoredir}/result.txt"

                log "Write ${_type} result in ${_scoredir}/result.txt"
                grep -e Avg -e SPKR -m 2 "${_scoredir}/result.txt"
            done

        done

        # Show results in Markdown syntax
        scripts/utils/show_asr_result.sh "${joint_exp}" > "${joint_exp}"/RESULTS.md
        cat "${joint_exp}"/RESULTS.md

    fi
else
    log "Skip the evaluation stages"
fi

if [ ${stage} -le 13 ] && [ ${stop_stage} -ge 13 ]; then
    log "[Option] Stage 13: Pack model: ${joint_exp}/packed.zip"

    _opts=
    if [ "${feats_normalize}" = global_mvn ]; then
        _opts+="--option ${joint_stats_dir}/train/feats_stats.npz "
    fi

    # shellcheck disable=SC2086
    python -m espnet2.bin.pack asr\
        --train_config.yaml "${joint_exp}"/config.yaml \
        --model_file.pth "${joint_exp}"/"${decode_joint_model}" \
        ${_opts} \
        --option "${joint_exp}"/RESULTS.TXT \
        --outpath "${joint_exp}/packed.zip"

fi

log "Successfully finished. [elapsed=${SECONDS}s]"