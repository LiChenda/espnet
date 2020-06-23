#!/bin/bash

# Copyright 2020  Shanghai Jiao Tong University (Authors: Wangyou Zhang)
# Apache 2.0

min_or_max=min

. utils/parse_options.sh
. ./path.sh

if [ $# -ne 2 ]; then
  echo "Arguments should be WSJ0-2MIX data directory and spatialized WSJ0-2MIX wav path, see local/data.sh for example."
  exit 1;
fi

# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

wsj0_2mix_datadir=$1
wsj0_2mix_spatialized_wavdir=$2

# check if the data dirs exist.
for f in $wsj0_2mix_datadir/tr $wsj0_2mix_datadir/cv $wsj0_2mix_datadir/tt; do
  if [ ! -d $f ]; then
    echo "Error: $f is not a directory."
    exit 1;
  fi
done
# check if the wav dirs exist.
for suffix in anechoic reverb; do
  for x in tr cv tt; do
    f=${wsj0_2mix_spatialized_wavdir}/2speakers_${suffix}/wav16k/${min_or_max}/${x}/mix
    if [ ! -d $f ]; then
      echo "Error: $f is not a directory."
      exit 1;
    fi
  done
done

data=./data
rm -r ${data}/{tr,cv,tt}_spatialized_anechoic_multich 2>/dev/null
rm -r ${data}/{tr,cv,tt}_spatialized_reverb_multich 2>/dev/null

for x in tr_spatialized_anechoic_multich cv_spatialized_anechoic_multich tt_spatialized_anechoic_multich \
         tr_spatialized_reverb_multich cv_spatialized_reverb_multich tt_spatialized_reverb_multich; do
  mkdir -p ${data}/$x
  x_ori=${x%%_*}    # tr, cv, tt
  suffix=$(echo $x | rev | cut -d"_" -f2 | rev)   # anechoic, reverb
  wavdir=${wsj0_2mix_spatialized_wavdir}/2speakers_${suffix}/wav16k/${min_or_max}/${x_ori}
  awk '{print $1}' ${wsj0_2mix_datadir}/${x_ori}/wav.scp | \
    cut -d"_" -f 3- | \
    awk -v dir="$wavdir" -v suffix="$suffix" '{printf("%s_%s %s/mix/%s.wav\n", $1, suffix, dir, $1)}' | \
    awk '{split($1, lst, "_"); spk=substr(lst[1],1,3)"_"substr(lst[3],1,3); print(spk"_"$0)}' | sort > ${data}/$x/wav.scp

  awk '{split($1, lst, "_"); spk=lst[1]"_"lst[2]; print($1, spk)}' ${data}/$x/wav.scp | sort > ${data}/$x/utt2spk
  utt2spk_to_spk2utt.pl ${data}/$x/utt2spk > ${data}/$x/spk2utt

  # transcriptions (only for 'max' version)
  if [[ "$min_or_max" = "max" ]]; then
    paste -d " " \
      <(awk -v suffix="$suffix" '{print($1 "_" suffix)}' ${wsj0_2mix_datadir}/${x_ori}/text_spk1) \
      <(cut -f 2- -d" " ${wsj0_2mix_datadir}/${x_ori}/text_spk1) | sort > ${data}/${x}/text_spk1
    paste -d " " \
      <(awk -v suffix="$suffix" '{print($1 "_" suffix)}' ${wsj0_2mix_datadir}/${x_ori}/text_spk2) \
      <(cut -f 2- -d" " ${wsj0_2mix_datadir}/${x_ori}/text_spk2) | sort > ${data}/${x}/text_spk2
  fi
done