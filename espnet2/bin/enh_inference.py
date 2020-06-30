#!/usr/bin/env python3
import logging
import sys
from typing import Optional
from typing import Sequence
from typing import Tuple
from typing import Union

import configargparse
import torch
from typeguard import check_argument_types

from espnet.utils.cli_utils import get_commandline_args
from espnet2.fileio.sound_scp import SoundScpWriter
from espnet2.tasks.frontend import FrontendTask

from espnet2.torch_utils.device_funcs import to_device
from espnet2.torch_utils.set_all_random_seed import set_all_random_seed
from espnet2.utils.types import str2bool
from espnet2.utils.types import str2triple_str
from espnet2.utils.types import str_or_none


def inference(
    output_dir: str,
    batch_size: int,
    dtype: str,
    ngpu: int,
    seed: int,
    num_workers: int,
    log_level: Union[int, str],
    data_path_and_name_and_type: Sequence[Tuple[str, str, str]],
    key_file: Optional[str],
    enh_train_config: str,
    enh_model_file: str,
    allow_variable_data_keys: bool,
):
    assert check_argument_types()
    if batch_size > 1:
        raise NotImplementedError("batch decoding is not implemented")
    if ngpu > 1:
        raise NotImplementedError("only single GPU decoding is supported")

    logging.basicConfig(
        level=log_level,
        format="%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s",
    )

    if ngpu >= 1:
        device = "cuda"
    else:
        device = "cpu"

    # 1. Set random-seed
    set_all_random_seed(seed)

    # 2. Build Enh model
    enh_model, enh_train_args = FrontendTask.build_model_from_file(
        enh_train_config, enh_model_file, device
    )
    enh_model.eval()

    num_spk = enh_model.num_spk
    fs = enh_model.fs

    # 3. Build data-iterator
    loader = FrontendTask.build_streaming_iterator(
        data_path_and_name_and_type,
        dtype=dtype,
        batch_size=batch_size,
        key_file=key_file,
        num_workers=num_workers,
        preprocess_fn=FrontendTask.build_preprocess_fn(enh_train_args, False),
        collate_fn=FrontendTask.build_collate_fn(enh_train_args),
        allow_variable_data_keys=allow_variable_data_keys,
        inference=True,
    )

    writers = []
    for i in range(num_spk):
        writers.append(
            SoundScpWriter(f"{output_dir}/wavs/{i + 1}", f"{output_dir}/spk{i + 1}.scp")
        )

    for keys, batch in loader:
        assert isinstance(batch, dict), type(batch)
        assert all(isinstance(s, str) for s in keys), keys
        _bs = len(next(iter(batch.values())))
        assert len(keys) == _bs, f"{len(keys)} != {_bs}"

        with torch.no_grad():
            # a. To device
            batch = to_device(batch, device)
            # b. Forward Enhancement Frontend
            waves, _, _ = enh_model.frontend.forward_rawwav(**batch)
            assert len(waves[0]) == batch_size, len(waves[0])

        # FIXME(Chenda): will be incorrect when
        #  batch size is not 1 or multi-channel case
        waves = [w.T.cpu().numpy() for w in waves]
        for (i, w) in enumerate(waves):
            writers[i][keys[0]] = fs, w

    for writer in writers:
        writer.close()


def get_parser():
    parser = configargparse.ArgumentParser(
        description="Frontend inference",
        config_file_parser_class=configargparse.YAMLConfigFileParser,
        formatter_class=configargparse.ArgumentDefaultsHelpFormatter,
    )

    # Note(kamo): Use '_' instead of '-' as separator.
    # '-' is confusing if written in yaml.
    parser.add_argument("--config", is_config_file=True, help="config file path")

    parser.add_argument(
        "--log_level",
        type=lambda x: x.upper(),
        default="INFO",
        choices=("INFO", "ERROR", "WARNING", "INFO", "DEBUG", "NOTSET"),
        help="The verbose level of logging",
    )

    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument(
        "--ngpu", type=int, default=0, help="The number of gpus. 0 indicates CPU mode",
    )
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument(
        "--dtype",
        default="float32",
        choices=["float16", "float32", "float64"],
        help="Data type",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=1,
        help="The number of workers used for DataLoader",
    )

    group = parser.add_argument_group("Input data related")
    group.add_argument(
        "--data_path_and_name_and_type",
        type=str2triple_str,
        required=True,
        action="append",
    )
    group.add_argument("--key_file", type=str_or_none)
    group.add_argument("--allow_variable_data_keys", type=str2bool, default=False)

    group = parser.add_argument_group("The model configuration related")
    group.add_argument("--enh_train_config", type=str, required=True)
    group.add_argument("--enh_model_file", type=str, required=True)

    group = parser.add_argument_group("Beam-search related")
    group.add_argument(
        "--batch_size", type=int, default=1, help="The batch size for inference",
    )

    return parser


def main(cmd=None):
    print(get_commandline_args(), file=sys.stderr)
    parser = get_parser()
    args = parser.parse_args(cmd)
    kwargs = vars(args)
    kwargs.pop("config", None)
    inference(**kwargs)


if __name__ == "__main__":
    main()