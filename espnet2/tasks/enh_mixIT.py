import argparse
from typing import Callable
from typing import Collection
from typing import Dict
from typing import List
from typing import Optional
from typing import Tuple

import numpy as np
import torch
from typeguard import check_argument_types
from typeguard import check_return_type


from espnet2.enh.abs_enh import AbsEnhancement
from espnet2.enh.nets.tasnet import TasNet
from espnet2.enh.nets.dprnn_raw import FaSNet_base as DPRNN
from espnet2.enh.nets.tf_mask_net import TFMaskingNet
from espnet2.enh.nets.beamformer_net import BeamformerNet
from espnet2.enh.espnet_model import ESPnetEnhancementModel_mixIT
from espnet2.tasks.abs_task import AbsTask
from espnet2.torch_utils.initialize import initialize
from espnet2.train.class_choices import ClassChoices
from espnet2.train.trainer import Trainer
from espnet2.utils.get_default_kwargs import get_default_kwargs
from espnet2.utils.nested_dict_action import NestedDictAction
from espnet2.utils.types import str2bool
from espnet2.utils.types import str_or_none

from egs2.wsj0_mixIT.enh1.codes.collate_fn import CommonCollateFn

enh_choices = ClassChoices(
    name="enh",
    classes=dict(
        tf_masking=TFMaskingNet,
        tasnet=TasNet,
        wpe_beamformer=BeamformerNet,
        dprnn=DPRNN,
    ),
    type_check=AbsEnhancement,
    default="tf_masking",
)

MAX_REFERENCE_NUM = 100


class EnhancementTask(AbsTask):
    # If you need more than one optimizers, change this value
    num_optimizers: int = 1

    class_choices_list = [
        # --enh and --enh_conf
        enh_choices,
    ]

    # If you need to modify train() or eval() procedures, change Trainer class here
    trainer = Trainer

    @classmethod
    def add_task_arguments(cls, parser: argparse.ArgumentParser):
        group = parser.add_argument_group(description="Task related")

        # NOTE(kamo): add_arguments(..., required=True) can't be used
        # to provide --print_config mode. Instead of it, do as
        # required = parser.get_default("required")

        group.add_argument(
            "--init",
            type=lambda x: str_or_none(x.lower()),
            default=None,
            help="The initialization method",
            choices=[
                "chainer",
                "xavier_uniform",
                "xavier_normal",
                "kaiming_uniform",
                "kaiming_normal",
                None,
            ],
        )

        group.add_argument(
            "--model_conf",
            action=NestedDictAction,
            default=get_default_kwargs(ESPnetEnhancementModel_mixIT),
            help="The keyword arguments for model class.",
        )

        group = parser.add_argument_group(description="Preprocess related")
        group.add_argument(
            "--use_preprocessor",
            type=str2bool,
            default=False,
            help="Apply preprocessing to data or not",
        )

        group = parser.add_argument_group(description="MIXit related")
        group.add_argument(
            "--N_per_mixture",
            type=int,
            default=4,
            help="Number of sources for each mixture",
        )
        group.add_argument(
            "--M_per_MoM",
            type=int,
            default=8,
            help="M for each mixture of mixtures",
        )
        group.add_argument(
            "--ratio_supervised",
            type=float,
            default=0.2,
        )
        group.add_argument(
            "--SNR_max",
            type=int,
            default=30,
        )
        for class_choices in cls.class_choices_list:
            # Append --<name> and --<name>_conf.
            # e.g. --encoder and --encoder_conf
            class_choices.add_arguments(group)

    @classmethod
    def build_collate_fn(
        cls, args: argparse.Namespace
    ) -> Callable[
        [Collection[Tuple[str, Dict[str, np.ndarray]]]],
        Tuple[List[str], Dict[str, torch.Tensor]],
    ]:
        # TODO:jing  here to mix the mixtures.
        assert check_argument_types()

        return CommonCollateFn(float_pad_value=0.0, int_pad_value=0)

    @classmethod
    def build_preprocess_fn(
        cls, args: argparse.Namespace, train: bool
    ) -> Optional[Callable[[str, Dict[str, np.array]], Dict[str, np.ndarray]]]:
        assert check_argument_types()
        retval = None
        assert check_return_type(retval)
        return retval

    @classmethod
    def required_data_names(
        cls, train: bool = True, inference: bool = False
    ) -> Tuple[str, ...]:
        if not inference:
            retval = ("speech_mix", "speech_ref1")
        else:
            # Recognition mode
            retval = ("speech_mix",)
        return retval

    @classmethod
    def optional_data_names(
        cls, train: bool = True, inference: bool = False
    ) -> Tuple[str, ...]:
        retval = ["dereverb_ref"]
        retval += ["mix_ref1"]
        retval += ["mix_ref2"]
        retval += ["mix_of_mixtures"]
        retval += ["speech_ref{}".format(n) for n in range(2, MAX_REFERENCE_NUM + 1)]
        retval += ["noise_ref{}".format(n) for n in range(1, MAX_REFERENCE_NUM + 1)]
        retval = tuple(retval)
        assert check_return_type(retval)
        return retval

    @classmethod
    def build_model(cls, args: argparse.Namespace) -> ESPnetEnhancementModel_mixIT:
        assert check_argument_types()

        enh_model = enh_choices.get_class(args.enh)(**args.enh_conf)

        # 1. Build model
        model = ESPnetEnhancementModel_mixIT(enh_model=enh_model)

        # FIXME(kamo): Should be done in model?
        # 2. Initialize
        if args.init is not None:
            initialize(model, args.init)

        assert check_return_type(model)
        return model