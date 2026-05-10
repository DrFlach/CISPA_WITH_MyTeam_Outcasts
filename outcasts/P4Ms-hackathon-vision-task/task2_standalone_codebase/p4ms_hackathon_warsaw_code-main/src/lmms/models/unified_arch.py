from copy import deepcopy
import torch
from torch import Tensor as T
from abc import ABC, abstractmethod
from hydra.utils import instantiate
from src.lmms.models.utils import (
    maybe_adjust_ckpt_keys,
    pretty_print_missing_keys_get_len,
)
from src.lmms.configs.unified_config import ModelArguments

import yaml

with open("config/vision.yaml", mode="r") as fd:
    vision = yaml.safe_load(fd)


import torch.distributed as dist


def is_main_process():
    if not dist.is_available() or not dist.is_initialized():
        return True
    return dist.get_rank() == 0


class UnifiedMetaModel:

    def __init__(self, config):
        super(UnifiedMetaModel, self).__init__(config)
        self.config = config

    def load_pretrained_weights(self, ckpt_path):
        if ckpt_path is None:
            return
        ckpt = maybe_adjust_ckpt_keys(torch.load(ckpt_path, map_location="cpu"))
        inc_keys = self.load_state_dict(ckpt, strict=False)
        pretty_print_missing_keys_get_len(inc_keys.missing_keys, "model.layers")
        if is_main_process():
            print(f"{inc_keys.unexpected_keys=}")
            print("loaded pretrained weights")

    def init_multimodal_modules(
        self,
        args: ModelArguments,
        visual_branch=False,
        speech_branch=False,
    ):
        if visual_branch:
            self.visual_encoder = instantiate(
                vision[args.visual_encoder_type]["encoder"]
            )
            self.vl_projector = instantiate(
                vision[args.visual_encoder_type]["projector"]
            )
            if args.visual_proj_ckpt_dir is not None:
                ckpt = maybe_adjust_ckpt_keys(
                    torch.load(args.visual_proj_ckpt_dir, map_location="cpu")
                )
                inc_keys = self.load_state_dict(ckpt, strict=False)
                pretty_print_missing_keys_get_len(
                    inc_keys.missing_keys, "vision_encoder"
                )
                pretty_print_missing_keys_get_len(inc_keys.missing_keys, "vl_projector")
                if is_main_process():
                    print(f"{inc_keys.unexpected_keys=}")
                    print("loaded visual processor weights")

    def encode_video(self, visual):
        vit_feature_list = self.visual_encoder(visual)  # [(b,t*n,d),(b,t*n,d),...]
        qformer_feature_list = []
        for vit_feature in vit_feature_list:
            qformer_feature = self.vl_projector(vit_feature)  # b,t*256 -> b,t*32
            qformer_feature_list.append(qformer_feature)
        return vit_feature_list, qformer_feature_list

    def encode_audio(self, audio):
        audio_feature = self.audio_encoder(audio)
        audio_feature = self.al_projector(audio_feature)
        return audio_feature

    def encode_mask(self, mask):
        return self.mask_encoder(mask)


class UnifiedMetaForCausalLM(ABC):

    @abstractmethod
    def get_model(self) -> UnifiedMetaModel:
        pass

    def encode_audio(self, audio, batch_first=True):
        if not batch_first:
            audio = audio.unsqueeze(0)
        audio_feature = self.get_model().encode_audio(audio)
        if not batch_first:
            audio_feature = audio_feature.squeeze(0)
        return audio_feature

    def encode_video(self, video, batch_first=True):
        if not batch_first:
            video = video.unsqueeze(0)
        vit_feature_list, qformer_feature_list = self.get_model().encode_video(video)
        if not batch_first:
            vit_feature_list = [item.squeeze(0) for item in vit_feature_list]
            qformer_feature_list = [item.squeeze(0) for item in qformer_feature_list]
        return vit_feature_list, qformer_feature_list

    def encode_ids(self, ids):
        return self.get_model().embed_tokens(ids)

    def prepare_multimodal_inputs(
        self,
        batch_input_ids,
        batch_labels,
        batch_X_modals,
        **kwargs,
    ):
        device = self.device

        def add_zeros(inputs: T, amount: int) -> None:
            inputs.append(torch.zeros((amount, 1), dtype=torch.long, device=device))

        def add_ones(inputs: T, amount: int) -> None:
            inputs.append(torch.ones((amount, 1), dtype=torch.long, device=device))

        bs = len(batch_input_ids)

        max_length = 0
        new_batch_inputs_embeds = []
        new_batch_attention_mask = []
        new_batch_labels = []

        new_batch_inputs_unimodal_mask_video = []
        new_batch_inputs_unimodal_mask_text = []
        new_batch_inputs_unimodal_mask_audio = []

        new_batch_inputs_unimodal_mask_question = []

        modalities_data = {
            k: [] for k in set([key for item in batch_X_modals for key in item])
        }
        for i in range(bs):
            input_ids = batch_input_ids[i]
            for key in modalities_data.keys():
                value = batch_X_modals[i].get(key, None)
                modalities_data[key].append(value)

        # --- Phantom term: accumulated 0-scaled encoder output for absent
        #     modalities, so DeepSpeed gradient buckets fire on every rank. ---
        self._phantom_loss = None
        inner = self.get_model()

        # VISION — always call if the model has a visual encoder
        if hasattr(inner, "visual_encoder") and inner.visual_encoder is not None:
            if "<image>" in modalities_data:
                nones = [item is not None for item in modalities_data["<image>"]]
                if any(nones):
                    zero_input = torch.zeros_like(
                        modalities_data["<image>"][
                            torch.argmax(torch.tensor(nones).long())
                        ]
                    )
                    for i in range(bs):
                        if modalities_data["<image>"][i] is None:
                            modalities_data["<image>"][i] = zero_input
                    stacked = torch.stack(modalities_data["<image>"], dim=0)
                    _, qformer_features_list = self.encode_video(
                        stacked, batch_first=True
                    )
                    modalities_data["<image>"] = [
                        qformer_features_list[-1][i] for i in range(bs)
                    ]
            else:
                # Modality completely absent — run encoder on dummy data,
                # keep output connected to graph via 0-scaling.
                img_size = getattr(inner.visual_encoder, "image_size", 384)
                dummy_img = torch.zeros(
                    1,
                    1,
                    3,
                    img_size,
                    img_size,
                    dtype=torch.bfloat16,
                    device=device,
                )
                _, qf = self.encode_video(dummy_img, batch_first=True)
                term = qf[-1].sum() * 0.0
                self._phantom_loss = (
                    term if self._phantom_loss is None else self._phantom_loss + term
                )

        # AUDIO — always call if the model has an audio encoder
        if hasattr(inner, "audio_encoder") and inner.audio_encoder is not None:
            if "<audio>" in modalities_data:
                nones = [item is not None for item in modalities_data["<audio>"]]
                if any(nones):
                    zero_input = torch.zeros_like(
                        modalities_data["<audio>"][
                            torch.argmax(torch.tensor(nones).long())
                        ]
                    )
                    for i in range(bs):
                        if modalities_data["<audio>"][i] is None:
                            modalities_data["<audio>"][i] = zero_input
                    stacked = torch.stack(modalities_data["<audio>"], dim=0)
                    audio_features = self.encode_audio(stacked, batch_first=True)
                    modalities_data["<audio>"] = [audio_features[i] for i in range(bs)]
            else:
                # Modality completely absent — dummy forward
                dummy_audio = torch.zeros(
                    1,
                    3000,
                    128,
                    dtype=torch.bfloat16,
                    device=device,
                )
                feats = self.encode_audio(dummy_audio, batch_first=True)
                term = feats.sum() * 0.0
                self._phantom_loss = (
                    term if self._phantom_loss is None else self._phantom_loss + term
                )

        for i in range(bs):
            input_ids = batch_input_ids[i]
            labels = batch_labels[i]

            X_token_indices = torch.where(
                torch.any(
                    torch.stack(
                        [
                            input_ids == self.SPECIAL_TOKEN_2_IDS[key]
                            for key in self.KEYS
                        ]
                    ),
                    dim=0,
                )
            )[0]
            X_token_indices = X_token_indices.tolist()
            inputs_embeds_seg = []

            inputs_unimodal_mask_video = []
            inputs_unimodal_mask_text = []
            inputs_unimodal_mask_audio = []
            inputs_unimodal_mask_question = []

            labels_seg = []
            pre_indice = 0
            text, question = True, False

            for idx, indice in enumerate(X_token_indices):
                special_token = self.IDS_2_SPECIAL_TOKEN[input_ids[indice].item()]

                if text:
                    idx = (
                        indice + 1
                        if not special_token in ["<image>", "<audio>"]
                        else indice
                    )  # handle cases when no <modality_start> token
                    tmp = self.encode_ids(
                        input_ids[pre_indice:idx]
                    )  # include <modality_start>

                    inputs_embeds_seg.append(tmp)
                    amount = tmp.size()[0]
                    add_ones(inputs_unimodal_mask_text, amount)
                    add_zeros(inputs_unimodal_mask_video, amount)
                    add_zeros(inputs_unimodal_mask_audio, amount)
                    add_zeros(inputs_unimodal_mask_question, amount)

                    labels_seg.append(labels[pre_indice:idx])
                    text = False

                if special_token in ["<image>", "<audio>"]:
                    assert not text, "No text after <modality_start> allowed"
                    masks = [
                        inputs_unimodal_mask_text,
                        inputs_unimodal_mask_video,
                        inputs_unimodal_mask_audio,
                        inputs_unimodal_mask_question,
                    ]

                    if special_token == "<image>":
                        feature = modalities_data[special_token][i]
                        mask = masks.pop(1)
                    elif special_token == "<audio>":
                        feature = modalities_data[special_token][i]
                        mask = masks.pop(2)

                    inputs_embeds_seg.append(feature)
                    amount = feature.size()[0]
                    add_ones(mask, amount)
                    for m in masks:
                        add_zeros(m, amount)

                    labels_seg.append(
                        torch.full((amount,), -100, dtype=torch.long, device=device)
                    )

                if special_token in ["<image_end>", "<video_end>", "<audio_end>"]:
                    assert not text, "No text after <modality_start> allowed"
                    assert pre_indice == indice, "Should be a single token operation"
                    tmp = self.encode_ids(
                        input_ids[[indice]]
                    )  # only the <modality_end> token

                    inputs_embeds_seg.append(tmp)
                    amount = 1
                    add_ones(inputs_unimodal_mask_text, amount)
                    add_zeros(inputs_unimodal_mask_video, amount)
                    add_zeros(inputs_unimodal_mask_audio, amount)
                    add_zeros(inputs_unimodal_mask_question, amount)

                    labels_seg.append(labels[[indice]])

                if special_token == "<question_start>":
                    assert not text, "No text after <modality_start> allowed"
                    assert (
                        not question
                    ), "There should not be two consecutive <question_start> tokens."
                    tmp = self.encode_ids(
                        input_ids[pre_indice : indice + 1]
                    )  # include <question_start>

                    inputs_embeds_seg.append(tmp)
                    amount = tmp.size()[0]
                    add_ones(inputs_unimodal_mask_text, amount)
                    add_zeros(inputs_unimodal_mask_video, amount)
                    add_zeros(inputs_unimodal_mask_audio, amount)
                    add_zeros(inputs_unimodal_mask_question, amount)

                    labels_seg.append(labels[pre_indice : indice + 1])
                    question = True

                if special_token == "<question_end>":
                    # token size * emb size
                    tmp = self.encode_ids(input_ids[pre_indice : indice + 1])

                    inputs_embeds_seg.append(tmp)
                    amount = tmp.size()[0]
                    add_ones(inputs_unimodal_mask_text, amount)
                    add_zeros(inputs_unimodal_mask_video, amount)
                    add_zeros(inputs_unimodal_mask_audio, amount)

                    add_ones(
                        inputs_unimodal_mask_question, amount - 1
                    )  # all except <question_end>
                    add_zeros(inputs_unimodal_mask_question, 1)

                    labels_seg.append(labels[pre_indice : indice + 1])
                    question = False

                pre_indice = indice + 1

            # add last tokens
            # text token
            tmp = self.encode_ids(input_ids[pre_indice:])

            inputs_embeds_seg.append(tmp)
            amount = tmp.size()[0]
            add_ones(inputs_unimodal_mask_text, amount)
            add_zeros(inputs_unimodal_mask_video, amount)
            add_zeros(inputs_unimodal_mask_audio, amount)
            add_zeros(inputs_unimodal_mask_question, amount)
            labels_seg.append(labels[pre_indice:])

            # concat segs
            inputs_embeds_seg = torch.cat(inputs_embeds_seg, dim=0)
            inputs_unimodal_mask_text = torch.cat(inputs_unimodal_mask_text, dim=0)
            inputs_unimodal_mask_video = torch.cat(inputs_unimodal_mask_video, dim=0)
            inputs_unimodal_mask_audio = torch.cat(inputs_unimodal_mask_audio, dim=0)

            inputs_unimodal_mask_question = torch.cat(
                inputs_unimodal_mask_question, dim=0
            )

            assert (
                inputs_unimodal_mask_question.size() == inputs_unimodal_mask_text.size()
            ), "The sizes of inputs_unimodal_mask_question and input_unimodal_mask_text do not match."

            attention_mask_seg = torch.ones(
                inputs_embeds_seg.shape[0], dtype=torch.long, device=device
            )
            labels_seg = torch.cat(labels_seg, dim=0)

            new_batch_inputs_embeds.append(inputs_embeds_seg)
            new_batch_inputs_unimodal_mask_text.append(inputs_unimodal_mask_text)
            new_batch_inputs_unimodal_mask_video.append(inputs_unimodal_mask_video)
            new_batch_inputs_unimodal_mask_audio.append(inputs_unimodal_mask_audio)

            new_batch_inputs_unimodal_mask_question.append(
                inputs_unimodal_mask_question
            )

            new_batch_attention_mask.append(attention_mask_seg)
            new_batch_labels.append(labels_seg)

            max_length = max(max_length, inputs_embeds_seg.shape[0])

        ### left padding
        padding_inputs_embeds = []
        padding_inputs_mask_text = []
        padding_inputs_mask_video = []
        padding_inputs_mask_audio = []
        padding_inputs_mask_question = []

        padding_attention_mask = []
        padding_labels = []
        padding_mask_token_mask = []

        for i in range(bs):
            embeds = new_batch_inputs_embeds[i]
            mask_text = new_batch_inputs_unimodal_mask_text[i]
            mask_video = new_batch_inputs_unimodal_mask_video[i]
            mask_audio = new_batch_inputs_unimodal_mask_audio[i]
            mask_question = new_batch_inputs_unimodal_mask_question[i]

            mask = new_batch_attention_mask[i]
            labels = new_batch_labels[i]

            L, d = embeds.shape
            pad_embeds = self.encode_ids(
                torch.full(
                    (max_length - L,),
                    self.get_model().pad_token_id,
                    dtype=torch.long,
                    device=device,
                )
            )

            padding_inputs_embeds.append(torch.cat([pad_embeds, embeds], dim=0))

            temp = torch.zeros(
                (pad_embeds.size()[0], 1), dtype=torch.long, device=device
            )

            padding_inputs_mask_text.append(torch.cat([temp, mask_text], dim=0))
            padding_inputs_mask_video.append(torch.cat([temp, mask_video], dim=0))
            padding_inputs_mask_audio.append(torch.cat([temp, mask_audio], dim=0))

            padding_inputs_mask_question.append(torch.cat([temp, mask_question], dim=0))

            padding_attention_mask.append(
                torch.cat(
                    [
                        torch.zeros((max_length - L), dtype=torch.long, device=device),
                        mask,
                    ],
                    dim=0,
                )
            )
            padding_labels.append(
                torch.cat(
                    [
                        torch.full(
                            (max_length - L,), -100, dtype=torch.long, device=device
                        ),
                        labels,
                    ],
                    dim=0,
                )
            )

        padding_inputs_embeds = torch.stack(padding_inputs_embeds, dim=0)
        padding_inputs_mask_text = torch.stack(padding_inputs_mask_text, dim=0)
        padding_inputs_mask_video = torch.stack(padding_inputs_mask_video, dim=0)
        padding_inputs_mask_audio = torch.stack(padding_inputs_mask_audio, dim=0)
        padding_inputs_mask_question = torch.stack(padding_inputs_mask_question, dim=0)

        padding_attention_mask = torch.stack(padding_attention_mask, dim=0)
        padding_labels = torch.stack(padding_labels, dim=0)
        if len(padding_mask_token_mask) > 0:
            padding_mask_token_mask = torch.stack(padding_mask_token_mask, dim=0)

        position_ids = torch.cumsum(padding_attention_mask, dim=-1) - 1
        position_ids[position_ids == -1] = 0

        mask_emb = [
            padding_inputs_embeds,
            padding_inputs_mask_text,
            padding_inputs_mask_video,
            padding_inputs_mask_audio,
            padding_inputs_mask_question,
        ]
        mask_emb = (
            mask_emb if self.get_model().inputs_embeds_with_mmask else mask_emb[0]
        )

        dict_data = {
            "input_ids": None,
            "inputs_embeds": mask_emb,
            "attention_mask": padding_attention_mask,
            "labels": padding_labels,
            "position_ids": position_ids,
        }

        return dict_data

    def get_dummy_encoder_loss(self, batch_X_modals):
        """
        Run absent-modality encoders+projectors on a minimal dummy input so
        that DeepSpeed properly manages their parameter lifecycle (offload,
        prefetch, gradient reduction).  The output is scaled by 0.0 to have
        zero numerical impact on the loss.
        """
        if batch_X_modals is None:
            batch_X_modals = []

        present_keys = set()
        for sample in batch_X_modals:
            present_keys.update(sample.keys())

        inner = self.get_model()
        phantom = None

        # --- VISION: run encoder + projector on tiny dummy image ---
        if "<image>" not in present_keys:
            if hasattr(inner, "visual_encoder") and inner.visual_encoder is not None:
                img_size = getattr(inner.visual_encoder, "image_size", 384)
                # visual_encoder.forward expects (B, T, C, H, W)
                dummy_img = torch.zeros(
                    1,
                    1,
                    3,
                    img_size,
                    img_size,
                    dtype=torch.bfloat16,
                    device=self.device,
                )
                _, qf = self.encode_video(dummy_img, batch_first=True)
                term = qf[-1].sum() * 0.0
                phantom = term if phantom is None else phantom + term

        # --- AUDIO: run encoder + projector on tiny dummy mel ---
        if "<audio>" not in present_keys:
            if hasattr(inner, "audio_encoder") and inner.audio_encoder is not None:
                # SpeechEncoder.encode_audio does audio.permute(0,2,1)
                # so input shape is (B, time, mel_bins).
                # 3000 frames is Whisper's expected length.
                dummy_audio = torch.zeros(
                    1,
                    3000,
                    128,
                    dtype=torch.bfloat16,
                    device=self.device,
                )
                feats = self.encode_audio(dummy_audio, batch_first=True)
                term = feats.sum() * 0.0
                phantom = term if phantom is None else phantom + term

        return phantom

    def initialize_MM_tokenizer(self, tokenizer):
        vocab_nums = len(tokenizer)
        added_tokens = []
        image_tokens = ["<image>", "<image_start>", "<image_end>"]
        added_tokens += image_tokens
        video_tokens = ["<video>", "<video_start>", "<video_end>"]
        added_tokens += video_tokens
        audio_tokens = ["<audio>", "<audio_start>", "<audio_end>"]
        added_tokens += audio_tokens
        num_new_tokens = tokenizer.add_tokens(added_tokens, special_tokens=True)

        question_tokens = ["<question_start>", "<question_end>"]
        num_new_tokens += tokenizer.add_tokens(question_tokens, special_tokens=True)
        added_tokens += question_tokens

        self.KEYS = deepcopy(added_tokens)

        self.SPECIAL_TOKEN_2_IDS = {
            token: i + vocab_nums for i, token in enumerate(added_tokens)
        }
        self.IDS_2_SPECIAL_TOKEN = {
            i + vocab_nums: token for i, token in enumerate(added_tokens)
        }

        self.resize_token_embeddings(len(tokenizer))

    @property
    def device(self):
        return list(self.parameters())[0].device
