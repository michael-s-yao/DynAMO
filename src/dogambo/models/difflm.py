#!/usr/bin/python3
"""
Diffusion language model.

Author(s):
    Michael Yao @michael-s-yao

Citation(s):
    [1] Lovelace J, Kishore V, Wan C, Shekhtman E, Weinberger KQ. Latent
        diffusion for language generation. Proc NeurIPS. (2023). doi:
        https://doi.org/10.48550/arXiv.2212.09462

Portions of this code were adapted from the latent-diffusion-for-language
repository by @justinlovelace at the following URL:
https://github.com/justinlovelace/latent-diffusion-for-language

Licensed under the MIT License. Copyright University of Pennsylvania 2024.
"""
from __future__ import annotations
import os
import json
import logging
import sys
import torch
import torch.nn as nn
from huggingface_hub import PyTorchModelHubMixin
from pathlib import Path
from transformers import (
    AutoTokenizer, AutoConfig, BartForConditionalGeneration
)
from transformers.modeling_outputs import BaseModelOutput
from typing import Final, Sequence, Union

DIFFUSIONLM_PATH: Final[Union[Path, str]] = os.environ.get(
    "DIFFUSIONLM_PATH", "latent-diffusion-for-language"
)
sys.path.append(DIFFUSIONLM_PATH)
from latent_models.bart_latent_model import BARTForConditionalGenerationLatent
from diffusion.text_denoising_diffusion import GaussianDiffusion
from model.diffusion_transformer import DiffusionTransformer


class DiffusionLM(nn.Module, PyTorchModelHubMixin):
    attn_head_dim: int = 64

    def __init__(
        self,
        latent_model_path: Union[Path, str],
        enc_dec_model: str = "facebook/bart-base",
        tx_dim: int = 768,
        tx_depth: int = 12,
        max_seq_len: int = 64,
        self_condition: bool = False,
        scale_shift: bool = True,
        dropout: float = 0.1,
        class_conditional: bool = False,
        num_dense_connections: int = 3,
        sampling_timesteps: int = 250,
        sampler: str = "ddpm",
        scale: float = 1.0,
        num_encoder_latents: int = 32,
        num_decoder_latents: int = 32,
        dim_ae: int = 64,
        num_layers: int = 3,
        l2_normalize_latents: bool = True,
        **kwargs
    ):
        """
        Args:
            latent_model_path: the path to the latent model.
            enc_dec_model: encoder-decoder model name.
            tx_dim: dimension of the denoising transformer. Default 768.
            tx_depth: depth of the denoising transformer. Default 12.
            max_seq_len: maximum generated sequence length. Default 64.
            self_condition: whether to do self-conditioning. Default False.
            scale_shift: whether to do scale shifting: Default True.
            dropout: dropout parameter. Default 0.1.
            class_conditional: whether to do class-conditional generation.
                Default False.
            num_dense_connections: number of dense connections. Default 3.
            sampling_timesteps: number of sampling timesteps. Default 250.
            sampler: noise schedule. Default `ddpm`.
            scale: scale parameter. Default 1.0
            num_encoder_latents: reconstructor encoder latent size. Default 32.
            num_decoder_latents: reconstructor decoder latent size. Default 32.
            dim_ae: reconstructor autoencoder dimensions. Default 64.
            num_layers: number of layers for the conditional BART model.
                Default 3.
            l2_normalize_latents: whether to L2 normalize the latent vectors.
                Default True.
        """
        super().__init__()
        self.latent_model_path: Final[Union[Path, str]] = latent_model_path
        self.enc_dec_model: Final[str] = enc_dec_model
        self.enc_dec_config = AutoConfig.from_pretrained(self.enc_dec_model)
        self.tx_dim: Final[int] = tx_dim
        self.tx_depth: Final[int] = tx_depth
        self.latent_dim: Final[int] = self.enc_dec_config.d_model
        self.max_seq_len: Final[int] = max_seq_len
        self.self_condition: Final[bool] = self_condition
        self.scale_shift: Final[bool] = scale_shift
        self.dropout: Final[float] = dropout
        self.class_conditional: Final[bool] = class_conditional
        self.num_dense_connections: Final[int] = num_dense_connections

        self.sampling_timesteps: Final[int] = sampling_timesteps
        self.sampler: Final[str] = sampler
        self.scale: Final[float] = scale

        self.num_encoder_latents: Final[int] = num_encoder_latents
        self.num_decoder_latents: Final[int] = num_decoder_latents
        self.dim_ae: Final[int] = dim_ae
        self.num_layers: Final[int] = num_layers
        self.l2_normalize_latents: Final[bool] = l2_normalize_latents

        for key, val in kwargs.items():
            setattr(self, key, val)

        assert (self.tx_dim % self.attn_head_dim) == 0
        model = DiffusionTransformer(
            tx_dim=self.tx_dim,
            tx_depth=self.tx_depth,
            heads=self.tx_dim // self.attn_head_dim,
            latent_dim=self.latent_dim,
            max_seq_len=self.max_seq_len,
            self_condition=self.self_condition,
            scale_shift=self.scale_shift,
            dropout=self.dropout,
            class_conditional=self.class_conditional,
            num_classes=0,
            class_unconditional_prob=0.1,
            seq2seq=False,
            seq2seq_context_dim=self.latent_dim,
            num_dense_connections=self.num_dense_connections
        )

        self.diffusion = GaussianDiffusion(
            model,
            max_seq_len=model.max_seq_len,
            sampling_timesteps=self.sampling_timesteps,
            sampler=self.sampler,
            train_schedule="cosine",
            sampling_schedule=None,
            loss_type="l2",
            objective="pred_v",
            train_prob_self_cond=0.5,
            seq2seq_unconditional_prob=0.1,
            scale=self.scale
        )

        self.tokenizer = AutoTokenizer.from_pretrained(self.enc_dec_model)
        self.bart_model = BartForConditionalGeneration.from_pretrained(
            self.enc_dec_model
        )
        self.bart_config = self.bart_model.config
        self.diffusion.context_encoder = self.bart_model.get_encoder()
        self.context_tokenizer = self.tokenizer
        self.bart_model = BARTForConditionalGenerationLatent.from_pretrained(
            self.enc_dec_model,
            config=self.bart_config,
            num_encoder_latents=self.num_encoder_latents,
            num_decoder_latents=self.num_decoder_latents,
            dim_ae=self.dim_ae,
            num_layers=self.num_layers,
            l2_normalize_latents=self.l2_normalize_latents,
            _fast_init=False
        )

        if torch.cuda.is_available():
            model = model.cuda()
            self.diffusion = self.diffusion.cuda()
            self.bart_model = self.bart_model.cuda()
        self.device = next(model.parameters()).device

        bart_ckpt = os.path.join(
            DIFFUSIONLM_PATH, self.latent_model_path, "model.pt"
        )
        data = torch.load(
            bart_ckpt, map_location=self.device, weights_only=True
        )
        self.bart_model.load_state_dict(data["model"])
        self.diffusion.context_encoder = self.bart_model.get_encoder()
        self.diffusion.using_latent_model = True

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the diffusion language model.
        Input:
            z: the latent space noise vector of shape D or BD, where B is the
                (optional) batch dimension and D is the number of input
                dimensions.
        Returns:
            The generated tokens.
        """
        z = z.reshape(-1, self.num_decoder_latents, self.latent_dim)
        z = self.bart_model.get_decoder_input(z)
        z = BaseModelOutput(last_hidden_state=z)
        return self.bart_model.generate(encoder_outputs=z)

    def decode(self, toks: torch.Tensor) -> Sequence[str]:
        """
        Decode a sequence of tokens into text.
        Input:
            toks: the batched sequence of tokens to decode.
        Returns:
            The decoded text.
        """
        texts = [
            self.tokenizer.decode(
                g, skip_special_tokens=True, clean_up_tokenization_spaces=True
            )
            for g in toks
        ]
        return [text.strip() for text in texts]

    @classmethod
    def load_from_checkpoint(
        cls, diffusion_ckpt_dir: Union[Path, str]
    ) -> DiffusionLM:
        """
        Loads the diffusion language model for inference from a checkpoint.
        Input:
            diffusion_ckpt_dir: the checkpoint directory of the trained
                generative diffusion model.
        Returns:
            The loaded diffusion language model.
        """
        with open(os.path.join(diffusion_ckpt_dir, "args.json"), "rt") as f:
            model_args = json.load(f)
        latent_args_fn = os.path.join(
            DIFFUSIONLM_PATH, model_args["latent_model_path"], "args.json"
        )
        with open(latent_args_fn, "rt") as f:
            model_args.update(json.load(f))
        model_args["latent_dim"] = model_args["dim_ae"]

        model = cls(**model_args)
        diffusion_ckpt_fn = os.path.join(diffusion_ckpt_dir, "model.pt")
        assert os.path.isfile(diffusion_ckpt_fn)
        data = torch.load(
            diffusion_ckpt_fn, map_location=model.device, weights_only=True
        )
        model.diffusion.load_state_dict(data["model"])
        logging.info(f"Loading diffusion model from {diffusion_ckpt_fn}")

        model.diffusion = model.diffusion.cuda()
        model.bart_model = model.bart_model.cuda()

        return model

    @torch.no_grad()
    def sample(self, num_samples: int = 1) -> Sequence[str]:
        """
        Generates sample text by sampling from the latent normal prior.
        Input:
            num_samples: the number of samples of text to generate. Default 1.
        Returns:
            A list of the generated text samples.
        """
        z = torch.randn(num_samples, self.num_decoder_latents, self.latent_dim)
        z = z.to(next(self.diffusion.parameters()).device)
        toks = self(z)
        return self.decode(toks)
