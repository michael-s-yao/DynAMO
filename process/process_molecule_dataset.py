#!usr/bin/python3
"""
Script to construct the Molecule Penalized LogP custom MBO offline dataset.

Author(s):
    Michael Yao @michael-s-yao

Citation(s):
    [1] Krenn M, Hase F, Nigam AK, Friederich P, Aspuru-Guzik A. Self-
        referencing embedded strings (SELFIES): A 100% robust molecular string
        representation. Mach Learn: Sci Technol 1: 045024. (2020). doi:
        10.1088/2632-2153/aba947
    [2] Brown N, Fiscato M, Segler MHS, Vaucher AC. GuacaMol: Benchmarking
        models for de novo molecular design. J Chem Inf Model 59(3): 1096-108.
        (2019). doi: 10.1021/acs.jcim.8b00839

Licensed under the MIT License. Copyright University of Pennsylvania 2024.
"""
import click
import numpy as np
import selfies as sf
from datasets import Dataset, DatasetDict
from pathlib import Path
from typing import Optional, Tuple, Union

import dogambo


@click.command()
@click.option(
    "--train-fn",
    "-t",
    type=str,
    default="train.smiles",
    show_default=True,
    help="Path to the file of training molecules."
)
@click.option(
    "--val-fn",
    "-v",
    type=str,
    default="val.smiles",
    show_default=True,
    help="Path to the file of validation molecules."
)
@click.option(
    "--dataset-savename",
    type=str,
    default="None",
    show_default=True,
    help="Path or Huggingface Repo to save the dataset to."
)
@click.option(
    "--tokenizer-savename",
    type=str,
    default="None",
    show_default=True,
    help="Path or Huggingface Repo to save the tokenizer to."
)
def main(
    train_fn: Union[Path, str],
    val_fn: Union[Path, str],
    dataset_savename: Optional[Union[Path, str]] = None,
    tokenizer_savename: Optional[Union[Path, str]] = None
) -> Tuple[DatasetDict, dogambo.data.SELFIESTokenizer]:
    """Construct the Penalized LogP custom MBO offline dataset."""
    train = []
    with open(train_fn, "r") as f:
        for x in f.readlines():
            try:
                train.append(sf.encoder(x.strip()))
            except sf.exceptions.EncoderError:
                continue

    val = []
    with open(val_fn, "r") as f:
        for x in f.readlines():
            try:
                val.append(sf.encoder(x.strip()))
            except sf.exceptions.EncoderError:
                continue
    vocab = sf.get_alphabet_from_selfies(train + val)

    tokenizer = dogambo.data.SELFIESTokenizer()
    tokenizer.add_tokens(sorted(list(vocab)))
    tokenizer.add_special_tokens({
        "bos_token": "<start>", "eos_token": "<stop>"
    })
    tokenizer.pad_token_id = tokenizer.eos_token_id
    if tokenizer_savename is not None and tokenizer_savename != "None":
        tokenizer.push_to_hub(tokenizer_savename)

    for i in range(len(val)):
        val[i] = tokenizer.bos_token + val[i] + tokenizer.eos_token

    train = tokenizer(train, return_tensors="pt", padding=True)["input_ids"]
    val = tokenizer(val, return_tensors="pt", padding=True)["input_ids"]

    oracle = dogambo.oracle.PenalizedLogPOracle(None)

    ytrain = oracle(train)[..., np.newaxis]
    yval = oracle(val)[..., np.newaxis]
    dataset = DatasetDict({
        "train": Dataset.from_dict({"designs": train, "scores": ytrain}),
        "validation": Dataset.from_dict({"designs": val, "scores": yval})
    })
    if dataset_savename is not None and dataset_savename != "None":
        dataset.push_to_hub(dataset_savename)
    return dataset, tokenizer


if __name__ == "__main__":
    main()
