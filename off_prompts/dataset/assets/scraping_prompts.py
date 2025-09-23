"""Scripts for scraping prompts online."""
from typing import List
from pathlib import Path
from tqdm.auto import tqdm
import time

import hydra
from omegaconf import DictConfig

import requests
from bs4 import BeautifulSoup

import torch
import pandas as pd

from utils import format_runtime


def process(setting: str, keywords: List[str]):
    words = []

    with tqdm(torch.arange(len(keywords))) as pbar:
        for i, ch in enumerate(pbar):
            pbar.set_description(
                f"[scrape candidate prompts online: keyword - {keywords[i]}]"
            )

            url = f"https://relatedwords.io/{keywords[i]}"

            html = requests.get(url)
            soup = BeautifulSoup(html.content, "html.parser")
            elem = soup.select("a")

            for a in elem:
                try:
                    words.append(a.attrs["data-href"][1:])
                except:
                    pass

            time.sleep(3)

    df = pd.DataFrame()
    df["vocab"] = list(set(words))
    df.to_csv(f"{setting}_benchmark_prompts.csv", index=False)

    print(f"total number of prompts: {len(df)}")


@hydra.main(config_path="conf/", config_name="config")
def main(cfg: DictConfig):
    print(cfg)
    print(f"The current working directory is {Path().cwd()}")
    print(f"The original working directory is {hydra.utils.get_original_cwd()}")
    print()

    process(
        setting=cfg.setting.setting,
        keywords=cfg.setting.keywords,
    )


if __name__ == "__main__":
    start = time.time()
    main()
    finish = time.time()
    print("total runtime:", format_runtime(start, finish))
