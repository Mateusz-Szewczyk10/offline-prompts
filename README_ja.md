# OfflinePrompts: ログデータを用いた文章生成個別最適化のためのPythonパッケージ

<div align="center"><img src="https://raw.githubusercontent.com/aiueola/offline-prompts/main/images/logo.png" width="100%"/></div>

[![pypi](https://img.shields.io/pypi/v/scope-rl.svg)](https://pypi.python.org/pypi/offline-prompts)
[![Python](https://img.shields.io/badge/python-3.9%20%7C%203.10%20%7C%203.11-blue)](https://www.python.org)
[![Downloads](https://pepy.tech/badge/offline-prompts)](https://pepy.tech/project/offline-prompts)
[![GitHub commit activity](https://img.shields.io/github/commit-activity/m/aiueola/offline-prompts)](https://github.com/aiueola/offline-prompts/graphs/contributors)
[![GitHub last commit](https://img.shields.io/github/last-commit/aiueola/offline-prompts)](https://github.com/aiueola/offline-prompts/graphs/commit-activity)
[![Documentation Status](https://readthedocs.org/projects/offline-prompts/badge/?version=latest)](https://offline-prompts.readthedocs.io/en/latest/)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![arXiv](https://img.shields.io/badge/arXiv-xxxx.xxxxx-b31b1b.svg)](https://arxiv.org/abs/xxxx.xxxxx)
[![arXiv](https://img.shields.io/badge/arXiv-xxxx.xxxxx-b31b1b.svg)](https://arxiv.org/abs/xxxx.xxxxx)

<details>
<summary><strong>Table of Contents </strong>(click to expand)</summary>

- [OfflinePrompts: ログデータを用いた文章生成個別最適化のためのPythonパッケージ](#offlineprompts-ログデータを用いた文章生成個別最適化のためのPythonパッケージ)
- [概要](#概要)
- [インストール](#インストール)
- [モジュールの紹介](#モジュールの紹介)
- [使用例](#使用例)
  - [xxx]()
- [引用](#引用)
- [プロジェクトチーム](#プロジェクトチーム)
- [ライセンス](#ライセンス)
- [連絡先](#連絡先)
- [参考文献等](#参考文献等)

</details>

**事前学習済みのベンチマーク用シミュレーションモデルとデータセットは [こちら]().**

**ドキュメンテーションは [こちら]().**

**PyPI の stable version は [こちら]().**

**解説スライドは [こちら]().**

**Enligh version is [here](README.md)。**

## 概要

推薦システムやオンライン広告、教育アプリなどではユーザー体験を改善するために文章を個別最適化することが非常に重要です。例えば、下記の図で示されるように、映画の推薦システムで "Wall-E (2008)" という映画を短いキャッチコピーを使って推薦する場面を考えてみます。この映画は一般的にサイエンス・フィクションやヒューマンドラマ、環境保全など様々な視点で語られる映画ですが、SF好きのユーザーにはSF要素に注目し、恋愛好きのユーザーにはヒューマンドラマや恋愛面の要素に注目した文章を生成することで、より "刺さる" 側面に注目して映画を紹介することができます。このような個別最適化は、ユーザー体験のためにも、企業の収益向上のためにも非常に有用です。

<div align="center"><img src="https://raw.githubusercontent.com/aiueola/offline-prompts/main/images/motivative_example_ja.png" width="100%"/></div>
<figcaption>
<p align="center">
  映画推薦において個別最適化された文章を生成したい場面の一例
</p>
</figcaption>

本パッケージ、**OfflinePrompt** は、上記のような生成文章最適化を、日常のサービス運用で集まるログデータとプロンプト最適化のメカニズムを使って行うことを目的としたライブラリです。特に OfflinePrompts では、以下の図に示される手順で集められたデータを使うことを考えています： (1)プラットフォーム上にやってくるユーザーに対し、そのユーザー情報をもとに "プロンプト方策" がどんなプロンプト（短い指示文）を使って文章を生成するか決定する。(2)ユーザーは "プロンプト方策" が選んだプロンプトによって大規模言語モデル (LLM) で生成された文章のみを受け取り、それに対しクリックや購入、いいねボタンなどのフィードバックを返す。こうしたアプリケーション上で何かしらのログ方策をサービスの運営のために動かせば自然にログデータが溜まっていくので、それを使って新たな方策の学習や評価を行おうというのが **オフ方策学習・評価（Off-policy evaluation and learning; OPE/L）** の考え方になります。

<div align="center"><img src="https://raw.githubusercontent.com/aiueola/offline-prompts/main/images/personalized_sentence_generation.png" width="100%"/></div>
<figcaption>
<p align="center">
  プロンプトを使った文章生成個別最適化の問題を、文脈付きバンディットのオフ方策評価・学習として定式化
</p>
</figcaption>

こうした文章生成の個別最適化のためのオフ方策評価の研究や実務応用を促進するため、**OfflinePrompts** は以下のサポートを行っています。

- プロンプト最適化に使える代表的なオフ方策学習手法の標準実装。
- 人工データと大規模言語モデルを使った文章生成タスクにおけるベンチマーク環境の実装。
- スムーズな実験のための、オフ方策評価と文章生成の一貫したパイプライン実装. 

中でも、大規模言語モデルを使った文章生成タスクにおけるベンチマーク環境の実装が特に力を入れた点です。このベンチマークは映画の説明文を生成してデータ拡張した [MovieLens](https://grouplens.org/datasets/movielens/) のデータセットをもとに、ユーザーがどのようなジャンルの映画を好むかといった傾向を学習した、データをもとにしたシミュレーションになっています。こうしたシミュレーションの実装にはいくつか満たすべき性質があり、本ベンチマークは以下の特徴を満たす初の実験環境になっています。

- 大規模言語モデル (LLM) がアイテム (映画) に関する情報を持っているため、アイテムの説明文やキャッチコピーの生成が可能。
- 各アイテム (映画) が二つ以上の側面 (例えば恋愛やアクションなど) を持っており、それによりプロンプト最適化が実際にユーザーの満足度や報酬の改善につながりうる。
- 上記のプロンプトによる期待報酬への影響 "プロンプト効果" がユーザーによって異なり、一様ではない。
- さらにデータから "プロンプト効果" が学習可能であり、それによりリアルな報酬シミュレータを学習可能 (例えば、 MovieLensのデータでは、ユーザーと映画の特徴についての親和性が学習できる) 。

<details>
<summary><strong>実装の詳細はこちら</strong></summary>

<div align="center"><img src="https://raw.githubusercontent.com/aiueola/offline-prompts/main/images/workflow.png" width="100%"/></div>
<figcaption>
<p align="center">
  オフ方策学習 (Off-policy learning; OPL) の流れと OfflinePrompts で実装されているモジュール
</p>
</figcaption>

OfflinePromptsは dataset, OPL, policy の3つのモジュールで構成されており、実装は全て [PyTorch](https://pytorch.org/get-started/locally/) を使って書かれています。大まかには、dataset モジュールではベンチマーク環境を実装しており、policy モジュールでプロンプト方策の実装、OPL モジュールでプロンプト方策を学習するための learner モジュールの実装をしています。

*synthetic (人工データ)* と *full-LLM (LLMを使った文章生成)* の2つのベンチマークは両方とも、標準実装とサブモジュールによってカスタマイズされた環境が使えるようになっています。もし環境をカスタマイズしたい場合は、以下の4つのサブモジュールを設定することでシミュレーション環境の詳細な変更が可能です。

<div align="center"><img src="https://raw.githubusercontent.com/aiueola/offline-prompts/main/images/dataset_module.png" width="100%"/></div>
<figcaption>
<p align="center">
  人工データとLLMを使ったベンチマークを構成するサブモジュールの一覧
</p>
</figcaption>

- `ContextQueryModule`: データセットやユーザー情報、アイテム情報の読み込みに関わるモジュール
- `CandidateActionsModule`: 候補のプロンプト (アクション) を扱うモジュール
- `AuxiliaryOutputGenerator`/`FrozenLLM`: プロンプトを入力として、(大規模言語モデルによる) 文章生成を扱うモジュール
- `RewardSimulator`: (大規模言語モデルによって) 生成された文章に対し、ユーザーのフィードバックをシミュレーションするモジュール

[obp](https://github.com/st-tech/zr-obp) や [scope-rl](https://github.com/hakuhodo-technologies/scope-rl) などの既存のベンチマークと比較して、 OfflinePromptsでは `AuxiliaryOutputGenerator`/`FrozenLLM` を実装していることが特徴です。これにより、生成された文章という、報酬以外の別の補助的なフィードバックの存在する文脈付きバンディットの設定をシミュレーションすることができます。また、`FrozenLLM` や `RewardSimulator` のモジュールは [Huggingface](https://huggingface.co/models) のライブラリと互換性があるので、OfflinePrompts で文章生成を伴う実験を行う場合には様々なモデルをベースの大規模言語モデルとして選ぶことができます。そして、大規模言語モデルを使った実験環境は自前のデータでカスタマイズすることが可能であり、標準実装されている映画の説明文生成と同じように扱うことができます。データの読み込みや環境をカスタマイズする手順については、[このページ](./src/dataset/assets/README.md) を参照してください。 

</details>

**Further details are available in the [preprint]().**


## インストール

OfflinePromptsはPythonのパッケージ管理ツール `pip` からインストールできます。
```
pip install offline-prompts
```

OfflinePrompts のソースコードは github からも利用可能です。
```bash
git clone https://github.com/aiueola/offline-prompts
cd offline-prompts
python setup.py install
```

OfflinePrompts は Python 3.9 以降をサポートしています。 ライブラリ等の準備については [requirements.txt](./requirements.txt) を参照してください。

## 使用例

## 引用

もしこのベンチマーク・ライブラリを使う場合、または実験結果が研究に役に立った場合は、以下の論文の引用をお願いします。

(ベンチマーク実験)

Haruka Kiyohara, Daniel Yiming Cao, Yuta Saito, Thorsten Joachims.<br>
**An Off-Policy Learning Approach for Steering Sentence Generation towards Personalization**<br>
[link](https://arxiv.org/abs/2504.02646)

```
@inproceedings{kiyohara2025off,
  title = {An Off-Policy Learning Approach for Steering Sentence Generation towards Personalization},
  author = {Kiyohara, Haruka and Cao, Daniel Yiming and Saito, Yuta and Joachims, Thorsten},
  booktitle = {Proceedings of the 19th ACM Conference on Recommender Systems},
  pages = {xxx--xxx},
  year = {2025},
}
```

(パッケージ)

Haruka Kiyohara, Daniel Yiming Cao, Yuta Saito, Thorsten Joachims.<br>
**OfflinePrompts: Benchmark Suites for Prompt-guided Text Personalization from Logged Data**<br>
[link]()

```
@article{kiyohara2025offline,
  title = {OfflinePrompts: Benchmark Suites for Prompt-guided Text Personalization from Logged Data},
  author = {Kiyohara, Haruka and Cao, Daniel Yiming and Saito, Yuta and Joachims, Thorsten},
  journal = {arXiv preprint arXiv:},
  year = {2025},
}
```

## プロジェクトチーム

- [Haruka Kiyohara](https://sites.google.com/view/harukakiyohara) (**Main Contributor**; コーネル大学)
- Daniel Yiming Cao (Cornell University)
- [Yuta Saito](https://usait0.com/en/) (コーネル大学)
- [Thorsten Joachims](https://www.cs.cornell.edu/people/tj/) (コーネル大学) 

この研究は NSF Awards IIS-2312865 と OAC-2311521 により出資されています。 Haruka Kiyohara と Yuta Saito は当該の研究期間に [Funai Overseas Scholarship](https://funaifoundation.jp/scholarship/en/scholarship_guidelines_phd.html) により奨学金を受けています。

## ライセンス

このプロジェクトは Apache 2.0 のライセンスを使用しています。 - 詳細については [LICENSE](LICENSE) を参照してください。

## 連絡先

質問等ある場合はこちらまで： hk844 [at] cornell.edu.

## 参考文献等

<details>
<summary><strong>実装されている論文 </strong>(クリックして表示)</summary>

- Ronald J. Whilliams. "Simple Statistical Gradient-following Algorithms for Connectionist Reinforcement Learning." 1992.

- Vijay Konda and John Tsitsiklis. "Actor-critic algorithms." 1999.

- Alina Beygelzimer and John Langford. "The offset tree for learning with partial labels." 2009.

- Alex Strehl, John Langford, Lihong Li, and Sham M Kakade. "Learning from logged implicit exploration data." 2010.

- Miroslav Dudík, John Langford, and Lihong Li. "Doubly robust policy evaluation and learning." 2011.

- Yuta Saito, Qingyang Ren, and Thorsten Joachims. "Off-policy evaluation for large action spaces via conjunct effect modeling." 2023.

- Adith Swaminathan and Thorsten Joachims. "Batch learning from logged bandit feedback through counterfactual risk minimization." 2015.

- Miroslav Dudík, John Langford, and Lihong Li. "Doubly robust policy evaluation and learning." 2011.

- Yuta Saito, Jihan Yao, and Thorsten Joachims. "POTEC: Off-policy learning for large action spaces via two-stage policy decomposition." 2024.

- Haruka Kiyohara, Daniel Yiming Cao, Yuta Saito, and Thorsten Joachims. "An Off-Policy Learning Approach for Steering Sentence Generation towards Personalization". 2025.

</details>

<details>
<summary><strong>その他の参考文献 </strong>(クリックして表示)</summary>

- Mingkai Deng, Jianyu Wang, Cheng-Ping Hsieh, Yihan Wang, Han Guo, Tianmin Shu, Meng Song, Eric Xing, and Zhiting Hu. "RLPrompt: Optimizing discrete text prompts with reinforcement learning." 2022.

- Yuta Saito and Thorsten Joachims. "Off-policy evaluation for large action spaces via embedding." 2022.

- Noveen Sachdeva, Yi Su, and Thorsten Joachims. "Off-Policy Bandits with Deficient Support." 2020.

- Maxwell F. Harper and Joseph A. Konstan. "The MovieLens Datasets: History and Context." 2015.

- Yuta Saito, Shunsuke Aihara, Megumi Matsutani, and Yusuke Narita. "Open Bandit Dataset and Pipeline: Towards Realistic and Reproducible Off-Policy Evaluation." 2021.

- Haruka Kiyohara, Ren Kishimoto, Kosuke Kawakami, Ken Kobayashi, Kazuhide Nakata, Yuta Saito. "SCOPE-RL: A Python Library for Offline Reinforcement Learning and Off-Policy Evaluation." 2023.

- Thomas Wolf, Lysandre Debut, Victor Sanh, Julien Chaumond, Clement Delangue, Anthony Moi, Pierric Cistac, Tim Rault, Rémi Louf, Morgan Funtowicz, Joe Davison, Sam Shleifer, Patrick von Platen, Clara Ma, Yacine Jernite, Julien Plu, Canwen Xu, Teven Le Scao, Sylvain Gugger, Mariama Drame, Quentin Lhoest, and Alexander M. Rush. "Huggingface's Transformers: State-of-the-art Natural Language Processing." 2019.

- Tom Brown, Benjamin Mann, Nick Ryder, Melanie Subbiah, Jared Kaplan, Prafulla Dhariwal, Arvind Neelakantan, Pranav Shyam, Girish Sastry, Amanda Askell, Sandhini Agarwal, Ariel Herbert-Voss, Gretchen Krueger, Tom Henighan, Rewon Child, Aditya Ramesh, Daniel M. Ziegler, Jeffrey Wu, Clemens Winter, Christopher Hesse, Mark Chen, Eric Sigler, Mateusz Litwin, Scott Gray, Benjamin Chess, Jack Clark, Christopher Berner, Sam McCandlish, Alec Radford, Ilya Sutskever, and Dario Amodei. "Language Models are Few-Shot Learners." 2020.

- Albert Q. Jiang, Alexandre Sablayrolles, Guillaume Lample, Lucile Saulnier, Lélio Renard Lavaud, Marie-Anne Lachaux, Pierre Stock, Teven Le Scao, Thibaut Lavril, Thomas Wang, Timothee Lacroix, William El Sayed. "Mistral-7B." 2023.

</details>

<details>
<summary><strong>関連するプロジェクト </strong>(クリックして表示)</summary>

- **Open Bandit Pipeline**  -- 一般的な文脈付きバンディットの設定におけるオフ方策評価手法の実装: [[github](https://github.com/st-tech/zr-obp)] [[documentation](https://zr-obp.readthedocs.io/en/latest/)] [[paper](https://arxiv.org/abs/2008.07146)]
- **scope-rl** -- 一般的な強化学習の設定におけるオフ方策評価手法の実装: [[github](https://github.com/hakuhodo-technologies/scope-rl)] [[documentation](https://scope-rl.readthedocs.io/en/latest/)] [[paper](https://arxiv.org/abs/2311.18206)]
- **MovieLens** -- 映画推薦のための、ユーザーとアイテム間の評価値を集めたデータセット: [[documentation](https://grouplens.org/datasets/movielens/)] [[paper](https://dl.acm.org/doi/10.1145/2827872)]

</details>

