# ZEN

ZEN is a BERT-based Chinese **(Z)** text encoder **E**nhanced by **N**-gram representations, where different combinations of characters are considered during training. The potential word or phrase boundaries are explicitly pre-trained and fine-tuned with the character encoder (BERT), so that ZEN incorporates the comprehensive information of both the character sequence and words or phrases it contains. The structure of ZEN is illustrated in the figure below.

　

![ZEN_model](docs/figures/zen.png)

　
## Citation

If you use or extend our work, please cite the following [**paper**](https://aclanthology.org/2020.findings-emnlp.425) published in EMNLP-2020:
```
@inproceedings{diao-etal-2020-zen,
    title = "ZEN: Pre-training Chinese Text Encoder Enhanced by N-gram Representations",
    author = "Diao, Shizhe and Bai, Jiaxin and Song, Yan and Zhang, Tong and Wang, Yonggang",
    booktitle = "Findings of the Association for Computational Linguistics: EMNLP 2020",
    month = nov,
    year = "2020",
    address = "Online",
    pages = "4729--4740",
}
```


## Quick tour of pre-training and fine-tune using ZEN

The library comprises several example scripts for conducting [**Chinese NLP tasks**](/datasets):

- tasks:
```
{'cola': <class 'utils_sequence_level_task.ColaProcessor'>, 
'mnli': <class 'utils_sequence_level_task.MnliProcessor'>, 
'mnli-mm': <class 'utils_sequence_level_task.MnliMismatchedProcessor'>, 
'mrpc': <class 'utils_sequence_level_task.MrpcProcessor'>, 
'sst-2': <class 'utils_sequence_level_task.Sst2Processor'>, 
'qqp': <class 'utils_sequence_level_task.QqpProcessor'>, 
'qnli': <class 'utils_sequence_level_task.QnliProcessor'>, 
'rte': <class 'utils_sequence_level_task.RteProcessor'>, 
'wnli': <class 'utils_sequence_level_task.WnliProcessor'>, 
'xnli': <class 'utils_sequence_level_task.XnliProcessor'>, 
'fudansmall': <class 'utils_sequence_level_task.FudansmallProcessor'>, 
'fudanlarge': <class 'utils_sequence_level_task.FudanlargeProcessor'>, 
'thucnews': <class 'utils_sequence_level_task.ThucnewsProcessor'>, 
'chnsenticorp': <class 'utils_sequence_level_task.ChnsenticorpProcessor'>, 
'lcqmc': <class 'utils_sequence_level_task.LcqmcProcessor'>}
```

- `run_pre_train.py`: an example pre-training ZEN
- `run_sequence_level_classification.py`: an example fine-tuning ZEN on DC, SA, SPM and NLI tasks (*sequence-level classification*)

```
python examples/run_sequence_level_classification.py --data_dir ./datasets/ChnSentiCorp --bert_model bert-base-chinese --task_name chnsenticorp
```


- `run_token_level_classification.py`: an example fine-tuning ZEN on CWS, POS and NER tasks (*token-level classification*)


[**Examples**](/examples) of pre-training and fine-tune using ZEN.


## Contact information

For help or issues using ZEN, please submit a GitHub issue.

For personal communication related to ZEN, please contact Yuanhe Tian (`yhtian94@gmail.com`).

## example usage
```python
python examples/run_sequence_level_classification.py --data_dir datasets/ChnSentiCorp --bert_model models/test_output/zen0317092603_epoch_2/ --task_name chnsenticorp --num_train_epochs 3

python examples/run_pre_train.py --pregenerated_data pretrain_data/policy/pretrain_data --output_dir models/test_output --bert_model models/test_ZEN_pretrain_base

python examples/run_pre_train.py --pregenerated_data pretrain_data/policy/pretrain_data --output_dir models/pretrained+gate-jieba/ --bert_model models/test_zen

python examples/run_sequence_level_classification.py --data_dir datasets/policy_SM --bert_model models/pretrained+gate-jieba/ --task_name lcqmc --do_train --do_eval
```