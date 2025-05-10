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
python  examples/create_pre_train_data.py --train_corpus pretrain_data/policy/policy_single_sentence.txt --output_dir pretrain_data/policy/pretrain_data_e10 --bert_model models/test_ZEN_pretrain_base --epochs_to_generate 10

python examples/run_sequence_level_classification.py --data_dir datasets/ChnSentiCorp --bert_model models/test_output/zen0317092603_epoch_2/ --task_name chnsenticorp --num_train_epochs 3

python examples/run_pre_train.py --pregenerated_data pretrain_data/policy/pretrain_data --output_dir models/test_output --bert_model models/test_ZEN_pretrain_base

python examples/run_pre_train.py --pregenerated_data pretrain_data/policy/pretrain_data --output_dir models/pretrained+gate-jieba/ --bert_model models/test_zen

python examples/run_sequence_level_classification.py --data_dir datasets/policy_SM --bert_model models/pretrained+gate-jieba/ --task_name lcqmc --do_train --do_eval
```

python examples/run_token_level_classification.py --data_dir datasets/CWS-MSR --bert_model models/ZEN_CWS/ --task cwsmsra --multift --do_train --do_eval