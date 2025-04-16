import torch
from torch.utils.data import DataLoader, SequentialSampler,TensorDataset
import torch.nn.functional as F
import sys
import os

import numpy as np
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ZEN import BertTokenizer, ZenNgramDict,PolicyBertForSingleSentenceEmbedding
from utils_sequence_level_task import convert_examples_to_features, SingleSentenceProcessor


def encode_sentence(sentence, model_path, tokenizer_path, device='cpu'):
    """
    对单独的一句话进行编码
    :param sentence: str, 输入的句子
    :param model_path: str, 预训练模型的路径
    :param tokenizer_path: str, 分词器的路径
    :param device: str, 使用的设备 ('cpu' 或 'cuda')
    :return: tensor, 编码后的句子向量
    """
    # 加载分词器和模型
    tokenizer = BertTokenizer.from_pretrained(tokenizer_path, do_lower_case=True)
    ngram_dict = ZenNgramDict(tokenizer_path, tokenizer=tokenizer)
    model = PolicyBertForSingleSentenceEmbedding.from_pretrained(model_path)  
    processor = SingleSentenceProcessor()

    model.to(device)
    model.eval()

    examples = processor.get_test_examples(sentence)
    features = convert_examples_to_features(examples, ["0","1"], 128, tokenizer, ngram_dict)

    all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
    all_input_mask = torch.tensor([f.input_mask for f in features], dtype=torch.long)
    all_segment_ids = torch.tensor([f.segment_ids for f in features], dtype=torch.long)
    all_label_ids = torch.tensor([f.label_id for f in features], dtype=torch.long)
    all_ngram_ids = torch.tensor([f.ngram_ids for f in features], dtype=torch.long)
    all_ngram_positions = torch.tensor([f.ngram_positions for f in features], dtype=torch.long)
    all_ngram_lengths = torch.tensor([f.ngram_lengths for f in features], dtype=torch.long)
    all_ngram_seg_ids = torch.tensor([f.ngram_seg_ids for f in features], dtype=torch.long)
    all_ngram_masks = torch.tensor([f.ngram_masks for f in features], dtype=torch.long)

    train_dataset = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids, all_ngram_ids,
                              all_ngram_positions, all_ngram_lengths, all_ngram_seg_ids, all_ngram_masks)


    train_sampler = SequentialSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=1)

    # 进行推理
    for batch in train_dataloader:
        batch = tuple(t.to(device) for t in batch)
        input_ids, input_mask, segment_ids, label_ids, input_ngram_ids, ngram_position_matrix, \
        ngram_lengths, ngram_seg_ids, ngram_masks = batch

        with torch.no_grad():
            logits = model(input_ids=input_ids,
                           input_ngram_ids=input_ngram_ids,
                           ngram_position_matrix=ngram_position_matrix,
                            head_mask=None)
            
            if isinstance(logits, tuple):
                # print(f"Logits is a tuple, length: {len(logits)}")
                logits = logits[0]  
            else:
                # print(f"Logits is not a tuple, type: {type(logits)}")
                pass

            if isinstance(logits, list):
                # create numpy array from list
                logits = torch.tensor(logits)
            logits = logits.detach().cpu().numpy()


    return logits



# 示例用法
if __name__ == "__main__":
    
    model_path = "models\\result-seqlevel-2025-04-16-10-11-03"
    tokenizer_path = model_path
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # sentence = "这是一个测试句子。"
    # encoded_sentence = encode_sentence(sentence, model_path, tokenizer_path, device)
    # print("Encoded sentence:", encoded_sentence)
    # print("Encoded sentence shape:", encoded_sentence.shape)

    sentence1 = '2024年数据建设推动要素数字化转型'
    sentence2 = '老鼠爱大米'
    
    # query
    sentence3 = '2024数字经济发展趋势分析'

    encode_sentence1 = encode_sentence(sentence1, model_path, tokenizer_path, device)
    encode_sentence2 = encode_sentence(sentence2, model_path, tokenizer_path, device)
    encode_sentence3 = encode_sentence(sentence3, model_path, tokenizer_path, device)

    cos_sim = F.cosine_similarity(torch.tensor(encode_sentence1), torch.tensor(encode_sentence2))
    print("Cosine similarity between sentence1 and sentence2:", cos_sim.item())

    cos_sim = F.cosine_similarity(torch.tensor(encode_sentence1), torch.tensor(encode_sentence3))
    print("Cosine similarity between sentence1 and sentence3:", cos_sim.item())