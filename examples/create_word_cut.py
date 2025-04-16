import os
import re
import time
import tqdm
from hanlp_restful import HanLPClient

#set api_key as environment variable
os.environ['API_KEY'] = 'ODIwNEBiYnMuaGFubHAuY29tOmJjazRWbUN0OWh6OUd2cEU='


class Hanlp_cutter:
    def __init__(self, stopwords_path='pretrain_data\stopword\stopword.txt'):
        self.hanlp =  HanLPClient('https://www.hanlp.com/api', auth=os.environ['API_KEY'] , language='zh')

        self.stopwords_path = stopwords_path
        self.stopwords = set()
        if stopwords_path:
            self.stopwords = self._read_stopwords(stopwords_path)

    def _read_stopwords(self, stopwords_path):
        """
        :param stopwords_path: str, the path to the stopwords file
        :return: set, the set of stopwords
        """
        with open(stopwords_path, 'r', encoding='utf-8') as f:
            stopwords = set(f.read().splitlines())
        print(f"stopwords loaded from {stopwords_path}, {len(stopwords)} stopwords")
        print(f"example stopword: {list(stopwords)[:10]}")
        return stopwords

    def cut(self, text):
        """
        :param text: str, the text to be tokenized
        :return: [['word1', 'word2', ...], ['word1', 'word2', ...], ...], the tokenized text
        """
        tokenized_sentences = self.hanlp.tokenize(text)
        # remove stopwords
        if self.stopwords:
            tokenized_sentences = [[word for word in sentence if word not in self.stopwords] for sentence in tokenized_sentences]
        return tokenized_sentences
    
    def cut_long_text(self, text, max_request_per_minute=60, max_sentence_length=14000):
        """
        split long text with "##第x条政策" 
        :param text: str, the text to be tokenized
        :return: [['word1', 'word2', ...], ['word1', 'word2', ...], ...], the tokenized text
        """
        # split text into sentences
        sentences = re.split(r'##第\d+条政策', text)
        # remove empty sentences
        sentences = [sentence.strip() for sentence in sentences if sentence.strip()]
        # print(sentences[:10])
        tokenized_sentences = []

        sentence_length = len(sentences)
        

        for i,sentence in enumerate(sentences):
            # 简单进度条
            print(f"\rProcessing {i+1}/{sentence_length} sentences,length: {len(sentence)}")

            # if sentence is too long, split it into smaller sentences
            if len(sentence) > max_sentence_length:
                sub_sentences = re.split(r'。', sentence)
                
                sub_sentences_list = []
                tmp_sentence = ''
                while sub_sentences:
                    sub_sentence = sub_sentences.pop(0)
                    if len(tmp_sentence) + len(sub_sentence) < max_sentence_length:
                        tmp_sentence += sub_sentence + '。'
                    else:
                        sub_sentences_list.append(tmp_sentence)
                        tmp_sentence = sub_sentence + '。'
                if tmp_sentence:
                    sub_sentences_list.append(tmp_sentence)

                for sub_sentence in sub_sentences_list:
                    # cut each sub_sentence
                    if len(sub_sentence) > max_sentence_length:
                        print(f"Warning: sentence length {len(sub_sentence)} exceeds max length {max_sentence_length},title: {sentence[:20]},sub_sentence: {sub_sentence[:20]}")
                        tokenized_sentences += self.cut(sub_sentence)
                    else:
                        tokenized_sentences += self.cut(sub_sentence)


            else:
                tokenized_sentences += self.cut(sentence)

            # sleep to avoid exceeding the request limit
            time.sleep(60 / max_request_per_minute)


        return tokenized_sentences




    def generate_word_dict(self, tokenized_text, threshold=10):
        """
        :param tokenized_text: [['word1', 'word2', ...], ['word1', 'word2', ...], ...], the tokenized text
        :param threshold: int, the frequency threshold for the word list
        :return: dict, the word list with frequency
        """
        word_dict = {}
        for sentence in tokenized_text:
            for word in sentence:
                if word not in self.stopwords:
                    if word in word_dict:
                        word_dict[word] += 1
                    else:
                        word_dict[word] = 1

        # sort word_dict by frequency
        word_dict = dict(sorted(word_dict.items(), key=lambda item: item[1], reverse=True))
                
        # remove words with frequency less than threshold
        word_dict = {word: freq for word, freq in word_dict.items() if freq >= threshold}
        return word_dict




if __name__ == '__main__':
    text = ''
    text_path = 'pretrain_data\policy\policy_strip.txt' 
    with open(text_path, 'r', encoding='utf-8') as f:
        text = f.read()
    print(f"read {len(text)} characters from {text_path}")
    
    hanlp_cutter = Hanlp_cutter()
    tokenized_text = hanlp_cutter.cut_long_text(text)
    print(f"tokenized text: {tokenized_text[:10]}")

    word_dict = hanlp_cutter.generate_word_dict(tokenized_text, threshold=10)
    print(f"word list length: {len(word_dict)}")
    # print(f"example word list: {list(word_dict.items())[:10]}")

    # save word list to file
    word_list_path = 'pretrain_data\policy\word_list.txt'
    with open(word_list_path, 'w', encoding='utf-8') as f:
        for word, freq in word_dict.items():
            f.write(f"{word},{freq}\n")
    