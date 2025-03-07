import re

def replace_punctuation(text):
    '''
    Replace all punctuations in the text with space.
    To achieve that, we replace all characters that are not Chinese characters, English characters or numbers with space.
    '''
    return re.sub(u'[^\u4e00-\u9fa5a-zA-Z0-9]+', ' ', text)

def main():
    text = "我想对你说：“I love You”。"
    print(replace_punctuation(text))

if __name__ == '__main__':
    main()