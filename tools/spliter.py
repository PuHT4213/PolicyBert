import re

def chinese_chracater_detector(character):
    '''
    Check if a character is a Chinese character.
    '''
    return re.match(u'[\u4e00-\u9fa5]', character)


def spliter(text):
    '''
    For text like "我想对你说：“I love You”。"，return["我", "想", "对", "你", "说", "“" ,"I", "love", "You", "”" , "。"]
    Which means split the text into words and special symbols.
    '''
    text = text.strip()
    if len(text) == 0:
        return []
    result = []
    i = 0
    while i < len(text):
        if chinese_chracater_detector(text[i]):
            result.append(text[i])
            i += 1
        elif re.match(u'[a-zA-Z0-9]', text[i]):
            word = ""
            while i < len(text) and re.match(u'[a-zA-Z0-9]', text[i]):
                word += text[i]
                i += 1
            result.append(word)
        else:
            result.append(text[i])
            i += 1

    return result

def main():
    text = "我想对你说：“I love You”。"
    print(spliter(text))

if __name__ == '__main__':
    main()