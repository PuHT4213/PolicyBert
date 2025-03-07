import re


def is_chinese_char(character):
    '''
    Check if a character is a Chinese character.
    '''
    return re.match(u'[\u4e00-\u9fa5]', character)

def main():
    print(is_chinese_char("我"))
    print(is_chinese_char("a"))

if __name__ == '__main__':
    main()