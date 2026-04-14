
def is_important_word(pos_tag):
    return pos_tag.startswith(('N', 'V', 'J', 'R'))
