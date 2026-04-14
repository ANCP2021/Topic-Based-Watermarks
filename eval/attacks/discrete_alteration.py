"""
An attacker's motivation is to induce alterations in tokens, subsequently introducing misspelling and grammar errors. 
These manipulations can be executed through insertion or deletion of single or multiple characters, or even entire 
stings of characters within the specified output sequence, with the aim of diminishing the effectiveness of the watermark in detection.
"""
import random
from nltk import pos_tag
from utils import is_important_word

class DiscreteAlterations:
    def __init__(self):
        super().__init__()

    # Add whitespaces 
    def add_whitespace(self, text, n_edits, inference=False):
        words = list(text)
        
        # Assumption that there is a watermark, important words are the target
        if inference:
            words = text.split()
            tagged_words = pos_tag(words) 
            important_words = []
            for word, tag in tagged_words:
                if is_important_word(tag):
                    important_words.append(word)
            words = list(' '.join(important_words))

        for _ in range(n_edits):
            pos = random.randint(0, len(words))
            words.insert(pos, ' ')

        return ''.join(words)
    
    # Add alphabetical characters
    def add_char(self, text, n_edits, inference=False):
        words = text.split()

        # Assumption that there is a watermark, important words are the target
        if inference:
            tagged_words = pos_tag(words)
            important_words = []
            for word, tag in tagged_words:
                if is_important_word(tag):
                    important_words.append(word)

            if len(important_words) < n_edits:
                n_edits = len(important_words)
        else:
            important_words = words

        for _ in range(n_edits):
            pos = random.randint(0, len(important_words) - 1)
            word = important_words[pos]

            if len(word) > 1:
                char_pos = random.randint(0, len(word) - 1)
                misspelled_word = word[:char_pos] + random.choice('abcdefghijklmnopqrstuvwxyz') + word[char_pos + 1:]
                words[pos] = misspelled_word

        return ' '.join(words)

    
    # Combination function for whitespace and character insertion
    def combination_modify_text(self, text, 
                                whitespace_n_edits=0, white_space_inference=False,
                                add_char_n_edits=0, add_char_inference=False):
        if whitespace_n_edits > 0:
            text = self.add_whitespace(text, whitespace_n_edits, white_space_inference)
        if add_char_n_edits > 0:
            text = self.add_char(text, add_char_n_edits, add_char_inference)

        return text

# Use Case Example
if __name__ == '__main__':
    # Hypothetical watermarked text
    watermarked_text = (
        "Sports have been an integral part of human culture for centuries, serving as a means of entertainment, "
        "physical fitness, and social interaction. They are not merely games but vital activities that contribute "
        "to the holistic development of individuals and communities. The significance of sports transcends the boundaries "
        "of competition."
    )
    
    # Example of combination of inserted whitespaces and misspellings, 3 insertions each
    alteration = DiscreteAlterations()
    text = alteration.add_char(watermarked_text, n_edits=20, inference=True)
