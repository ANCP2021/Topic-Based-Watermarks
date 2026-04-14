"""
A tokenization attack alters text that causes significant changes in how the text is tokenized
into subwords. This type of attack is can drastically alter the sequence of tokens without changing 
the overall meaning of the text. By introducing minor changes—such as inserting special
characters (e.g., underscores '_', asterisks '*') or replacing spaces, newline characters, or punctuation—the attack
splits what would normally be a single, valid token into multiple sub-tokens.
"""
import random
from nltk import pos_tag
from utils import is_important_word

class TokenizationAttack:
    def __init__(self):
        super().__init__()

    # Function to modify text to change tokenization
    def tokenization_attack(self, text, n_edits, inference=False):
        words = text.split()
        tagged_words = pos_tag(words)
        important_words = []
        for word, tag in tagged_words:
            if is_important_word(tag):
                important_words.append(word)

        for _ in range(n_edits):
            # Assumption that there is a watermark, important words are the target
            if inference and important_words:
                word = random.choice(important_words)
                pos = words.index(word)
            else:
                pos = random.randint(0, len(words) - 1)
                word = words[pos]

            if '\n' in word:
                words[pos] = word.replace('\n', ' ')
            elif '.' in word:
                words[pos] = word.replace('.', ' ')
            elif ' ' in word:
                words[pos] = word.replace(' ', '_')
            else:
                if len(word) > 1:
                    insert_pos = random.randint(1, len(word) - 1)
                    modified_word = word[:insert_pos] + '_' + word[insert_pos:]
                    words[pos] = modified_word

            if inference:
                important_words = [w for w in important_words if w != word]

        return ' '.join(words) 

# Use Case Example
if __name__ == '__main__':
    # Hypothetical watermarked text
    watermarked_text = (
        "Sports have been an integral part of human culture for centuries, serving as a means of entertainment, "
        "physical fitness, and social interaction. They are not merely games but vital activities that contribute "
        "to the holistic development of individuals and communities. The significance of sports transcends the boundaries "
        "of competition."
    )

    tokenization = TokenizationAttack()
    text = tokenization.tokenization_attack(watermarked_text, n_edits=3, inference=True)
