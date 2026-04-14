"""
Baseline attack consists of the insertion, substitution, and deletion of text for a given output
sequence. The attacker selects a single or a combination of techniques with the objective to diminish 
detection accuracy.
"""
import random
import string
from nltk.corpus import reuters
from nltk.probability import FreqDist
from nltk.corpus import wordnet
from nltk import pos_tag
# from utils import is_important_word


def is_important_word(pos_tag):
    return pos_tag.startswith(('N', 'V', 'J', 'R'))

class BaselineAttack:
    def __init__(self):
        super().__init__()

    # Synonym helper function for substitution attack
    def get_synonym(self, word):
        synonyms = set()
        for syn in wordnet.synsets(word):
            for lemma in syn.lemmas():
                synonyms.add(lemma.name())
                
        synonyms.discard(word)
        return list(synonyms)

    # Modify text randomly through insertion, deletion, and substitution
    def modify_text(self, text, n_edits, edit_type='insert'):
        words = text.split()
        common_words = FreqDist(reuters.words()).most_common(1000)
        substituted_words = set()
        filtered_words = [w for w, _ in common_words if w not in string.punctuation]
        
        substituted_words = set()
        for _ in range(n_edits):
            if edit_type == 'insert':
                pos = random.randint(0, len(words))
                words.insert(pos, random.choice(filtered_words))
            elif edit_type == 'delete' and len(words) > 1:
                pos = random.randint(0, len(words) - 1)
                words.pop(pos)
            elif edit_type == 'substitute':
                pos = random.randint(0, len(words) - 1)
                if words[pos] not in substituted_words:
                    words[pos] = random.choice(filtered_words)
                    substituted_words.add(words[pos])
        
        return ' '.join(words)

    # Modify text under the assumption that there is a watermark, important words are the target
    # choosing more important words (excluding 'the', 'and', etc.) randomly 
    def inference_modify_text(self, text, n_edits, edit_type='insert'):
        words = text.split()
        tagged_words = pos_tag(words) 
        important_words = [w for w, tag in tagged_words if is_important_word(tag)]
        substituted_words = set()

        for _ in range(n_edits):
            if edit_type == 'insert':
                # pick a random position, insert from the pool of important words
                pos = random.randint(0, len(words))
                if important_words:
                    word_to_insert = random.choice(important_words)
                else:
                    word_to_insert = "insertedword"  # fallback if none are "important"
                words.insert(pos, word_to_insert)

            elif edit_type == 'delete' and important_words:
                # randomly delete from the list of important words, if present
                word_to_delete = random.choice(important_words)
                if word_to_delete in words:
                    pos = words.index(word_to_delete)
                    words.pop(pos)
                # else do nothing if the random choice isn't in the text

            elif edit_type == 'substitute':
                pos = random.randint(0, len(words) - 1)
                word = words[pos]
                synonyms = self.get_synonym(word)
                if synonyms and len(synonyms) > 0 and word not in substituted_words:
                    new_word = random.choice(synonyms)
                    words[pos] = new_word
                    substituted_words.add(new_word)

        return ' '.join(words)

    # Combination function for insertion, deletion, and substitution
    def combination_modify_text(self, text, 
                                insertion_n_edits=0, insertion_is_inferenced=False, 
                                deletion_n_edits=0, deletion_is_inferenced=False, 
                                substitution_n_edits=0, substitution_is_inferenced=False):
        if insertion_n_edits > 0:
            if insertion_is_inferenced:
                text = self.inference_modify_text(text, insertion_n_edits, edit_type='insert')
            else:
                text = self.modify_text(text, insertion_n_edits, edit_type='insert')
        
        if deletion_n_edits > 0:
            if deletion_is_inferenced:
                text = self.inference_modify_text(text, deletion_n_edits, edit_type='delete')
            else:
                text = self.modify_text(text, deletion_n_edits, edit_type='delete')
        
        if substitution_n_edits > 0:
            if substitution_is_inferenced:
                text = self.inference_modify_text(text, substitution_n_edits, edit_type='substitute')
            else:
                text = self.modify_text(text, substitution_n_edits, edit_type='substitute')

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

    # Example of 3 random insertions where there is no assumption of a watermark
    baseline = BaselineAttack()
    text = baseline.inference_modify_text(watermarked_text, 3, 'insert')
    