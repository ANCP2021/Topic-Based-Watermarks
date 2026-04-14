"""
A paraphrasing attack is a category of a baseline substitution attack.
Execution of this attack may be manual by an individual or by rephrasing the output via an LLM.
"""
import torch
from transformers import (
    PegasusForConditionalGeneration, 
    PegasusTokenizer,
    AutoTokenizer,
    AutoModelForSeq2SeqLM
)
import nltk
import ssl
import math

try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context


class ParaphrasingAttack:
    """
    Paraphrasing class that can handle both:
      1) Pegasus paraphrasing
      2) Dipper paraphrasing
    """

    def __init__(self, model_type="pegasus"):
        self.model_type = model_type
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        if self.model_type == "pegasus":
            self.model_name = "tuner007/pegasus_paraphrase"
            self.tokenizer = PegasusTokenizer.from_pretrained(self.model_name)
            self.model = PegasusForConditionalGeneration.from_pretrained(
                self.model_name
            ).to(self.device)

        elif self.model_type == "dipper":
            self.model_name = "SamSJackson/paraphrase-dipper-no-ctx"
            self.tokenizer = AutoTokenizer.from_pretrained("google/t5-efficient-large-nl32")
            self.model = AutoModelForSeq2SeqLM.from_pretrained(self.model_name).to(self.device)

        else:
            raise ValueError("model_type must be 'pegasus' or 'dipper'.")
        
    def _chunk_sentence(self, sentence, max_tokens=60):
        """
        Break a single sentence into sub-chunks if it is longer than `max_tokens`.
        """
        tokens = self.tokenizer.tokenize(sentence)
        if len(tokens) <= max_tokens:
            return [sentence]

        chunks = []
        for i in range(0, len(tokens), max_tokens):
            chunk_tokens = tokens[i : i + max_tokens]
            chunk_text = self.tokenizer.convert_tokens_to_string(chunk_tokens)
            # Avoid empty chunks:
            if chunk_text.strip():
                chunks.append(chunk_text)
        return chunks


    def rephrase(self, input_text, num_return_sequences=1, num_beams=1, lexical=20, order=40):
        if self.model_type == "pegasus":
            batch = self.tokenizer(
                input_text,
                truncation=True,
                padding='longest',
                return_tensors="pt",
                max_length=60,
            ).to(self.device)

            outputs = self.model.generate(
                **batch,
                max_length=60,
                num_beams=num_beams,
                num_return_sequences=num_return_sequences,
                length_penalty=1.0,
                temperature=1.0,
            )

            paraphrases = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
            return paraphrases

        elif self.model_type == "dipper":
            prompt = f"lexical = {lexical}, order = {order} {input_text}"
            
            batch = self.tokenizer(
                prompt,
                return_tensors='pt',
                padding="longest",
                truncation=True,
                max_length=1000,
            ).to(self.device)

            outputs = self.model.generate(
                **batch,
                top_p=0.75,
                max_new_tokens=300,
                do_sample=True,
            )

            paraphrases = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
            return paraphrases

    def paraphrase_processor(self, watermarked_text_list, num_sequences=1, num_beams=1, lexical=20, order=40):
        paraphrased_list = []
        
        count = 0
        for watermarked_text in watermarked_text_list:
            print(f"text {count}")
            sentences = nltk.sent_tokenize(watermarked_text)

            paraphrased_sentences = []
            for sentence in sentences:
                if self.model_type == 'pegasus':
                    chunks = self._chunk_sentence(sentence, max_tokens=60)
                    paraphrased_chunks = []
                    for chunk in chunks:
                        paraphrases = self.rephrase(
                            chunk,
                            num_return_sequences=num_sequences,
                            num_beams=num_beams,
                            lexical=lexical,
                            order=order
                        )
                        paraphrased_chunks.append(paraphrases[0])

                    paraphrased_sentence = " ".join(paraphrased_chunks)

                else:
                    paraphrases = self.rephrase(
                        sentence,
                        num_return_sequences=num_sequences,
                        num_beams=num_beams,
                        lexical=lexical,
                        order=order
                    )
                    paraphrased_sentence = paraphrases[0]

                paraphrased_sentences.append(paraphrased_sentence)

            final_paraphrase = " ".join(paraphrased_sentences)
            print(final_paraphrase)
            count+=1
            paraphrased_list.append(final_paraphrase)


        return paraphrased_list


if __name__ == '__main__':
    watermarked_text_list = [' new report has revealed new data.\n“Clearly, processed meats are more powerful than other forms of cancer treatment. However, they also act as passive barriers to cancer prevention – it’s not always a combination of factors other than genetics or age or alcohol or other factors –” explains Dr. Carsten Krause, head of cancer at the IARC. “The new research reveals that processed meats are more likely to be cancerous if they come from or are grown on farms or plantations – or else from human beings who have not travelled outside France or elsewhere in France for long periods of time.”\nThe new report is available now through the IARC’s website – which, if you want it updated by January 14th, will be updated via RSS feed. It also highlights new data from more than 5,000 trials involving over 110,000 people who participated in the new study. That’s more than double the number of trials conducted by cancer']

    print("Original Text List:")
    print(watermarked_text_list, "\n")

    print("Pegasus Paraphrase")
    pegasus_paraphraser = ParaphrasingAttack(model_type="pegasus")
    paraphrased_list_pegasus = pegasus_paraphraser.paraphrase_processor(
        watermarked_text_list=watermarked_text_list,
        num_sequences=1, 
        num_beams=3
    )
    print(paraphrased_list_pegasus, "\n")

    print("Dipper Paraphrase")
    dipper_paraphraser = ParaphrasingAttack(model_type="dipper")
    paraphrased_list_dipper = dipper_paraphraser.paraphrase_processor(
        watermarked_text_list=watermarked_text_list,
        lexical=20,
        order=40
    )
    print(paraphrased_list_dipper, "\n")
