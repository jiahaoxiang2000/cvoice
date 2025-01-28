from transformers import pipeline

class TextImprover:
    @staticmethod
    def improve_text(text):
        corrector = pipeline("text2text-generation", model="facebook/bart-large-cnn")
        improved_text = corrector(text, max_length=130)[0]["generated_text"]
        return improved_text
