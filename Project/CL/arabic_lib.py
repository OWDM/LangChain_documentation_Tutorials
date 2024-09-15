from translate import Translator

Translator = Translator(to_lang="Arabic")

transalation = Translator.translate("How are you?")
print(transalation)