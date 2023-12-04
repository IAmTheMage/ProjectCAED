import spacy
import nltk
import sys
from nltk.corpus import floresta
from nltk import CFG
from nltk.stem import WordNetLemmatizer
from nltk.stem import RSLPStemmer
from nltk.tag import pos_tag
import json
import os
print("Baixando os binários do spacy")
nlp = spacy.load("pt_core_news_lg")

import nltk
from nltk import CFG
print("Baixando os verbos")
nltk.download('floresta')
nltk.download('wordnet')
nltk.download('rslp')
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')

lemmatizer = WordNetLemmatizer()
stemmer = RSLPStemmer()

# Encontre o lema


# Conjugação de verbos

def save_to_file(file_name, data):
    with open(file_name, 'w') as file:
        json.dump(data, file)

def load_from_file(file_name):
    with open(file_name, 'r') as file:
        data = json.load(file)
        return data


'''
Código para inversão de frases, _mount_1 equivale a primeira Heuristica, _mount_2 a terceira, _mount_3 a segunda e _mount_4 a quarta
'''
class Dependency:
    def __init__(self, text): 
        self.doc = nlp(text)
        self.has_subject = self._has_subject()
        self.auxiliary_tense = ""
        self.auxiliary_aspect = ""
        self.has_auxiliary_verb = self._has_auxiliary_verb()
        self.cache = {}
        self.passive_voice = self._identify_passive_voice()
        self.verbs_dict = []
        #self.print_dependencies()
        self.generate_cache()
        print("Lendo os verbos do dicionário")
        self.generate_verbs_dict()
        #print(self.verbs_dict)

    def generate_cache(self):
        for token in self.doc:
            self.cache[token.text] = token

    def print_dependencies(self):
        for token in self.doc:
            print(token.text, token.lemma_, token.pos_, token.tag_, token.dep_,
            token.shape_, token.is_alpha, token.is_stop, token.morph.get("Tense"))

    def _has_subject(self) -> bool:
        for token in self.doc:
            if "NSUBJ" in token.dep_.upper():
                return True
        return False

    def _has_auxiliary_verb(self):
        for token in self.doc:
            if "aux" in token.pos_.lower():
                self.auxiliary_tense = token.morph.get("Tense")[0]
                self.auxiliary_aspect = token.morph.get("VerbForm")[0]
                return True
        return False
    
    def _identify_passive_voice(self):
        if not self.has_subject:
            return False
        else:
            for token in self.doc:
                if "agent" in token.dep_.lower():
                    return True
        return False
    
    def mount(self):
        #print(f'Data: {self.has_subject}, {self.phrase_has(self.doc, "dep_", "appos")}, {self.phrase_has(self.doc, "dep_", "flat:name")}')
        ret = ""
        if self.has_subject and self.phrase_has(self.doc, "dep_", "appos"):
            ret = self._mount_4()
            print()
        elif "det" in self.doc[0].pos_.lower():
            ret = self._mount_2()
        
        elif "adp" in self.doc[0].pos_.lower() or "adv" in self.doc[0].pos_.lower():
            ret = self._mount_1()
        
        elif self.has_subject == False:
            if "verb" in self.doc[0].pos_.lower() and self.has_subject == False:
                ret = self._mount_1_adapted()
            else:
                ret = self._mount_3()
        
        else:
            print("Não foi encontrado uma heuristica capaz de resolver esse problema")
            self.print_dependencies()
            pass

        return self.gramatical_corrector(ret)

    
    def find_lemma(self, token,verbs, doc, aspect = None):
        _person = doc.morph.get("Person")
        _number = doc.morph.get("Number")
        # print(f"Text: {token}")
        # if _aspect is not None:
        #     print(f"Aspect: {_aspect}")
        # if _tense is not None:
        #     print(f"Tense: {_tense}")
        # if _person is not None:
        #     print(f"Person: {_person}")
        # if _number is not None:
        #     print(f"Number: {_number}")
        # if _voice is not None:
        #     print(f"Voice: {_voice}")
        # if _mood is not None:
        #     print(f"Mood: {_mood}")
        # if _aux is not None:
        #     print(f"Aux: {_aux}")
        # if _reflex is not None:
        #     print(f"Reflex: {_reflex}")
        tok_data = nlp(token)[0]
        lemma = tok_data.lemma_
        find = {}
        for element in self.verbs_dict:
            if element['key'] == lemma:
                find = element
                break

        verbs = None
        current = 0
        if not 'data' in find:
            return None
        
        for element in find['data']:
            s_value = 0
            if len(element['tense']) > 0 and element['tense'][0] == self.auxiliary_tense:
                s_value += 10
            if _number == element['number']:
                s_value += 5
            if _person == element['person']:
                s_value += 4
            
            if s_value > current:
                current = s_value
                verbs = element
                
        return verbs
        
        
    def get_adp_gender(self, token):
        if len(token) == 0:
            return ""
        return token[len(token) - 1]

    #Função para gerar o dicionário(lemms.json)
    def generate_verbs_dict(self):

        file_name = 'lemms.json'

        if os.path.exists(file_name):
            self.verbs_dict = load_from_file("lemms.json")
        else:
            tagged_words = floresta.tagged_words()
            tagged_verbs = [(word.lower(), pos) for word, pos in tagged_words if 'v-' in pos]

            verb_forms = []
            index = 0
            for word, tag in tagged_verbs:
                print("Process {} of {}, tag form: {}".format(index, len(tagged_verbs), tag))
                index += 1
                is_finded = False
                word_data = nlp(word)[0]
                lemm = word_data.lemma_
                aspect = word_data.morph.get("VerbForm")
                tense = word_data.morph.get("Tense")
                person = word_data.morph.get("Person")
                number = word_data.morph.get("Number")
                voice = word_data.morph.get("Voice")
                mood = word_data.morph.get("Mood")
                aux = word_data.morph.get("Aux")
                reflex = word_data.morph.get("Reflex")

                # Verifica se o verbo já existe no dicionário de verbos
                is_found = False
                for i, verb_info in enumerate(self.verbs_dict):
                    if verb_info['key'] == lemm:
                        self.verbs_dict[i]['data'].append({
                            'key': tag,
                            'word': word,
                            'aspect': aspect,
                            'tense': tense,
                            'person': person,
                            'number': number,
                            'voice': voice,
                            'mood': mood,
                            'aux': aux,
                            'reflex': reflex
                        })
                        is_found = True
                        break

                # Se o verbo não existir no dicionário, adiciona-o
                if not is_found:
                    self.verbs_dict.append({'key': lemm, 'data': [{
                        'key': tag,
                        'word': word,
                        'aspect': aspect,
                        'tense': tense,
                        'person': person,
                        'number': number,
                        'voice': voice,
                        'mood': mood,
                        'aux': aux,
                        'reflex': reflex
                    }]})
            
            save_to_file("lemms.json", self.verbs_dict)

    def _determine_positions(self, tag, property_1):
        position = []
        index = 0
        cursor = 0
        for item in self.doc:
            if tag in getattr(item, property_1).lower():
                position.append(index)
            index += 1
        
        return position
    
    def determine_positions(self, doc, tag, property_1):
        position = []
        index = 0
        cursor = 0
        for item in doc:
            if tag in getattr(item, property_1).lower():
                position.append(index)
            index += 1
        
        return position

    def reversion_after(self, cut_after) -> str:
        first_phase = []
        second_phase = []
        index = 0
        for item in self.doc:
            if index < cut_after:
                second_phase.append(item.text)
            else:
                first_phase.append(item.text)
            
            index += 1
        
        res = ""
        for item in first_phase:
            res += item + " "
        
        for item in second_phase:
            res += item + " "
        return res
    
    def _mount_1(self):
        det_positions = self._determine_positions("det", "pos_")
        nsubj_position = self._determine_positions("nsubj", "dep_")[0]
        cut_after = nsubj_position
        for det_position in det_positions:
            if det_position == nsubj_position - 1:
                cut_after = det_position
        
        reversed_txt = self.reversion_after(cut_after)
        reversed_doc = nlp(reversed_txt)
        reversed_sanitized = self.removeIf(reversed_doc, "pos_", "punc")
        return self.getText(reversed_sanitized)

    def _mount_1_adapted(self):
        aux_position = self._determine_positions("aux", "pos_")[0]
        reversed_txt = self.reversion_after(aux_position)
        reversed_doc = nlp(reversed_txt)
        reversed_sanitized = self.removeIf(reversed_doc, "pos_", "punc")
        return self.getText(reversed_sanitized)


    def generate_alternatives(self, original = [], changes = [], changes_position = []):
        new_sentences = []
        for item, index in changes:
            print(item)
            print(index)

    def swap(self, doc, swap_tag_1, swap_tag_2, property_1, property_2):
        complete_sentece = ""
        swap1 = 0
        swap2 = 0
        index = 0
        for item in doc:
            if swap_tag_1 in getattr(item, property_1).lower():
                swap1 = index
            elif swap_tag_2 in getattr(item, property_2).lower():
                swap2 = index
            index += 1
        
        index = 0
        for item in doc:
            if index == swap1:
                text = doc[swap2].text
                if swap2 != 0:
                    text = text.lower()
                complete_sentece += text + " "
            elif index == swap2:
                text = doc[swap1].text
                if swap1 != 0:
                    text = text.lower()
                complete_sentece += text + " "
            else:
                complete_sentece += item.text + " "
            
            index += 1
        
        return complete_sentece
    
    def removeIf(self, doc, property, value_in):
        index = 0
        neo_doc = []
        for element in doc:
            if value_in not in getattr(element, property).lower():
                neo_doc.append(element)
            
            index += 1
        
        return neo_doc
    
    def getText(self, doc):
        txt = ""
        for item in doc:
            txt += item.text + " "
        
        return txt

    def phrase_has(self, doc, attribute, value) -> bool:
        for element in doc:
            if value in getattr(element, attribute).lower():
                return True
        return False
    
    def swap_by_tag(self, base_doc, tag_1, value_1, tag_2, value_2):
        position_1 = self._determine_positions(tag_1, value_1)[0]
        position_2 = self._determine_positions(tag_2, value_2)[0]

        neo_doc = list(base_doc)
        neo_doc[position_1], neo_doc[position_2] = neo_doc[position_2], neo_doc[position_1]

        return neo_doc
    
    def change_position(self, doc: list, current_position, new_position):
        neo_doc = list(doc)
        popped_item = neo_doc.pop(current_position)
        neo_doc.insert(new_position, popped_item)
        return neo_doc
        

    
    def _mount_2(self):
        #print("Mounting 2 {}".format(self.auxiliary_tense))
        tmp = ""
        verbs = ""
        current_verb = ""
        det = ""
        adp = ""
        index = 0
        neo_doc = self.removeIf(self.doc, "pos_", "aux")
        swaped = self.swap(neo_doc, "agent", "nsubj", "dep_", "dep_")
        swapped_description = nlp(swaped)
        neo_swap = self.swap(swapped_description, "det", "adp", "pos_", "tag_")
        swapped_description = nlp(swaped)
        has_not_lemma = True
        if self.passive_voice:
            for token in self.doc:
                if "VERB" in token.tag_.upper():
                    verbs = self.find_lemma(token.text, token.morph.get('Tense'), token, "Fin")
                    #print(f"Verbs: {verbs}")
                    if verbs != None:
                        has_not_lemma = False
                    current_verb = token.text
                    #print(verbs)
                elif "adp" in token.tag_.lower():
                    adp = token.text
                
                elif "det" in token.pos_.lower():
                    det = token.text
                if index == 2:
                    break

        #print(f"Lemma size: {verbs['word']}")
        neo_swap = neo_swap.replace(adp, self.get_adp_gender(adp).upper())   
        neo_swap = neo_swap.replace(det, det.lower())    
        if has_not_lemma == False:
            neo_swap = neo_swap.replace(current_verb, verbs['word'])  
        return neo_swap

    def _mount_3(self):
        root_position = self._determine_positions("root", "dep_")[0]
        cut_after = root_position + 1
        
        reversed_txt = self.reversion_after(cut_after)
        reversed_doc = nlp(reversed_txt)
        reversed_sanitized = self.removeIf(reversed_doc, "pos_", "punc")
        return self.getText(reversed_sanitized)

    def gramatical_corrector(self, sentence: str):
        neo_sentence = ""
        sentence_tokens = nlp(sentence)
        index = 0
        for token in sentence_tokens:
            if token.dep_ == "flat:name" or index == 0:
                neo_sentence += token.text.capitalize()
            else:
                neo_sentence += token.text.lower()
            
            if(token.pos_ != "PUNCT"):
                neo_sentence += " "
            index += 1
        
        return neo_sentence


    def _mount_4(self):
        #if self._determine_positions("appos", "dep_")[0] == self._determine_positions("flat:name", "dep_")[0] - 1:
        #print("Mounting 4")
        #self.print_dependencies()
        first_swap = self.swap_by_tag(self.doc, "det", "pos_", "appos", "dep_")
        second_swap = first_swap
        _position_det = self.determine_positions(second_swap, "det", "pos_")[0]
        if self.phrase_has(self.doc, "dep_", "flat:name"):
            second_swap = self.swap_by_tag(first_swap, "nsubj", "dep_", "flat:name", "dep_")
        else:
            second_swap = self.change_position(second_swap, self.determine_positions(second_swap, "nsubj", "dep_")[0], _position_det)
        
        position_nsubj = self.determine_positions(second_swap, "nsubj", "dep_")[0]
        position_det = self.determine_positions(second_swap, "det", "pos_")[0]
        first_str = ""
        index = 0
        for elem in second_swap:
            first_str += elem.text
            if(index == position_det - 1 or index == position_nsubj):
                first_str += ","
            else:
                first_str += " "
            index += 1
        return first_str
