from num2words import num2words

def convert_numericals_to_words(sentences):
    def number_to_words(number):
        # Convert number to words
        words = num2words(number)
        return words
    
    def process_sentence(sentence):
        words = []
        current_word = ""
        for char in sentence:
            if char.isdigit():
                current_word += char
            elif current_word:
                words.append(number_to_words(int(current_word)))
                current_word = ""
                words.append(char)
            else:
                words.append(char)
        if current_word:
            words.append(number_to_words(int(current_word)))
        return ''.join(words).replace('-', ' ').replace(':', ' ').replace('.', ' ').replace(',', ' ').replace('   ', ' ').replace('  ', ' ').replace(".", "dot")
    
    converted_sentences = [[process_sentence(word) for word in sentence] for sentence in sentences]
    return converted_sentences

# # Example usage:
# directory_path = r'C:\Users\Saad Khan\OneDrive - UNSW\University\5th Yr\T1\Thesis A\Data'

# # Fetch sentences from prepare_data function
# sentences, _, _, _ = prepare_data.prepare_data(r'C:\Users\Saad Khan\OneDrive - UNSW\University\5th Yr\T1\Thesis A\Data\planex_smacam_pantilt.json', r'C:\Users\Saad Khan\OneDrive - UNSW\University\5th Yr\T1\Thesis A\Data\jvc_kenwood_hdtv_ip_camera.json')

# # Convert numericals to words
# converted_sentences = convert_numericals_to_words(sentences)
# print(converted_sentences)
