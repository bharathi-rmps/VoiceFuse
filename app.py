from flask import *
import pickle    
import io
from gtts import gTTS
#!pip install torchtext==0.6.0 --quiet
import torch
import torch.nn as nn
import torch.optim as optim
from torchtext.datasets import Multi30k
from torchtext.data import Field, BucketIterator
import numpy as np
import pandas as pd
import spacy
import random
from torchtext.data import TabularDataset
import io
import urllib.request
from pydub import AudioSegment
import speech_recognition as sr
import tempfile
import os
import firebase_admin
from firebase_admin import credentials, firestore, storage
import tempfile
from nltk.tokenize.treebank import TreebankWordDetokenizer

app = Flask(__name__)
creds = {
  "type": "service_account",
  "project_id": "voicefuse-e6a23",
  "private_key_id": "af92728ae38d3bb13e48353e0e36b9febb0ad92c",
  "private_key": "-----BEGIN PRIVATE KEY-----\nMIIEvwIBADANBgkqhkiG9w0BAQEFAASCBKkwggSlAgEAAoIBAQDcbXsTvHABwu6L\nJ1AKymZwNyfEijcF8Qwh+O+W4crPOCz1PCADkI774T1qnwjLD085vUEINp/vQJBB\nv7t9yRHucJ9PnYZhxli/0IBqWjCAqQqug2oIhpcOC4ARiynF5jhRjTyfAf4J8/5H\nHUDGcbchrnvd3EHKmlx0uMVrtys+TDK2KOrefj3iTeYyndLRXGjMWLggi/ZdycZi\nuIaRgkZgMA60ZmBwr4/c2EXLx3w7s0D05ZP7nptvqLJblUGF19HPWsHSmtju6fki\nHe5J13UHikn7t3qob5TVd0H+mSVsGd405LwOe8AgZn6HWY3PUfmHEDJVwZvmBApr\ngeCxaHNBAgMBAAECggEAajFgSvjMaTzVGTr+R8SMp2EbCoefzH/9XVyFaIYs47nn\nhtnJfo4lJFm18ejeHp2oiGORsfhHAsdy+An7tLSqq9mcFooIVJBu7zHuu83pcgJ+\nP+bVrXfRzBVkRI9rt4ia90b4vo4CKh9fEDmanD/qfxKFYDQdihLlmeHuUl9zONYt\n16Oa238xWVTJgB1k1wyLwQJkrb1cSlE0qz8m9uWrnrmScpazjDcW/yXuzyZGjbOw\nO/ov7S3y/7HPvR41zw83YSp6g6ExlsHTm5B56eT7wKthWRfXX2J0kxaAh2tJilMq\nLRkDL8EnEy1KHWXRfvrGqAu15XYOZQV3FAX+peF25QKBgQD0stl2INvQO7x44O5X\nwxFTYEmDIFvdii5U3Dr1azVLuWYdLhiuQC14haHdKwXyckWGSeE/3tdhVVHa0hac\n3RP+qxj82uXKWnNfZofGDYFTQq3zilfYSL32oX09XqK+dbvCgY/OS7cZEC2JG9H/\nBXu43s5tf4bUDTQsrTakIdg80wKBgQDmm6r2CH89cXVTTPm4PJaEC7mJ09wtS1Y9\neaYIhJ5AXuRVj+09aGiz5EmeC0fgj5dGrMf0qvsRvkGJx7OASKA56wGQLrD3+YU7\nnBPSev1Cx78Cm8DpoHSSXFSbHRLMI7osNOhCq/UC4LzqyvBuFORrqA0dYH2spsLg\nROW9qRQzGwKBgQDskvFdnN0H0IkiEM09+jEI++F2rdVDNbIfhyBVX8YSJPfNpGBm\nL1QG3qOkUVEZmlMPRuRIPOjciIFv3ofQNol7QO4SoItjfNloVZdU6n+rAJ9vAsR1\nLbbC+FQ9/f23x9m0blCbMWafC54KneQD+8gm7vqCsLWo0+8qdniKbNJD0wKBgQCn\nY+l1y3coz3l9bMt7KyeKU8Rqwkj4681+tBWL60+/o+GUJfPr9iTCJ3w0ZzXWUARb\nvcEq3Q1/tJ13+GhYPt9nCynIUcwNQ6atPT66MqIxXjJNH2epbdoP/0s+iJ0DZw+V\nVRYehxlC7ITU3VgmX63qY0KZx1eSAj5Ecl5dDCTobwKBgQCdF0dW+pkFO5QmMZIv\nPPxi21O1xgcJbKNXlAkMSDrkthkaoU1ICu6ZZkziQGRT4YsEJxxtAMMzJ6B2mpuQ\nrGfR8J0LlxuDhhLra6GW65CzjOtpvcz4yD3acibFqDnbqpOA6XMylcKYCtdf+h5L\nWgswjePYbZ3o28iFkXyAx9Y9tQ==\n-----END PRIVATE KEY-----\n",
  "client_email": "firebase-adminsdk-ud68w@voicefuse-e6a23.iam.gserviceaccount.com",
  "client_id": "104191210572344804283",
  "auth_uri": "https://accounts.google.com/o/oauth2/auth",
  "token_uri": "https://oauth2.googleapis.com/token",
  "auth_provider_x509_cert_url": "https://www.googleapis.com/oauth2/v1/certs",
  "client_x509_cert_url": "https://www.googleapis.com/robot/v1/metadata/x509/firebase-adminsdk-ud68w%40voicefuse-e6a23.iam.gserviceaccount.com"
}
cd = credentials.Certificate(creds)
firebase_admin.initialize_app(cd, options={
    'storageBucket': 'voicefuse-e6a23.appspot.com'
})
datab = firestore.client()

# src_lang = input("Enter the source language: ")
# trg_lang = input("Enter the target language: ")
src_lang = "Tamil"
trg_lang = "English"
uid = 'JCnVRvZB9EVq6tdjphQAm6xuJj33'

if src_lang == "English" and trg_lang == "Tamil":
    in_language = 'en-in'
    sp_src_path = "en"
    sp_trg_path = "ta"
    out_language = "ta"
    folder = "5Lang/English to Tamil"
elif src_lang == "Tamil" and trg_lang == "English":
    in_language = 'ta-in'
    sp_src_path = "ta"
    sp_trg_path = "en"
    out_language = "en"
    folder = "5Lang/Tamil to English"
elif src_lang == "Spanish" and trg_lang == "Turkish":
    in_language = 'es'
    sp_src_path = "es"
    sp_trg_path = "tr"
    out_language = "tr"
    folder = "5Lang/Spanish to Turkish"
elif src_lang == "Hindi" and trg_lang == "English":
    in_language = 'hi-in'
    sp_src_path = "hi"
    sp_trg_path = "en"
    out_language = "en"
    folder = "5Lang/Hindi to English"
elif src_lang == "English" and trg_lang == "Hindi":
    in_language = 'en-in'
    sp_src_path = "en"
    sp_trg_path = "hi"
    out_language = "hi"
    folder = "5Lang/English to Hindi"
elif src_lang == "English" and trg_lang == "Tamil":
    in_language = 'en-in'
    sp_src_path = "en"
    sp_trg_path = "ta"
    out_language = "ta"
    folder = "5Lang/English to Tamil"
else:
    print("Invalid")
    exit()
print(sp_src_path+"\n"+sp_trg_path)
data_path = folder+"/data.csv"
model_path = folder+"/model.pkl"
ts_path = folder+"/ts1.pkl"


# df_eng_word.to_csv('/content/Words.csv')
class EncoderLSTM(nn.Module):
  def __init__(self, input_size, embedding_size, hidden_size, num_layers, p):
    super(EncoderLSTM, self).__init__()

    # Size of the one hot vectors that will be the input to the encoder
    #self.input_size = input_size

    # Output size of the word embedding NN
    #self.embedding_size = embedding_size

    # Dimension of the NN's inside the lstm cell/ (hs,cs)'s dimension.
    self.hidden_size = hidden_size

    # Number of layers in the lstm
    self.num_layers = num_layers

    # Regularization parameter
    self.dropout = nn.Dropout(p)
    self.tag = True

    # Shape --------------------> (5376, 300) [input size, embedding dims]
    self.embedding = nn.Embedding(input_size, embedding_size)
    
    # Shape -----------> (300, 2, 1024) [embedding dims, hidden size, num layers]
    self.LSTM = nn.LSTM(embedding_size, hidden_size, num_layers, dropout = p)

  # Shape of x (26, 32) [Sequence_length, batch_size]
  def forward(self, x):

    # Shape -----------> (26, 32, 300) [Sequence_length , batch_size , embedding dims]
    embedding = self.dropout(self.embedding(x))
    
    # Shape --> outputs (26, 32, 1024) [Sequence_length , batch_size , hidden_size]
    # Shape --> (hs, cs) (2, 32, 1024) , (2, 32, 1024) [num_layers, batch_size size, hidden_size]
    outputs, (hidden_state, cell_state) = self.LSTM(embedding)

    return hidden_state, cell_state


class DecoderLSTM(nn.Module):
  def __init__(self, input_size, embedding_size, hidden_size, num_layers, p, output_size):
    super(DecoderLSTM, self).__init__()

    # Size of the one hot vectors that will be the input to the encoder
    #self.input_size = input_size

    # Output size of the word embedding NN
    #self.embedding_size = embedding_size

    # Dimension of the NN's inside the lstm cell/ (hs,cs)'s dimension.
    self.hidden_size = hidden_size

    # Number of layers in the lstm
    self.num_layers = num_layers

    # Size of the one hot vectors that will be the output to the encoder (English Vocab Size)
    self.output_size = output_size

    # Regularization parameter
    self.dropout = nn.Dropout(p)

    # Shape --------------------> (5376, 300) [input size, embedding dims]
    self.embedding = nn.Embedding(input_size, embedding_size)

    # Shape -----------> (300, 2, 1024) [embedding dims, hidden size, num layers]
    self.LSTM = nn.LSTM(embedding_size, hidden_size, num_layers, dropout = p)

    # Shape -----------> (1024, 4556) [embedding dims, hidden size, num layers]
    self.fc = nn.Linear(hidden_size, output_size)

  # Shape of x (32) [batch_size]
  def forward(self, x, hidden_state, cell_state):

    # Shape of x (1, 32) [1, batch_size]
    x = x.unsqueeze(0)

    # Shape -----------> (1, 32, 300) [1, batch_size, embedding dims]
    embedding = self.dropout(self.embedding(x))

    # Shape --> outputs (1, 32, 1024) [1, batch_size , hidden_size]
    # Shape --> (hs, cs) (2, 32, 1024) , (2, 32, 1024) [num_layers, batch_size size, hidden_size] (passing encoder's hs, cs - context vectors)
    outputs, (hidden_state, cell_state) = self.LSTM(embedding, (hidden_state, cell_state))

    # Shape --> predictions (1, 32, 4556) [ 1, batch_size , output_size]
    predictions = self.fc(outputs)

    # Shape --> predictions (32, 4556) [batch_size , output_size]
    predictions = predictions.squeeze(0)

    return predictions, hidden_state, cell_state


class Seq2Seq(nn.Module):
  def __init__(self, Encoder_LSTM, Decoder_LSTM):
    super(Seq2Seq, self).__init__()
    self.Encoder_LSTM = Encoder_LSTM
    self.Decoder_LSTM = Decoder_LSTM

  def forward(self, source, target, tfr=0.5):
    # Shape - Source : (10, 32) [(Sentence length German + some padding), Number of Sentences]
    batch_size = source.shape[1]

    # Shape - Source : (14, 32) [(Sentence length English + some padding), Number of Sentences]
    target_len = target.shape[0]
    target_vocab_size = len(english.vocab)
    
    # Shape --> outputs (14, 32, 5766) 
    outputs = torch.zeros(target_len, batch_size, target_vocab_size).to(device)

    # Shape --> (hs, cs) (2, 32, 1024) ,(2, 32, 1024) [num_layers, batch_size size, hidden_size] (contains encoder's hs, cs - context vectors)
    hidden_state, cell_state = self.Encoder_LSTM(source)

    # Shape of x (32 elements)
    x = target[0] # Trigger token <SOS>

    for i in range(1, target_len):
      # Shape --> output (32, 5766) 
      output, hidden_state, cell_state = self.Decoder_LSTM(x, hidden_state, cell_state)
      outputs[i] = output
      best_guess = output.argmax(1) # 0th dimension is batch size, 1st dimension is word embedding
      x = target[i] if random.random() < tfr else best_guess # Either pass the next word correctly from the dataset or use the earlier predicted word

    # Shape --> outputs (14, 32, 5766) 
    return outputs
# Hyperparameters


def translate_sentence(model, sentence, german, english, device, max_length=50):
    spacy_ger = spacy.blank(sp_src_path)

    if type(sentence) == str:
        tokens = [token.text.lower() for token in spacy_ger(sentence)]
    else:
        tokens = [token.lower() for token in sentence]
    tokens.insert(0, german.init_token)
    tokens.append(german.eos_token)
    text_to_indices = [german.vocab.stoi[token] for token in tokens]
    sentence_tensor = torch.LongTensor(text_to_indices).unsqueeze(1).to(device)

    # Build encoder hidden, cell state
    with torch.no_grad():
        hidden, cell = model.Encoder_LSTM(sentence_tensor)

    outputs = [english.vocab.stoi["<sos>"]]

    for _ in range(max_length):
        previous_word = torch.LongTensor([outputs[-1]]).to(device)

        with torch.no_grad():
            output, hidden, cell = model.Decoder_LSTM(previous_word, hidden, cell)
            best_guess = output.argmax(1).item()

        outputs.append(best_guess)

        # Model predicts it's the end of the sentence
        if output.argmax(1).item() == english.vocab.stoi["<eos>"]:
            break

    translated_sentence = [english.vocab.itos[idx] for idx in outputs]
    return translated_sentence[1:]

class CPU_Unpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module == 'torch.storage' and name == '_load_from_bytes':
            return lambda b: torch.load(io.BytesIO(b), map_location='cpu')
        else:
            return super().find_class(module, name)
spacy_german = spacy.blank(sp_src_path)
spacy_english = spacy.blank(sp_trg_path) 
def tokenize_german(text):
    return [token.text for token in spacy_german.tokenizer(text)]
def tokenize_english(text):
    return [token.text for token in spacy_english.tokenizer(text)]
@app.route('/')
def predict():
    # Take language input from the user
  # replace with the UID you want to search for
    users_ref = datab.collection(u'users')
    docs = users_ref.where(u'uid', u'==', uid).stream()
    for doc in docs:
        data = doc.to_dict()
        audio_url = data['audioUrl']
    # Firebase storage URL of the audio file
    url = audio_url
    # Read the audio file from the URL
    req = urllib.request.urlopen(url)
    audio_data = req.read()
    #Create a file-like object from the audio data
    audio_file = io.BytesIO(audio_data)
    # Convert the audio file to an AudioSegment
    audio_segment = AudioSegment.from_file(audio_file, format='3gp', codec='libopencore_amrnb', parameters=["-ar", "16000"])
    # Save the AudioSegment to a temporary file
    temp_file = tempfile.NamedTemporaryFile(delete=False)
    temp_file_path = temp_file.name
    audio_segment.export(temp_file_path, format='wav')
    r = sr.Recognizer()
    # Pass the file name of the temporary file to the AudioFile constructor
    with sr.AudioFile(temp_file_path) as source:
        audio = r.record(source)
    # Remove the temporary file
    temp_file.close()
    os.remove(temp_file_path)
    input_text = r.recognize_google(audio, language=in_language)
    print(input_text)
    german = Field(tokenize=tokenize_german,
                lower=True,
                init_token="<sos>",
                eos_token="<eos>")

    english = Field(tokenize=tokenize_english,
                lower=True,
                init_token="<sos>",
                eos_token="<eos>")

    train_data, valid_data, test_data = TabularDataset.splits(path="", train=data_path, validation=data_path, test=data_path,
        format="csv", fields=[ ("English", english),("Hindi", german)],
        skip_header=True)
    german.build_vocab(train_data, max_size=10000, min_freq=3)
    english.build_vocab(train_data, max_size=10000, min_freq=3)

    german.build_vocab(train_data, max_size=10000, min_freq=3)
    english.build_vocab(train_data, max_size=10000, min_freq=3)

    e = list(german.vocab.__dict__.values())
    word_2_idx = dict(e[3])
    idx_2_word = {}
    for k,v in word_2_idx.items():
        idx_2_word[v] = k
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    BATCH_SIZE = 32

    train_iterator, valid_iterator, test_iterator = BucketIterator.splits((train_data, valid_data, test_data), 
                                                                        batch_size = BATCH_SIZE, 
                                                                        sort_within_batch=True,
                                                                        sort_key=lambda x: len(x.Hindi),
                                                                        device = device)
    count = 0
    max_len_eng = []
    max_len_ger = []
    for data in train_data:
        max_len_ger.append(len(data.Hindi))
        max_len_eng.append(len(data.English))
        if count < 10 :
            count += 1
    count = 0
    for data in train_iterator:
        if count < 1 :
            temp_ger = data.Hindi
            temp_eng = data.English
            count += 1
    temp_eng_idx = (temp_eng).cpu().detach().numpy()
    temp_ger_idx = (temp_ger).cpu().detach().numpy()
    df_eng_idx = pd.DataFrame(data = temp_eng_idx, columns = [str("S_")+str(x) for x in np.arange(1, 33)])
    df_eng_idx.index.name = 'Time Steps'
    df_eng_idx.index = df_eng_idx.index + 1 
    # df_eng_idx.to_csv('/content/idx.csv')
    df_eng_word = pd.DataFrame(columns = [str("S_")+str(x) for x in np.arange(1, 33)])
    df_eng_word = df_eng_idx.replace(idx_2_word)

    input_size_encoder = len(german.vocab)
    encoder_embedding_size = 300
    hidden_size = 1024
    num_layers = 2
    encoder_dropout = 0.5

    encoder_lstm = EncoderLSTM(input_size_encoder, encoder_embedding_size,
                            hidden_size, num_layers, encoder_dropout).to(device)

    input_size_decoder = len(english.vocab)
    decoder_embedding_size = 300
    hidden_size = 1024
    num_layers = 2
    decoder_dropout = 0.5
    output_size = len(english.vocab)

    decoder_lstm = DecoderLSTM(input_size_decoder, decoder_embedding_size,
                            hidden_size, num_layers, decoder_dropout, output_size).to(device)
    for batch in train_iterator:
        break

    x = batch.English[1]

    learning_rate = 0.001
    step = 0

    model = Seq2Seq(encoder_lstm, decoder_lstm).to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    pad_idx = english.vocab.stoi["<pad>"]
    criterion = nn.CrossEntropyLoss(ignore_index=pad_idx)

    #contents = pickle.load(f) becomes...
    with open(model_path, 'rb') as f:
        model = CPU_Unpickler(f).load()
    # load the saved model from file
    with open(ts_path, 'rb') as f:
        ts1 = pickle.load(f)
    progress  = []
    for i,sen in enumerate(ts1):
        progress.append(TreebankWordDetokenizer().detokenize(sen))
    translated_sentence = translate_sentence(model, input_text, german, english, device, max_length=50)
    progress.append(TreebankWordDetokenizer().detokenize(translated_sentence))
    tts = gTTS(text=progress[-1], lang=out_language)
    # use the engine to convert text to speech
    with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as f:
        tts.write_to_fp(f)
        file_path = f.name
    # upload the file to Firebase storage in a folder
    bucket = storage.bucket()
    blob = bucket.blob('outputAudio/enTota' + uid)  # add the folder name to the blob name
    blob.upload_from_filename(file_path)
    # get the URL of the file in Firebase storage
    url = blob.public_url
    # print the URL
    print(progress[-1])
    return ("Predicted Sentence : {}".format(progress[-1]))

if __name__ == '__main__':
    app.run(debug=True)