import argparse
import numpy as np
import os, pickle

from keras.models import Model, load_model
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

from data_helpers import Dataloader

class bc_LSTM:
    def __init__(self, args):
        self.classification_mode = args.classify
        self.modality = args.modality
        
        self.PATH = "./data/models/{}_weights_{}.hdf5".format(args.modality,self.classification_mode.lower())
        self.OUTPUT_PATH = "./data/pickles/{}_{}.pkl".format(args.modality,self.classification_mode.lower())
        
        print("Model initiated for {} classification".format(self.classification_mode))
    
    def load_data(self,):
        print('Loading data')
        
        self.data = Dataloader(mode = self.classification_mode)
        
        if self.modality == "text":
            self.data.load_text_data()
        elif self.modality == "audio":
            self.data.load_audio_data()
        elif self.modality == "bimodal":
            self.data.load_bimodal_data()
        else:
            exit()
        
        self.train_x = self.data.train_dialogue_features
        self.val_x = self.data.val_dialogue_features
        self.test_x = self.data.test_dialogue_features
        
        self.train_y = self.data.train_dialogue_label
        self.val_y = self.data.val_dialogue_label
        self.test_y = self.data.test_dialogue_label
        
        self.train_mask = self.data.train_mask
        self.val_mask = self.data.val_mask
        self.test_mask = self.data.test_mask
        
        self.train_id = self.data.train_dialogue_ids.keys()
        self.val_id = self.data.val_dialogue_ids.keys()
        self.test_id = self.data.test_dialogue_ids.keys()
        
        self.sequence_length = self.train_x.shape[1]
        self.classes = self.train_y.shape[2]
    
    def calc_test_result(self, pred_label, test_label, test_mask):
        true_label = []
        predicted_label = []
        
        for i in range(pred_label.shape[0]):
            for j in range(pred_label.shape[1]):
                if test_mask[i,j] == 1:
                    true_label.append(np.argmax(test_label[i,j]))
                    predicted_label.append(np.argmax(pred_label[i,j]))
        
        print("Confusion Matrix :")
        print(confusion_matrix(true_label, predicted_label))
        
        print("Classification Report :")
        print(classification_report(true_label, predicted_label, digits = 4))
        
        print('Weighted FScore: \n ', precision_recall_fscore_support(true_label, predicted_label, average = 'weighted'))
    
    def test_model(self):
        model = load_model(self.PATH)
        intermediate_layer_model = Model(input = model.input, output = model.get_layer("utter").output)
        
        intermediate_output_train = intermediate_layer_model.predict(self.train_x)
        intermediate_output_val = intermediate_layer_model.predict(self.val_x)
        intermediate_output_test = intermediate_layer_model.predict(self.test_x)
        
        train_emb, val_emb, test_emb = {}, {}, {}
        
        for idx, ID in enumerate(self.train_id):
            train_emb[ID] = intermediate_output_train[idx]
        
        for idx, ID in enumerate(self.val_id):
            val_emb[ID] = intermediate_output_val[idx]
        
        for idx, ID in enumerate(self.test_id):
            test_emb[ID] = intermediate_output_test[idx]
        
        pickle.dump([train_emb, val_emb, test_emb], open(self.OUTPUT_PATH, "wb"))
        
        self.calc_test_result(model.predict(self.test_x), self.test_y, self.test_mask)

print("Hello")
parser = argparse.ArgumentParser()
parser.required = True
parser.add_argument("-classify", help = "Set the classifiction to be 'Emotion' or 'Sentiment'", required = True)
parser.add_argument("-modality", help = "Set the modality to be 'text' or 'audio' or 'bimodal'", required = True)
args = parser.parse_args()

if args.classify.lower() not in ["emotion", "sentiment"]:
    print("Classification mode hasn't been set properly. Please set the classifiction flag to be: -classify Emotion/Sentiment")
    exit()

if args.modality.lower() not in ["text", "audio", "bimodal"]:
    print("Modality hasn't been set properly. Please set the modality flag to be: -modality text/audio/bimodal")
    exit()

args.classify = args.classify.title()
args.modality = args.modality.lower()

for directory in ["./data/pickles", "./data/models"]:
    if not os.path.exists(directory):
        os.makedirs(directory)

model = bc_LSTM(args)
model.load_data()
model.test_model()