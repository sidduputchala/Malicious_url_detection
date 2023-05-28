import pandas as pd
import numpy as np
import itertools
import matplotlib.pyplot as plt
np.random.seed(1337) 

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import (precision_score, recall_score,f1_score, accuracy_score,mean_squared_error,mean_absolute_error)
from sklearn.metrics import confusion_matrix

def LoadData():
    train_labels = pd.read_csv('data/trainlabel.csv',encoding="ISO-8859-1", header=None) 
    train_label = train_labels.iloc[:, 0:1]
    test_labels = pd.read_csv('data/testlabel.csv',encoding="ISO-8859-1", header=None)
    test_label = test_labels.iloc[:, 0:1]

    train = pd.read_csv('data/train.txt', encoding="ISO-8859-1", header=None)
    test = pd.read_csv('data/test.txt', encoding="ISO-8859-1", header=None)

    train = train.values.tolist()
    train = list(itertools.chain(*train))
    test = test.values.tolist()
    test = list(itertools.chain(*test))

    train_label = train_label.values.tolist()
    train_label = list(itertools.chain(*train_label))
    test_label = test_label.values.tolist()
    test_label = list(itertools.chain(*test_label))
    return train, test, train_label, test_label

def preprocessing(train, test) :
    # preprocessing the training data
    for i in range(len(train)) :
      train[i] = train[i].replace('"','')
      train[i] = train[i].replace("'",'')
      if "www." in train[i] :
          train[i] = train[i].replace("www.","")
      if "//" in train[i]:
          train[i] = train[i].replace("//","/")
      if ';' in train[i]:
          url_parts = train[i].split(';')
          path = url_parts[0]
          train[i] = path + ';'.join(url_parts[1:])

    # preprocessing the test data
    for i in range(len(test)) :
        test[i] = test[i].replace('"','')
        test[i] = test[i].replace("'",'')
        if "www." in test[i] :
            test[i] = test[i].replace("www.","")
        if "//" in test[i]:
            test[i] = test[i].replace("//","/")
        if ';' in test[i]:
            url_parts = test[i].split(';')
            path = url_parts[0]
            test[i] = path + ';'.join(url_parts[1:])

    return train, test                                                

def one_hot_encoding(train, test, max_url_len): 
    # Create a dictionary of all characters in the URL
    vocab = {'a','b','c','d','e','f','g','h','i','j','k','l','m','n','o','p','q','r','s', 't','u','v','w','x','y','z','A','B','C','D','E','F','G','H', 'I','J','K','L','M','N','O','P','Q','R','S', 'T','U','V','W','X','Y','Z','0','1','2','3','4','5','6','7','8', '9','/','.',':','-','_','?','=','&','%','+','*','$','@','!','~','^','(',')','[',']','{','}','|','<','>','`',';',',',' ',}
    char_to_index = {'a': 0,'b': 1,'c':2, 'd':3, 'e':4, 'f':5, 'g':6, 'h':7, 'i':8, 'j':9, 'k':10,'l':11,'m':12,'n':13,'o':14,'p':15,'q':16,'r':17,'s':18,'t':19,'u':20,'v':21,'w':22,'x':23,'y':24,'z':25,'A':26,'B':27,'C':28,'D':29,'E':30,'F':31,'G':32,'H':33,'I':34,'J':35,'K':36,'L':37,'M':38,'N':39,'O':40,'P':41,'Q':42,'R':43,'S':44,'T':45,'U':46,'V':47,'W':48,'X':49,'Y':50,'Z':51,'0':52,'1':53,'2':54,'3':55,'4':56,'5':57,'6':58,'7':59,'8':60,'9':61,'/':62,':':63,'.':64,'-':65,'_':66,'?':67,'=':68,'&':69,'%':70,'+':71,'*':72,'$':73,'@':74,'!':75,'~':76,'^':77,'(':78,')':79,'[':80,']':81,'{':82,'}':83,'|':84,'<':85,'>':86,'`':87,';':88,',':89,' ':90 }
    
    # One-hot encode URL sequence
    one_hot_encoded_train = np.zeros((len(train),max_url_len,len(char_to_index)), dtype=np.int32)
    for i, url in enumerate(train):
         for j, char in enumerate(url[:max_url_len]):
              if char in  vocab:
                one_hot_encoded_train[i, j, char_to_index[char]] = 1

    one_hot_encoded_test = np.zeros((len(test),max_url_len,len(char_to_index)), dtype=np.int32)
    for i, url in enumerate(test):
         for j, char in enumerate(url[:max_url_len]):
             if char in  vocab:
               one_hot_encoded_test[i, j, char_to_index[char]] = 1
    return one_hot_encoded_train, one_hot_encoded_test


# Define the CNN model
class CNN(nn.Module):
    def __init__(self, input_dim):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv1d(input_dim, 64, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool1d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv1d(64, 128, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool1d(kernel_size=2, stride=2)
        self.conv3 = nn.Conv1d(128, 256, kernel_size=3, padding=1)
        self.pool3 = nn.MaxPool1d(kernel_size=2, stride=2)

        # self.fc1 = nn.Linear(256 * max_len_url, 512)
        # self.fc2 = nn.Linear(512, num_classes)
        # self.dropout = nn.Dropout(0.5)
        self.relu = nn.ReLU()
    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = self.relu(self.conv1(x))
        x = self.pool1(x)
        x = self.relu(self.conv2(x))
        x = self.pool2(x)
        x = self.relu(self.conv3(x))
        x = self.pool3(x)
        x = x.permute(0, 2, 1)
        # x = x.view(x.size(0), -1)
        # x = self.relu(self.fc1(x))
        # x = self.dropout(x)
        # x = self.fc2(x)
        return x

# Define the LSTM model
class LSTM(nn.Module):
    def __init__(self, input_dim, hidden_size, num_layers, num_classes):
        super(LSTM, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)
        # self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        fc_out = self.fc(lstm_out[:, -1, :])
        # x = self.sigmoid(fc_out)
        return fc_out

# Define the combined CNN-LSTM model
class CNN_LSTM(nn.Module):
    def __init__(self, input_dim, hidden_size, num_layers, num_classes):
        super(CNN_LSTM, self).__init__()
        self.cnn = CNN(input_dim)
        self.lstm =LSTM(256, hidden_size, num_layers, num_classes)

    def forward(self, x):
        cnn_out = self.cnn(x)
        lstm_out = self.lstm(cnn_out)
        return lstm_out
    
def main():

# Define the hyperparameters
    batch_size = 32
    learning_rate = 0.001
    num_epochs = 10
    input_dim = 91
    num_classes = 2
    max_len_url = 100
    hidden_size = 128
    num_layers = 1
    x_train,x_test,trainlabels,testlabels = LoadData()
    x_train,x_test = preprocessing(x_train,x_test)
    x_train,x_test = one_hot_encoding(x_train,x_test,max_len_url)

    # Split the data into batches
    x_train = torch.Tensor(x_train)
    y_train = torch.Tensor(trainlabels)
    train_data = TensorDataset(x_train, y_train)
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)

    # Initialize the model, optimizer, and loss function
    # CNN_Model = CNN(input_dim, num_classes,max_len_url)
    
    # Initialize model and move to device
    model = CNN_LSTM(input_dim, hidden_size, num_layers, num_classes)
    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)


    # Train the model
    for epoch in range(num_epochs):
        running_loss = 0.0
        for i, data in enumerate(train_loader, 0):
            inputs, labels = data
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels.long())
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        print(' In Epoch %d the loss is: %.3f' % (epoch + 1, running_loss / len(train_loader)))

    # Evaluate the model on the test set
    x_test = torch.Tensor(x_test)
    y_test = torch.Tensor(testlabels)
    test_data = TensorDataset(x_test, y_test)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)

    correct = 0
    total = 0
    with torch.no_grad():
        # print the confusion matrix
        total_predictions = []
        for data in test_loader:
            inputs, labels = data
            predicted_prob = model(inputs)
            # predicted = (predicted_prob >= 0.5).int().squeeze(dim=1)
            _, predicted = torch.max(predicted_prob.data, 1)
            total_predictions.append(predicted)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        conf_mat = confusion_matrix(y_test, torch.cat(total_predictions))    
        print('Confusion matrix: \n', conf_mat)
        tp = conf_mat[1][1]
        fp = conf_mat[0][1]
        fn = conf_mat[1][0]
        tn = conf_mat[0][0]
        print("\ntrue positives :",conf_mat[1][1])
        print("true negatives :",conf_mat[0][0])
        print("false positivies :",conf_mat[0][1])
        print("false negatives :",conf_mat[1][0])
        print("\nprecision :",tp/(tp+fp))
        print("recall :",tp/(tp+fn))
        print("f1 score :",2*tp/(2*tp+fp+fn))
        
    print('Accuracy on the test set: %f %%' % (100 * correct / total))

main()