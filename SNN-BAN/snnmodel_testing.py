
from scipy import signal
import numpy as np

import torch
import torchaudio
import matplotlib.pyplot as plt

from spikingjelly.activation_based import encoding
from spikingjelly import visualizing
import os
from librosa import cqt
import librosa


import torch
from spikingjelly.activation_based import encoding
import os


Max_num=83
class SNNforauditory(nn.Module):
    def __init__(self, tau):
        super().__init__()

        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(83 * Max_num, 1024, bias=False),
            neuron.LIFNode(tau=tau, surrogate_function=surrogate.ATan()),
            layer.Dropout(0.5),


        )
        self.fc1 = nn.Sequential(
            nn.Linear(1024, 11, bias=False), # For TID dataset
            # nn.Linear(1024, 10, bias=False), # For RWCP dataset
            neuron.LIFNode(tau=tau, surrogate_function=surrogate.ATan()),

        )

    def forward(self, x: torch.Tensor):
        x = self.fc(x)
        x = self.fc1(x)
        return x

# Loading model
modelname=findfile('./snnmodel/Model_TID_R_MSE/BAN_encoding/T10/','.pth') 

# Loading data
mat_data = loadmat('./After_encoding_data/TID/BAN_encoding/TEST/test_data_10T.mat')
mat_data_label = loadmat('./After_encoding_data/TID/BAN_encoding/TEST/labeltest_data.mat')
variable_name = 'Test_data'
variable_name_label = 'test_labels'
Test_data = mat_data[variable_name]
test_labels = mat_data_label[variable_name_label]
test_labels=np.squeeze(test_labels)

#%
accuracyScore_ALL = []
macro_precisionScore_ALL = []
macro_recallScore_ALL = []
macro_f1Score_ALL = []
macro_AUC_ALL = []
for idx in range(len(modelname)):
  
    modelpath=modelname[idx]
    savepath='./snnmodel/Model_TID_R_MSE/BAN_encoding/T10/figure_T10/'

    net = torch.load(modelpath, map_location='cpu')
    net.eval()
    with torch.inference_mode():
        SUM = 0
        pred = []
        pred_probabilities = []
        # T=15
        for idx in range(Test_data.shape[0]):
            test_output = 0
            result1 = Test_data[idx,:, :, :]
            result1 = result1[np.newaxis, :]
            result1=np.array(result1)
            result1 = torch.from_numpy(result1).type(torch.FloatTensor)
            result1 = result1.permute(3, 0, 2, 1)

            for t in range(T):
                # encoded_data=test_data[:,t,:,:]
                # encoded_data = encoder(test_data)
                # test_output += net(encoded_data)
                temp=result1[t]
                # temp=temp[np.newaxis,:,:]
                # temp=torch.from_numpy(temp).type(torch.FloatTensor)
                test_output += net(temp)
            test_output = test_output / T

            Label = np.argmax(test_output.numpy())
            pred.append(Label)
            pred_probabilities.append(torch.softmax(test_output, dim=1).numpy())
            if Label == test_labels[idx]:
                SUM = SUM + 1
            print(idx)
        print(SUM /Test_data.shape[0])
    #%
    if os.path.exists(savepath):
        print('ready exist')
    else:
        print('ok I make it')
        os.makedirs(savepath)
    figurename=str.split(str.split(modelpath,'/')[-1],'.pth')[0]


    # Visualization
    cm = confusion_matrix(true_label, pred_label)
    plt.imshow(cm, cmap=plt.cm.Blues)
    plt.colorbar()
    plt.xlabel('Predicted labels')
    plt.ylabel('True labels')
    plt.xticks(np.arange(11))
    plt.yticks(np.arange(11))
    plt.savefig(savepath+figurename+'_ACC:'+str(SUM /Test_data.shape[0])+'_2'+'.tif', dpi=300)
    plt.show()

    # Evaluating model
    from sklearn.metrics import accuracy_score
    from sklearn.metrics import confusion_matrix
    from sklearn.metrics import precision_score
    from sklearn.metrics import recall_score
    from sklearn.metrics import f1_score
    
    from sklearn.metrics import classification_report
    from sklearn.preprocessing import label_binarize
    from sklearn.metrics import roc_auc_score
    from sklearn.metrics import roc_curve
    import csv

    def cal_metrics(y_test,y_predict):
    
        accuracyScore = accuracy_score(y_test, y_predict)

        macro_precisionScore = precision_score(y_test, y_predict, average='macro')

        macro_recallScore = recall_score(y_test, y_predict, average='macro')

        macro_f1Score = f1_score(y_test, y_predict,average='macro')
        n_classes = np.unique(y_test).size
        y_true_binary = label_binarize(y_test, classes=range(n_classes))
    
        weigthed_AUC = roc_auc_score(y_true_binary, pred_probabilities,average='weighted',multi_class='ovo')
 
        macro_AUC = roc_auc_score(y_true_binary, pred_probabilities,average='macro',multi_class='ovo')
    
    
        # report = classification_report(y_test, y_predict)
    
        print("accuracy_score = {}".format(accuracyScore))
  
        print("macro_precisionScore = {}".format(macro_precisionScore))
    
        print("macro_recallScore  = {}".format(macro_recallScore))

        print("macro_f1Score = {}".format(macro_f1Score))
    
        print("macro_AUC = {}".format(macro_AUC))
    
        return accuracyScore,macro_precisionScore,macro_recallScore,macro_f1Score,macro_AUC
    
    
    accuracyScore,macro_precisionScore,macro_recallScore,macro_f1Score,macro_AUC=cal_metrics(true_label,pred_label)
    
    
    accuracyScore_ALL.append(accuracyScore)
    macro_precisionScore_ALL.append(macro_precisionScore)
    macro_recallScore_ALL.append(macro_recallScore)
    macro_f1Score_ALL.append(macro_f1Score)
    macro_AUC_ALL.append(macro_AUC)
    
    filename=savepath+'model_result.csv'
    final_name=final_name =str.split(figurename,'_')
    with open(filename, 'w', newline='') as file:
        fieldnames = ['name', 'accuracyScore_all', 'macro_precisionScore_all', 'macro_recallScore_all', 'macro_f1Score_all','macro_AUC_all']
        writer = csv.DictWriter(file, fieldnames=fieldnames)
    
        writer.writeheader()
        for accuracyScore_alls, macro_precisionScore_alls, macro_recallScore_alls, macro_f1Score_alls,macro_AUC_alls in zip(np.array(accuracyScore_ALL),np.array(macro_precisionScore_ALL),np.array(macro_recallScore_ALL),np.array(macro_f1Score_ALL),np.array(macro_AUC_ALL)):
            writer.writerow({'name': final_name[-1], 'accuracyScore_all': accuracyScore_alls, 'macro_precisionScore_all': macro_precisionScore_alls,
                             'macro_recallScore_all': macro_recallScore_alls,
                             'macro_f1Score_all': macro_f1Score_alls,
                             'macro_AUC_all':macro_AUC_alls})