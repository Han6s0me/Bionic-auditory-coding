#%%
import os
import numpy as np
figpath='/home/handsome/PythonProject/SNN/snnmodel/Model_RWCP_R_MSE/Freq_encoding/T10_1/figure_T10_snr20_final/'
def findfile(path, file_last_name):
    file_name = []
    for file in os.listdir(path):
        file_path = os.path.join(path, file)
        # 如果是文件夹，则递归
        if os.path.isdir(file_path):
            findfile(file_path, file_last_name)
        elif os.path.splitext(file_path)[1] == file_last_name:
            file_name.append(file_path)
    return file_name

modelname=findfile(figpath,'.tif')
#%
Timestamp_check=[]
ACC_check=[]
for idx in range(len(modelname)):
    figname=modelname[idx]
    figurename = str.split(str.split(figname, '/')[-1], '.tif')[0]
    final_name =str.split(figurename,'_')
    if final_name[-1]=='1':
        Timestamp_check.append(final_name[-3])
        ACC_check.append(str.split(final_name[-2],':')[-1])

#%
import csv
savepath='/home/handsome/PythonProject/SNN/snnmodel/Model_RWCP_R_MSE/Freq_encoding/T10_1/T10_snr20.csv'
with open(savepath,'w',newline='') as file:
    fieldnames = ['Timestamp', 'Model_ACC']
    writer = csv.DictWriter(file, fieldnames=fieldnames)
    writer.writeheader()
    for timestamps,model_accs in zip(np.array(Timestamp_check),np.array(ACC_check)):
        writer.writerow({'Timestamp':timestamps,'Model_ACC':model_accs})

