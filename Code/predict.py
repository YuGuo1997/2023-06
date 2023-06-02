import pandas as pd
import warnings
from tqdm import tqdm
from sklearn import metrics
import gc
import time
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from keras.models import load_model
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

def data_slicing(data):
    # data_sel = data.iloc[0:4143619, ]       ##SWARM-A
    # data_sel=data.iloc[4143620:8273814,]    ##SWARM-B
    data_sel = data.iloc[8273815:19372516, ]  ##SWARM-C
    # data_sel = data.iloc[19372517:29305303, ] ##GRACE-A
    # data_sel=data_sel[(data.date=='2013-06-28')|(data.date=='2013-06-29')|(data.date=='2013-06-30')]
    data_sel = data_sel[(data.date == '2016-06-06')]
    index=data_sel.index.values
    data_x0 = data.drop('f107s', axis=1)
    data_x = data_x0.drop('density_msise', axis=1)
    x0 = data_x0.iloc[:, 9:15].values
    x = data_x.iloc[:, 8:14].values
    x0=x0[index]
    scale=MinMaxScaler()
    dataXS=scale.fit_transform(x)
    dataXS=dataXS[index]
    y = dataXS[:, 0]
    y0 = data.iloc[:, 8].values
    y0=y0[index]
    return dataXS,y,x0,y0,scale

def create_dataset(x,y, seq_len):
    features = []
    targets = []
    for i in range(0, len(x) - seq_len, 1):
        data = x[i:i + seq_len]
        label = y[i + seq_len]
        features.append(data)
        targets.append(label)
    # return dataset
    return np.array(features), np.array(targets)

def predict(save_path,model_path,test_dataset,x_test0,y_test0,scale):
    # f = h5py.File(model_path, 'r')
    # print(f.attrs.get('keras_version'))
    model=load_model(model_path)
    y_pred = model.predict(test_dataset, verbose=1)
    y_pred = np.repeat(y_pred, 6, axis=1)
    y_pred = scale.inverse_transform(y_pred)
    y_pred = y_pred[:, 0]
    density = pd.DataFrame()
    y_true = []
    x_test0 = x_test0[window:]
    y_test0 = y_test0[window:]
    msise = x_test0[:, 0]
    y_test_tem = np.ndarray.flatten(y_test0)
    for i in tqdm(range(len(y_pred))):
        y_true.append(abs(y_test_tem[i]))
    # print(y_true)
    density['true'] = y_true
    density['msise'] = msise
    density['predict'] = y_pred
    density['altitude'] = x_test0[:, 3]
    density['longitude'] = x_test0[:, 4]
    density['latitude'] = x_test0[:, 5]
    density.to_csv(save_path, sep='\t', index=True, header=True)
    # nrlmsise-00 error
    print('Mean Absolute Error:', metrics.mean_absolute_error(x_test0[:, 0], y_test0))
    print('Mean Squared Error:', metrics.mean_squared_error(x_test0[:, 0], y_test0))
    print('Root Mean Squared Error:',np.sqrt(metrics.mean_squared_error(x_test0[:, 0], y_test0)))
    ## prediction error
    print('Mean Absolute Error:', metrics.mean_absolute_error(density['predict'].values, y_test0))
    print('Mean Squared Error:', metrics.mean_squared_error(density['predict'].values, y_test0))
    print('Root Mean Squared Error:',np.sqrt(metrics.mean_squared_error(density['predict'].values, y_test0)))

if __name__ == '__main__':
    warnings.filterwarnings('ignore')
    ###############file path
    save_path = 'C:\\Users\\Administrator\\Desktop\\results(density)\\re\\storm_test\\gru.txt'
    model_path='G:\\model\\GRU.h5'                #GRU
    # model_path='G:\\model\\LSTM.h5'             #LSTM
    # model_path='G:\\model\\AMAD.h5'             #AMAD-NET

    start = time.perf_counter()
    data_org = pd.read_table("C:\\Users\\Administrator\\Desktop\\data.txt", index_col=0) ## loading data
    data_org['altitudecopy']=data_org.iloc[:,3]
    data_org['glongitudecopy'] = data_org.iloc[:, 4]
    data_org['glatitudecopy'] = data_org.iloc[:, 5]
    end1 = time.perf_counter()
    print("Data loading:", round(end1 - start), 'seconds')
    dataXS,y,x0,y0,scale=data_slicing(data_org)
    end2 = time.perf_counter()
    print("Data slicing:", round(end2 - end1), 'seconds')
    window=3  #sliding window
    test_dataset, test_labels = create_dataset(dataXS, y, window)
    end3 = time.perf_counter()
    print("Batch data generation:", round(end3 - end2), 'seconds')
    predict(save_path,model_path,test_dataset,x0,y0,scale)
    end4 = time.perf_counter()
    print("Model prediction:", round(end4 - end3), 'seconds')
    del data_org
    gc.collect()