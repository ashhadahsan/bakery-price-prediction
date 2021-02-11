#!/usr/bin/env python
# coding: utf-8

# In[463]:

import streamlit as st
import streamlit.components.v1 as stc
from time import sleep
import pandas as pd
import numpy as np
import time
from tqdm import tqdm
from datetime import datetime
import os
import joblib
import uuid 
from sklearn.preprocessing import StandardScaler
import base64
from sklearn.preprocessing import LabelEncoder

from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dense
from time import sleep
from pandas.tseries.offsets import DateOffset
from keras.callbacks import ReduceLROnPlateau

import pandas as pd
import numpy as np
import time
from tqdm import tqdm
from datetime import datetime
import os
import joblib
import uuid
from sklearn.preprocessing import LabelEncoder
from numpy import hstack
import base64
def get_table_download_link(df,filename):
    """Generates a link allowing the data in a given panda dataframe to be downloaded
    in:  dataframe
    out: href string
    """
    csv = df.to_csv()
    b64 = base64.b64encode(csv.encode()).decode()  # some strings <-> bytes conversions necessary here
    return f'<a href="data:file/csv;base64,{b64}" download="{filename}.csv">Download csv file</a>'
def main():

    st.set_page_config(page_title="Bakery Sales")

    st.title("Bakery Sales Predictions")
    data_file = st.file_uploader("Upload CSV",type=['csv'])
    if st.button("Process"):
        if data_file is not None:
            df=pd.read_csv(data_file,parse_dates=[0],index_col=0)
            df= df[df.Item !='NONE']
            df=df.dropna()
            df=df.drop(columns=['Time','Transaction'])
            un=str(uuid.uuid1())
            df=pd.DataFrame(df.groupby('Date')['Item'].value_counts()).to_csv(f'count{un}.csv')
            df=pd.read_csv(f'count{un}.csv',parse_dates=[0],index_col=0)
            lbl=LabelEncoder()
            df.Item=lbl.fit_transform(df.Item)
            df.columns=['Item','val']
            df2=df
            df2=df2.reset_index()
            df2['Year']= df2.Date.dt.year
            df2['Month']=df2.Date.dt.month
            df2=df2.drop('Date',axis=1)
            y=df2.val.values
            X=df2.drop('val',axis=1)
            scaler=StandardScaler()
            # X=scaler.fit_transform(X)
            X_train=X.values
            d=[df.index[-1] + DateOffset(days=x) for x in range(0,1) ] *len(df2.Item.unique().tolist())
            df_seven_days = pd.DataFrame({'Item':df2.Item.unique(),'Date':d})
            df_seven_days['Year']= df_seven_days.Date.dt.year
            df_seven_days['Month']=df_seven_days.Date.dt.month
            dates_daily=df_seven_days.Date
            df_seven_days = df_seven_days.drop('Date',axis=1)
            x_train = X_train.reshape(-1, 1, X_train.shape[1])
            model = Sequential()
            model.add(LSTM(100,activation='relu',return_sequences=False,input_shape=(x_train.shape[1],x_train.shape[2])))
            # model.add(Dropout(0.5))
            # model.add(Dense(100, activation='relu'))
            model.add(Dense(1))
            model.compile(loss='mse', optimizer='nadam')

            red_lr= ReduceLROnPlateau(monitor='loss',patience=3,verbose=0,factor=0.1) #define early stop criteria

            with st.spinner('Training may take a while'):
                model.fit(x_train,y,epochs=200,callbacks=red_lr,batch_size=10,verbose=0)
            df_seven_days_array = df_seven_days.values

            preds = model.predict(df_seven_days_array.reshape(-1,1,df_seven_days_array.shape[1]))*np.random.randint(2,7)
            df_seven_days['preds']=abs(preds).astype('int')
            df_seven_days['Date']=dates_daily
            # In[476]:


            df_seven_days['Item'] =lbl.inverse_transform(df_seven_days.Item)
            dateee=datetime.now().strftime(r'%Y-%m-%d ')
            dateee=dateee+"daily"
            st.markdown(get_table_download_link(df_seven_days,dateee), unsafe_allow_html=True)
            st.write("Calculating predictions for the next 7 days")





# In[478]:


            preds = [df.index[-1] + DateOffset(days=x) for x in range(0,8) ][1:] * len(df2.Item.unique().tolist())
            preds = pd.DataFrame({'Date':preds})
            preds['Items']=0
            start=0
            leng=len(df2.Item.unique().tolist())

            for xx in range(7):
                if xx == 0:
                    preds['Items'][start:leng] =df2.Item.unique().tolist()
                    start=leng
                else:
                    preds['Items'][start:(start+leng)] =df2.Item.unique().tolist()
                    start=start+leng
            preds['Year'] = preds.Date.dt.year
            preds['Month'] = preds.Date.dt.month
            dates_week=preds.Date

            preds = preds.drop('Date',axis=1)
            preds_array=preds.values


# In[479]:


            preds_wek = model.predict(preds_array.reshape(-1,1,preds_array.shape[1]))*np.random.randint(2,7)


            # In[480]:


            preds['preds']=0
            preds['preds']=preds_wek.astype('int')
            preds['Date']=dates_week
            preds.drop(['Year','Month'],axis=1)


            # In[481]:


            preds['Items']=lbl.inverse_transform(preds['Items'])
            
            dateee=datetime.now().strftime(r'%Y-%m-%d ')
            dateee=dateee+"next week"
            st.markdown(get_table_download_link(preds,dateee), unsafe_allow_html=True)






    