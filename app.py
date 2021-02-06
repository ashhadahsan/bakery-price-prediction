import streamlit as st
import streamlit.components.v1 as stc
from time import sleep
from stqdm import stqdm
import pandas as pd
import numpy as np
from fbprophet import Prophet
import time
from tqdm import tqdm
from datetime import datetime
import os
import joblib
import uuid 
  
import base64
from sklearn.preprocessing import LabelEncoder
def get_table_download_link(df,filename):
    """Generates a link allowing the data in a given panda dataframe to be downloaded
    in:  dataframe
    out: href string
    """
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()  # some strings <-> bytes conversions necessary here
    return f'<a href="data:file/csv;base64,{b64}" download="{filename}.csv">Download csv file</a>'

def main():
	st.set_page_config(page_title="Bakery Sales")

	st.title("Bakery Sales Predictions")
	data_file = st.file_uploader("Upload CSV",type=['csv'])
	if st.button("Process"):
		if data_file is not None:

			file_details = {"Filename":data_file.name,"FileType":data_file.type,"FileSize":data_file.size}
			# st.write(file_details)
			df=pd.read_csv(data_file)
			df= df[df.Item !='NONE']
			df=df.dropna()
			df=df.drop(columns=['Time','Transaction'])
			un=str(uuid.uuid1())
			df=pd.DataFrame(df.groupby('Date')['Item'].value_counts()).to_csv(f'count{un}.csv')
			df=pd.read_csv(f'count{un}.csv')
			lbl=LabelEncoder()
			df.Item=lbl.fit_transform(df.Item)
			df.columns=['ds','Item','y']
			df2=df
			emp=[]
			# st.dataframe(df)
			st.write("Training is started ")
			# for x in stqdm(df2.index.tolist()):
			m = Prophet()
			m.add_regressor('Item')
			m.fit(df2)
			future=m.make_future_dataframe(periods=1,freq='D',include_history=False)

			for x in stqdm((df2.Item.unique().tolist())):

					future['Item']=x
					emp.append(m.predict(future))

			joblib.dump(m,"model.pkl")
			dfs = [df2.set_index('ds') for df2 in emp]
			dfpred=pd.concat(dfs)
			preds = dfpred.reset_index()[['ds','yhat']]
			preds['Item']= df2.Item.unique().tolist()
			preds.columns=['Date','Sales','Item']
			preds.Sales = preds.Sales.astype('int').abs()
			preds['Item']=lbl.inverse_transform(preds['Item'])
			result = preds.groupby(['Date','Item']).sum()
			dateee=datetime.now().strftime(r'%Y-%m-%d ')
			dateee=dateee+"daily"

			# st.write("Predictions for next day saved at ",os.getcwd())
			st.markdown(get_table_download_link(result,dateee), unsafe_allow_html=True)

			# result.to_csv(f'{dateee}_nextday.csv')


			st.write("Calculating predictions for the next 7 days")
			future=m.make_future_dataframe(periods=8,freq='D',include_history=False)
			future=future.drop(0).reset_index(drop=True)

			emp=[]
			for x in stqdm((df2.Item.unique().tolist())):

					future['Item']=x
					emp.append(m.predict(future))

			dfs = [df2.set_index('ds') for df2 in emp]
			dfpred=pd.concat(dfs)
			preds = dfpred.reset_index()[['ds','yhat']]
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
			# # preds['Items'][0:94] =df2.Item.unique().tolist()
			# # preds['Items'][94:188] =df2.Item.unique().tolist()
			# # preds['Items'][188:282] =df2.Item.unique().tolist()
			# # preds['Items'][282:376] =df2.Item.unique().tolist()
			# # preds['Items'][376:470] =df2.Item.unique().tolist()
			# # preds['Items'][470:564] =df2.Item.unique().tolist()
			# # preds['Items'][470:564] =df2.Item.unique().tolist()
			# # preds['Items'][564:658] =df2.Item.unique().tolist()
			preds.columns=['Date','Sales','Item']
			preds.Sales = preds.Sales.astype('int').abs()
			preds['Item']=lbl.inverse_transform(preds['Item'])
			result = preds.groupby(['Date','Item']).sum()
			dateee=datetime.now().strftime(r'%Y-%m-%d ')
			dateee=dateee+"next week"
			st.markdown(get_table_download_link(result,dateee), unsafe_allow_html=True)


			# st.write("Predictions for next seven days saved at ",os.getcwd())
			
			# result.to_csv(f'{dateee}_weekly.csv')


if __name__ == '__main__':
	
	main()
	
	     

	
  


