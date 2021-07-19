import streamlit as st 
from PIL import Image
import pickle
import numpy as np
#import matplotlib.pyplot as plt
import pandas as pd
st.set_option('deprecation.showfileUploaderEncoding', False)
# Load the pickled model
model = pickle.load(open('secondmidkmeans.pkl','rb'))   
dataset= pd.read_csv('clustering dataset 33.csv')
X = dataset.iloc[:, [0,1,2,3,4,5,6,7,8]].values
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X = sc.fit_transform(X)
def predict_note_authentication(meanfreq,sd,median,IQR,skew,kurt,mode,centroid,dfrange):
  output= model.predict(sc.transform([[meanfreq,sd,median,IQR,skew,kurt,mode,centroid,dfrange]]))
  print("Gender is: ", output)
  if output==[1]:
    prediction="Male"
  else:
    prediction="Female"
  print(prediction)
  return prediction

def main():
    html_temp = """
   <div class="" style="background-color:blue;" >
   <div class="clearfix">           
   <div class="col-md-12">
   <center><p style="font-size:40px;color:white;margin-top:10px;">Poornima Institute of Engineering & Technology</p></center> 
   <center><p style="font-size:30px;color:white;margin-top:10px;">Department of Computer Engineering</p></center> 
   <center><p style="font-size:25px;color:white;margin-top:10px;">RTU End Term Practical Examination</p></center> 
   </div>
   </div>
   </div>
   """
    st.markdown(html_temp,unsafe_allow_html=True)
    st.header("Gender Prediction")
    meanfreq = st.text_input("Meanfreq","")
    sd = st.text_input("SD","")
    median = st.text_input("Median","")
    IQR = st.text_input("IQR","")
    skew = st.text_input("Skew","")
    kurt = st.text_input("Hurt","")
    mode = st.text_input("Mode","")
    centroid = st.text_input("Centroid","")
    dfrange = st.text_input("DFrange","")
    resul=""
    if st.button("Predict"):
      result=predict_note_authentication(meanfreq,sd,median,IQR,skew,kurt,mode,centroid,dfrange)
      st.success('Model has predicted {}'.format(result))
    if st.button("About"):
      st.text("Developed by Prabhjeet Singh")
      st.text("Student , Department of Computer Engineering")

if __name__=='__main__':
  main()
