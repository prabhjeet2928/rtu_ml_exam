# -*- coding: utf-8 -*-
"""PIET18CS106-Prabhjeet_Singh-Set(7).ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/193I7JdpjAg8LGNsGHGgYWZhxYfo3Eb_7
"""

# # Data Preprocessing
# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from google.colab import files
uploaded = files.upload()

dataset= pd.read_csv('clustering dataset 33.csv')
dataset

#Check Missing Data
dataset.isnull().sum()

# Extracting dependent and independent variables:
# Extracting independent variable:
X = dataset.iloc[:, 0:].values

print(X)

# Taking care of missing data
#handling missing data (Replacing missing data with the mean value)  
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(missing_values= np.NAN, strategy= 'mean', fill_value=None, verbose=1, copy=True)
#Fitting imputer object to the independent variables x.   
imputer = imputer.fit(X[:, 1:8]) 
#Replacing missing data with the calculated mean value  
X[:, 1:8]= imputer.transform(X[:, 1:8])

print(X)

types = dataset.dtypes
print(types)

# standardizing the data
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X = sc.fit_transform(X)
# statistics of scaled data
pd.DataFrame(X).describe()

print(X)

print(np.isnan(X).sum())

# Using the elbow method to find the optimal number of clusters
from sklearn.cluster import KMeans
wcss = []
for i in range(1, 20):
    kmeans = KMeans(n_clusters = i, init = 'k-means++', random_state = 42)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)

plt.plot(range(1, 20), wcss)
plt.title('The Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
plt.show()

# Fitting K-Means to the dataset
kmeans = KMeans(n_clusters = 2, init = 'k-means++', random_state = 42)
print(kmeans)
y_kmeans = kmeans.fit_predict(X)

frame = pd.DataFrame(X)
frame['cluster'] = y_kmeans
frame['cluster'].value_counts()

print("Within cluster sum of square when k=2", kmeans.inertia_)

print("center of Cluster are", kmeans.cluster_centers_ )

print("Number of iterations", kmeans.n_iter_)

# Visualising the clusters
plt.scatter(X[:,0], X[:,3], s = 100, c = 'black', label = 'Data Distribution')
plt.title('Customer Distribution before clustering')
plt.xlabel('Channel')
plt.ylabel('Region)')
plt.legend()
plt.show()

frame = pd.DataFrame(X)
frame['cluster'] = y_kmeans
frame['cluster'].value_counts()

meanfreq =  0.059780985#@param {type:"number"}
sd = 0.064241268#@param {type:"number"}
median = 0.032026913#@param {type:"number"}
IQR = 0.075121951#@param {type:"number"}
skew = 12.86346184#@param {type:"number"}
kurt = 274.4029055#@param {type:"number"}
mode = 0#@param {type:"number"}
centroid = 0.59780985#@param {type:"number"}
dfrange = 0#@param {type:"number"}


output= kmeans.predict(sc.transform([[meanfreq,sd,median,IQR,skew,kurt,mode,centroid,dfrange]]))
print("Gender is: ",output)
if output==[1]:
  print("Male")
else:
  print("Female")

# Visualising the clusters
plt.scatter(X[y_kmeans == 0, 0], X[y_kmeans == 0, 1], s = 100, c = 'red', label = 'Cluster 1')
plt.scatter(X[y_kmeans == 1, 0], X[y_kmeans == 1, 1], s = 100, c = 'blue', label = 'Cluster 2')
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 3], s = 300, c = 'yellow', label = 'Centroids')
plt.title('Clusters of Gender')
plt.xlabel('Spending on meanfreq')
plt.ylabel('Spending on IQR')
plt.legend()
plt.show()

import pickle 
  
# Save the trained model as a pickle string. 
saved_model = pickle.dumps(kmeans) 
  
# Load the pickled model 
Saved_Model = pickle.loads(saved_model) 
  
# Use the loaded pickled model to make predictions 
Saved_Model.predict(X)

import pickle 
print("[INFO] Saving model...")
# Save the trained model as a pickle string. 
saved_model=pickle.dump(kmeans,open('secondmidkmeans.pkl', 'wb')) 
# Saving model to disk

# Load the pickled model 
model = pickle.load(open('secondmidkmeans.pkl','rb'))  
# Use the loaded pickled model to make predictions 
model.predict(X)

#!pip install streamlit

#!pip install pyngrok

#!ngrok authtoken 1sO9O2v7CGlRWPKUgjrtmB7tWIa_6unaAnmQHqgRwQhgTz8Jg

# Commented out IPython magic to ensure Python compatibility.
# %%writefile app.py
# import streamlit as st 
# from PIL import Image
# import pickle
# import numpy as np
# import matplotlib.pyplot as plt
# import pandas as pd
# st.set_option('deprecation.showfileUploaderEncoding', False)
# # Load the pickled model
# model = pickle.load(open('/content/secondmidkmeans.pkl','rb'))   
# dataset= pd.read_csv('/content/clustering dataset 33.csv')
# X = dataset.iloc[:, [0,1,2,3,4,5,6,7,8]].values
# from sklearn.preprocessing import StandardScaler
# sc = StandardScaler()
# X = sc.fit_transform(X)
# def predict_note_authentication(meanfreq,sd,median,IQR,skew,kurt,mode,centroid,dfrange):
#   output= model.predict(sc.transform([[meanfreq,sd,median,IQR,skew,kurt,mode,centroid,dfrange]]))
#   print("Gender is: ", output)
#   if output==[1]:
#     prediction="Male"
#   else:
#     prediction="Female"
#   print(prediction)
#   return prediction
# 
# def main():
#     html_temp = """
#    <div class="" style="background-color:blue;" >
#    <div class="clearfix">           
#    <div class="col-md-12">
#    <center><p style="font-size:40px;color:white;margin-top:10px;">Poornima Institute of Engineering & Technology</p></center> 
#    <center><p style="font-size:30px;color:white;margin-top:10px;">Department of Computer Engineering</p></center> 
#    <center><p style="font-size:25px;color:white;margin-top:10px;">RTU End Term Practical Examination</p></center> 
#    </div>
#    </div>
#    </div>
#    """
#     st.markdown(html_temp,unsafe_allow_html=True)
#     st.header("Gender Prediction")
#     meanfreq = st.text_input("Meanfreq","")
#     sd = st.text_input("SD","")
#     median = st.text_input("Median","")
#     IQR = st.text_input("IQR","")
#     skew = st.text_input("Skew","")
#     kurt = st.text_input("Hurt","")
#     mode = st.text_input("Mode","")
#     centroid = st.text_input("Centroid","")
#     dfrange = st.text_input("DFrange","")
#     resul=""
#     if st.button("Predict"):
#       result=predict_note_authentication(meanfreq,sd,median,IQR,skew,kurt,mode,centroid,dfrange)
#       st.success('Model has predicted {}'.format(result))
#     if st.button("About"):
#       st.text("Developed by Prabhjeet Singh")
#       st.text("Student , Department of Computer Engineering")
# 
# if __name__=='__main__':
#   main()

#!nohup streamlit run  app.py &

#from pyngrok import ngrok
#url=ngrok.connect(port='8050')
#url

#!streamlit run --server.port 80 app.py

