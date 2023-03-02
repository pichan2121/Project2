# import thư viện
import numpy as np
import pandas as pd
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split  
from sklearn.metrics import accuracy_score 
from sklearn.metrics import confusion_matrix
from sklearn. metrics import classification_report, roc_auc_score, roc_curve
import pickle
import streamlit as st
import matplotlib.pyplot as plt
from sklearn import metrics
import seaborn as sns
import glob
from popularity import popularity_based_recommender_model
from sklearn.model_selection import train_test_split


# set page
st.set_page_config(page_title="E-commerce", page_icon=":money_with_wings:")


# import thư viện Content Base
# add styling
with open('style.css') as f:
    st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)


# Define page layout
menu = ["Home","Recommended Items", "Personalized Items"]
choice = st.sidebar.selectbox("You are at ", menu)
# upload data

uploaded_files = st.file_uploader("Choose a CSV file", accept_multiple_files=True)
for file in uploaded_files:
    if file.name=="consine_similarity.npy" :
        result=np.load(file)
        
    elif file.name=="newdata.csv" :
        data=pd.read_csv(file)
    elif file.name=="reviewdata.csv" :
        data2=pd.read_csv(file)
        train_data, test_data = train_test_split(data2, test_size =.20, random_state=10)
        pr = popularity_based_recommender_model(train_data=train_data, test_data=test_data, user_id='userId', item_id='productId')
        
# chia dữ liệu đọc popularity


with st.container():
    item=st.sidebar.text_input(' ')
    st.sidebar.write("Bạn muốn tìm sản phẩm nào")
    clicked= st.sidebar.button("Search")   
    clicked="Enter"



if choice == "Home":
    # Create a container for the header
    with st.container():
        st.title("WELCOME TO OUR E-COMMERCE SITE!")
        st.write("Here are the top recommended products which are used by the Popularity Based Recommender Model. Popularity based recommendation system uses the items that are in trend right now. It ranks products based on its highest the rating count. If a product is highly rated then it is most likely to be ranked higher and hence will be recommended. As it is based on the products popularity, this can not be personalized and hence same set of products will be recommended for all the users.")
   
            
if choice == "Recommended Items" and len(item)==0:
    # Create a container for the header
    with st.container():
        st.title("WELCOME TO OUR E-COMMERCE SITE!")
        st.write("Here are the top recommended products which are used by the Popularity Based Recommender Model. Popularity based recommendation system uses the items that are in trend right now. It ranks products based on its highest the rating count. If a product is highly rated then it is most likely to be ranked higher and hence will be recommended. As it is based on the products popularity, this can not be personalized and hence same set of products will be recommended for all the users.")
        
        
    # Create a container for the featured products
    with st.container():
        st.header("Top Highest Rank Products")
        col1, col2, col3, col4= st.columns(4)
        
        # Display a sample product in each column
        with col1:
            st.image(data.iloc[1,11], use_column_width=True)
            st.write("Product Name")
            st.write("$20.00")
            
            
        with col2:
            st.image(data.iloc[2,11], use_column_width=True)
            st.write("Product Name")
            st.write("$25.00")
           
        
        with col3:
            st.image(data.iloc[2,11], use_column_width=True)
            st.write("Product Name")
            st.write("$25.00")
           
            
        with col4:
            st.image(data.iloc[2,11], use_column_width=True)
            st.write("Product Name")
            st.write("$25.00")
           
elif  choice == "Recommended Items" and len(item)!=0:        
    with st.container():
        st.title("Một số đề xuất cho bạn tham khảo!")
    
        item.lower()
        item_index=data.loc[data['name'].str.lower().str.startswith(item)].index[0]  ##this is the problem
        distances=result[item_index]
        item_list=sorted(list(enumerate(distances)), reverse=True , key=lambda x:x[1])[1:10]
        col1, col2 = st.columns([5, 3])
    
    for i in item_list:

        if i[0]%2==0:
            with col1:
                st.image(data.iloc[i[0],11],use_column_width=True)
                st.write(data.iloc[i[0],3])
                st.write(data.iloc[i[0],8])
                st.write(data.iloc[i[0],5])
        else: 
            with col2:
                st.image(data.iloc[i[0],11],use_column_width=True)
                st.write(data.iloc[i[0],3])
                st.write(data.iloc[i[0],8])
                st.write(data.iloc[i[0],5])
        
            

            
elif choice == "Personalized item":
    # Create a container for the header
    with st.container():
        st.header("About Us")
        st.write("We are a small e-commerce company based in Streamlitville, USA.")
        
    # Create a container for the team members
    with st.container():
        st.subheader("Meet the Team")
        col1, col2, col3 = st.columns(3)









   





