#!/usr/bin/env python
# coding: utf-8

import setuptools.dist
import time
import streamlit as st
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
import cv2
import os
from PIL import Image
import pickle
import matplotlib.pyplot as plt
#import pydicom
from sklearn.svm import SVC
import joblib
import lightgbm

st.set_page_config(layout="wide")
st.title('FluDeep Grading System')
st.header('FluDeep')
st.subheader('Prediction of influenza pneumonia 30-day mortality')
col1, col2, col3, col4 = st.columns((1,1,1,2))



with col1:
    # Gender
    status = st.radio("Gender", ('male', 'female'))
    Gender = 1 if status == 'male' else 0
    # Age
    Age = st.number_input(label = 'Age (years old)', min_value = 0.0)
    # BMI
    BMI = st.number_input(label = 'BMI (kg/m2)', min_value = 0.0, step = 0.1)
    # Diastolic_blood_pressure
    Diastolic_blood_pressure = st.number_input(label = 'Diastolic blood pressure (mmHg)', min_value = 0.0, step = 0.1)
    # Heart_rate
    Heart_rate = st.number_input(label = 'Heart rate (beats per minute)', min_value = 0.0, step = 0.1)
    # Respiratory_rate
    Respiratory_rate = st.number_input(label = 'Respiratory rate (breaths per minute)', min_value = 0.0, step = 0.1)
    # Creatinine
    Creatinine = st.number_input(label = 'Creatinine (mg/dL)', min_value = 0.0, step = 0.1)

with col2:
    # CRP
    CRP = st.number_input(label = 'CRP (mg/dL)', min_value = 0.0, step = 0.1)
    # Hemoglobin
    Hemoglobin = st.number_input(label = 'Hemoglobin (g/dL)', min_value = 0.0, step = 0.1)
    # Bicarbonate
    Bicarbonate = st.number_input(label = 'Bicarbonate (mmol/L)', min_value = 0.0, step = 0.1)
    # PCO2
    PCO2 = st.number_input(label = 'PCO2 (mmHg)', min_value = 0.0, step = 0.1)
    # pH
    pH = st.number_input(label = 'pH of arterial blood', min_value = 0.0, step = 0.1)
    # Platelet
    Platelet = st.number_input(label = 'Platelet count (k/µL)', min_value = 0.0, step = 0.1)

with col3:
    # PO2
    PO2 = st.number_input(label = 'PO2 (mmHg)', min_value = 0.0, step = 0.1)
    # Blood_urea_nitrogen
    Blood_urea_nitrogen = st.number_input(label = 'Blood urea nitrogen (mg/dL)', min_value = 0.0, step = 0.1)
    # Lactic_acid
    Lactic_acid = st.number_input(label = 'Lactic acid (mmol/L)', min_value = 0.0, step = 0.1)
    # INR
    INR = st.number_input(label = 'INR', min_value = 0.0, step = 0.1)
    # Glucose
    Glucose = st.number_input(label = 'Glucose (mg/dL)', min_value = 0.0, step = 0.1)
    # Hematocrit
    Hematocrit = st.number_input(label = 'Hematocrit (%)', min_value = 0.0, step = 0.1)


with col4:
    # upload X-ray image and return score(1-5)
    uploaded_file = st.file_uploader("Please upload a X-ray image(.jpeg, .dicom)：")
    
    uploaded_image = []
    if uploaded_file is not None and st.button('Submit'):
        st.write("Loading....")
        image = Image.open(uploaded_file)
        st.image(image, caption='X-ray', width = 320)
        
        image = cv2.cvtColor(np.array(image), cv2.COLOR_GRAY2BGR)
        image = cv2.resize(image, (299, 299))
        uploaded_image.append(image/255.0)

        model = keras.models.load_model('DeepFluXR_MSE0.9446.h5')
        pred = model.predict(np.array(uploaded_image))
        if pred > 5.0:  pred = 5.0
        if pred < 1.0:  pred = 1.0  
        
                      
        dataset = pd.DataFrame([[Gender, Age, BMI, Diastolic_blood_pressure, Heart_rate, Respiratory_rate, Creatinine, CRP, Hemoglobin, Bicarbonate, 
                   PCO2, pH, Platelet, PO2, Blood_urea_nitrogen, Lactic_acid, INR, Glucose, Hematocrit]], 
        columns = ['Gender', 'Age', 'BMI', 'Diastolic_blood_pressure', 'Heart_rate', 'Respiratory_rate', 'Creatinine', 'CRP', 'Hemoglobin', 'Bicarbonate', 
                   'PCO2', 'pH', 'Platelet', 'PO2',  'Blood_urea_nitrogen', 'Lactic_acid', 'INR', 'Glucose', 'Hematocrit'])
        
        with open('LightGBM_withoutCXR_v2_AUC0.8103.pickle', 'rb') as f:
            LightGBM = pickle.load(f)
        LightGBM_result = LightGBM.predict_proba(dataset)[:, 1]

        df = pd.DataFrame({"pred":[float(pred)],"LightGBM":[float(LightGBM_result)]})
        clf = joblib.load('Late_Fusion_RF_lr_AUC0.8371.pickle')
        SVC_result = clf.predict_proba(df)[:, 1]
        SVC_result2 = clf.predict(df)

        st.info("FluDeep-XR score is {:.2f}".format(float(pred)))
        # st.write(SVC_result2)  # alive within 30 days(0 = alive/1 = not alive)
        st.info("30 days survival rate is {:.2f}%".format((1 - SVC_result[0])*100))