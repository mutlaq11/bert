import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import openpyxl

#-----------------------------------------------------------------------------
hide_st_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            header {visibility: hidden;}
            </style>
            """
st.markdown(hide_st_style, unsafe_allow_html=True)

st.markdown("""
<div style="background-color:#FF9B52;padding:10px">
<h2 style="color:white;text-align:center;">Satisfaction Measurement 🙁😐🙂</h2>
</div>
""", unsafe_allow_html=True)

# Create two columns
left_column, right_column = st.columns(2)
# Use the right column for the image
right_column.image(r"suummaiafinmo.gif")
# Use the left column for the title
with left_column:
    st.title("Data Visualization 📊")
    longText = "Once the classification process is complete, the program will generate various graphs and visualizations that provide a clear representation of the classified data 📈📉. These visualizations help to easily understand the distribution and patterns of satisfaction and dissatisfaction among the pilgrims 🙂🙁."
    st.markdown(longText)

#-----------------------------------------------------------------------------
