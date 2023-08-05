import streamlit as st
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification

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
    st.title("Data Classification🎯")
    longText = "The program allows you to upload an Excel file containing complaints 🗳. You can select specific services you want to measure 🤔, and the program automatically classifies the complaints related to those services into two categories: satisfied or unsatisfied 🙂🙁. This classification enables a detailed analysis of the feedback received from customers 📉📑."
    st.markdown(longText)

tokenizer = AutoTokenizer.from_pretrained("textattack/bert-base-uncased-imdb")
model = AutoModelForSequenceClassification.from_pretrained("textattack/bert-base-uncased-imdb")
labels_sentiment = ["positive", "negative"]

# Streamlit application

# Upload file
uploaded_file = st.file_uploader("Upload a file", type=["xlsx", "xls"])
if uploaded_file is not None:
    # Read file
    df = pd.read_excel(uploaded_file)

    # Select column
    columns = list(df.columns)
    column_to_classify = st.selectbox("Select a column to classify", options=columns)

    # Classify
    if st.button("Classify"):
        if column_to_classify:
            segmented_df = pd.DataFrame()
            segmented_df['Segmented Text'] = df[column_to_classify]

            results_sentiment = []
            total_rows = len(segmented_df)
            progress_bar = st.progress(0)
            progress_text = st.empty()
            for i, row in segmented_df.iterrows():
                sequence_to_classify = row['Segmented Text']
                inputs = tokenizer(sequence_to_classify, padding=True, truncation=True, return_tensors="pt")
                outputs = model(**inputs)
                predictions = outputs.logits.argmax(dim=1).item()
                sentiment = labels_sentiment[predictions]
                results_sentiment.append(sentiment)

                progress_bar.progress((i + 1) / total_rows)
                progress_text.text(f"{int((i + 1) / total_rows * 100)}%")

            segmented_df['Sentiment'] = results_sentiment

            st.write(segmented_df)

        else:
            st.warning("Please select a column.")
