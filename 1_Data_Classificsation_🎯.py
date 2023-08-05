import streamlit as st
import pandas as pd
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
import nltk
from nltk.tokenize import sent_tokenize
from mtranslate import translate
import io

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

model_name = "MoritzLaurer/mDeBERTa-v3-base-mnli-xnli"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)
classifier = pipeline("zero-shot-classification", model=model, tokenizer=tokenizer)

# Function for translation and segmentation
def TranslateAndSegment(df, complaint_column):
    all_rows = []
    for index, row in df.iterrows():
        if pd.notnull(row[complaint_column]):  # Exclude rows with null values
            translated_text = translate(row[complaint_column], 'en')
            sentences = sent_tokenize(translated_text)
            for sentence in sentences:
                # Tokenize the sentence
                tokens = tokenizer.tokenize(sentence)

                # Check if any token is unknown
                if tokenizer.unk_token in tokens:
                    # Handle unknown token
                    special_token = "<UNK>"
                    tokens = [special_token if token == tokenizer.unk_token else token for token in tokens]

                # Convert tokens back to segmented text
                segmented_text = " ".join(tokens)

                new_row = row.copy()
                new_row[complaint_column] = segmented_text
                all_rows.append(new_row)
    return pd.DataFrame(all_rows)

# Streamlit application

# Upload file
uploaded_file = st.file_uploader("Upload a file", type=["xlsx", "xls"])
if uploaded_file is not None:
    # Read file
    df = pd.read_excel(uploaded_file)

    # Select column
    columns = list(df.columns)
    column_to_classify = st.selectbox("Select a column to classify", options=columns)

    # Select labels
    labels_sentiment = ["positive", "negative"]

    default_labels_topic = ['Accommodation', 'Transportation', 'Guidance', 'Catering',
                            'Holy sites', 'Haram', 'Airport', 'Nusk app', 'Restrooms']
    user_labels_topic = st.multiselect("Select topic labels", options=default_labels_topic,
                                       default=default_labels_topic)

    # Classify
    if st.button("Classify"):
        if column_to_classify:
            # Translate and segment
            segmented_df = TranslateAndSegment(df, column_to_classify)
            segmented_df.rename(columns={column_to_classify: 'Segmented Text'}, inplace=True)

            results_sentiment = []
            results_topic = []
            total_rows = len(segmented_df)
            progress_bar = st.progress(0)
            progress_text = st.empty()
            for i, row in segmented_df.iterrows():
                sequence_to_classify = row['Segmented Text']
                output_sentiment = classifier(sequence_to_classify, candidate_labels=labels_sentiment,
                                              multi_label=False)
                results_sentiment.append(output_sentiment['labels'][0])

                output_topic = classifier(sequence_to_classify, candidate_labels=user_labels_topic + ['other'],
                                          multi_label=False)
                results_topic.append(output_topic['labels'][0])

                progress_bar.progress((i + 1) / total_rows)
                progress_text.text(f"{int((i + 1) / total_rows * 100)}%")

            segmented_df['Sentiment'] = results_sentiment
            segmented_df['Topic'] = results_topic

            st.write(segmented_df)

            # Convert DataFrame to Excel and enable downloading
            output = io.BytesIO()
            excel_writer = pd.ExcelWriter(output, engine='openpyxl')
            segmented_df.toExcel(excel_writer, index=False, sheet_name='Sheet1')
            excel_writer.save()
            excel_data = output.getvalue()
            st.download_button("Download Classification Results", data=excel_data, file_name="classification_results.xlsx",
                               mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet')
        else:
            st.warning("Please select a column.")
