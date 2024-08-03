import streamlit as st
from helper import process_text
import time

st.set_page_config(page_title="LLM Text Chunker", layout="wide")
st.header(body="LLM Text Chunker", divider="green")

col_input, col_dummy, col_chunks = st.columns([3,0.05,2])

with col_input:

    input_text = st.text_area("**:green[Text to be Chunked:]**", height=800, label_visibility="collapsed")

    process_text_btn = st.button(label="**:green[Chunk the Text]**", use_container_width=True)

with col_dummy:
    st.empty()

if process_text_btn:
    with col_chunks:

        if input_text:
            try:
                with st.spinner("Processing text..."):
                    start_time = time.perf_counter()
                    chunks = process_text(input_text)
                    end_time = time.perf_counter()
                    elapsed_time = round(end_time - start_time, 2)
                    
                if chunks:
                    st.success(body=f"**Text splitted into {len(chunks)} chunks successfully! | {elapsed_time} seconds**")

                    for chunk in chunks:
                        with st.expander(f"**:green[Chunk {chunk.order} | Sentences {chunk.start}-{chunk.end} | Chunk Length: {len(chunk.text)}]**"):
                            context_container = st.container(border=True)
                            context_container.markdown(body=f"**:green[Context:]** {chunk.context}")
                            chunk_text_container = st.container(border=True)
                            chunk_text_container.markdown(body=f"**:green[Text:]** {chunk.text}")

            except Exception as e:
                st.error(f"An error occurred: {str(e)}")
        else:
            st.warning("Please enter some text to process.")