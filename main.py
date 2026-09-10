import streamlit as st
import tempfile
from pathlib import Path

from db import SessionDB, RepositoryDB
from rag import RagPipeline
from preprocessing import document_pipeline, generate_query_embedding, read_image, read_audio



# ---------------- INITIALIZATION ----------------
repo_db = RepositoryDB()
rag = RagPipeline()
include_session = False
if "conversation_history" not in st.session_state:
    st.session_state.conversation_history = []
if "session_db" not in st.session_state:
    st.session_state.session_db = SessionDB()

# ---------------- UI ----------------

## SIDEBAR
with st.sidebar:
    st.header("Settings")
    files = st.file_uploader(
        "Upload documents",
        type=["pdf", "txt","ppt","docx","md"],
        accept_multiple_files=True
    )
    add_to_repository = st.checkbox("Add to repository")
    upload_document = st.button("Upload")
    if upload_document and files:
        for file in files:
            with tempfile.NamedTemporaryFile(delete=False, suffix=Path(file.name).suffix) as tmp:
                tmp.write(file.getbuffer())
                tmp_path = tmp.name
            with st.spinner("Processing file(s) ..."): 
                processed_chunks = document_pipeline(tmp_path)
                include_session = True
            if add_to_repository:
                repo_db.add_documents(processed_chunks)
                st.success("File added to repository")
            else:
                st.session_state.session_db.add_documents(processed_chunks)
                st.success("File added to current session")
                
            


## MAIN PANNEL        
col1,col2 = st.columns([6, 1])
with col1: query = st.text_input("Your query")
with col2: 
    st.markdown("<div style='height: 28px'></div>", unsafe_allow_html=True)
    send = st.button("➤")

col3, col4, col5 = st.columns([0.5, 0.5, 0.5])

with col3:
    audio = st.audio_input("Record")
    if audio:
        with open("recorded_audio.wav", "wb") as f:
            f.write(audio.getbuffer())

with col4:
    image = st.file_uploader(
        "📷",
        type=["png", "jpg"],
        label_visibility="collapsed",
        key="ocr_image"

    )
with col5:
    include_repository = st.checkbox("Include repository search")

    

if send:
    if not query and not audio:
        st.warning("Write a query")
        st.stop()
    if not include_repository and not files:
        st.warning("Upload a file or check 'Include repository' box")
        st.stop()
    combined_query_parts = []
    if query:
        combined_query_parts.append(query)       
    # Read Image
# Audio
    if audio:
        with st.spinner("Transcribing audio..."):
            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
                tmp.write(audio.getbuffer())
                tmp_audio = tmp.name
                
                audio_text = read_audio(tmp_audio)
                combined_query_parts.append(audio_text)

    # Image
    if image:
        with st.spinner("Transcribing image..."):
            tmp_suffix = Path(image.name).suffix
            with tempfile.NamedTemporaryFile(delete=False, suffix=tmp_suffix) as tmp:
                tmp.write(image.getbuffer())
                tmp_path = tmp.name  

            image_text = read_image(tmp_path)
            combined_query_parts.append(image_text)


    # Combine query
    with st.spinner("Combining & Embedding Query..."):
        combined_query = " | ".join(combined_query_parts)
        embedded_query = generate_query_embedding(combined_query)

    # Retrieve
    with st.spinner("Retrieving Relevant Information..."):
        if include_repository:
            results = repo_db.retrieve(embedded_query, 5)
            print(f"Retrieval Result {results}")
        else:
            results = st.session_state.session_db.retrieve(embedded_query)

        if results and "documents" in results and results["documents"]:
            retrieved_chunks = results["documents"][0]  
        else:
            retrieved_chunks = []

    with st.spinner("Generating Answer..."):
        prompt = rag.build_prompt(retrieved_chunks, combined_query, conversation_history=st.session_state.conversation_history)
        print(f"Prompt: {prompt}")
        answer = rag.generate_answer(prompt)
        st.session_state.conversation_history.append({
            "user": combined_query,
            "assistant": answer
        })

        st.chat_message("user").write(combined_query)
        st.chat_message("assistant").write(answer)
