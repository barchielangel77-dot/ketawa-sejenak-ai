import streamlit as st
import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

# 1. Konfigurasi Halaman & Tema Web
st.set_page_config(page_title="Ketawa Sejenak AI", page_icon="🤣", layout="centered")

st.title("🤣 Platform Interaktif: Ketawa Sejenak")
st.write("Platform asisten AI yang pengetahuannya murni dari buku pribadi kamu.")
st.write("---")

# 2. Setup API Key (Masukkan OpenAI API Key kamu di sini atau lewat environment variable)
# Kamu bisa mendapatkan API Key dari platform.openai.com
os.environ["OPENAI_API_KEY"] = "MASUKKAN_API_KEY_OPENAI_KAMU_DI_SINI"

# Tempat penyimpanan database lokal untuk otak AI
PERSIST_DIR = "./chroma_db"

# 3. SIDEBAR: Fitur Upload Buku Otomatis
with st.sidebar:
    st.header("📚 Dashboard Admin")
    st.subheader("Upload atau Perbarui Buku")
    
    # Input file PDF mandiri
    uploaded_file = st.file_uploader("Unggah file 'KETAWA SEJENAK Obat anti stress.pdf' di sini", type=["pdf"])
    
    if uploaded_file:
        # Simpan file sementara ke dalam sistem
        temp_file_path = f"./temp_{uploaded_file.name}"
        with open(temp_file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
            
        with st.spinner("AI sedang membaca dan mempelajari isi bukumu..."):
            try:
                # Load PDF menggunakan PyPDFLoader
                loader = PyPDFLoader(temp_file_path)
                docs = loader.load()
                
                # Potong teks buku menjadi bagian-bagian kecil agar AI mudah mengingat konteksnya
                text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
                splits = text_splitter.split_documents(docs)
                
                # Buat Vector Store (Otak AI) dan simpan secara lokal di laptop
                embeddings = OpenAIEmbeddings()
                vectorstore = Chroma.from_documents(documents=splits, embedding=embeddings, persist_directory=PERSIST_DIR)
                
                st.success("🎉 Sukses! AI sudah selesai mempelajari buku baru kamu.")
                
                # Hapus file sampah sementara
                os.remove(temp_file_path)
                st.rerun()
            except Exception as e:
                st.error(f"Gagal memproses file: {e}")

# 4. SISTEM UTAMA: Memanggil Otak AI yang Sudah Disimpan
if os.path.exists(PERSIST_DIR) and len(os.listdir(PERSIST_DIR)) > 0:
    # Inisialisasi ulang database dan model AI
    embeddings = OpenAIEmbeddings()
    vectorstore = Chroma(persist_directory=PERSIST_DIR, embedding_function=embeddings)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3}) # Mengambil 3 potongan cerita paling relevan
    
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.7)
    
    # Mengatur kepribadian AI (Prompt Engineering) agar bergaya akrab ("bro")
    system_prompt = (
        "Kamu adalah asisten AI yang asyik, akrab, humoris, dan menggunakan panggilan 'bro' ke pengguna.\n"
        "Tugas utama kamu adalah menjawab pertanyaan atau menceritakan kembali humor hanya berdasarkan teks buku yang diberikan di bawah ini.\n"
        "Jika cerita yang diminta tidak ada di dalam teks buku, katakan saja dengan jujur dan santai kalau cerita itu belum ada di koleksi buku saat ini.\n\n"
        "Konteks Buku:\n{context}"
    )
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{input}"),
    ])
    
    # Membuat RAG Chain
    question_answer_chain = create_stuff_documents_chain(llm, prompt)
    rag_chain = create_retrieval_chain(retriever, question_answer_chain)
    
    # 5. ANTARMUKA CHAT (Sama seperti Gemini/ChatGPT)
    if "messages" not in st.session_state:
        st.session_state.messages = [
            {"role": "assistant", "content": "Halo bro! File bukumu sudah tersimpan di otak gw. Mau denger atau diskusi cerita humor yang mana nih? Tanya aja!"}
        ]
        
    # Tampilkan riwayat chat
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])
            
    # Input dari pengguna
    if user_input := st.chat_input("Tanyakan sesuatu (misal: Ceritakan tentang papeda masuk hotel dong, bro)..."):
        # Tampilkan chat user
        st.session_state.messages.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.write(user_input)
            
        # AI memproses jawaban berdasarkan isi buku
        with st.chat_message("assistant"):
            with st.spinner("Bentar bro, gw inget-inget dulu ceritanya..."):
                response = rag_chain.invoke({"input": user_input})
                answer = response["answer"]
                st.write(answer)
                st.session_state.messages.append({"role": "assistant", "content": answer})
else:
    # Tampilan jika admin belum mengunggah buku sama sekali
    st.info("👋 Halo bro! Platform sudah siap. Silakan unggah file PDF buku kamu di menu 'Dashboard Admin' sebelah kiri terlebih dahulu ya biar AI-nya punya otak!")
