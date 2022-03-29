from io import StringIO
from Bio import SeqIO
import pandas as pd
import streamlit as st
from PIL import Image
from keras.models import load_model
import numpy as np
import joblib as jl

icon = Image.open('fav.png')
st.set_page_config(page_title='DeepDBS', page_icon = icon)

def encodeSeq(seq):
    encoder = ['X', 'A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y']
    encSeq = [0 for x in range(41)]
    i = 0
    ra = len(seq)
    for i in range(ra):
        value = encoder.index(seq[i])
        encSeq[i] = value
    seqArray=np.asarray(encSeq)
    return seqArray.reshape(1,41)


def modelLoader():
    myLSTM = load_model("./models/lstm.h5")
    myRF = jl.load("./models/rf.joblib")
    return myLSTM, myRF


def seqValidator(seq):
    checkSet = {'X', 'A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y'}
    if set(seq).issubset(checkSet):
            return True
    return False


final_df = pd.DataFrame(columns=['Sequence ID', 'Sub Sequence','Location','Label'])
seq = ""
len_seq = 0
image = Image.open('WebPic.png')
st.subheader("""DeepDBS""")
st.image(image, use_column_width=True)
st.sidebar.subheader(("Input Sequence(s) (FASTA FORMAT ONLY)"))
fasta_string  = st.sidebar.text_area("Sequence Input", height=200)       
st.subheader("Click the Example Button for Sample Data")


if st.button('Example'):
    st.code(">sp|Q5SHR1|IF1_THET8 Translation initiation factor IF-1 OS=Thermus thermophilus (strain ATCC 27634 / DSM 579 / HB8) OX=300852 GN=infA PE=1 SV=1\nMAKEKDTIRTEGVVTEALPNATFRVKLDSGPEILAYISGKMRMHYIRILPGDRVVVEITPYDPTRGRIVYRK", language="markdown")
    st.code(">tr|Q924Q6|Q924Q6_MOUSE VH186.2-D-J-C mu protein (Fragment) OS=Mus musculus OX=10090 GN=Ighm PE=1 SV=1\nQVQLQQPGAELVKPGASVKLSCKASGYTFTSYWMHWVKQRPGRGLEWIGRIDPNSGGTKYNEKFKSKATLTVDKPSSTAYMQLSSLTSEDSAVYYCARSTLSHYYAMDYWGQGTSVTVSSESQSFPNVFPLVSCESPLSDKNLVA", language="markdown")
    

if st.sidebar.button("SUBMIT"):
    if(fasta_string==""):
        st.info("Please input the sequence first.")
    fasta_io = StringIO(fasta_string) 
    records = SeqIO.parse(fasta_io, "fasta") 
    for rec in records:
        seq_id = str(rec.id)
        seq=str(rec.seq).upper()
        if(seqValidator(seq)):
            seq = "XXXXXXXXXXXXXXXXXXXX"+seq+"XXXXXXXXXXXXXXXXXXXX"
            seqLen = len(seq)
            for i in range(20, seqLen-20):
                sub_seq = seq[i-20: i+21]
                df_temp = pd.DataFrame([[seq_id, sub_seq,str(i+1-20),'None']], columns=['Sequence ID', 'Sub Sequence','Location','Label'] )
                final_df = pd.concat([final_df,df_temp], ignore_index=True)
        else:
            st.info("Sequence with Sequence ID: " + str(seq_id) + " is invalid, containing letters other than standard amino acids")
    fasta_io.close()
    if(final_df.shape[0]!=0):
        myLSTM, myRF = modelLoader()
        for iter in range(final_df.shape[0]):
            tempSeq =  final_df.iloc[iter, 1]
            seqArray = encodeSeq(tempSeq)
            fvArray = myLSTM.predict(seqArray)
            score = myRF.predict(fvArray)
            pred_label = np.round_(score, decimals=0, out=None)
            if(pred_label==1):
                pred_label="Non-DNA Binding Site"
            else:
                pred_label="DNA Binding Site"
            final_df.iloc[iter, 3] = str(pred_label)
    st.dataframe(final_df)

