# +
import streamlit as st
import os
import shutil
import subprocess

INPUTS_FOLDER = 'inputs'
OUTPUTS_FOLDER = 'outputs'
VIDEO_EXTENSIONS = ['mp4']
MODEL_PATH = './models/NOX.pth'
PROCESS_SCRIPT = 'main.py'

st.title('NOX VIDEO')

def clear_folder(folder):
    if os.path.exists(folder):
        shutil.rmtree(folder)
    os.makedirs(folder)

def cache_data(func):
    def wrapper(*args, **kwargs):
        unique_key = ' '.join(map(str, args)) + ' ' + ' '.join(map(str, kwargs.values()))
        if unique_key not in st.session_state:
            st.session_state[unique_key] = func(*args, **kwargs)
        return st.session_state[unique_key]
    return wrapper

@cache_data
def process_video(uploaded_video_path, result_video_path, result_filename):
    if not os.path.exists(result_video_path):
        file_name = os.path.basename(uploaded_video_path) 
        with st.spinner(f'{file_name} 처리 중...'):
            subprocess.run([
                "python", PROCESS_SCRIPT,
                "--input", uploaded_video_path,
                "--result_dir", OUTPUTS_FOLDER,
                "--output_name", result_filename,
                "--nox_path", MODEL_PATH
            ])
    return os.path.exists(result_video_path)

if __name__ == "__main__":
    if 'need_clear' not in st.session_state or st.session_state.need_clear:
        clear_folder(INPUTS_FOLDER)
        clear_folder(OUTPUTS_FOLDER)
        st.session_state.need_clear = False

    uploaded_files = st.file_uploader("동영상 업로드", type=VIDEO_EXTENSIONS, accept_multiple_files=True)

    if uploaded_files:
        for uploaded_file in uploaded_files:
            if uploaded_file is not None:
                original_filename = os.path.splitext(uploaded_file.name)[0]
                result_filename = f"{original_filename}_NOX"
                result_video_path = os.path.join(OUTPUTS_FOLDER, f"{result_filename}.mp4")
                uploaded_video_path = os.path.join(INPUTS_FOLDER, uploaded_file.name)

                with open(uploaded_video_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())

                if process_video(uploaded_video_path, result_video_path, result_filename):
                    with open(result_video_path, "rb") as file:
                        st.download_button(
                            label=f"{result_filename} 다운로드",
                            data=file,
                            file_name=f"{result_filename}.mp4",
                            mime="video/mp4"
                        )

