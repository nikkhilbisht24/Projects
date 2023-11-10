import pyaudio
import os
import wave
import librosa
import numpy as np
from sys import byteorder
from array import array
from struct import pack


THRESHOLD = 500
CHUNK_SIZE = 1024
FORMAT = pyaudio.paInt16
RATE = 16000

SILENCE = 30

def is_silent(snd_data):
    "Returns 'True' if below the 'silent' threshold"
    return max(snd_data) < THRESHOLD

def normalize(snd_data):
    "Average the volume out"
    MAXIMUM = 16384
    times = float(MAXIMUM)/max(abs(i) for i in snd_data)

    r = array('h')
    for i in snd_data:
        r.append(int(i*times))
    return r

def trim(snd_data):
    "Trim the blank spots at the start and end"
    def _trim(snd_data):
        snd_started = False
        r = array('h')

        for i in snd_data:
            if not snd_started and abs(i)>THRESHOLD:
                snd_started = True
                r.append(i)

            elif snd_started:
                r.append(i)
        return r

    # Trim to the left
    snd_data = _trim(snd_data)

    # Trim to the right
    snd_data.reverse()
    snd_data = _trim(snd_data)
    snd_data.reverse()
    return snd_data

def add_silence(snd_data, seconds):
    "Add silence to the start and end of 'snd_data' of length 'seconds' (float)"
    r = array('h', [0 for i in range(int(seconds*RATE))])
    r.extend(snd_data)
    r.extend([0 for i in range(int(seconds*RATE))])
    return r

def record():
    """
    Record a word or words from the microphone and 
    return the data as an array of signed shorts.
    Normalizes the audio, trims silence from the 
    start and end, and pads with 0.5 seconds of 
    blank sound to make sure VLC et al can play 
    it without getting chopped off.
    """
    p = pyaudio.PyAudio()
    stream = p.open(format=FORMAT, channels=1, rate=RATE,
        input=True, output=True,
        frames_per_buffer=CHUNK_SIZE)

    num_silent = 0
    snd_started = False

    r = array('h')

    while 1:
        # little endian, signed short
        snd_data = array('h', stream.read(CHUNK_SIZE))
        if byteorder == 'big':
            snd_data.byteswap()
        r.extend(snd_data)

        silent = is_silent(snd_data)

        if silent and snd_started:
            num_silent += 1
        elif not silent and not snd_started:
            snd_started = True

        if snd_started and num_silent > SILENCE:
            break

    sample_width = p.get_sample_size(FORMAT)
    stream.stop_stream()
    stream.close()
    p.terminate()

    r = normalize(r)
    r = trim(r)
    r = add_silence(r, 0.5)
    return sample_width, r

def record_to_file(path):
    "Records from the microphone and outputs the resulting data to 'path'"
    sample_width, data = record()
    data = pack('<' + ('h'*len(data)), *data)

    wf = wave.open(path, 'wb')
    wf.setnchannels(1)
    wf.setsampwidth(sample_width)
    wf.setframerate(RATE)
    wf.writeframes(data)
    wf.close()



def extract_feature(file_name, **kwargs):
    """
    Extract feature from audio file `file_name`
        Features supported:
            - MFCC (mfcc)
            - Chroma (chroma)
            - MEL Spectrogram Frequency (mel)
            - Contrast (contrast)
            - Tonnetz (tonnetz)
        e.g:
        `features = extract_feature(path, mel=True, mfcc=True)`
    """
    mfcc = kwargs.get("mfcc")
    chroma = kwargs.get("chroma")
    mel = kwargs.get("mel")
    contrast = kwargs.get("contrast")
    tonnetz = kwargs.get("tonnetz")
    X, sample_rate = librosa.core.load(file_name)
    if chroma or contrast:
        stft = np.abs(librosa.stft(X))
    result = np.array([])
    if mfcc:
        mfccs = np.mean(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=40).T, axis=0)
        result = np.hstack((result, mfccs))
    if chroma:
        chroma = np.mean(librosa.feature.chroma_stft(S=stft, sr=sample_rate).T,axis=0)
        result = np.hstack((result, chroma))
    if mel:
        mel = np.mean(librosa.feature.melspectrogram(y=X, sr=sample_rate).T,axis=0)
        result = np.hstack((result, mel))
    if contrast:
        contrast = np.mean(librosa.feature.spectral_contrast(S=stft, sr=sample_rate).T,axis=0)
        result = np.hstack((result, contrast))
    if tonnetz:
        tonnetz = np.mean(librosa.feature.tonnetz(y=librosa.effects.harmonic(X), sr=sample_rate).T,axis=0)
        result = np.hstack((result, tonnetz))
    return result


if __name__ == "__main__":
    # load the saved model (after training)
    # model = pickle.load(open("result/mlp_classifier.model", "rb"))
    from utils import load_data, split_data, create_model
    import argparse
    parser = argparse.ArgumentParser(description="""Gender recognition script, this will load the model you trained, 
                                    and perform inference on a sample you provide (either using your voice or a file)""")
    parser.add_argument("-f", "--file", help="The path to the file, preferred to be in WAV format")
    args = parser.parse_args()
    file = args.file
    # construct the model
    model = create_model()
    # load the saved/trained weights
    model.load_weights(r"C:\Users\Nikkhil_Bisht\Desktop\Coding\Voice Detection\gender-recognition-by-voice-master\FINAL\results\model.h5")
    if not file or not os.path.isfile(file):
        # if file not provided, or it doesn't exist, use your voice
        print("Please Start talking for gender recognition ")
        # put the file name here
        file = "test.wav"
        # record the file (start talking)
        record_to_file(file)
    # extract features and reshape it
    features = extract_feature(file, mel=True).reshape(1, -1)
    # predict the gender!
    male_prob = model.predict(features)[0][0]
    female_prob = 1 - male_prob
    gender = "male" if male_prob > female_prob else "female"
    # show the result!
    print("Voice is of:", gender)
    print(f"Probabilities:     Male: {male_prob*100:.2f}%    Female: {female_prob*100:.2f}%")

# if __name__ == "__main__":
#     import argparse
#     import tkinter as tk
#     from tkinter import filedialog, messagebox
#     import os
#     from utils import create_model
#     from PIL import Image, ImageDraw, ImageFont
#     import playsound
#     import threading

# # Create a Tkinter root window
#     root = tk.Tk()
#     root.withdraw()  # Hide the root window

# # Display a messagebox to choose between selecting a file or speaking
#     response = messagebox.askquestion("File Selection", "Do you want to select a WAV file?")

#     if response == "yes":
#     # Open a file dialog to select the WAV file
#         file_path = filedialog.askopenfilename(filetypes=[("WAV files", "*.wav")])

#     # Check if a file was selected
#         if file_path:
#             print("Selected file:", file_path)
#         else:
#             print("No file selected.")
#             exit()

#         if not os.path.isfile(file_path):
#             print("Invalid file:", file_path)
#             print("Please talk")
#             file = "test.wav"
#             record_to_file(file)
#         else:
#             file = file_path

#     else:
#         print("Winter is one of the most important seasons in India. It is a part of the four seasons that occur in India. Winters are the coolest season that starts from December and last till March. The peak time when winter is experienced the most in December and January. In India, winters hold great importance. In addition, the essence it has is admired by many people. Winters give you the time to indulge in various activities like snowball fighting, building snowmen, ice hockey and more. It is a great time for kids to enjoy their vacations and get cozy in their blankets.")
#         file = "test.wav"
#         record_to_file(file)

# # Confirm with the user to play the audio file
#     confirmation = messagebox.askquestion("Play Audio", "Do you want to listen?")

#     def play_audio():
#     # Play the selected audio file
#         playsound.playsound(file)

#     if confirmation == "yes":
#     # Show a messagebox while playing the audio file
#         messagebox.showinfo("Listening", "You are listening to the selected audio file and the gender detection analysis is running in the background press ok.")

#     # Start a new thread to play the audio file
#         audio_thread = threading.Thread(target=play_audio)
#         audio_thread.start()            

# # Load the trained weights
#     model = create_model()
#     model.load_weights(r"C:\Users\Nikkhil_Bisht\Desktop\Coding\Voice Detection\gender-recognition-by-voice-master\FINAL\results\model.h5")

# # Rest of your code
# # extract features and reshape it
#     features = extract_feature(file, mel=True).reshape(1, -1)
# # predict the gender!
#     male_prob = model.predict(features)[0][0]
#     female_prob = 1 - male_prob
#     gender = "male" if male_prob > female_prob else "female"

# # Display the result image
#     if gender == "male":
#         image_path = r"C:\Users\Nikkhil_Bisht\Desktop\Coding\Voice Detection\gender-recognition-by-voice-master\FINAL\male.png" # Path to the male image file
#     else:
#         image_path = r"C:\Users\Nikkhil_Bisht\Desktop\Coding\Voice Detection\gender-recognition-by-voice-master\FINAL\female.png"  # Path to the female image file

#     result_image = Image.open(image_path)

# # Draw the text on the image
#     draw = ImageDraw.Draw(result_image)
#     title_text = "Voice-based Gender Detection Mini Project by Nikhil Bisht"
#     gender_text = f"Gender: {gender}\nProbabilities: Male: {male_prob*100:.2f}% Female: {female_prob*100:.2f}%"
#     font = ImageFont.truetype("arial.ttf", size=20)

#     title_text_bbox = draw.textbbox((0, 0), title_text, font=font)
#     gender_text_bbox = draw.textbbox((0, 0), gender_text, font=font)

#     title_text_position = (10, 10)
#     gender_text_position = (10, title_text_bbox[3] + 10)

#     draw.rectangle(title_text_bbox, fill='white')
#     draw.rectangle(gender_text_bbox, fill='white')

#     draw.text(title_text_position, title_text, fill='black', font=font)
#     draw.text(gender_text_position, gender_text, fill='black', font=font)

# # Show the result image
#     result_image.show()


