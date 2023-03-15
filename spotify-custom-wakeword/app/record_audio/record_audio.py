import sounddevice as sd 
from scipy.io.wavfile import write 


def record_audio_and_save(save_path):
    """
    This function will record the audio and save it in file path
    ----------
    save_path: str
        Where to save the wav file
    """
    input("To start recording Wake Word press Enter: ")
    fs = 44100
    seconds = 2

    myrecording = sd.rec(int(seconds * fs), samplerate=fs, channels=1)
    sd.wait()
    write(save_path +"prediction" + ".wav", fs, myrecording)