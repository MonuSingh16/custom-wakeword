import sounddevice as sd # Record sound device and write as numpy array
from scipy.io.wavfile import write # Convert the numpy array to wave file

def record_audio_and_save(save_path, n_times=10):
    """
    This function will run `n_times` and everytime you press Enter you have to speak the wake word
    Parameters
    ----------
    n_times: int, default=50
        The function will run n_times default is set to 50.
    save_path: str
        Where to save the wav file which is generated in every iteration.
    """

    input("To start recording Wake Word press Enter: ")
    for i in range(n_times):
        fs = 44100
        seconds = 2

        myrecording = sd.rec(int(seconds * fs), samplerate=fs, channels=1)
        sd.wait()
        write(save_path + str(i) + ".wav", fs, myrecording)
        input(f"Press to record next or to stop press ctrl + C ({i + 1}/{n_times}): ")

# Step 1: Record yourself saying the Wake Word
print("Recording the Wake Word:\n")