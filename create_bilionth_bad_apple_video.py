# %%
import numpy as np
import librosa
import cv2
import matplotlib.pyplot as plt
import os
from tqdm import tqdm

class BadApple():
    def __init__(self, input_video_path, frames_folder):
        self.input_video_path = input_video_path
        self.frames_folder = frames_folder
        self.max_freq = 360
        os.mkdir(frames_folder)
    
    def load_video(self):
        cap = cv2.VideoCapture(self.input_video_path)
        self.frameCount = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frameWidth = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frameHeight = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        print('Frames to generate:', self.frameCount)

        buf = np.empty((self.frameCount, frameHeight, frameWidth, 3), np.dtype('uint8'))

        fc = 0
        ret = True

        while (fc < self.frameCount  and ret):
            ret, buf[fc] = cap.read()
            fc += 1
            
        cap.release()
        return buf[..., 0]
    
    def generate_frequency_swipe(self):        
        frequencies = np.linspace(1, self.max_freq, self.max_freq).astype('int32')     
        time = np.linspace(0, 1, self.max_freq*2) # 1 second with 2*max_freq sample rate        
        frequency_swipe = []
        for frequency in frequencies:
                sine = np.sin(time*frequency*2*np.pi)
                frequency_swipe.append(sine)
        return np.array(frequency_swipe)

    def audio_from_frame(self, frame, frequency_swipe):
        _, binary_frame = cv2.threshold(frame, 127, 255, cv2.THRESH_BINARY)
        binary_frame = ((binary_frame/255-1)*(-1)).astype('int8')
        
        audio_frame = []
        for column in binary_frame.T:
            column = np.expand_dims(column, 1)
            column = np.repeat(column, self.max_freq*2, axis=1)
            audio_frame.append(np.sum(frequency_swipe*column, axis=0))
            
        audio_frame = np.concatenate(audio_frame)
        return audio_frame

    def save_output_frame(self, audio_frame, frame_name):
        spec = librosa.stft(audio_frame, n_fft=1024)
        fig, ax = plt.subplots(figsize=(8, 8), nrows=2, gridspec_kw={'height_ratios': [3, 1]})
        ax[0].imshow(np.abs(spec),
                interpolation='bilinear',
                aspect='auto',
                cmap='plasma',
                vmin=0,
                vmax=1467,
                extent=[0,480,360,0])
        ax[0].set(ylabel='Frequency [Hz]', xlabel='Time [s]')
        ax[1].plot(audio_frame)
        ax[1].set(ylabel='Amplitude', xlabel='Time [s]')
        plt.savefig(os.path.join(self.frames_folder, frame_name), dpi=100)
        fig.clear()
        plt.close(fig)
        
    def create_video(self):
        self.video_name = 'output_video.avi'        
        
        filenames = [f'bad_apple_frame{i}.png' for i in range(1, self.frameCount+1)]

        frame = cv2.imread(os.path.join(self.frames_folder, filenames[0]))
        height, width, _ = frame.shape

        video = cv2.VideoWriter(self.video_name, 0, 30, (width,height))
        for name in os.listdir(self.frames_folder):
            print(name)
            break

        print("Generating video...")
        for image in tqdm(filenames):
            video.write(cv2.imread(os.path.join(self.frames_folder, image)))

        cv2.destroyAllWindows()
        video.release()
        
    def play_video(self):
        cap = cv2.VideoCapture(self.video_name)
        cv2.namedWindow('BadApple on Sepctrogram')
        frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        for _ in range(frames):
            ret_val, frame = cap.read()
            cv2.imshow('BadApple on Sepctrogram', frame)
            if cv2.waitKey(1000//30) == 27:
                break  # esc to quit
        cv2.destroyAllWindows()
        
# Closes all the frames
cv2.destroyAllWindows()
if __name__=="__main__":    
    GoodApple = BadApple('Touhou - Bad Apple.mp4', frames_folder='frames')
    video_buf = GoodApple.load_video()
    frequency_swipe = GoodApple.generate_frequency_swipe()
    
    # for i, frame in tqdm(enumerate(video_buf), total=GoodApple.frameCount):        
    #     audio_frame = GoodApple.audio_from_frame(frame, frequency_swipe)
    #     GoodApple.save_output_frame(audio_frame, f'bad_apple_frame{i+1}')
        
        
    print('Done')
    GoodApple.create_video()
    # GoodApple.play_video()