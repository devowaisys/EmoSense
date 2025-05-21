import queue

import librosa
import numpy as np
import sounddevice as sd


class AudioAnalysis:
    def __init__(self):
        self.sample_rate = 22050
        self.chunk_duration = 3  # seconds
        self.chunk_samples = int(self.sample_rate * self.chunk_duration)
        self.audio_queue = queue.Queue()
        self.is_recording = False

    def extract_features(self, audio_data, sr=None):
        """Extract advanced audio features for deep learning"""
        try:
            # If audio_data is a file path
            if isinstance(audio_data, str):
                y, sr = librosa.load(audio_data, duration=3, offset=0.5)
            else:
                y = audio_data
                if sr is None:
                    sr = self.sample_rate

            # More comprehensive feature extraction for deep learning
            features = []

            # 1. MFCC
            mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)
            mfcc_scaled = np.mean(mfcc.T, axis=0)
            features.extend(mfcc_scaled)

            # 2. Pitch
            pitch, _ = librosa.piptrack(y=y, sr=sr)
            pitch_features = [
                np.mean(pitch),
                np.std(pitch),
                np.max(pitch),
                np.min(pitch)
            ]
            features.extend(pitch_features)

            # 3. Energy and Spectral Features
            rmse = librosa.feature.rms(y=y)
            spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
            spectral_bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)
            spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)

            spectral_features = [
                np.mean(rmse),
                np.mean(spectral_centroid),
                np.mean(spectral_bandwidth),
                np.mean(spectral_rolloff)
            ]
            features.extend(spectral_features)

            # 4. Zero Crossing Rate
            zcr = librosa.feature.zero_crossing_rate(y)
            features.append(np.mean(zcr))

            return np.array(features)

        except Exception as e:
            print(f"Error extracting features: {str(e)}")
            return None

    def audio_callback(self, indata, frames, time, status):
        """Callback for audio stream"""
        if status:
            print(status)
        if self.is_recording:
            self.audio_queue.put(indata.copy())

    def start_audio_stream(self):
        """Start the audio stream and return the stream object"""
        return sd.InputStream(
            channels=1,
            samplerate=self.sample_rate,
            callback=self.audio_callback,
            blocksize=self.chunk_samples
        )