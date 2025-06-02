
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

    def extract_enhanced_features(self, audio_input, sr=22050):
        """Enhanced feature extraction - handles both file paths and audio data"""
        try:
            # Handle both file paths and audio data
            if isinstance(audio_input, str):
                audio_data, sr = librosa.load(audio_input, sr=sr)
            else:
                audio_data = audio_input

            # Ensure audio is 1D
            if len(audio_data.shape) > 1:
                audio_data = np.mean(audio_data, axis=1)

            # Skip if audio is too short
            if len(audio_data) < 1024:
                return None

            features = []

            # 1. MFCC features (most important for emotion) - with error handling
            try:
                mfccs = librosa.feature.mfcc(y=audio_data, sr=sr, n_mfcc=13, window='hann')
            except:
                # Fallback without window parameter
                mfccs = librosa.feature.mfcc(y=audio_data, sr=sr, n_mfcc=13)

            features.extend([
                np.mean(mfccs, axis=1),
                np.std(mfccs, axis=1),
                np.max(mfccs, axis=1),
                np.min(mfccs, axis=1)
            ])

            # 2. Spectral features (critical for emotion detection)
            spectral_centroids = librosa.feature.spectral_centroid(y=audio_data, sr=sr)[0]
            spectral_rolloff = librosa.feature.spectral_rolloff(y=audio_data, sr=sr)[0]
            spectral_bandwidth = librosa.feature.spectral_bandwidth(y=audio_data, sr=sr)[0]
            zero_crossing_rate = librosa.feature.zero_crossing_rate(audio_data)[0]

            features.extend([
                [np.mean(spectral_centroids), np.std(spectral_centroids)],
                [np.mean(spectral_rolloff), np.std(spectral_rolloff)],
                [np.mean(spectral_bandwidth), np.std(spectral_bandwidth)],
                [np.mean(zero_crossing_rate), np.std(zero_crossing_rate)]
            ])

            # 3. Chroma features (for tonal content) - with error handling
            try:
                chroma = librosa.feature.chroma_stft(y=audio_data, sr=sr)
                features.append([np.mean(chroma), np.std(chroma)])
            except:
                # Fallback with default values
                features.append([0.5, 0.1])

            # 4. Tempo and rhythm (important for emotional state) - with error handling
            try:
                tempo, _ = librosa.beat.beat_track(y=audio_data, sr=sr)
                features.append([tempo])
            except:
                # Default tempo if beat tracking fails
                features.append([120.0])

            # 5. Energy and power features
            rms = librosa.feature.rms(y=audio_data)[0]
            features.append([np.mean(rms), np.std(rms)])

            # Flatten all features
            feature_vector = np.concatenate([np.array(f).flatten() for f in features])

            return feature_vector

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