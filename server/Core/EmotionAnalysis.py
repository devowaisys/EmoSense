import os
import queue
import time
import warnings
from collections import defaultdict
from datetime import datetime

import joblib
import keyboard
import librosa
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm

from Operation.Analysis import Analysis
from Core.AudioAnalysis import AudioAnalysis
from Operation.Patient import Patient
from Operation.Therapist import Therapist


# Suppress TensorFlow warnings
tf.get_logger().setLevel('ERROR')
warnings.filterwarnings('ignore')



class EmotionAnalysis:
    def __init__(self, patient=None, therapist=None):
        """Initialize EmotionAnalysis with Patient and Therapist objects"""
        self.emotions = {
            '01': 'neutral', '02': 'calm', '03': 'happy', '04': 'sad',
            '05': 'angry', '06': 'fearful', '07': 'disgust', '08': 'surprised'
        }
        self.model = None
        self.label_encoder = LabelEncoder()
        self.session_emotions = []
        self.session_start_time = None
        self.session_end_time = None
        self.audio_analyzer = AudioAnalysis()

        # Store objects for aggregation
        self.patient = patient
        self.therapist = therapist
        self.patient_email = patient.email if patient else None
        self.therapist_id = therapist.therapist_id if therapist else None

    def extract_enhanced_features(self, audio_input, sr=22050):
        """Use the AudioAnalysis class method to maintain consistency"""
        return self.audio_analyzer.extract_enhanced_features(audio_input, sr)

    def start_session(self):
        """Initialize a new session"""
        self.session_emotions = []
        self.session_start_time = datetime.now()

    def end_session(self):
        """End the current session"""
        self.session_end_time = datetime.now()

    def generate_session_summary(self):
        """Generate a structured summary of the therapy session"""
        if not self.session_emotions:
            return {"status": "error", "message": "No emotional data was recorded during this session."}

        duration = (self.session_end_time - self.session_start_time).total_seconds() / 60
        emotion_counts = defaultdict(int)
        emotion_confidences = defaultdict(list)
        emotional_transitions = []
        all_confidence_scores = []

        for i, entry in enumerate(self.session_emotions):
            emotion = entry['predicted_emotion']
            emotion_counts[emotion] += 1

            if i > 0:
                prev_emotion = self.session_emotions[i - 1]['predicted_emotion']
                if prev_emotion != emotion:
                    emotional_transitions.append({
                        'from': prev_emotion,
                        'to': emotion,
                        'timestamp': entry['timestamp']
                    })

            for emotion_name, confidence in entry['confidence_scores'].items():
                emotion_confidences[emotion_name].append(confidence)
                all_confidence_scores.append(confidence)

        total_chunks = len(self.session_emotions)
        emotion_percentages = {emotion: (count / total_chunks) * 100 for emotion, count in emotion_counts.items()}
        avg_confidences = {emotion: np.mean(confidences) for emotion, confidences in emotion_confidences.items()}

        confidence_stats = {
            "overall_average": round(np.mean(all_confidence_scores), 2),
            "max": round(np.max(all_confidence_scores), 2),
            "min": round(np.min(all_confidence_scores), 2),
            "total_predictions": len(all_confidence_scores)
        }

        dominant_emotions = {emotion: percentage for emotion, percentage in emotion_percentages.items() if
                             percentage > 20}
        high_confidence_emotions = {emotion: conf for emotion, conf in avg_confidences.items() if conf > 0.7}

        return {
            "status": "success",
            "session_info": {
                "start_time": self.session_start_time.strftime('%Y-%m-%d %H:%M:%S'),
                "end_time": self.session_end_time.strftime('%Y-%m-%d %H:%M:%S'),
                "duration_minutes": round(duration, 2)
            },
            "confidence_metrics": confidence_stats,
            "emotion_analysis": {
                "total_chunks_analyzed": total_chunks,
                "emotion_counts": dict(emotion_counts),
                "emotion_percentages": {k: round(v, 2) for k, v in emotion_percentages.items()},
                "average_confidences": {k: round(v, 2) for k, v in avg_confidences.items()},
                "dominant_emotions": {k: round(v, 2) for k, v in dominant_emotions.items()},
                "high_confidence_emotions": {k: round(v, 2) for k, v in high_confidence_emotions.items()}
            },
            "emotional_progression": {
                "transitions": emotional_transitions,
                "total_transitions": len(emotional_transitions),
                "transition_rate": round(len(emotional_transitions) / duration, 2) if duration > 0 else 0
            },
            "raw_data": {
                "emotional_timeline": [
                    {
                        "timestamp": entry['timestamp'],
                        "emotion": entry['predicted_emotion'],
                        "confidence_scores": {k: round(v, 2) for k, v in entry['confidence_scores'].items()}
                    }
                    for entry in self.session_emotions
                ]
            }
        }

    def check_data_path(self, data_path=r"C:\Users\Owais\PycharmProjects\EmoSense-server\ravdess_data"):
        """Check if the data path exists and contains actor folders"""
        if not os.path.exists(data_path):
            print(f"Error: Directory '{data_path}' not found!")
            return False

        actor_folders = [f for f in os.listdir(data_path) if f.startswith('Actor_')]
        if not actor_folders:
            print(f"Error: No actor folders found in '{data_path}'!")
            return False

        total_wav_files = sum(len([f for f in os.listdir(os.path.join(data_path, folder)) if f.endswith('.wav')])
                              for folder in actor_folders)
        print(f"Found {len(actor_folders)} actors and {total_wav_files} audio files in the dataset.")
        return True

    def prepare_training_data(self, data_path):
        """Prepare training data using enhanced features"""
        features, labels = [], []
        actor_folders = sorted([f for f in os.listdir(data_path) if f.startswith('Actor_')])

        print(f"Processing {len(actor_folders)} actors...")

        for actor_folder in tqdm(actor_folders, desc="Processing actors"):
            actor_path = os.path.join(data_path, actor_folder)
            wav_files = [f for f in os.listdir(actor_path) if f.endswith('.wav')]

            for filename in tqdm(wav_files, desc=f"Processing {actor_folder}", leave=False):
                try:
                    emotion_code = filename.split("-")[2]
                    if emotion_code in self.emotions:
                        emotion = self.emotions[emotion_code]
                        file_path = os.path.join(actor_path, filename)
                        feature_vector = self.extract_enhanced_features(file_path)

                        if feature_vector is not None:
                            features.append(feature_vector)
                            labels.append(emotion)
                except Exception as e:
                    print(f"Error processing {filename}: {str(e)}")
                    continue

        if not features:
            raise ValueError("No features could be extracted from the dataset!")

        labels_encoded = self.label_encoder.fit_transform(labels)
        print(f"\nExtracted features from {len(features)} audio files")

        unique_labels, counts = np.unique(labels, return_counts=True)
        print("Emotion distribution:")
        for emotion, count in zip(unique_labels, counts):
            print(f"{emotion}: {count} samples")

        return np.array(features), labels_encoded

    def create_improved_model(self, input_shape, num_classes):
        """Create improved model architecture with better regularization"""
        model = tf.keras.Sequential([
            tf.keras.layers.Input(shape=(input_shape,)),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dense(256, activation='relu'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dropout(0.4),
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dropout(0.3),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(num_classes, activation='softmax')
        ])

        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.0005),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        return model

        # Combined and optimized training method

    def train_model(self, X, y):
        """Enhanced training with better data preprocessing and validation"""
        try:
            if len(X) == 0 or len(y) == 0:
                raise ValueError("Empty training data!")

            # Better train-test split with stratification
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y
            )

            # Improved normalization - fit on training data only
            self.scaler_mean = np.mean(X_train, axis=0)
            self.scaler_std = np.std(X_train, axis=0)
            self.scaler_std[self.scaler_std == 0] = 1

            X_train_scaled = (X_train - self.scaler_mean) / self.scaler_std
            X_test_scaled = (X_test - self.scaler_mean) / self.scaler_std

            print("Training Enhanced Neural Network...")
            self.model = self.create_improved_model(X_train_scaled.shape[1], len(np.unique(y)))

            # Enhanced callbacks
            callbacks = [
                tf.keras.callbacks.EarlyStopping(
                    monitor='val_accuracy', patience=15, restore_best_weights=True, verbose=1
                ),
                tf.keras.callbacks.ReduceLROnPlateau(
                    monitor='val_loss', factor=0.5, patience=8, min_lr=0.00001, verbose=1
                )
            ]

            # Class weights for imbalanced data
            from sklearn.utils.class_weight import compute_class_weight
            class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
            class_weight_dict = dict(enumerate(class_weights))

            history = self.model.fit(
                X_train_scaled, y_train,
                validation_data=(X_test_scaled, y_test),
                epochs=150, batch_size=16,
                callbacks=callbacks,
                class_weight=class_weight_dict,
                verbose=1
            )

            # Evaluate model
            test_loss, test_accuracy = self.model.evaluate(X_test_scaled, y_test, verbose=0)
            print(f"\nTest Accuracy: {test_accuracy:.2%}")

            # Classification report
            y_pred = self.model.predict(X_test_scaled)
            y_pred_classes = np.argmax(y_pred, axis=1)

            print("\nClassification Report:")
            report = classification_report(y_test, y_pred_classes, target_names=self.label_encoder.classes_,
                                           output_dict=True)

            for emotion in self.label_encoder.classes_:
                if emotion in report:
                    print(f"{emotion}: Precision={report[emotion]['precision']:.2f}, "
                          f"Recall={report[emotion]['recall']:.2f}, F1={report[emotion]['f1-score']:.2f}")

            return test_accuracy

        except Exception as e:
            print(f"Error training model: {str(e)}")
            return None

    def process_audio_chunk(self, audio_chunk):
        """Enhanced audio processing with better preprocessing"""
        try:
            if self.model is None:
                raise ValueError("Model not trained! Please train the model first.")

            # Convert to 1D if needed
            if len(audio_chunk.shape) > 1:
                audio_data = np.mean(audio_chunk, axis=1)
            else:
                audio_data = audio_chunk.flatten()

            # Skip if audio is too short or too quiet
            if len(audio_data) < 1024 or np.max(np.abs(audio_data)) < 0.01:
                return None

            # Extract features
            features = self.extract_enhanced_features(audio_data)
            if features is None:
                return None

            # Apply normalization
            if hasattr(self, 'scaler_mean') and hasattr(self, 'scaler_std'):
                features = (features - self.scaler_mean) / self.scaler_std
            else:
                features = (features - np.mean(features)) / (np.std(features) + 1e-8)

            features = features.reshape(1, -1)

            # Get prediction
            probabilities = self.model.predict(features, verbose=0)[0]
            predicted_class_index = np.argmax(probabilities)
            predicted_emotion = self.label_encoder.classes_[predicted_class_index]

            max_confidence = np.max(probabilities)
            if max_confidence < 0.3:
                predicted_emotion = 'neutral'

            result = {
                'predicted_emotion': predicted_emotion,
                'confidence_scores': {emotion: float(prob) for emotion, prob in
                                      zip(self.label_encoder.classes_, probabilities)},
                'max_confidence': float(max_confidence),
                'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
            }

            self.session_emotions.append(result)
            return result

        except Exception as e:
            print(f"Error processing audio chunk: {str(e)}")
            return None

    def save_model(self, path=r'C:\Users\Owais\PycharmProjects\EmoSense-server\Core\therapy_emotion_model.h5'):
        """Save the trained TensorFlow model"""
        try:
            if self.model is None:
                raise ValueError("No trained model to save!")

            self.model.save(path)
            joblib.dump(self.label_encoder, r'C:\Users\Owais\PycharmProjects\EmoSense-server\Core\label_encoder.joblib')
            print(f"Model saved successfully to {path}")
        except Exception as e:
            print(f"Error saving model: {str(e)}")

    def load_model(self, path=r'C:\Users\Owais\PycharmProjects\EmoSense-server\Core\therapy_emotion_model.h5'):
        """Load a trained TensorFlow model"""
        try:
            if not os.path.exists(path):
                raise FileNotFoundError(f"Model file not found: {path}")

            self.model = tf.keras.models.load_model(path)
            encoder_path = r'C:\Users\Owais\PycharmProjects\EmoSense-server\Core\label_encoder.joblib'

            if os.path.exists(encoder_path):
                self.label_encoder = joblib.load(encoder_path)

            print("Model loaded successfully")
        except Exception as e:
            print(f"Error loading model: {str(e)}")
            self.model = None

    def _analyze_audio_with_chunks(self, audio_data, sr, chunk_duration=3.0, overlap=0.5):
        """Common method for chunked audio analysis"""
        chunk_samples = int(chunk_duration * sr)
        hop_samples = int(chunk_samples * (1 - overlap))
        results = []

        for i in range(0, len(audio_data) - chunk_samples, hop_samples):
            chunk = audio_data[i:i + chunk_samples]
            if np.max(np.abs(chunk)) > 0.01:  # Skip quiet segments
                result = self.process_audio_chunk(chunk)
                if result:
                    results.append(result)

        return results

    def start_recording(self):
        """Start recording and analyzing audio in real-time"""
        if self.model is None:
            return {'status': 'error', 'message': "Model not trained! Please train the model first."}

        try:
            self.audio_analyzer.is_recording = True
            self.start_session()
            self.audio_analyzer.audio_queue = queue.Queue()
            results = []

            def on_press(event):
                self.audio_analyzer.is_recording = False
                return False

            keyboard.on_press(on_press)
            stream = self.audio_analyzer.start_audio_stream()

            print("\nStarting therapy session analysis...")
            print("Press any key to stop recording...")

            with stream:
                while self.audio_analyzer.is_recording:
                    try:
                        audio_chunk = self.audio_analyzer.audio_queue.get(timeout=0.5)
                        result = self.process_audio_chunk(audio_chunk)

                        if result:
                            results.append(result)
                            print(f"\nTimestamp: {result['timestamp']}")
                            print(f"Predicted Emotion: {result['predicted_emotion']}")
                            print("Confidence Scores:")
                            for emotion, score in result['confidence_scores'].items():
                                print(f"{emotion}: {score:.2f}")
                            print("-" * 50)

                    except queue.Empty:
                        continue
                    except Exception as e:
                        print(f"\nError processing audio: {str(e)}")
                        break

            self.end_session()
            summary = self.generate_session_summary()

            if summary['status'] == 'success':
                summary_description = self.generate_summary_description(summary)
                summary['summary_description'] = summary_description
                save_result = self.save_session_analysis('realtime')

                if save_result['status'] == 'error':
                    print(f"\nWarning: Failed to save session: {save_result['message']}")
                else:
                    print(f"\nSession saved with ID: {save_result['analysis_id']}")

                return {
                    'status': 'success',
                    'message': 'Real-time analysis completed successfully',
                    'summary': summary,
                    'emotions': results,
                    'save_result': save_result
                }
            else:
                return {'status': 'error', 'message': 'Failed to generate session summary', 'summary': summary}

        except Exception as e:
            return {'status': 'error', 'message': f'Error during real-time analysis: {str(e)}'}
        finally:
            self.audio_analyzer.is_recording = False
            keyboard.unhook_all()
            while not self.audio_analyzer.audio_queue.empty():
                try:
                    self.audio_analyzer.audio_queue.get_nowait()
                except queue.Empty:
                    break

    def analyze_prerecorded_audio(self, audio_path):
        """Analyze prerecorded audio file"""
        try:
            if self.model is None:
                raise ValueError("Model not trained! Please train the model first.")
            if not os.path.exists(audio_path):
                raise FileNotFoundError(f"Audio file not found: {audio_path}")

            self.start_session()
            y, sr = librosa.load(audio_path, sr=22050, duration=None)
            results = self._analyze_audio_with_chunks(y, sr)
            self.end_session()

            if not results:
                return {'status': 'error', 'message': 'No valid audio segments found for analysis'}

            summary = self.generate_session_summary()
            summary_description = self.generate_summary_description(summary)
            summary['summary_description'] = summary_description
            save_result = self.save_session_analysis("pre-recorded")

            return {
                'status': 'success',
                'message': 'Audio analysis completed successfully',
                'summary': summary,
                'emotions': results,
                'save_result': save_result
            }

        except Exception as e:
            return {'status': 'error', 'message': str(e)}

    def save_session_analysis(self, analysis_mode):
        """Save analysis results to database"""
        if not self.session_emotions or not self.session_start_time or not self.session_end_time:
            return {'status': 'error', 'message': 'No session data available to save'}

        if not self.patient_email or not self.therapist_id:
            return {'status': 'error', 'message': 'Patient email and therapist ID are required'}

        try:
            session_summary = self.generate_session_summary()
            if session_summary['status'] != 'success':
                raise ValueError("Failed to generate session summary")

            session_duration = self.session_end_time - self.session_start_time
            summary_description = self.generate_summary_description(session_summary)

            dominant_emotions = session_summary['emotion_analysis']['dominant_emotions']
            dominant_emotion = max(dominant_emotions.items(), key=lambda x: x[1])[0] if dominant_emotions else 'neutral'

            analysis = Analysis(
                therapist_id=self.therapist_id,
                patient_email=self.patient_email,
                analysis_mode=analysis_mode,
                analysis_duration=session_duration,
                dominant_emotion=dominant_emotion,
                analysis_summary=summary_description,
                date=self.session_start_time.date(),
                session_duration=session_duration,
                session_start=self.session_start_time,
                session_end=self.session_end_time
            )

            analysis_id = analysis.add_analysis()
            return {'status': 'success', 'analysis_id': analysis_id, 'message': 'Analysis saved successfully'}

        except Exception as e:
            return {'status': 'error', 'message': f'Failed to save analysis: {str(e)}'}

    def generate_summary_description(self, summary):
        """Generate brief narrative summary"""
        try:
            if summary['status'] != 'success':
                return "Unable to generate summary: No emotional data was recorded."

            dominant_emotions = summary['emotion_analysis']['dominant_emotions']
            if not dominant_emotions:
                return "No clear emotional patterns were detected in this audio."

            primary_emotion = max(dominant_emotions.items(), key=lambda x: x[1])
            emotion_name = primary_emotion[0]

            narratives = {
                'happy': "The speaker maintained a predominantly positive emotional state, expressing happiness and contentment.",
                'sad': "The speaker exhibited signs of sadness or melancholy, suggesting emotional distress.",
                'angry': "The speaker displayed significant signs of frustration or anger during the recording.",
                'neutral': "The speaker maintained a mostly neutral emotional state with minimal variation.",
                'calm': "The speaker demonstrated a consistently calm and composed demeanor.",
                'fearful': "The speaker showed signs of anxiety or fear throughout the recording.",
                'disgust': "The speaker exhibited strong negative reactions during the recording.",
                'surprised': "The speaker displayed heightened alertness or surprise through the recording."
            }

            return narratives.get(emotion_name,
                                  f"The speaker primarily expressed {emotion_name} throughout the recording.")

        except Exception as e:
            return f"Error generating summary description: {str(e)}"

    def clear_session_data(self):
        """Clear all session-related data"""
        self.session_emotions = []
        self.session_start_time = None
        self.session_end_time = None


def main():
    try:
        # Create EmotionAnalysis instance
        john_pat = Patient(email="john@email.com", full_name="John Muller", contact=12345)
        kevin = Therapist(therapist_id=1, email="owais@email.com", full_name="Owais", password="owais123")
        analyzer = EmotionAnalysis(patient=john_pat, therapist=kevin)

        # Check for pre-trained model
        model_path = r'C:\Users\Owais\PycharmProjects\EmoSense-server\Core\therapy_emotion_model.h5'
        if os.path.exists(model_path):
            print("Loading pre-trained model...")
            analyzer.load_model()
        else:
            print("No pre-trained model found. Training a new model...")

            # Check if dataset exists
            data_path = r"C:\Users\Owais\PycharmProjects\EmoSense-server\ravdess_data"
            if analyzer.check_data_path(data_path):
                # Prepare data and train model
                print("\nPreparing training data...")
                X, y = analyzer.prepare_training_data(data_path)
                print("\nTraining model...")
                test_accuracy = analyzer.train_model(X, y)

                if test_accuracy is not None:
                    print(f"\nModel training completed with test accuracy: {test_accuracy:.2%}")

                    # Save the trained model
                    print("\nSaving trained model...")
                    analyzer.save_model()
                else:
                    print("\nModel training failed. Please check your data and try again.")
                    return
            else:
                print("\nCannot train model: Dataset not found or invalid.")
                return

        while True:
            print("\nEmotion Analysis Options:")
            print("1. Real-time recording analysis")
            print("2. Analyze prerecorded audio file")
            print("3. Train new model")
            print("4. Exit")

            choice = input("\nEnter your choice (1-4): ").strip()

            if choice == '1':
                result = analyzer.start_recording()
                if result['status'] == 'success':
                    print("\nAnalysis Summary:")
                    print(result['summary']['summary_description'])
                else:
                    print(f"\nError: {result['message']}")

            elif choice == '2':
                audio_path = input("\nEnter the path to your audio file: ").strip()
                if os.path.exists(audio_path):
                    result = analyzer.analyze_prerecorded_audio(audio_path)
                    if result['status'] == 'success':
                        print("\nAnalysis Summary:")
                        print(result['summary']['summary_description'])
                    else:
                        print(f"\nError: {result['message']}")
                else:
                    print(f"Error: File not found at {audio_path}")

            elif choice == '3':
                data_path = input(
                    "\nEnter the path to your dataset (default: ravdess_data): ").strip() or "ravdess_data"
                if analyzer.check_data_path(data_path):
                    print("\nPreparing training data...")
                    X, y = analyzer.prepare_training_data(data_path)
                    print("\nTraining model...")
                    test_accuracy = analyzer.train_model(X, y)

                    if test_accuracy is not None:
                        print(f"\nModel training completed with test accuracy: {test_accuracy:.2%}")

                        # Save the trained model
                        save_option = input("\nDo you want to save this model? (y/n): ").strip().lower()
                        if save_option == 'y':
                            analyzer.save_model()
                    else:
                        print("\nModel training failed.")
                else:
                    print(f"\nDataset not found or invalid at path: {data_path}")

            elif choice == '4':
                print("\nExiting program...")
                break

            else:
                print("\nInvalid choice. Please try again.")

    except KeyboardInterrupt:
        print("\nProgram interrupted by user. Exiting...")
    except Exception as e:
        print(f"\nAn error occurred: {str(e)}")
    finally:
        keyboard.unhook_all()


if __name__ == "__main__":
    main()
