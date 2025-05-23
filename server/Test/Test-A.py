import pytest
import json
import base64
import numpy as np
import sys
import os
from unittest.mock import patch, MagicMock
from datetime import datetime
from io import BytesIO
from werkzeug.datastructures import FileStorage

# Add the parent directory to the Python path to find the modules
# This allows importing from the project root
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

# Import the main application
from Application.EmoSenseApp import EmoSenseApp


@pytest.fixture
def app():
    """Fixture to create an app instance with mocked dependencies."""
    # Use string paths for patching to avoid import issues during test discovery
    with patch('Application.EmoSenseApp.EmotionAnalysis') as mock_analyzer, \
            patch('Application.EmoSenseApp.Patient') as mock_patient, \
            patch('Application.EmoSenseApp.Therapist') as mock_therapist:
        # Configure the mock EmotionAnalysis behavior
        mock_analyzer_instance = mock_analyzer.return_value
        mock_analyzer_instance.model = MagicMock()
        mock_analyzer_instance.process_audio_chunk.return_value = {
            'emotion': 'happy',
            'confidence': 0.85,
            'timestamp': str(datetime.now())
        }
        mock_analyzer_instance.analyze_prerecorded_audio.return_value = {
            'status': 'success',
            'emotion_data': [
                {'emotion': 'happy', 'confidence': 0.85, 'timestamp': '2025-03-16T10:00:00'},
                {'emotion': 'sad', 'confidence': 0.75, 'timestamp': '2025-03-16T10:00:05'}
            ],
            'summary': {
                'dominant_emotion': 'happy',
                'emotion_percentages': {'happy': 60, 'sad': 40},
                'summary_description': 'Patient showed mostly positive emotions'
            }
        }
        mock_analyzer_instance.generate_session_summary.return_value = {
            'status': 'success',
            'dominant_emotion': 'happy',
            'emotion_percentages': {'happy': 60, 'sad': 40}
        }
        mock_analyzer_instance.generate_summary_description.return_value = 'Patient showed mostly positive emotions'
        mock_analyzer_instance.save_session_analysis.return_value = {'analysis_id': 1, 'success': True}

        # Configure mock Patient behavior
        mock_patient_instance = mock_patient.return_value
        mock_patient_instance.get_patient_by_email.return_value = {
            'email': 'test@example.com',
            'full_name': 'Test Patient'
        }
        mock_patient_instance.add_patient.return_value = 'test@example.com'

        # Create and configure the app
        app_instance = EmoSenseApp()
        app_instance.analyzer = mock_analyzer_instance
        app_instance.patient_manager = mock_patient_instance

        # Configure the test client
        app_instance.app.config['TESTING'] = True
        app_instance.socketio.server = MagicMock()

        return app_instance


@pytest.fixture
def client(app):
    """Fixture for the test client."""
    return app.app.test_client()


@pytest.fixture
def socket_client(app):
    """Fixture for socket testing that directly calls the internal methods."""

    class MockSocketClient:
        def __init__(self, app):
            self.app = app
            self.emit_responses = {}
            self.app.socketio.emit = MagicMock(side_effect=self._mock_emit)

        def _mock_emit(self, event, data, **kwargs):
            self.emit_responses[event] = data

        def emit(self, event, data):
            """Simulate emitting a socket event by directly calling the internal methods."""
            # Skip trying to use socketio's event handlers and directly call the app's methods
            if event == 'start_analysis':
                # Extract required fields
                session_id = data.get('session_id')
                patient_email = data.get('patient_email')
                patient_fullname = data.get('patient_fullname', '')
                patient_contact = data.get('patient_contact', '')
                therapist_id = data.get('therapist_id')

                # Check patient exists or create
                if patient_email:
                    patient = self.app.patient_manager
                    patient.email = patient_email
                    patient.full_name = patient_fullname
                    patient.contact = patient_contact

                    test_var = patient.get_patient_by_email(patient_email)
                    if not test_var or "email" not in test_var:
                        print(f"{patient_email} {patient_fullname} {patient_contact} {therapist_id}")
                        patient.add_patient()

                # Start the session
                success = self.app._start_analysis_session(session_id, patient_email, therapist_id)
                if success:
                    self.app.socketio.emit('analysis_started', {
                        'status': 'success',
                        'session_id': session_id,
                        'message': 'Real-time analysis session started'
                    })
                else:
                    self.app.socketio.emit('error', {'message': 'Session already exists'})

            elif event == 'audio_chunk':
                session_id = data.get('session_id')
                audio_data = data.get('audio_data')

                if session_id and audio_data:
                    result = self.app._process_audio_chunk(session_id, audio_data)
                    if result:
                        self.app.socketio.emit('analysis_result', {
                            'session_id': session_id,
                            'result': result
                        })
                else:
                    self.app.socketio.emit('error', {'message': 'Missing session_id or audio data'})

            elif event == 'end_analysis':
                session_id = data.get('session_id')

                if session_id:
                    summary = self.app._end_analysis_session(session_id)
                    if summary:
                        self.app.socketio.emit('analysis_ended', {
                            'status': 'success',
                            'session_id': session_id,
                            'summary': summary
                        })
                    else:
                        self.app.socketio.emit('error', {'message': 'Failed to generate session summary'})
                else:
                    self.app.socketio.emit('error', {'message': 'Missing session_id'})

        def get_response(self, event):
            """Get the response for a specific event."""
            return self.emit_responses.get(event)

    return MockSocketClient(app)


class TestPrerecordedAudio:
    """Test cases for prerecorded audio analysis."""

    @patch('werkzeug.utils.secure_filename',
           return_value=r'C:\Users\Owais\PycharmProjects\EmoSense-server\ravdess_data\Actor_01\03-01-01-01-01-01-01.wav')
    @patch('os.path.join',
           return_value=r'C:\Users\Owais\PycharmProjects\EmoSense-server\ravdess_data\Actor_01\03-01-01-01-01-01-01.wav')
    @patch('os.path.exists', return_value=True)
    @patch('os.remove')
    def test_analyze_file_success(self, mock_remove, mock_exists, mock_join, mock_secure, client, app):
        """Test successful analysis of a prerecorded audio file."""
        # Create a mock file
        mock_audio = FileStorage(
            stream=BytesIO(b'dummy audio data'),
            filename=r'../ravdess_data/Actor_01/03-01-01-01-01-01-01.wav',
            content_type='audio/wav',
        )

        # Create form data
        data = {
            'audio': mock_audio,
            'patientEmail': 'test@example.com',
            'patientFullname': 'Test Patient',
            'patientContact': '1234567890',
            'therapistId': '1'
        }

        # Make the request
        response = client.post('/api/analyze/file', data=data, content_type='multipart/form-data')

        # Check response
        assert response.status_code == 200
        result = json.loads(response.data)
        assert result['status'] == 'success'
        assert 'emotion_data' in result
        assert result['summary']['dominant_emotion'] == 'happy'

        # Verify analyzer was called with correct parameters
        app.analyzer.analyze_prerecorded_audio.assert_called_once()

    def test_analyze_file_missing_data(self, client):
        """Test analysis with missing audio file."""
        # Missing audio file
        response = client.post('/api/analyze/file', data={
            'patientEmail': 'test@example.com',
            'patientFullname': 'Test Patient',
            'patientContact': '1234567890',
            'therapistId': '1'
        }, content_type='multipart/form-data')

        assert response.status_code == 400
        result = json.loads(response.data)
        assert 'error' in result


class TestRealtimeAudio:
    """Test cases for realtime audio analysis."""

    def test_realtime_session_lifecycle(self, socket_client, app):
        """Test the complete lifecycle of a realtime analysis session."""
        # Configure the mock analyzer
        mock_analyzer = app.analyzer
        mock_analyzer.process_audio_chunk.return_value = {
            'emotion': 'happy',
            'confidence': 0.85,
            'timestamp': str(datetime.now())
        }

        # Mock the audio analyzer's sample rate if needed
        mock_analyzer.audio_analyzer = MagicMock()
        mock_analyzer.audio_analyzer.sample_rate = 22050

        # 1. Start a session
        session_data = {
            'session_id': 'test-session',
            'patient_email': 'test@example.com',
            'patient_fullname': 'Test Patient',
            'patient_contact': '1234567890',
            'therapist_id': 1
        }

        # Start the session
        socket_client.emit('start_analysis', session_data)

        # Verify session was created
        assert 'test-session' in app.active_sessions
        mock_analyzer.start_session.assert_called_once()

        # 2. Process audio chunk - create valid test data
        sample_audio = np.random.randint(-32768, 32767, 8000, dtype=np.int16)
        audio_b64 = base64.b64encode(sample_audio.tobytes()).decode('utf-8')

        # Process the chunk
        with patch('librosa.resample') as mock_resample:
            # Mock librosa to return the input unchanged
            mock_resample.return_value = np.zeros(8000, dtype=np.float32)

            chunk_data = {
                'session_id': 'test-session',
                'audio_data': audio_b64
            }
            socket_client.emit('audio_chunk', chunk_data)

            # Verify processing happened
            mock_analyzer.process_audio_chunk.assert_called_once()

            # Verify the call arguments
            args = mock_analyzer.process_audio_chunk.call_args[0]
            assert isinstance(args[0], np.ndarray)
            assert args[0].dtype == np.float32

        # 3. End the session
        end_data = {'session_id': 'test-session'}
        socket_client.emit('end_analysis', end_data)

        # Verify cleanup
        assert 'test-session' not in app.active_sessions
        mock_analyzer.end_session.assert_called_once()
        mock_analyzer.generate_session_summary.assert_called_once()
        mock_analyzer.save_session_analysis.assert_called_once_with('realtime')