from datetime import timedelta
import os
from typing import Set, Dict, Any, List, Optional, Tuple
import base64

import numpy as np
from flask import Flask, jsonify, request, Response
from flask_cors import CORS
from flask_jwt_extended import JWTManager, create_access_token, jwt_required, get_jwt
from werkzeug.utils import secure_filename
from flask_socketio import SocketIO, emit
from datetime import datetime
from Operation.Patient import Patient
from Operation.Therapist import Therapist
from Operation.Analysis import Analysis
from Core.EmotionAnalysis import EmotionAnalysis

class EmoSenseApp:
    ALLOWED_EXTENSIONS: Set[str] = {'wav', 'mp3'}
    def __init__(self):
        # Flask app configuration
        self.app: Flask = Flask(__name__)
        self.jwt: JWTManager = JWTManager(self.app)
        self.blacklist: Set[str] = set()

        # Configure JWT settings
        self._configure_jwt()

        # Configure CORS
        self._configure_cors()

        # Configure upload folder
        self.upload_folder: str = 'uploads'
        self.app.config['UPLOAD_FOLDER'] = self.upload_folder
        os.makedirs(self.upload_folder, exist_ok=True)

        # Initialize managers
        self.patient_manager: Patient = Patient() # composition of Patient class
        self.therapist_manager: Therapist = Therapist() # composition of Therapist class
        self.analysis_manager: Analysis = Analysis()  # composition of Core class
        self.analyzer: EmotionAnalysis = EmotionAnalysis()  # composition of EmotionAnalysis class

        # Initialize SocketIO and active sessions tracking
        self.socketio: SocketIO = SocketIO(self.app, cors_allowed_origins="*")
        self.active_sessions: Dict[str, Dict[str, Any]] = {}

        # Register routes and WebSocket handlers
        self._register_routes()
        self._register_websocket_handlers()
        self._register_before_request()

    def _configure_jwt(self) -> None:
        """Configure JWT settings for the application."""
        self.app.config['JWT_SECRET_KEY'] = 'EmoSense'
        self.app.config['JWT_BLACKLIST_ENABLED'] = True
        self.app.config['JWT_ACCESS_TOKEN_EXPIRES'] = timedelta(minutes=5)

    def _configure_cors(self) -> None:
        """Configure CORS settings for the application."""
        CORS(self.app, resources={
            r"/api/*": {
                "origins": ["http://localhost:3000"],
                "methods": ["GET", "POST", "PUT", "DELETE", "OPTIONS"],
                "allow_headers": ["Content-Type", "Authorization"],
                "supports_credentials": True
            }
        })

    def _register_routes(self) -> None:
        """Register all API routes."""
        # Basic routes
        self.app.route('/')(self.test)
        self.app.route('/health', methods=['GET'])(self.health_check)

        # Therapist routes
        self.app.route('/api/add_therapist', methods=['POST'])(self.add_therapist)
        self.app.route('/api/login_therapist', methods=['POST'])(self.login_therapist)
        self.app.route('/api/logout_therapist', methods=['POST'])(self.logout_therapist)
        self.app.route('/api/update_therapist/<int:therapist_id>', methods=['PUT'])(self.update_therapist)
        self.app.route('/api/delete_therapist/<int:therapist_id>', methods=['DELETE'])(self.delete_therapist)

        # Patient routes
        self.app.route('/api/add_patient', methods=['POST'])(self.add_patient)
        self.app.route('/api/get_patient/<patient_email>', methods=['GET'])(self.get_patient)
        self.app.route('/api/update_patient/<patient_prev_email>', methods=['PUT'])(self.update_patient)
        self.app.route('/api/delete_patient/<patient_email>', methods=['DELETE'])(self.delete_patient)
        self.app.route('/api/email_patient', methods=['POST'])(
            self.email_patient_route)  # New route for email functionality
        self.app.route('/api/get_analysis_by_therapist_id/<therapist_id>', methods=['GET'])(
            self.get_analysis_by_therapist_id)
        # self.app.route('/api/get_analysis_by_patient_email/<patient_email>', methods=['GET'])(self.get_analysis_by_patient_email)
        self.app.route("/api/update_analysis_text/<analysis_id>", methods=['PUT'])(self.update_analysis_text)
        # Core routes
        self.app.route('/api/analyze/file', methods=['POST'])(self.analyze_file)

    def allowed_file(self, filename: str) -> bool:
        """Check if the file extension is allowed."""
        return '.' in filename and filename.rsplit('.', 1)[1].lower() in self.ALLOWED_EXTENSIONS

    def parameters_checker(self, required_fields: Optional[List[str]] = None,
                           data: Optional[Dict[str, Any]] = None) -> List[str]:
        """Check for missing fields in the provided data."""
        if data is None:
            data = {}
        if required_fields is None:
            required_fields = []
        return [field for field in required_fields if field not in data]

    # Route handler methods
    def test(self) -> str:
        return 'EmoSense Server Up & Running!'

    def add_therapist(self) -> Tuple[Response, int]:
        data = request.get_json()
        missing_fields = self.parameters_checker(
            required_fields=['full_name', 'email', 'password'],
            data=data
        )
        print(data)
        if missing_fields:
            return jsonify({
                'message': f'Missing fields {", ".join(missing_fields)}',
                'success': False
            }), 400
        print(missing_fields)
        try:
            therapist = Therapist(
                email=data['email'],
                full_name=data['full_name'],
                password=data['password']
            )
            therapist_id = therapist.add_therapist()
            return jsonify({
                'id': therapist_id,
                'message': f'User {data["full_name"]} registered successfully',
                'success': True
            }), 200
        except Exception as error:
            return jsonify({
                'message': 'Failed to register user',
                'error': str(error),
                'success': False
            }), 400

    def login_therapist(self) -> Tuple[Response, int]:
        """Authenticate a therapist."""
        data = request.get_json()
        print(data)
        if not data:
            return jsonify({'message': 'No JSON payload', 'success': False}), 400

        missing_fields = self.parameters_checker(
            required_fields=['email', 'password'],
            data=data
        )
        if missing_fields:
            return jsonify({
                'message': f'Missing fields {", ".join(missing_fields)}',
                'success': False
            }), 400

        try:
            result = self.therapist_manager.get_therapist_by_email_and_password(
                data['email'],
                data['password']
            )

            if not result:
                return jsonify({
                    'message': f'Therapist {data["email"]} not found',
                    'success': False
                }), 404
            if isinstance(result, str):  # Error message for incorrect password
                return jsonify({'message': result, 'success': False}), 401

            access_token = create_access_token(identity=str(result['therapist_id']))
            return jsonify({
                'message': 'Login Successful',
                'success': True,
                'therapist': result,
                'access_token': access_token
            }), 200
        except Exception as error:
            return jsonify({
                'message': 'Login failed',
                'error': str(error),
                'success': False
            }), 400

    @jwt_required()
    def update_therapist(self, therapist_id: int) -> Tuple[Response, int]:
        """Update therapist details."""
        data = request.get_json()
        missing_fields = self.parameters_checker(
            required_fields=['full_name', 'email', 'curr_password', 'new_password'],
            data=data
        )
        if missing_fields:
            return jsonify({
                'message': f'Missing fields {", ".join(missing_fields)}',
                'success': False
            }), 400

        try:
            success = self.therapist_manager.update_therapist(
                therapist_id,
                data['full_name'],
                data['email'],
                data['curr_password'],
                data['new_password']
            )
            if success:
                return jsonify({
                    'success': True,
                    'message': 'User updated successfully',
                    'id': therapist_id
                }), 200
            return jsonify({
                'success': False,
                'message': 'User not found or no changes made',
                'id': therapist_id
            }), 404
        except Exception as error:
            return jsonify({
                'message': 'Failed to update user',
                'error': str(error),
                'success': False
            }), 400

    @jwt_required()
    def delete_therapist(self, therapist_id: int) -> Tuple[Response, int]:
        """Delete a therapist."""
        try:
            success = self.therapist_manager.delete_therapist(therapist_id)
            if success:
                return jsonify({
                    'success': True,
                    'message': 'User deleted successfully'
                }), 200
            return jsonify({
                'success': False,
                'message': 'User not found'
            }), 404
        except Exception as error:
            return jsonify({
                'message': 'Failed to delete user',
                'error': str(error),
                'success': False
            }), 400

    def add_patient(self) -> Tuple[Response, int]:
        """Register a new patient."""
        data = request.get_json()
        missing_fields = self.parameters_checker(
            required_fields=['email', 'full_name', 'contact'],
            data=data
        )
        if missing_fields:
            return jsonify({
                'message': f'Missing fields {", ".join(missing_fields)}',
                'success': False
            }), 400

        try:
            patient = Patient(
                email=data['email'],
                full_name=data['full_name'],
                contact=data['contact']
            )
            patient_email = patient.add_patient()
            return jsonify({
                'email': patient_email,
                'message': f'Patient {patient_email} registered successfully',
                'success': True
            }), 200
        except Exception as error:
            return jsonify({
                'message': 'Failed to register patient',
                'error': str(error),
                'success': False
            }), 400

    def get_patient(self, patient_email: str) -> Tuple[Response, int]:
        """Get patient details."""
        try:
            result = self.patient_manager.get_patient_by_email(patient_email)
            if not result:
                return jsonify({
                    'message': f'Patient {patient_email} not found',
                    'success': False
                }), 404
            return jsonify({
                'message': f'Patient {patient_email} found',
                'success': True,
                'patient': result
            }), 200
        except Exception as error:
            return jsonify({
                'message': 'Failed to get patient',
                'error': str(error),
                'success': False
            }), 400

    def update_patient(self, patient_prev_email: str) -> Tuple[Response, int]:
        """Update patient details."""
        data = request.get_json()
        missing_fields = self.parameters_checker(
            required_fields=['email', 'full_name', 'contact'],
            data=data
        )
        if missing_fields:
            return jsonify({
                'message': f'Missing fields {", ".join(missing_fields)}',
                'success': False
            }), 400

        try:
            patient = Patient(
                email=data['email'],
                full_name=data['full_name'],
                contact=data['contact']
            )
            success = patient.update_patient(patient_prev_email)
            if success:
                return jsonify({
                    'success': True,
                    'message': 'Patient updated successfully'
                }), 200
            return jsonify({
                'success': False,
                'message': 'Patient not found or no changes made'
            }), 404
        except Exception as error:
            return jsonify({
                'message': 'Failed to update patient',
                'error': str(error),
                'success': False
            }), 400

    def delete_patient(self, patient_email: str) -> Tuple[Response, int]:
        """Delete a patient."""
        try:
            success = self.patient_manager.delete_patient(patient_email)
            if success:
                return jsonify({
                    'success': True,
                    'message': 'Patient deleted successfully'
                }), 200
            return jsonify({
                'success': False,
                'message': 'Patient not found'
            }), 404
        except Exception as error:
            return jsonify({
                'message': 'Failed to delete patient',
                'error': str(error),
                'success': False
            }), 400

    @jwt_required()
    def logout_therapist(self) -> Tuple[Response, int]:
        jti = get_jwt()["jti"]
        if jti in self.blacklist:
            return jsonify({
                'message': 'Token already blacklisted',
                'success': False
            }), 400
        self.blacklist.add(jti)
        return jsonify({
            'message': 'Successfully logged out',
            'success': True
        }), 200

    # @jwt_required()
    def get_analysis_by_therapist_id(self, therapist_id: int) -> Tuple[Response, int]:
        try:
            result = self.analysis_manager.get_analysis_by_therapist_id(therapist_id)
            print(result)
            if not result:
                return jsonify({
                    'message': f'No data found.',
                    'success': False
                }), 404
            return jsonify({
                'message': f'Core with therapist: {therapist_id} found',
                'success': True,
                'analysis_results': result
            }), 200
        except Exception as error:
            return jsonify({
                'message': 'Failed to get analysis result',
                'error': str(error),
                'success': False
            }), 400

    # @jwt_required()
    def get_analysis_by_patient_email(self, patient_email: str) -> Tuple[Response, int]:
        try:
            result = self.analysis_manager.get_analysis_by_patient_email(patient_email)
            if not result:
                return jsonify({
                    'message': f'Core with patient email: {patient_email} not found',
                    'success': False
                }), 404
            return jsonify({
                'message': f'Core with patient email: {patient_email} found',
                'success': True,
                'patient': result
            }), 200
        except Exception as error:
            return jsonify({
                'message': 'Failed to get analysis result',
                'error': str(error),
                'success': False
            }), 400

    def email_patient_route(self) -> Tuple[Response, int]:
        """Route handler for sending emails to patients via API endpoint."""
        try:
            data = request.get_json()
            missing_fields = self.parameters_checker(
                required_fields=['patient_name', 'patient_email', 'patient_contact', 'analysis_summary'],
                data=data
            )

            if missing_fields:
                return jsonify({
                    'message': f'Missing fields {", ".join(missing_fields)}',
                    'success': False
                }), 400

            patient_name = data['patient_name']
            patient_email = data['patient_email']
            patient_contact = data['patient_contact']

            return self.emailPatient(patient_name, patient_email, patient_contact, data)

        except Exception as error:
            print(f"Email route error: {str(error)}")
            return jsonify({
                'message': 'Failed to send email',
                'error': str(error),
                'success': False
            }), 400

    def update_analysis_text(self, analysis_id: int) -> Tuple[Response, int]:
        try:
            data = request.get_json()
            missing_fields = self.parameters_checker(
                required_fields=['analysis_summary', 'patientInfo'],
                data=data
            )

            if missing_fields:
                return jsonify({
                    'message': f'Missing fields {", ".join(missing_fields)}',
                    'success': False
                }), 400

            # Extract patient information from the request
            patient_info = data.get('patientInfo', {})
            patient_email = patient_info.get('email', '')
            patient_name = patient_info.get('fullname', '')
            patient_contact = patient_info.get('contact', '')

            # First update the analysis text in the database
            analysis_updater = Analysis(
                analysis_id=analysis_id,
                analysis_summary=data['analysis_summary']
            )
            success = analysis_updater.update_analysis_text()

            if not success:
                return jsonify({
                    'success': False,
                    'message': 'Analysis not found or no changes made'
                }), 404

            # If update successful, send email and return its response
            return self.emailPatient(patient_name, patient_email, patient_contact, data)

        except Exception as error:
            print(f"Update analysis error: {str(error)}")
            return jsonify({
                'message': 'Failed to update analysis',
                'error': str(error),
                'success': False
            }), 400

    def emailPatient(self, patient_name: str, patient_email: str, patient_contact: str, data: Dict[str, Any]) -> Tuple[Response, int]:

        try:
            import win32com.client as win32
            import pythoncom

            # Verify that analysis_summary exists in the data
            analysis_summary = data.get('analysis_summary', '')
            if not analysis_summary:
                return jsonify({
                    'success': False,
                    'message': 'Missing analysis summary data'
                }), 400

            pythoncom.CoInitialize()
            try:
                olApp = win32.Dispatch("Outlook.Application")
                olNS = olApp.GetNamespace("MAPI")

                mail_item = olApp.CreateItem(0)
                mail_item.Subject = f"Session Report for Patient: {patient_name}"

                if patient_email and '@' in patient_email:
                    mail_item.To = patient_email
                else:
                    print(f"Warning: Invalid or missing patient email: '{patient_email}'")
                    mail_item.To = "your-default-email@example.com"

                mail_item.BodyFormat = 2
                mail_item.HTMLBody = f"""
                    <html>
                    <body>
                    <h1>Session Analysis Report</h1>
                    <p>Hi {patient_name},</p>
                    <p>Here is your session summary:</p>
                    <p>{analysis_summary}</p>
                    <p>Patient Email: {patient_email}</p>
                    </body>
                    </html>
                """

                mail_item.Send()

                print("Email sent successfully via Outlook")

                return jsonify({
                    'success': True,
                    'message': 'Analysis updated successfully and email sent via Outlook'
                }), 200

            finally:
                pythoncom.CoUninitialize()

        except ImportError as import_error:
            error_details = f"Import error: {str(import_error)}"
            print(error_details)
            return jsonify({
                'success': True,
                'message': 'Analysis processed, but email libraries not available',
                'email_error': error_details
            }), 207  # 207: Multi-Status (Partial success)

        except Exception as email_error:
            error_details = f"Email sending failed: {str(email_error)}"
            print(error_details)
            return jsonify({
                'success': True,
                'message': 'Analysis processed, but email sending failed',
                'email_error': error_details
            }), 207

    # PreRecorded Route
    def analyze_file(self) -> tuple[Response, int] | None:
        try:
            if 'audio' not in request.files:
                return jsonify({'error': 'No audio file provided'}), 400

            file = request.files['audio']
            if file.filename == '':
                return jsonify({'error': 'No selected file'}), 400

            # Get patient and therapist information from form data
            patient_email = request.form.get('patientEmail')
            patient_fullname = request.form.get('patientFullname')
            patient_contact = request.form.get('patientContact')
            therapist_id = request.form.get('therapistId')  # Add this line

            if not all([patient_email, patient_fullname, patient_contact, therapist_id]):  # Update check
                return jsonify({'error': 'Missing patient or therapist information'}), 400
            print(patient_email, patient_fullname, patient_contact, therapist_id)
            # Create Patient object for this analysis
            patient = Patient(
                email=patient_email,
                full_name=patient_fullname,
                contact=patient_contact
            )
            test_var = patient.get_patient_by_email(patient_email)
            if not test_var or "email" not in test_var:
                print("No patient record")
                patient.add_patient()

            # Update analyzer with patient and therapist information
            self.analyzer.patient = patient
            self.analyzer.patient_email = patient_email
            self.analyzer.therapist_id = int(therapist_id)  # Add this line

            # Clear any previous session data
            self.analyzer.clear_session_data()

            if file and self.allowed_file(file.filename):
                filename = secure_filename(file.filename)
                filepath = os.path.join(self.app.config['UPLOAD_FOLDER'], filename)
                file.save(filepath)

                try:
                    analysis_result = self.analyzer.analyze_prerecorded_audio(filepath)
                    if analysis_result['status'] == 'error':
                        print(f"Core Error: {analysis_result['message']}")  # Add debugging
                    return jsonify(analysis_result), 200
                finally:
                    if os.path.exists(filepath):
                        os.remove(filepath)

            return jsonify({'error': 'Invalid file type'}), 400

        except Exception as e:
            print(f"Exception in analyze_file: {str(e)}")  # Add debugging
            return jsonify({'error': str(e)}), 500
    # Realtime Sockets
    def _register_before_request(self) -> None:
        @self.app.before_request
        def load_model() -> None:
            if os.path.exists(r'C:\Users\Owais\PycharmProjects\EmoSense-server\Core\therapy_emotion_model.h5'):
                self.analyzer.load_model()
            else:
                data_path = "../ravdess_data"
                if self.analyzer.check_data_path(data_path):
                    X, y = self.analyzer.prepare_training_data(data_path)
                    if self.analyzer.train_model(X, y):
                        self.analyzer.save_model()
                    else:
                        raise Exception("Failed to train model")
                else:
                    raise Exception("RAVDESS dataset not found")

    def _start_analysis_session(self, session_id: str, patient_email: str, therapist_id: int) -> bool:
        """Start a new analysis session"""
        if session_id in self.active_sessions:
            return False

        self.active_sessions[session_id] = {
            'patient_email': patient_email,
            'therapist_id': therapist_id,
            'start_time': datetime.now(),
            'analysis_results': []
        }
        self.analyzer.start_session()
        return True

    def _end_analysis_session(self, session_id: str) -> Optional[Dict[str, Any]]:
        """End an analysis session and generate summary"""
        if session_id not in self.active_sessions:
            return None

        session_data = self.active_sessions.pop(session_id)
        self.analyzer.end_session()

        summary = self.analyzer.generate_session_summary()
        if summary['status'] == 'success':
            summary['summary_description'] = self.analyzer.generate_summary_description(summary)

            self.analyzer.patient_email = session_data['patient_email']
            self.analyzer.therapist_id = session_data['therapist_id']
            save_result = self.analyzer.save_session_analysis('realtime')

            return {
                'status': 'success',
                'summary': summary,
                'save_result': save_result
            }
        return None

    def _process_audio_chunk(self, session_id: str, audio_data: str) -> Optional[Dict[str, Any]]:
        """Process an audio chunk for emotion analysis with improved handling"""
        if session_id not in self.active_sessions:
            return None

        try:
            # Decode base64 string to bytes
            audio_bytes = base64.b64decode(audio_data)

            # Convert to numpy array (assuming 16-bit PCM from frontend)
            audio_array = np.frombuffer(audio_bytes, dtype=np.int16)

            if len(audio_array) == 0:
                return None

            print(f"Received audio chunk: {len(audio_array)} samples")

            # Pass the sample rate to the emotion analyzer
            result = self.analyzer.process_audio_chunk(audio_array, sample_rate=16000)
            if result:
                self.active_sessions[session_id]['analysis_results'].append(result)
                print(f"Analysis result: {result['predicted_emotion']} (confidence: {result['max_confidence']:.2f})")
            return result

        except Exception as e:
            print(f"Error processing audio chunk: {str(e)}")
            import traceback
            traceback.print_exc()
            return None

    def _register_websocket_handlers(self) -> None:
        """Register WebSocket event handlers"""

        @self.socketio.on('connect')
        def handle_connect() -> None:
            emit('connected', {'message': 'Connected to real-time analysis server'})

        @self.socketio.on('disconnect')
        def handle_disconnect() -> None:
            print('Client disconnected')

        @self.socketio.on('start_analysis')
        # @jwt_required()
        def handle_start_analysis(data: Dict[str, Any]) -> None:
            try:
                session_id = data.get('session_id')
                patient_fullname = data.get("patient_fullname")
                patient_email = data.get('patient_email')
                patient_contact = data.get("patient_contact")
                therapist_id = data.get('therapist_id')

                if not all([session_id, patient_fullname, patient_email, patient_contact, therapist_id]):
                    emit('error', {'message': 'Missing required session information'})
                    return
                patient = Patient(
                    email=patient_email,
                    full_name=patient_fullname,
                    contact=patient_contact
                )
                test_var = patient.get_patient_by_email(patient_email)
                if not test_var or "email" not in test_var:
                    print("No patient record")
                    patient.add_patient()

                success = self._start_analysis_session(session_id, patient_email, therapist_id)
                if success:
                    emit('analysis_started', {
                        'status': 'success',
                        'session_id': session_id,
                        'message': 'Real-time analysis session started'
                    })
                else:
                    emit('error', {'message': 'Session already exists'})

            except Exception as e:
                emit('error', {'message': f'Failed to start analysis: {str(e)}'})

        @self.socketio.on('audio_chunk')
        # @jwt_required()
        def handle_audio_chunk(data: Dict[str, Any]) -> None:
            try:
                session_id = data.get('session_id')
                audio_data = data.get('audio_data')

                if not all([session_id, audio_data]):
                    emit('error', {'message': 'Missing session_id or audio data'})
                    return

                result = self._process_audio_chunk(session_id, audio_data)
                if result:
                    emit('analysis_result', {
                        'session_id': session_id,
                        'result': result
                    })

            except Exception as e:
                emit('error', {'message': f'Failed to process audio: {str(e)}'})

        @self.socketio.on('end_analysis')
        # @jwt_required()
        def handle_end_analysis(data: Dict[str, Any]) -> None:
            try:
                session_id = data.get('session_id')
                if not session_id:
                    emit('error', {'message': 'Missing session_id'})
                    return

                summary = self._end_analysis_session(session_id)
                if summary:
                    emit('analysis_ended', {
                        'status': 'success',
                        'session_id': session_id,
                        'summary': summary
                    })
                else:
                    emit('error', {'message': 'Failed to generate session summary'})

            except Exception as e:
                emit('error', {'message': f'Failed to end analysis: {str(e)}'})

    def health_check(self) -> Tuple[Response, int]:
        return jsonify({
            'status': 'healthy',
            'model_loaded': self.analyzer.model is not None
        }), 200

    def run(self, host: str = '0.0.0.0', port: int = 5000, debug: bool = False) -> None:
        """Run the Flask application."""
        self.app.run(host=host, port=port, debug=debug)


if __name__ == '__main__':
    app = EmoSenseApp()
    app.run(debug=True)