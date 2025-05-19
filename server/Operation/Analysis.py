from Configuration.DatabaseConfig import DatabaseConfig

class Analysis:
    def __init__(self, analysis_id=None, therapist_id=None, patient_email=None,
                 analysis_mode=None, analysis_duration=None, dominant_emotion=None,
                 analysis_summary=None, date=None, session_duration=None,
                 session_start=None, session_end=None):
        self.analysis_id = analysis_id
        self.therapist_id = therapist_id
        self.patient_email = patient_email
        self.analysis_mode = analysis_mode
        self.analysis_duration = analysis_duration
        self.dominant_emotion = dominant_emotion
        self.analysis_summary = analysis_summary
        self.date = date
        self.session_duration = session_duration
        self.session_start = session_start
        self.session_end = session_end

    def add_analysis(self):
        db_config = DatabaseConfig(r'../config.ini') # dependency of DatabaseConfig class
        conn = db_config.get_connection()
        cursor = conn.cursor()


        try:
            # Print debug information
            print("\nDebug: Attempting to save analysis with values:")
            print(f"THERAPIST_ID: {self.therapist_id}")
            print(f"PATIENT_EMAIL: {self.patient_email}")
            print(f"ANALYSIS_MODE: {self.analysis_mode}")
            print(f"ANALYSIS_DURATION: {self.analysis_duration}")
            print(f"DOMINANT_EMOTION: {self.dominant_emotion}")
            print(f"SUMMARY: {self.analysis_summary}")
            print(f"DATE: {self.date}")
            print(f"SESSION_DURATION: {self.session_duration}")
            print(f"SESSION_START: {self.session_start}")
            print(f"SESSION_END: {self.session_end}")

            cursor.execute('''
                INSERT INTO "ANALYSIS" (
                    "THERAPIST_ID", "PATIENT_EMAIL", "ANALYSIS_MODE", 
                    "ANALYSIS_DURATION", "DOMINANT_EMOTION", "ANALYSIS_SUMMARY", 
                    "DATE", "SESSION_DURATION", "SESSION_START", "SESSION_END"
                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s) 
                RETURNING "ANALYSIS_ID"''',
                           (self.therapist_id, self.patient_email, self.analysis_mode,
                            self.analysis_duration, self.dominant_emotion, self.analysis_summary,
                            self.date, self.session_duration, self.session_start, self.session_end))

            self.analysis_id = cursor.fetchone()[0]
            conn.commit()
            print(f"Debug: Analysis saved successfully with ID: {self.analysis_id}")
            return self.analysis_id
        except Exception as e:
            conn.rollback()
            print(f"Debug: Database error: {str(e)}")
            raise e
        finally:
            cursor.close()
            conn.close()

    def get_analysis_by_therapist_id(self, therapist_id):
        db_config = DatabaseConfig(r'C:\Users\Owais\GitHub\EmoSense\server\config.ini')
        conn = db_config.get_connection()
        cursor = conn.cursor()
        try:
            cursor.execute('''
                SELECT "ANALYSIS_ID", "THERAPIST_ID", "PATIENT_EMAIL", 
                       "ANALYSIS_MODE", "ANALYSIS_DURATION", "DOMINANT_EMOTION", 
                       "ANALYSIS_SUMMARY", "DATE", "SESSION_DURATION", 
                       "SESSION_START", "SESSION_END" 
                FROM "ANALYSIS" 
                WHERE "THERAPIST_ID" = %s''', (therapist_id,))
            analyses = cursor.fetchall()
            conn.commit()

            if not analyses:
                return None

            results = []
            for analysis in analyses:
                result = {
                    "analysis_id": analysis[0],
                    "therapist_id": analysis[1],
                    "patient_email": analysis[2],
                    "analysis_mode": analysis[3],
                    "analysis_duration": str(analysis[4]) if analysis[4] else None,
                    "dominant_emotion": analysis[5],
                    "analysis_summary": analysis[6],
                    "date": analysis[7].isoformat() if analysis[7] else None,
                    "session_duration": str(analysis[8]) if analysis[8] else None,
                    "session_start": analysis[9].isoformat() if analysis[9] else None,
                    "session_end": analysis[10].isoformat() if analysis[10] else None,
                }
                results.append(result)

            return results

        except Exception as e:
            conn.rollback()
            raise e
        finally:
            cursor.close()
            conn.close()

    def get_analysis_by_patient_email(self, patient_email):
        db_config = DatabaseConfig(r'C:\Users\Owais\GitHub\EmoSense\server\config.ini')
        conn = db_config.get_connection()
        cursor = conn.cursor()
        try:
            cursor.execute('''
                SELECT "ANALYSIS_ID", "THERAPIST_ID", "PATIENT_EMAIL", 
                       "ANALYSIS_MODE", "ANALYSIS_DURATION", "DOMINANT_EMOTION", 
                       "ANALYSIS_SUMMARY", "DATE", "SESSION_DURATION", 
                       "SESSION_START", "SESSION_END" 
                FROM "ANALYSIS" 
                WHERE "PATIENT_EMAIL" = %s''', (patient_email,))

            analyses = cursor.fetchall()
            conn.commit()

            if not analyses:
                return None

            # Set instance variables from query results
            self.analysis_id = analyses[0]
            self.therapist_id = analyses[1]
            self.patient_email = analyses[2]
            self.analysis_mode = analyses[3]
            self.analysis_duration = analyses[4]
            self.dominant_emotion = analyses[5]
            self.analysis_summary = analyses[6]
            self.date = analyses[7]
            self.session_duration = analyses[8]
            self.session_start = analyses[9]
            self.session_end = analyses[10]

            results = []
            for analysis in analyses:
                result = {
                    "analysis_id": analysis[0],
                    "therapist_id": analysis[1],
                    "patient_email": analysis[2],
                    "analysis_mode": analysis[3],
                    "analysis_duration": str(analysis[4]) if analysis[4] else None,
                    "dominant_emotion": analysis[5],
                    "analysis_summary": analysis[6],
                    "date": analysis[7].isoformat() if analysis[7] else None,
                    "session_duration": str(analysis[8]) if analysis[8] else None,
                    "session_start": analysis[9].isoformat() if analysis[9] else None,
                    "session_end": analysis[10].isoformat() if analysis[10] else None,
                }
                results.append(result)

            return results
        except Exception as e:
            conn.rollback()
            raise e
        finally:
            cursor.close()
            conn.close()

    def update_analysis(self, analysis_id):
        # Set instance variables
        db_config = DatabaseConfig(r'C:\Users\Owais\GitHub\EmoSense\server\config.ini')
        conn = db_config.get_connection()
        cursor = conn.cursor()
        try:
            cursor.execute('''
                UPDATE "ANALYSIS" 
                SET "THERAPIST_ID" = %s, "PATIENT_EMAIL" = %s, 
                    "ANALYSIS_MODE" = %s, "ANALYSIS_DURATION" = %s, 
                    "DOMINANT_EMOTION" = %s, "ANALYSIS_SUMMARY" = %s, 
                    "DATE" = %s, "SESSION_DURATION" = %s, 
                    "SESSION_START" = %s, "SESSION_END" = %s 
                WHERE "ANALYSIS_ID" = %s''',
                (self.therapist_id, self.patient_email, self.analysis_mode,
                 self.analysis_duration, self.dominant_emotion, self.analysis_summary,
                 self.date, self.session_duration, self.session_start,
                 self.session_end, analysis_id))
            rows_affected = cursor.rowcount
            conn.commit()
            return rows_affected > 0
        except Exception as e:
            conn.rollback()
            return e
        finally:
            cursor.close()
            conn.close()

    def update_analysis_text(self):
        db_config = DatabaseConfig(r'C:\Users\Owais\GitHub\EmoSense\server\config.ini')
        conn = None
        cursor = None

        try:
            conn = db_config.get_connection()
            cursor = conn.cursor()

            # More efficient SQL query that only updates the necessary field
            cursor.execute('''
                UPDATE "ANALYSIS" 
                SET "ANALYSIS_SUMMARY" = %s
                WHERE "ANALYSIS_ID" = %s
                RETURNING "ANALYSIS_ID"''',
                           (self.analysis_summary, self.analysis_id))

            # Check if any row was actually updated
            result = cursor.fetchone()
            conn.commit()

            return result is not None

        except Exception as e:
            if conn:
                conn.rollback()
            raise e
        finally:
            if cursor:
                cursor.close()
            if conn:
                conn.close()

    def delete_analysis(self, analysis_id):

        db_config = DatabaseConfig(r'C:\Users\Owais\GitHub\EmoSense\server\config.ini')
        conn = db_config.get_connection()
        cursor = conn.cursor()
        try:
            cursor.execute('DELETE FROM "ANALYSIS" WHERE "ANALYSIS_ID" = %s',
                         (analysis_id,))
            rows_affected = cursor.rowcount
            conn.commit()
            if rows_affected > 0:
                # Clear instance variables after successful deletion
                self.analysis_id = None
                self.therapist_id = None
                self.patient_email = None
                self.analysis_mode = None
                self.analysis_duration = None
                self.dominant_emotion = None
                self.analysis_summary = None
                self.date = None
                self.session_duration = None
                self.session_start = None
                self.session_end = None
                return True
            return False
        except Exception as e:
            conn.rollback()
            return e
        finally:
            cursor.close()
            conn.close()