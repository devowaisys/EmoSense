from Configuration.DatabaseConfig import DatabaseConfig


class Patient:
    def __init__(self, email=None, full_name=None, contact=None):
        self.email = email
        self.full_name = full_name
        self.contact = contact

    def add_patient(self):
        db_config = DatabaseConfig(r'C:\Users\Owais\GitHub\EmoSense\server\config.ini') # dependency of DatabaseConfig class
        conn = db_config.get_connection()
        cursor = conn.cursor()
        try:
            cursor.execute(
                '''
                INSERT INTO "PATIENTS" ("PATIENT_EMAIL", "PATIENT_FULLNAME", "PATIENT_CONTACT") 
                VALUES (%s, %s, %s) RETURNING "PATIENT_EMAIL"
                ''',
                (self.email, self.full_name, self.contact)
            )
            self.email = cursor.fetchone()[0]
            conn.commit()
            return self.email
        except Exception as e:
            conn.rollback()
            raise e
        finally:
            cursor.close()
            conn.close()

    def get_patient_by_email(self, email):
        db_config = DatabaseConfig(r'C:\Users\Owais\GitHub\EmoSense\server\config.ini')
        conn = db_config.get_connection()
        cursor = conn.cursor()
        try:
            cursor.execute(
                '''
                SELECT "PATIENT_EMAIL", "PATIENT_FULLNAME", "PATIENT_CONTACT" 
                FROM "PATIENTS" WHERE "PATIENT_EMAIL" = %s
                ''',
                (email,)
            )
            patient = cursor.fetchone()
            conn.commit()
            if patient is None:
                return None

            # Set instance variables from the query result
            self.email = patient[0]
            self.full_name = patient[1]
            self.contact = patient[2]

            return {
                "email": self.email,
                "full_name": self.full_name,
                "contact": self.contact,
            }
        except Exception as e:
            conn.rollback()
            raise e
        finally:
            cursor.close()
            conn.close()

    def update_patient(self, prev_email):
        db_config = DatabaseConfig(r'C:\Users\Owais\GitHub\EmoSense\server\config.ini')
        conn = db_config.get_connection()
        cursor = conn.cursor()
        try:
            cursor.execute(
                '''
                UPDATE "PATIENTS" 
                SET "PATIENT_FULLNAME" = %s, "PATIENT_EMAIL" = %s, "PATIENT_CONTACT" = %s 
                WHERE "PATIENT_EMAIL" = %s
                ''',
                (self.full_name, self.email, self.contact, prev_email)
            )
            rows_affected = cursor.rowcount
            conn.commit()
            return rows_affected > 0
        except Exception as e:
            conn.rollback()
            raise e
        finally:
            cursor.close()
            conn.close()

    def delete_patient(self, email):
        db_config = DatabaseConfig(r'C:\Users\Owais\GitHub\EmoSense\server\config.ini')
        conn = db_config.get_connection()
        cursor = conn.cursor()
        try:
            cursor.execute(
                '''
                DELETE FROM "PATIENTS" WHERE "PATIENT_EMAIL" = %s
                ''',
                (email,)
            )
            rows_affected = cursor.rowcount
            conn.commit()
            if rows_affected > 0:
                # Clear instance variables after deletion
                self.email = None
                self.full_name = None
                self.contact = None
                return True
            return False
        except Exception as e:
            conn.rollback()
            raise e
        finally:
            cursor.close()
            conn.close()
