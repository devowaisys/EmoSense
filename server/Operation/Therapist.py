from Configuration.DatabaseConfig import DatabaseConfig

class Therapist:
    def __init__(self, therapist_id=None, email=None, full_name=None, password=None):
        self.therapist_id = therapist_id
        self.email = email
        self.full_name = full_name
        self.password = password

    def get_therapist_by_email_and_password(self, email, password):
        db_config = DatabaseConfig(r'C:\Users\Owais\GitHub\EmoSense\server\config.ini') # dependency of DatabaseConfig class
        conn = db_config.get_connection()
        cursor = conn.cursor()
        try:
            cursor.execute('''
                SELECT "THERAPIST_ID", "THERAPIST_FULLNAME", 
                       "THERAPIST_EMAIL", "THERAPIST_PASSWORD" 
                FROM "THERAPIST" 
                WHERE "THERAPIST_EMAIL" = %s''', (email,))
            therapist = cursor.fetchone()
            conn.commit()

            if therapist is None:
                return None

            if therapist[3] != password:
                return "Incorrect password. Please try again."

            self.therapist_id = therapist[0]
            self.full_name = therapist[1]
            self.email = therapist[2]
            self.password = therapist[3]

            return {
                "therapist_id": self.therapist_id,
                "full_name": self.full_name,
                "email": self.email,
                "password": self.password
            }
        except Exception as e:
            conn.rollback()
            raise e
        finally:
            cursor.close()
            conn.close()

    def add_therapist(self):
        db_config = DatabaseConfig(r'C:\Users\Owais\GitHub\EmoSense\server\config.ini')
        conn = db_config.get_connection()
        cursor = conn.cursor()
        try:
            cursor.execute('''
                INSERT INTO "THERAPIST" 
                ("THERAPIST_FULLNAME", "THERAPIST_EMAIL", "THERAPIST_PASSWORD") 
                VALUES (%s, %s, %s) 
                RETURNING "THERAPIST_ID"''', (self.full_name, self.email, self.password))
            self.therapist_id = cursor.fetchone()[0]
            conn.commit()
            return self.therapist_id
        except Exception as e:
            conn.rollback()
            raise e
        finally:
            cursor.close()
            conn.close()

    def update_therapist(self, therapist_id, full_name, email, curr_password, new_password):
        db_config = DatabaseConfig(r'C:\Users\Owais\GitHub\EmoSense\server\config.ini')
        conn = db_config.get_connection()
        cursor = conn.cursor()
        try:
            cursor.execute('''
                SELECT "THERAPIST_PASSWORD" 
                FROM "THERAPIST" 
                WHERE "THERAPIST_ID" = %s''', (therapist_id,))
            result = cursor.fetchone()

            if result is None:
                return "Therapist not found."

            stored_password = result[0]
            if stored_password != curr_password:
                return "Current password is incorrect."

            self.therapist_id = therapist_id
            self.full_name = full_name
            self.email = email
            self.password = new_password

            cursor.execute('''
                UPDATE "THERAPIST" 
                SET "THERAPIST_FULLNAME" = %s, 
                    "THERAPIST_EMAIL" = %s, 
                    "THERAPIST_PASSWORD" = %s 
                WHERE "THERAPIST_ID" = %s''',
                           (self.full_name, self.email, self.password, self.therapist_id))
            rows_affected = cursor.rowcount
            conn.commit()
            return rows_affected > 0
        except Exception as e:
            conn.rollback()
            return "An error occurred while updating the therapist."
        finally:
            cursor.close()
            conn.close()

    def delete_therapist(self, therapist_id):
        db_config = DatabaseConfig(r'C:\Users\Owais\GitHub\EmoSense\server\config.ini')
        conn = db_config.get_connection()
        cursor = conn.cursor()
        try:
            cursor.execute('DELETE FROM "THERAPIST" WHERE "THERAPIST_ID" = %s',
                           (therapist_id,))
            rows_affected = cursor.rowcount
            conn.commit()
            if rows_affected > 0:
                self.therapist_id = None
                self.full_name = None
                self.email = None
                self.password = None
                return True
            return False
        except Exception as e:
            conn.rollback()
            return e
        finally:
            cursor.close()
            conn.close()
