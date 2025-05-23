import unittest
from Application.EmoSenseApp import EmoSenseApp  # Ensure your Flask app is properly imported


class EmoSenseTestCase(unittest.TestCase):
    def setUp(self):
        self.app = EmoSenseApp().app
        self.client = self.app.test_client()
        self.test_therapist = {
            "full_name": "Test Therapist",
            "email": "test_therapist@example.com",
            "password": "TestPass123"
        }

        self.test_patient = {
            "full_name": "Test Patient",
            "email": "test_patient@example.com",
            "contact": "1234567890"
        }

    def test_missing_fields_add_therapist(self):
        response = self.client.post('/api/add_therapist', json={})
        data = response.get_json()
        self.assertEqual(response.status_code, 400)
        self.assertFalse(data['success'])
        self.assertIn('Missing fields', data['message'])

    def test_invalid_login_therapist(self):
        response = self.client.post('/api/login_therapist', json={"email": "wrong@example.com", "password": "wrong"})
        data = response.get_json()
        self.assertEqual(response.status_code, 404)
        self.assertFalse(data['success'])
        self.assertIn('not found', data['message'])

    def test_missing_fields_add_patient(self):
        response = self.client.post('/api/add_patient', json={})
        data = response.get_json()
        self.assertEqual(response.status_code, 400)
        self.assertFalse(data['success'])
        self.assertIn('Missing fields', data['message'])

    def test_get_nonexistent_patient(self):
        response = self.client.get('/api/get_patient/nonexistent@example.com')
        data = response.get_json()
        self.assertEqual(response.status_code, 404)
        self.assertFalse(data['success'])
        self.assertIn('not found', data['message'])

    def test_update_patient_missing_fields(self):
        response = self.client.put('/api/update_patient/nonexistent@example.com', json={})
        data = response.get_json()
        self.assertEqual(response.status_code, 400)
        self.assertFalse(data['success'])
        self.assertIn('Missing fields', data['message'])

    def test_delete_nonexistent_patient(self):
        response = self.client.delete('/api/delete_patient/nonexistent@example.com')
        data = response.get_json()
        self.assertEqual(response.status_code, 404)
        self.assertFalse(data['success'])
        self.assertIn('not found', data['message'])


    def test_valid_login_and_access_protected_route(self):
        # Create a therapist first
        self.client.post('/api/add_therapist', json=self.test_therapist)
        response = self.client.post('/api/login_therapist', json=self.test_therapist)
        data = response.get_json()
        token = data.get('access_token')

        headers = {"Authorization": f"Bearer {token}"}
        protected_response = self.client.get('/api/get_analysis_by_therapist_id/1', headers=headers)
        self.assertNotEqual(protected_response.status_code, 401)

    def test_successful_add_therapist(self):
        response = self.client.post('/api/add_therapist', json=self.test_therapist)
        data = response.get_json()
        self.assertEqual(response.status_code, 200)
        self.assertTrue(data['success'])
        self.assertIn('registered successfully', data['message'])
        self.assertIn('id', data)

    def test_successful_update_therapist(self):
        # First add a therapist
        add_response = self.client.post('/api/add_therapist', json=self.test_therapist)
        add_data = add_response.get_json()
        therapist_id = add_data['id']

        # Login to get token
        login_response = self.client.post('/api/login_therapist', json=self.test_therapist)
        login_data = login_response.get_json()
        token = login_data.get('access_token')

        # Update the therapist
        update_data = {
            "full_name": "Updated Test Therapist",
            "email": "updated_test@example.com",
            "curr_password": self.test_therapist['password'],
            "new_password": "NewTestPass123"
        }
        headers = {"Authorization": f"Bearer {token}"}
        update_response = self.client.put(
            f'/api/update_therapist/{therapist_id}',
            json=update_data,
            headers=headers
        )
        update_data = update_response.get_json()

        self.assertEqual(update_response.status_code, 200)
        self.assertTrue(update_data['success'])
        self.assertIn('updated successfully', update_data['message'])
        self.assertEqual(update_data['id'], therapist_id)

    def test_successful_add_patient(self):
        """Test successful patient registration with all required fields"""
        # Test data with all required fields
        test_patient = {
            "email": "test_patient1@example.com",
            "full_name": "Test Patient",
            "contact": "1234567890"
        }

        # Make the request
        response = self.client.post(
            '/api/add_patient',
            json=test_patient
        )
        data = response.get_json()

        # Debug output if needed
        print("Response Status:", response.status_code)
        print("Response Data:", data)

        # Assertions
        self.assertEqual(response.status_code, 200)
        self.assertTrue(data['success'])
        self.assertIn('registered successfully', data['message'])
        self.assertEqual(data['email'], test_patient['email'])

if __name__ == '__main__':
    unittest.main()
