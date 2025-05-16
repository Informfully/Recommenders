import unittest
import pandas as pd
from unittest.mock import MagicMock, patch
from concurrent.futures import Future
from cornac.augmentation.region import get_region, requests, make_request_with_retries, is_valid_string, get_english_label, get_region_data

class TestGetRegion(unittest.TestCase):

    @patch('cornac.augmentation.region.requests.get')
    def test_make_request_with_retries_success(self, mock_get):
        # Mock successful response
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_get.return_value = mock_response

        url = "http://example.com"
        response = make_request_with_retries(url)

        self.assertEqual(response, mock_response)
        mock_get.assert_called_once_with(url,timeout=10)

    @patch('cornac.augmentation.region.requests.get')
    def test_make_request_with_retries_rate_limit(self, mock_get):
        # Mock rate-limit response
        mock_response = MagicMock()
        mock_response.status_code = 429
        mock_response.headers = {'Retry-After': '2'}
        mock_get.side_effect = [mock_response, MagicMock(status_code=200)]

        url = "http://example.com"
        with patch('time.sleep') as mock_sleep:
            response = make_request_with_retries(url)
            self.assertEqual(response.status_code, 200)
            mock_sleep.assert_called_with(2)

    @patch('cornac.augmentation.region.requests.get')
    def test_make_request_with_retries_failure(self, mock_get):
        # Mock request exception
        mock_get.side_effect = requests.RequestException("Error")
        url = "http://example.com"

        response = make_request_with_retries(url, retries=2)
        self.assertIsNone(response)

    @patch('cornac.augmentation.region.make_request_with_retries')
    def test_is_valid_string_true(self, mock_make_request):
        # Mock valid string response
        mock_search_response = MagicMock()
        mock_search_response.json.return_value = {
            'search': [{'label': 'Test String', 'id': 'Q123'}]
        }

        mock_claim_response = MagicMock()
        mock_claim_response.json.return_value = {'claims': {'P1566': 'Geonames ID'}}
        mock_make_request.side_effect = [mock_search_response, mock_claim_response]

        self.assertTrue(is_valid_string("Test String"))

    @patch('cornac.augmentation.region.make_request_with_retries')
    def test_is_valid_string_false(self, mock_make_request):
        # Mock invalid string response
        mock_search_response = MagicMock()
        mock_search_response.json.return_value = {}
        mock_make_request.return_value = mock_search_response

        self.assertFalse(is_valid_string("Invalid String"))

    @patch('cornac.augmentation.region.requests.get')
    def test_get_english_label_success(self, mock_get):
        # Mock successful response
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            'search': [{'label': 'English Label'}]
        }
        mock_get.return_value = mock_response

        result = get_english_label("Test String", "en")
        self.assertEqual(result, "English Label")

    @patch('cornac.augmentation.region.requests.get')
    def test_get_english_label_failure(self, mock_get):
        # Mock a response with status code 404 (not found)
        mock_response = MagicMock()
        mock_response.raise_for_status.side_effect = requests.RequestException("404 Client Error")
        mock_response.json.return_value = {}
        mock_get.return_value = mock_response
        result = get_english_label("Test String", "en")
        self.assertIsNone(result)

    @patch('cornac.augmentation.region.is_valid_string')
    def test_get_region_data(self, mock_is_valid_string):
        mock_is_valid_string.return_value = True

        entity = {'text': 'City', 'alternative': ['Alternative City']}
        lookup_dict = {}

        result = get_region_data(entity, lookup_dict)
        self.assertEqual(result, 'city')
        self.assertIn('city', lookup_dict)

    @patch('cornac.augmentation.region.get_region_data')
    def test_get_region(self, mock_get_region_data):
        # Mock region data responses
        future_1 = Future()
        future_1.set_result("region1")

        future_2 = Future()
        future_2.set_result("region2")

        mock_get_region_data.side_effect = [future_1.result, future_2.result]

        ne_list = [
            {'text': 'Region1', 'alternative': [], 'label': 'GPE'},
            {'text': 'Region2', 'alternative': [], 'label': 'LOC'}
        ]
        lookup_dict = {}

        with patch('concurrent.futures.ThreadPoolExecutor.submit', side_effect=[future_1, future_2]):
            result = get_region(ne_list, lookup_dict)
            self.assertEqual(set(result), {'region1', 'region2'})


if __name__ == "__main__":
    unittest.main()
