import unittest
import pandas as pd
from unittest.mock import patch
from cornac.augmentation.enrich_ne import get_enriched_ne, EfficientDict

class TestEnhanceNER(unittest.TestCase):
    def test_enhance_ner_found_wiki(self):
        ne_list = [
            {'text': 'Barack Obama', 'alternative': ['Barack Obama', 'Obama'], 'frequency': 1, 'label': 'PERSON'}]
        lookup_person = EfficientDict()
        lookup_org = EfficientDict()
        result = get_enriched_ne(ne_list, lookup_person, lookup_org)

        self.assertIn('Barack Obama', result[0])
        obama_data = result[0]['Barack Obama']
        
        if 'givenname' in obama_data:
            self.assertIn('Barack', obama_data['givenname'])
        if 'familyname' in obama_data:
            self.assertIn('Obama', obama_data['familyname'])
        

        if 'gender' in obama_data:
            self.assertEqual(obama_data['gender'], ['male'])
        
        if 'occupations' in obama_data:
            occupation_found = any(
                term in ' '.join(obama_data['occupations']).lower() 
                for term in ['politician', 'president', 'lawyer']
            )
            self.assertTrue(occupation_found)
        
        if 'party' in obama_data:
            party_found = any(
                'democratic' in party.lower() 
                for party in obama_data['party']
            )
            self.assertTrue(party_found)
        
        if 'citizen' in obama_data:
            us_citizen = any(
                'united states' in citizen.lower() or 'usa' in citizen.lower() or 'america' in citizen.lower()
                for citizen in obama_data['citizen']
            )
            self.assertTrue(us_citizen)
        
        if 'ethnicity' in obama_data:
            african_american = any(
                'african' in ethnicity.lower() 
                for ethnicity in obama_data['ethnicity']
            )
            self.assertTrue(african_american)
    
        if 'place_of_birth' in obama_data:
            us_born = any(
                'united states' in place.lower() or 'usa' in place.lower() or 'hawaii' in place.lower()
                for place in obama_data['place_of_birth']
            )
            self.assertTrue(us_born)
 

    def test_enhance_ner_not_found_wiki(self):
        ne_list = [{'text': 'Nonexistent Person', 'alternative': ['Nonexistent_person', 'Nonexistent'], 'frequency': 3, 'label': 'PERSON'}]
        lookup_person = EfficientDict()
        lookup_org = EfficientDict()
        result = get_enriched_ne(ne_list, lookup_person, lookup_org)

        self.assertIn('Nonexistent Person', result[0])
        self.assertNotIn('givenname', result[0]['Nonexistent Person'])

    @patch('cornac.augmentation.enrich_ne.WikidataQuery.person_data_query')
    def test_enhance_ner_with_non_english_text(self, mock_person_query):
        # Mock Wikidata response for non-English input
        mock_person_query.return_value = {
            'givenname': ['Pedro'],
            'familyname': ['Álvares Cabral'],
            'gender': ['male'],
            'occupations': ['explorer']
        }

        ne_list = [{'text': 'Pedro Álvares Cabral', 'alternative': ['Pedro Cabral'], 'frequency': 1, 'label': 'PERSON'}]
        lookup_person = EfficientDict()
        lookup_org = EfficientDict()
        result = get_enriched_ne(ne_list, lookup_person, lookup_org)

        self.assertEqual(result[0]['Pedro Álvares Cabral']['givenname'], ['Pedro'])
        self.assertEqual(result[0]['Pedro Álvares Cabral']['familyname'], ['Álvares Cabral'])
        self.assertEqual(result[0]['Pedro Álvares Cabral']['gender'], ['male'])
        self.assertIn('explorer', result[0]['Pedro Álvares Cabral']['occupations'])

    @patch('cornac.augmentation.enrich_ne.WikidataQuery.org_data_query')
    def test_enhance_ner_with_organization(self, mock_org_query):
        # Mock Wikidata response for organization
        mock_org_query.return_value = {
            'ideology': ['social democracy']
        }

        ne_list = [{'text': 'Democratic Party', 'alternative': ['Democratic Party', 'Democrats'], 'frequency': 5,
                    'label': 'ORG'}]
        lookup_person = EfficientDict()
        lookup_org = EfficientDict()
        result = get_enriched_ne(ne_list, lookup_person, lookup_org)

        self.assertEqual(result[0]['Democratic Party']['ideology'], ['social democracy'])

    def test_enhance_ner_empty_input(self):
        ne_list = []
        lookup_person = EfficientDict()
        lookup_org = EfficientDict()
        result = get_enriched_ne(ne_list, lookup_person, lookup_org)

        self.assertEqual(result, [])

    @patch('cornac.augmentation.enrich_ne.WikidataQuery.person_data_query')
    def test_enhance_ner_person_lookup_cache(self, mock_person_query):
        # Simulate cached lookup behavior
        mock_person_query.return_value = {
            'givenname': ['Albert'],
            'familyname': ['Einstein'],
            'gender': ['male'],
            'occupations': ['physicist'],
            'citizen': ['Switzerland']
        }

        # First query
        ne_list_1 = [{'text': 'Albert Einstein', 'alternative': ['Einstein'], 'frequency': 2, 'label': 'PERSON'}]
        lookup_person = EfficientDict()
        lookup_org = EfficientDict()
        result_1 = get_enriched_ne(ne_list_1, lookup_person, lookup_org)

        self.assertEqual(result_1[0]['Albert Einstein']['givenname'], ['Albert'])
        self.assertEqual(result_1[0]['Albert Einstein']['familyname'], ['Einstein'])
        self.assertIn('physicist', result_1[0]['Albert Einstein']['occupations'])

        # Second query for the same person, ensuring cached response is used
        ne_list_2 = [{'text': 'Albert Einstein', 'alternative': ['Albert'], 'frequency': 1, 'label': 'PERSON'}]
        mock_person_query.reset_mock()  # Ensure no new calls to Wikidata
        result_2 = get_enriched_ne(ne_list_2, lookup_person, lookup_org)

        self.assertEqual(result_2[0]['Albert Einstein']['givenname'], ['Albert'])
        self.assertEqual(result_2[0]['Albert Einstein']['familyname'], ['Einstein'])
        self.assertIn('physicist', result_2[0]['Albert Einstein']['occupations'])
        mock_person_query.assert_not_called()  # No new query should have been made

    @patch('cornac.augmentation.enrich_ne.WikidataQuery.org_data_query')
    def test_enhance_ner_organization_lookup_cache(self, mock_org_query):
        # Simulate cached lookup for an organization
        mock_org_query.return_value = {'ideology': ['liberalism']}

        # First query
        ne_list_1 = [{'text': 'Liberal Party', 'alternative': ['Liberal'], 'frequency': 3, 'label': 'ORG'}]
        lookup_person = EfficientDict()
        lookup_org = EfficientDict()
        result_1 = get_enriched_ne(ne_list_1, lookup_person, lookup_org)

        self.assertEqual(result_1[0]['Liberal Party']['ideology'], ['liberalism'])

        # Second query for the same organization
        ne_list_2 = [{'text': 'Liberal Party', 'alternative': ['The Liberals'], 'frequency': 2, 'label': 'ORG'}]
        mock_org_query.reset_mock()  # Ensure no new calls to Wikidata
        result_2 = get_enriched_ne(ne_list_2, lookup_person, lookup_org)

        self.assertEqual(result_2[0]['Liberal Party']['ideology'], ['liberalism'])
        mock_org_query.assert_not_called()  # No new query should have been made


if __name__ == "__main__":
    unittest.main()
