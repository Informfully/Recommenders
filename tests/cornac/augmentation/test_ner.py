import unittest
from cornac.augmentation.ner import get_ner, set_ner_lang

class TestGetNER(unittest.TestCase):

    def setUp(self):
        # Load the English NER model
        self.ner_model = set_ner_lang('en')

    def test_get_ner_with_entities(self):
        sample_text = "Barack Obama visited New York. Microsoft are tech giants."
        user_entities = ['PERSON', 'GPE', 'ORG']
        result = get_ner(sample_text, ner_model=self.ner_model, entities=user_entities)

        # Check extracted entities and their properties
        entities_list = [r['label'] for r in result]
        self.assertTrue(set(entities_list) == set(user_entities))

        # PERSON entity verification
        person = next((r for r in result if r['label'] == 'PERSON'), None)
        self.assertIsNotNone(person)
        self.assertEqual(person['text'], 'Barack Obama')
        self.assertEqual(person['frequency'], 1)
        self.assertEqual(person['spans'], [(0, 12)])

        # GPE entity verification
        location = next((r for r in result if r['label'] == 'GPE'), None)
        self.assertIsNotNone(location)
        self.assertEqual(location['text'], 'New York')
        self.assertEqual(location['frequency'], 1)
        self.assertEqual(location['spans'], [(21, 29)])

        # ORG entity verification
        organization = next((r for r in result if r['label'] == 'ORG'), None)
        self.assertIsNotNone(organization)
        self.assertEqual(organization['text'], 'Microsoft')
        self.assertEqual(organization['frequency'], 1)
        self.assertEqual(organization['spans'], [(31, 40)])

    def test_get_ner_with_alternative_entities(self):
        sample_text = "Barack Obama visited New York. Obama visited Microsoft in New York City. Obama was happy."
        user_entities = ['PERSON', 'GPE']
        result = get_ner(sample_text, ner_model=self.ner_model, entities=user_entities)

        # PERSON entity verification with multiple mentions
        person = next((r for r in result if r['label'] == 'PERSON'), None)
        self.assertIsNotNone(person)
        self.assertEqual(person['alternative'], ['Barack Obama', 'Obama'])
        self.assertEqual(person['frequency'], 3)
        self.assertEqual(person['spans'], [(0, 12), (31, 36), (73, 78)])

        # GPE entity verification with multiple locations
        location = next((r for r in result if r['label'] == 'GPE'), None)
        self.assertIsNotNone(location)
        self.assertEqual(location['alternative'], ['New York', 'New York City'])
        self.assertEqual(location['frequency'], 2)
        self.assertEqual(location['spans'], [(21, 29), (58, 71)])

    def test_get_ner_with_no_entities(self):
        sample_text = "This text contains no notable entities."
        user_entities = ['PERSON', 'GPE', 'ORG']
        result = get_ner(sample_text, ner_model=self.ner_model, entities=user_entities)

        # Expecting empty result
        self.assertEqual(result, [])

    def test_get_ner_with_unsupported_language(self):
        with self.assertRaises(ValueError) as context:
            set_ner_lang('sample') 
        self.assertEqual(str(context.exception), "Language 'sample' is not supported. Available options: ['en', 'pt', 'de', 'fr', 'es', 'zh', 'ca', 'hr', 'da', 'nl', 'fi', 'el', 'it', 'ja', 'ko', 'lt', 'mk', 'xx', 'mul', 'nb', 'pl', 'ro', 'ru', 'sl', 'sv', 'uk']")   

    def test_get_ner_with_non_english_text(self):
        # Load the Portuguese NER model
        ner_model = set_ner_lang('pt')

        # Sample Portuguese text
        sample_text = "Pelé nasceu em Três Corações e jogou no Santos Futebol Clube."
        user_entities = ['PER', 'LOC', 'ORG']
        result = get_ner(sample_text, ner_model=ner_model, entities=user_entities)

        # PER entity verification
        person = next((r for r in result if r['label'] == 'PER'), None)
        self.assertIsNotNone(person)
        self.assertEqual(person['text'], 'Pelé')
        self.assertEqual(person['frequency'], 1)
        self.assertEqual(person['spans'], [(0, 4)])

        # LOC entity verification for Três Corações
        location = next((r for r in result if r['label'] == 'LOC'), None)
        self.assertIsNotNone(location)
        self.assertEqual(location['text'], 'Três Corações')
        self.assertEqual(location['frequency'], 1)
        self.assertEqual(location['spans'], [(15, 28)])

        # ORG entity verification for Santos Futebol Clube
        organization = next((r for r in result if r['label'] == 'ORG'), None)
        self.assertIsNotNone(organization)
        self.assertEqual(organization['text'], 'Santos Futebol Clube')
        self.assertEqual(organization['frequency'], 1)
        self.assertEqual(organization['spans'], [(40, 60)])


if __name__ == "__main__":
    unittest.main()
