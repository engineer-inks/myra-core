from ink.core.forge.joins.text import testing, datasets

class SimpleIrisTest(testing.SparkTestCase):
    def setUp(self):
        self.iris = datasets.iris()

    def test_set_has_150_samples(self):
        expected = 150
        actual = self.iris.count()

        self.assertEqual(expected, actual)
