import unittest

from src.services.textrow_extractor import extract_rows as X

class T(unittest.TestCase):
    def test_simple(self):
        self.assertEqual(X('Total assets $ 27,787 $ 26,075'), [{'label': 'Total assets', 'values': ['27787', '26075']}])
    def test_negative_paren(self):
        r = X('Interest expense (433) (403) (397)')[0]
        self.assertEqual(r['values'], ['-433', '-403', '-397'])
    def test_dash(self):
        self.assertEqual(X('Assets held for sale — 683')[0]['values'], ['—', '683'])
    def test_dollar_attached(self):
        self.assertEqual(X('Operating revenues $3,792 $3,930 $3,777')[0]['values'], ['3792', '3930', '3777'])
    def test_concatenated(self):
        r = X('Goodwill 1,143 1,139 Postretirement benefit assets — 193 Other 379 242')
        self.assertEqual([x['label'] for x in r], ['Goodwill', 'Postretirement benefit assets', 'Other'])
    def test_label_with_inline_numbers(self):
        r = X('Accounts receivable, net of allowance of $2 and $4, respectively 334 271')
        self.assertEqual(len(r), 1); self.assertEqual(r[0]['values'], ['334', '271'])
    def test_decimal(self):
        self.assertEqual(X('Basic $ 4.51 $ 6.96 $ 3.91')[0]['values'], ['4.51', '6.96', '3.91'])
    def test_prose_no_rows(self):
        self.assertEqual(X('The Company has operations in 14 states.'), [])


def test_text_to_table_rows_shape():
    from src.services.textrow_extractor import text_to_table_rows
    assert text_to_table_rows("Total assets $ 27,787 $ 26,075") == [["Total assets", "27787", "26075"]]
