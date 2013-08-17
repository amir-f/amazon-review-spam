from scrapy.contrib.loader import ItemLoader

__author__ = 'amir'


PROD_TYPE = 'p'
MEMBER_TYPE = 'm'


def filter_empty(values):
    return [value for value in values if value]


def only_elem(list_of_elems):
    assert len(list_of_elems) == 1
    return list_of_elems[0]


def only_elem_or_default(iterable, default=None):
    assert len(iterable) <= 1
    if iterable:
        for item in iterable:
            return item
    return default


class SingleValItemLoader(ItemLoader):

    def __init__(self, item=None, **context):
        super(SingleValItemLoader, self).__init__(item=item, **context)
        self._values = dict()

    def _add_value(self, field_name, value):
        processed_value = self._process_input_value(field_name, value)
        if processed_value or processed_value == 0:
            self._values[field_name] = processed_value
