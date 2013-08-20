# Define here the models for your scraped items
#
# See documentation in:
# http://doc.scrapy.org/topics/items.html
from scrapy.contrib.loader.processor import Compose, MapCompose, TakeFirst
from scrapy.item import Item, Field
from scraper.utils import only_elem, only_elem_or_default, filter_empty


def remove_comma(s):
    if isinstance(s, unicode) or isinstance(s, str):
        return s.replace(',', '')


class Member(Item):
    id = Field()
    fullname = Field(input_processor=Compose(only_elem, unicode.strip))
    location = Field(input_processor=Compose(only_elem_or_default, unicode.strip))
    badges = Field(input_processor=MapCompose(unicode.strip, unicode.upper))
    ranking = Field(input_processor=Compose(only_elem_or_default, unicode.strip, remove_comma, int))
    helpfulStat = Field(input_processor=MapCompose(int))
    reviewStat = Field(input_processor=lambda v: v if isinstance(v, int) else Compose(only_elem, unicode.strip, remove_comma, int)(v))

    @property
    def key(self):
        return self._values['id']

    @property
    def export_filename(self):
        return 'member'


class Product(Item):
    id = Field()
    name = Field(input_processor=Compose(TakeFirst(), unicode.strip))
    price = Field(input_processor=Compose(TakeFirst(), unicode.strip, remove_comma, float))
    cat = Field()
    avgStars = Field(input_processor=Compose(only_elem, float))
    nReviews = Field(input_processor=Compose(only_elem, unicode.strip, remove_comma, int))
    salesRank = Field(input_processor=Compose(unicode.strip, remove_comma, int))
    subCatRank = Field(input_processor=Compose(only_elem_or_default, unicode.strip, remove_comma, int))
    subCat = Field(input_processor=Compose(only_elem_or_default, unicode.strip))

    @property
    def export_filename(self):
        return 'product'

    @property
    def key(self):
        return self._values['id']


class Review(Item):
    id = Field(input_processor=Compose(only_elem))
    productId = Field()
    memberId = Field()
    helpful = Field(input_processor=MapCompose(int))
    starRating = Field(input_processor=Compose(only_elem, float))
    title = Field(input_processor=Compose(only_elem_or_default, unicode.strip))
    date = Field(input_processor=Compose(only_elem_or_default, unicode.strip))
    verifiedPurchase = Field(input_processor=Compose(bool))
    reviewTxt = Field(input_processor=Compose(MapCompose(unicode.strip), filter_empty, lambda vs: ' '.join(vs)))
    nComments = Field(input_processor=Compose(int))
    vine = Field(input_processor=Compose(bool))

    @property
    def key(self):
        return self._values['id']

    @property
    def export_filename(self):
        return 'review'
