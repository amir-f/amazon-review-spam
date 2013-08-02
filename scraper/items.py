# Define here the models for your scraped items
#
# See documentation in:
# http://doc.scrapy.org/topics/items.html

from scrapy.item import Item, Field

# TODO comment count

class Member(Item):
    id = Field()
    fullname = Field()
    location = Field()
    isRealName = Field()

    @property
    def key(self):
        return self._values['id']

    def updateItem(self, newItem):
        missingKeys = [k for k in set(self.keys()) & (set(newItem.keys())) if self[k] is None and newItem[k] is not None]
        success = len(missingKeys) > 0
        for k in missingKeys:
            self[k] = newItem[k]

        return len(missingKeys) > 0


class Product(Item):
    id = Field()
    name = Field()
    price = Field()
    avail = Field()
    cat = Field()

    @property
    def key(self):
        return self._values['id']

    def updateItem(self, newItem):
        missingKeys = [k for k in set(self.keys()) & (set(newItem.keys())) if self[k] is None and newItem[k] is not None]
        for k in missingKeys:
            self[k] = newItem[k]

        return len(missingKeys) > 0


class Review(Item):
    productId = Field()
    memberId = Field()
    helpful = Field()
    starRating = Field()
    title = Field()
    date = Field()
    verifiedPurchase = Field()
    reviewTxt = Field()

    @property
    def key(self):
        return self._values['productId'], self._values['memberId']

    def updateItem(self, newItem):
        missingKeys = [k for k in set(self.keys()) & set(newItem.keys()) if self[k] is None and newItem[k] is not None]
        success = len(missingKeys) > 0
        for k in missingKeys:
            self[k] = newItem[k]
        return len(missingKeys) > 0