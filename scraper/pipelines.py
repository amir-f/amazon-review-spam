# Define your item pipelines here
#
# Don't forget to add your pipeline to the ITEM_PIPELINES setting
# See: http://doc.scrapy.org/topics/item-pipeline.html

from scrapy.conf import settings
from scrapy.contrib.exporter import CsvItemExporter
from scrapy.exceptions import DropItem
from scraper.items import Member, Product, Review


class ExportPipeline(object):
    def __init__(self):
        # set of items
        self._reviews = {}
        self._members = {}
        self._products = {}
        self._type_to_set = { Member:self._members, Product:self._products, Review:self._reviews }

        # Read filenames from settings
        self._revFile = open( settings['DATA_SET_DIR'] + '/' + settings.get('OUTPUT_FILE_REVIEW', 'revs.csv'), 'wb' )
        self._memberFile = open( settings['DATA_SET_DIR'] + '/' + settings.get('OUTPUT_FILE_MEMBER', 'members.csv'), 'wb' )
        self._prodFile = open( settings['DATA_SET_DIR'] + '/' + settings.get('OUTPUT_FILE_PRODUCT', 'product.csv'), 'wb' )
        self._outputFiles = [self._revFile, self._memberFile, self._prodFile]

        # Create item exporters given each filename
        self._revXport = CsvItemExporter(file=self._revFile, include_headers_line=True)
        self._memberXport = CsvItemExporter(file=self._memberFile, include_headers_line=True)
        self._prodXport = CsvItemExporter(file=self._prodFile, include_headers_line=True)

        self._type_to_xp = { Member:self._memberXport, Product:self._prodXport, Review:self._revXport }


    def process_item(self, item, spider):
        """
        Adds the item to the set and overwrites the set value if the new value is richer in attributes
        """

        set = self._type_to_set[item.__class__]
        k = item.key

        if k not in set:
            set[k] = item
        else:
            success = set[k].updateItem(item)
            if not success:
                raise DropItem("Duplicate")

        return item


    def close_spider(self, spider):
        """
        Writes everything to file
        """

        # The writing takes place at the end in order to let the set of items be filled with items richest in terms of attribute
        for type in self._type_to_set:
            set = self._type_to_set[type]
            xp = self._type_to_xp[type]

            xp.start_exporting()
            for item in set.values():
                xp.export_item(item)
            xp.finish_exporting()

        for file in self._outputFiles:
            file.close()


class InstantExportPipeline(object):
    def __init__(self):
        # set of items
        self._reviews = set()
        self._members = set()
        self._products = set()
        self._type_to_set = { Member:self._members, Product:self._products, Review:self._reviews }

        # Read filenames from settings
        self._revFile = open( settings.get('OUTPUT_FILE_REVIEW', 'io/revs.csv'), 'wb' )
        self._memberFile = open( settings.get('OUTPUT_FILE_MEMBER', 'io/members.csv'), 'wb' )
        self._prodFile = open( settings.get('OUTPUT_FILE_PRODUCT', 'io/product.csv'), 'wb' )

        # Create item exporters given each filename
        self._revXport = CsvItemExporter(file=self._revFile, include_headers_line=True)
        self._memberXport = CsvItemExporter(file=self._memberFile, include_headers_line=True)
        self._prodXport = CsvItemExporter(file=self._prodFile, include_headers_line=True)

        self._type_to_xp = { Member:self._memberXport, Product:self._prodXport, Review:self._revXport }


    def process_item(self, item, spider):
        """
        writes the item to output
        """

        set = self._type_to_set[item.__class__]
        if item.key not in set:
            set.add(item.key)
            xp = self._type_to_xp[item.__class__]
            xp.export_item(item)
            return item
        else:
            raise DropItem("Duplicate item found: %s" % item)


    def open_spider(self, spider):
        for xp in self._type_to_xp.values():
            xp.start_exporting()


    def close_spider(self, spider):
        """
        Finishes writing
        """

        for xp in self._type_to_xp.values():
            xp.finish_exporting()
        for file in self._outputFiles:
            file.close()
