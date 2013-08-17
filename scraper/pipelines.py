# Define your item pipelines here
#
# Don't forget to add your pipeline to the ITEM_PIPELINES setting
# See: http://doc.scrapy.org/topics/item-pipeline.html

import errno
from scrapy.contrib.exporter import JsonLinesItemExporter
from os import path
import os
from scrapy.exceptions import DropItem


class InstantExportPipeline(object):
    def __init__(self):
        self.xporters = {}

    def process_item(self, item, spider):
        """
        Writes the item to output
        """

        # create the output file for a new class of item per spider
        settings = spider.crawler.settings
        if item.__class__ not in self.xporters[spider.name]:
            filename = '%s.json' % item.export_filename
            dirpath = path.join(settings.get('IO_PATH', 'io'), settings['DATA_SET'])
            _mkdir_p(dirpath)
            file_h = open(path.join(dirpath, filename), 'w')
            xporter = JsonLinesItemExporter(file=file_h)
            xporter.start_exporting()
            self.xporters[spider.name][item.__class__] = (file_h, xporter)

        xporter = self.xporters[spider.name][item.__class__][1]
        xporter.export_item(item)
        return item

    def open_spider(self, spider):
        self.xporters[spider.name] = {}

    def close_spider(self, spider):
        """
        Finishes writing
        """

        for file_h, xporter in self.xporters[spider.name].values():
            xporter.finish_exporting()
            file_h.close()


def _mkdir_p(path):
    try:
        os.makedirs(path)
    except OSError as exc:
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else:
            raise


class DuplicatesPipeline(object):

    def __init__(self):
        self.ids_seen = set()

    def process_item(self, item, spider):
        itm_id = (item.export_filename, item.key)
        if itm_id in self.ids_seen:
            raise DropItem("Duplicate item found: %s" % str(itm_id))
        else:
            self.ids_seen.add(itm_id)
            return item