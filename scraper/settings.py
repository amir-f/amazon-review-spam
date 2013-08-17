# Scrapy settings for AmazonScraper project
#
# For simplicity, this file contains only the most important settings by
# default. All the other settings are documented here:
#
#     http://doc.scrapy.org/topics/settings.html
#
import os
from os import path


BOT_NAME = 'The Mighty Amazon Scraper'
USER_AGENT = 'Mozilla/5.0 (Windows; U; MSIE 9.0; Windows NT 9.0; en-US)'
COOKIES_ENABLED = False
ROBOTSTXT_OBEY = False
LOG_LEVEL = 'INFO'

# Pipeline
ITEM_PIPELINES = ['scraper.pipelines.DuplicatesPipeline', 'scraper.pipelines.InstantExportPipeline']

# IO
PROJECT_PATH = os.path.split(os.path.dirname(os.path.abspath(__file__)))[0]
DATA_SET = 'same_cat'
DATA_SET_DIR = path.join(PROJECT_PATH, 'io', DATA_SET)

# Scheduler
DOWNLOAD_DELAY = 0.2
SCHEDULER_DISK_QUEUE = 'scrapy.squeue.PickleFifoDiskQueue'
SCHEDULER_MEMORY_QUEUE = 'scrapy.squeue.FifoMemoryQueue'

# Spider
SPIDER_MODULES = ['scraper.spiders']
NEWSPIDER_MODULE = 'scraper.spiders'
SPIDER_MIDDLEWARES = {
    'scrapy.contrib.spidermiddleware.depth.DepthMiddleware': None,
    'scraper.middlewares.AmazonDepthMiddleware': 901,
    'scraper.middlewares.AmazonMaxPageMiddleware': 902,
}
SPIDER_SEED_FILENAME = path.join(DATA_SET_DIR, 'seed.csv')
SPIDER_PROD_MAX_NPAGE = 50
SPIDER_MEMBER_MAX_NPAGE = 50

# Depth
DEPTH_LIMIT = 2
DEPTH_PRIORITY = 1
DEPTH_STATS_VERBOSE = True
