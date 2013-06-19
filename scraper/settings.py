# Scrapy settings for AmazonScraper project
#
# For simplicity, this file contains only the most important settings by
# default. All the other settings are documented here:
#
#     http://doc.scrapy.org/topics/settings.html
#


BOT_NAME = 'The Mighty Amazon Scraper'
BOT_VERSION = '1.0'
USER_AGENT = 'Mozilla/5.0 (Windows; U; MSIE 9.0; Windows NT 9.0; en-US)'
COOKIES_ENABLED = False
ROBOTSTXT_OBEY = False
LOG_LEVEL = 'INFO'


# Pipeline
ITEM_PIPELINES = ['scraper.pipelines.ExportPipeline']


# IO
import os
PROJECT_PATH = os.path.split(os.path.dirname(os.path.abspath(__file__)))[0]
DATA_SET = 'spam'
DATA_SET_DIR = r'%s/io/%s'%(PROJECT_PATH, DATA_SET)
OUTPUT_FILE_REVIEW = 'Rev.csv'
OUTPUT_FILE_MEMBER = 'Member.csv'
OUTPUT_FILE_PRODUCT = 'Product.csv'


# Scheduler
#DOWNLOAD_DELAY = 1
SCHEDULER_DISK_QUEUE = 'scrapy.squeue.PickleFifoDiskQueue'
SCHEDULER_MEMORY_QUEUE = 'scrapy.squeue.FifoMemoryQueue'


# Spider
SPIDER_MODULES = ['scraper.spiders']
NEWSPIDER_MODULE = 'scraper.spiders'
SPIDER_MIDDLEWARES = {
    'scraper.middlewares.AmazonDepthMiddleware': 901,
    'scrapy.contrib.spidermiddleware.depth.DepthMiddleware': None,
    }
SPIDER_SEED_FILENAME = 'Seed.csv'
SPIDER_PROD_POPULARITY_NPAGE_THRESH = 30
SPIDER_MEMBER_MAX_NPAGE = 10
SPIDER_MINIMUM_FAVORABLE = 1.0/5.0


# Depth
DEPTH_LIMIT = 2
DEPTH_PRIORITY = 1
DEPTH_STATS_VERBOSE = True
