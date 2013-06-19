#!/bin/sh
export TORSOCKS_CONF_FILE=/home/amir/.tor/torsocks.conf
nohup torsocks scrapy crawl AmazonSpider &
