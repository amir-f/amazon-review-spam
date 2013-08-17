"""
Depth Spider Middleware for Amazon product and member pages

Requests for subsequent pages of the same product are regarded to be at the same depth
"""
from scraper.utils import MEMBER_TYPE, PROD_TYPE

__author__ = 'Amir'


from scrapy.contrib.spidermiddleware.depth import DepthMiddleware
from scrapy.http import Request
from scrapy import log


class AmazonDepthMiddleware(DepthMiddleware):
    def process_spider_output(self, response, result, spider):
        def _filter(request):
            if isinstance(request, Request):
                if (request.meta['id'], request.meta['type']) == (response.meta['id'], response.meta['type']):
                    depth = response.meta['depth']
                else:
                    depth = response.meta['depth'] + 1
                if depth > self.maxdepth:
                    log.msg(format="Ignoring link (depth > %(maxdepth)d): %(requrl)s ",
                            level=log.DEBUG, spider=spider,
                            maxdepth=self.maxdepth, requrl=request.url)
                    return False
                # If we won't be able to crawl new items retrieved from reviews, then don't crawl the reviews
                if 'page' in request.meta and depth >= self.maxdepth:
                    log.msg(format="Ignoring review page request when items won't be crawled: depth == %(maxdepth)d %(itm_type)s %(itm_id)s",
                            level=log.DEBUG, spider=spider,
                            maxdepth=self.maxdepth, itm_type=request.meta['type'], itm_id=request.meta['id'])
                    return False
                request.meta['depth'] = depth
                if self.prio:
                    request.priority -= depth * self.prio
                if self.stats:
                    if self.verbose_stats:
                        self.stats.inc_value('request_depth_count/%s' % depth, spider=spider)
                    self.stats.max_value('request_depth_max', depth, spider=spider)
            return True

        # base case (depth=0)
        if self.stats and 'depth' not in response.meta:
            response.meta['depth'] = 0
            if self.verbose_stats:
                self.stats.inc_value('request_depth_count/0', spider=spider)

        return (r for r in result or () if _filter(r))


type_2_maxpage = {MEMBER_TYPE: 'SPIDER_MEMBER_MAX_NPAGE', PROD_TYPE: 'SPIDER_PROD_MAX_NPAGE'}


class AmazonMaxPageMiddleware(object):
    """
    Filters those request which ask for pages of reviews further than the max specified
    Assumes it is run after DepthMiddleware so the depth values are calculated
    """

    def __init__(self, settings):
        self.settings = settings

    @classmethod
    def from_crawler(cls, crawler):
        return cls(crawler.settings)

    def process_spider_output(self, response, result, spider):
        def _filter(request):
            # check if this is a request for pages of reviews
            if isinstance(request, Request) and 'page' in request.meta:
                settings_key = type_2_maxpage[request.meta['type']]
                if request.meta['page'] > self.settings[settings_key]:
                    log.msg(format='Skipping page (%(page_n)d > MAX_NPAGE): %(req_url)s',
                            page_n=request.meta['page'], req_url=request.url, level=log.DEBUG, spider=spider)
                    return False
            return True

        return (r for r in result or () if _filter(r))