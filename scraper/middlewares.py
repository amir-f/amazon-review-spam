"""
Depth Spider Middleware for Amazon product and member pages

Requests for subsequent pages of the same product are regarded to be at the same depth
"""

__author__ = 'Amir'


from scrapy.contrib.spidermiddleware.depth import DepthMiddleware
from scrapy import log
from scrapy.http import Request


class AmazonDepthMiddleware(DepthMiddleware):
    def process_spider_output(self, response, result, spider):
        def _filter(request):
            if isinstance(request, Request):
                if (request.meta['id'], request.meta['type']) == (response.request.meta['id'], response.request.meta['type']):
                    depth = response.request.meta['depth']
                else:
                    depth = response.request.meta['depth'] + 1

                request.meta['depth'] = depth
                if self.prio:
                    request.priority -= depth * self.prio
                if depth > self.maxdepth:
                    log.msg("Ignoring link (depth > %d): %s " % (self.maxdepth, request.url),\
                        level=log.DEBUG, spider=spider)
                    return False
                elif self.stats:
                    if self.verbose_stats:
                        self.stats.inc_value('request_depth_count/%s' % depth, spider=spider)
                    self.stats.max_value('request_depth_max', depth, spider=spider)
            return True

        # base case (depth=0)
        if self.stats and 'depth' not in response.request.meta:
            response.request.meta['depth'] = 0
            if self.verbose_stats:
                self.stats.inc_value('request_depth_count/0', spider=spider)

        return (r for r in result or () if _filter(r))