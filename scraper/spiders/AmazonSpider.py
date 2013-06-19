"""
This modules implements a crawler to scrape Amazon web site based on a seed list of
products and member pages.

"""
import csv
import re
import math

from scrapy.selector import HtmlXPathSelector
from scrapy.utils.misc import arg_to_iter
from scraper.items import Member, Product, Review
from scrapy import log


from scrapy.http import Request
from scrapy.spider import BaseSpider
from scrapy.conf import settings

class AmazonSpider(BaseSpider):
    """
    A spider to crawl Amazon product and member pages
    """

    name = 'AmazonSpider'
    allowed_domains = ['amazon.com']

    _PROD_TYPE = 'p'
    _MEMBER_TYPE = 'm'

    _inParenthesis_RE = re.compile(r'(?<=\().+(?=\))')
    _productID_RE = re.compile('(?<=/dp/)\w+(?=(/|/.+)?)')

    def __init__(self, *a, **kw):
        super(AmazonSpider, self).__init__(*a, **kw)
        self._MIN_FAV = settings['SPIDER_MINIMUM_FAVORABLE']

    def _firstElemOrElse(self, l, e):
        if len(l) > 0:
            return l[0]
        else:
            return e

    def _member_rev_url(self, id, page = 1):
        return r'http://www.amazon.com/gp/cdp/member-reviews/%s?page=%d' % (id, page)

    def _product_rev_url(self, id, page = 1):
        return r'http://www.amazon.com/product-reviews/%s?showViewpoints=0&pageNumber=%d' % (id, page)

    def is_favorable(self, rating):
        return float(rating[0])/rating[1] >= self._MIN_FAV

    def parse_product_page(self, response):
        """
        parses a single product page and makes requests for subsequent pages
        """

        hxs = HtmlXPathSelector(response)
        # Find out the number of review pages
        noPagesXPath = '//table[@class="CMheadingBar"]//span[@class="paging"]//a[following-sibling::a[1][starts-with(text(),"Next")]]/text()'
        noPages = int( self._firstElemOrElse(hxs.select(noPagesXPath).re(r'\d+'), '1') )

        #skip vastly popular products: those with large number of reviews
        if noPages > settings['SPIDER_PROD_POPULARITY_NPAGE_THRESH']:
            log.msg('Skipping product (noPages > PROD_POPULARITY_NPAGE_THRESH): %s' % response.url, level=log.DEBUG, spider=self)
            return

        # Start parsing this page
        log.msg('Parsing product %s : p%d' % (response.meta['id'], response.meta['page']), level=log.INFO, spider=self)

        # yield the product itself. It suffices to do it only on the first page
        productID = response.meta['id']
        if response.meta['page'] == 1:
            productName = hxs.select('//table//h1[@class]/div[contains(text(), "Customer Reviews")]/following-sibling::div/a/text()').extract()[0]
            yield Product(id=productID, name = None, price = None, cat = None, avail=True)

        # yield reviews and members posting them
        reviewsXPath = '//table[@id="productReviews"]//td/div'
        revElems = hxs.select(reviewsXPath)
        for rev in revElems:
            # skip if the reviewer is anonymous
            memberElem = rev.select('.//div[contains(text(), "By")]/following-sibling::div/a[contains(@href, "profile")]')
            if not len(memberElem):
                continue

            # populate review data
            memberID = memberElem.select('self::a/@href').re(r'(?<=profile/)\w+(?=/)')[0].upper()
            helpful_tmp = rev.select('div[contains(text(), "helpful")]/text()').re(r'\d+')
            helpful = [int(i) for i in helpful_tmp]
            starRating_tmp = rev.select('.//span/span[contains(@class, "swSprite")]/@title').re(r'\d+(?:\.\d+)?')
            starRating = [math.trunc(float(i)) for i in starRating_tmp]
            title = self._firstElemOrElse(rev.select('div/span/b[1]/text()').extract(), None)
            date = self._firstElemOrElse(rev.select('div/span/nobr/text()').extract(), None)
            verifiedPurchase = len(  rev.select('.//span[contains(@class, "crVerifiedStripe")]')  ) > 0
            reviewTxtListElem = rev.select('text()').extract()
            reviewTxt = ''.join(reviewTxtListElem).strip()

            # yield the review
            yield Review(productId=productID, memberId=memberID, helpful=helpful, starRating=starRating, title=title,
                            date=date, verifiedPurchase=verifiedPurchase, reviewTxt=reviewTxt)

            # yield the reviewer
            memberFullName = self._firstElemOrElse(memberElem.select('span/text()').extract(), None)
            m = self._inParenthesis_RE.search(memberElem[0]._root.tail)
            if m is not None:
                location = m.group(0)
            else:
                location = None
            isRealName = len(  memberElem.select('following-sibling::*//span[contains(@class, "BadgeRealName")]')  ) > 0

            yield Member(id=memberID, fullname=memberFullName, location=location, isRealName=isRealName)

            # if the reviewer has given favorable review, crawl it
            if self.is_favorable(starRating):
                pageURL = self._member_rev_url(memberID)
                req_meta = {'id':memberID, 'type':self._MEMBER_TYPE ,'page':1}
                yield Request(pageURL, callback=self.parse_member_page, meta=req_meta)

        # request subsequent pages to be downloaded
        if response.meta['page'] < noPages:
            yield self._product_successor_page_request(response)


    def _product_successor_page_request(self, response):
        next_p = response.meta['page'] + 1
        pageURL = self._product_rev_url(response.meta['id'], next_p)
        req_meta = response.meta
        req_meta['page'] = next_p
        return Request(pageURL, callback=self.parse_product_page, meta=req_meta)

    def parse_member_page(self, response):
        """
        Parses a member reviews page and makes requests for subsequent pages
        """

        # skip if too many pages of a member have been crawled
        p = response.meta['page']
        memberID = response.meta['id']
        if p > settings['SPIDER_MEMBER_MAX_NPAGE']:
            log.msg(r'Skipping member page (p > SPIDER_MEMBER_MAX_NPAGE): %s' % response.url, level=log.DEBUG, spider=self)
            return

        # start parsing the page
        log.msg(r'Parsing member %s : p%d' % (memberID, p), level=log.INFO, spider=self)

        # yielding the member itself
        hxs = HtmlXPathSelector(response)
        revBodiesXPath = '//table//td[not(@width)]//table//tr[not(@valign)]/td[@class="small"]/div'
        revHeadersXPath = '//table//td[not(@width)]//table//tr[@valign]/td[@align][2]//table[@class="small"]'
        userXPath = '//table//td//b[@class="h1"]'
        memberElem = hxs.select(userXPath)[0]
        member_FullName = memberElem.select('br[1]')[0]._root.tail.strip()
        loc_tmp = memberElem.select('span[@class="reviewsRSSIcon"]')
        m = self._inParenthesis_RE.search(loc_tmp[0]._root.tail or '')
        if m is not None:
            member_Location = m.group(0)
        else:
            member_Location = None
        member_isRealName = len(  memberElem.select('descendant::a/img[contains(@alt,"REAL NAME")]')  ) > 0

        yield Member(id=memberID, fullname=member_FullName, location=member_Location, isRealName=member_isRealName)

        # yield each review
        revBodyElems = hxs.select(revBodiesXPath)
        revHeaderElems = hxs.select(revHeadersXPath)
        for revHeader, revBody in zip(revHeaderElems,revBodyElems):

            # if the product is not available skip the review
            productURLElem = revBody.select('.//b/span[contains(text(), "This review is from")]/following-sibling::a/@href')
            if not len(productURLElem):
                continue

            # populating review data
            productID = self._productID_RE.search(productURLElem.extract()[0]).group(0).upper()
            helpful_tmp = revBody.select('div[contains(text(), "helpful")]/text()').re(r'\d+')
            helpful = [int(i) for i in helpful_tmp]
            starRating_tmp = revBody.select('.//span/img[contains(@title, "stars")]/@title').re(r'\d+(?:\.\d+)?')
            starRating = [math.trunc(float(i)) for i in starRating_tmp]
            title = self._firstElemOrElse(revBody.select('div/b[1]/text()').extract(), None)
            date = self._firstElemOrElse(revBody.select('div/nobr/text()').extract(), None)
            verifiedPurchase = len(  revBody.select('.//span[contains(@class, "crVerifiedStripe")]')  ) > 0
            reviewTxtListElem = revBody.select('text()').extract()
            reviewTxt = ''.join(reviewTxtListElem).strip()

            # yield the review
            yield Review(productId=productID, memberId=memberID, helpful=helpful, starRating=starRating, title=title,
                            date=date, verifiedPurchase=verifiedPurchase, reviewTxt=reviewTxt)

            # yield the product
            productName = revBody.select('.//div[@class="tiny"]//span[contains(text(), "This review is from")]/following-sibling::a/text()').extract()[0]
            m = self._inParenthesis_RE.search(productName)
            if m is not None:
                productCat = m.group(0)
            else:
                productCat = None
            productPrice_tmp = self._firstElemOrElse(revHeader.select('.//span[@class="price"]//span'), None)
            if productPrice_tmp is not None:
                productPrice = productPrice_tmp._root.tail.strip()
            else:
                productPrice = None
            productAvail_tmp = revHeader.select('descendant::tr//b[contains(text(), "Availability")]')
            productAvail = "unavailable" not in productAvail_tmp[0]._root.tail.lower()

            yield Product(id=productID, name=productName, price=productPrice, avail=productAvail, cat=productCat)

            # if the product has been rated favorably, crawl it
            if self.is_favorable(starRating):
                pageURL = self._product_rev_url(productID)
                req_meta = {'id':productID, 'type':self._PROD_TYPE ,'page':1}
                yield Request(pageURL, callback=self.parse_product_page, meta=req_meta)


    #make request for subsequent pages
        noPagesXPath = '//table//table//td[@class="small"]/b/text()'
        page_number = hxs.select(noPagesXPath).re(r'\d+\s+\|')
        if len(page_number) > 0:
            yield self._member_successor_page_request(response)

    def _member_successor_page_request(self, response):
        next_p = response.meta['page'] + 1
        pageURL = self._member_rev_url(response.meta['id'], next_p)
        req_meta = response.meta
        req_meta['page'] = next_p
        return Request(pageURL, callback=self.parse_member_page, meta=req_meta)

    def start_requests(self):
        reqs = []
        with open( settings['DATA_SET_DIR'] + '/' + settings.get('SPIDER_SEED_FILENAME'), 'r' ) as readFile:
            reader = csv.DictReader(readFile)
            for seed in reader:
                # forming the url for each seed ID
                type = seed['Type']
                id = seed['ID']

                if type == self._PROD_TYPE:
                    url = self._product_rev_url(seed['ID'])
                    cb = self.parse_product_page
                elif type == self._MEMBER_TYPE:
                    url = self._member_rev_url(seed['ID'])
                    cb = self.parse_member_page
                else:
                    raise ValueError( "The type of seed with ID %s was %s. Expected 'p' or 'm'" % (seed['ID'], seed['Type']) )

                reqs.extend(  arg_to_iter(Request(url, meta={'id': id, 'type':type, 'page':1}, callback=cb))  )

        return reqs

