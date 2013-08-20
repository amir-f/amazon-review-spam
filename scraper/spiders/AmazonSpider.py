"""
This modules implements a crawler to scrape Amazon web site based on a seed list of
products and member pages.

"""

import csv
from scrapy.selector import HtmlXPathSelector
from scrapy.utils.misc import arg_to_iter
from scraper.items import Member, Product, Review
from scrapy import log
from scrapy.http import Request
from scrapy.spider import BaseSpider
from scraper.utils import SingleValItemLoader, only_elem_or_default, MEMBER_TYPE, PROD_TYPE
from lxml import etree


review_id_re = r'(?<=review/)\w+(?=/)'
number_digit_grpd = r'\d+(?:,\d+)*'
cat_name_re = r'\w+(?:\s+(?:\w+|\&))*'
sales_rank_re = r'#(%s)\s+(?:\w+\s+)?in\s+(%s)' % (number_digit_grpd, cat_name_re)
price_re = r'\$(%s\.\d\d)' % number_digit_grpd
star_rating_re = r'(\d+(?:\.\d+)?)\s+out\s+of'
product_url_id_re = r'(?<=/dp/)\w+(?=/)'
member_url_id_re = r'(?<=profile/)\w+(?=/)'


def xpath_lower_case(context, a):
    return [s.lower() for s in a]


ns = etree.FunctionNamespace(None)
ns['lower-case'] = xpath_lower_case


def _member_rev_url(m_id, page=1):
    return 'http://www.amazon.com/gp/cdp/member-reviews/%s?sort_by=MostRecentReview&page=%d' % (m_id, page)


def _member_profile_url(m_id):
    return 'http://www.amazon.com/gp/pdp/profile/%s/' % m_id


def _product_rev_url(p_id, page=1):
    return 'http://www.amazon.com/product-reviews/%s?sortBy=bySubmissionDateDescending&showViewpoints=0&pageNumber=%d' % (p_id, page)


def _product_details_url(p_id):
    return 'http://www.amazon.com/dp/%s/' % p_id


class AmazonSpider(BaseSpider):
    """
    A spider to crawl Amazon product and member pages
    """

    name = 'amazon_spider'
    allowed_domains = ['amazon.com']

    def __init__(self, **kwargs):
        super(AmazonSpider, self).__init__(**kwargs)
        self.rev_req_param = {MEMBER_TYPE: (_member_rev_url, self.parse_member_rev_page),
                              PROD_TYPE: (_product_rev_url, self.parse_product_rev_page)}
        self.item_req_param = {MEMBER_TYPE: (_member_profile_url, self.parse_member_profile_page),
                               PROD_TYPE: (_product_details_url, self.parse_product_details_page)}

    def _rev_page_request(self, id_, type_, p=1):
        rev_url_gen, cb = self.rev_req_param[type_]
        req_meta = {'id': id_, 'type': type_, 'page': p}
        return Request(rev_url_gen(id_, p), callback=cb, meta=req_meta)

    def _item_page_request(self, id_, type_):
        item_url_gen, cb = self.item_req_param[type_]
        req_meta = {'id': id_, 'type': type_}
        return Request(item_url_gen(id_), callback=cb, meta=req_meta)

    def _successor_page_request(self, response):
        type_, id_ = response.meta['type'], response.meta['id']
        next_p = response.meta['page'] + 1
        return self._rev_page_request(id_, type_, next_p)

    def parse_product_category_page(self, response):
        """
        Parses a page of same category products and yields all the products in the first page
        """
        log.msg('Parsing products of the same category as: %s' % response.meta['id'], level=log.INFO, spider=self)
        # from scrapy.shell import inspect_response
        # inspect_response(response)
        settings = self.crawler.settings
        hxs = HtmlXPathSelector(response)

        product_ids = hxs.select('//body//div[@id="zg_centerListWrapper"]//a[img]//@href').re(product_url_id_re)
        n_same_cat = 0
        for product_id in product_ids:
           yield self._rev_page_request(str(product_id), PROD_TYPE)
           n_same_cat += 1
           if n_same_cat > settings['SPIDER_MAX_SAME_CAT']:
                break
  
    def parse_product_details_page(self, response):
        """
        Extracts information from a product page and yields its review page and pages of products in the same category
        """
        log.msg('Parsing product info: %s' % response.meta['id'], level=log.INFO, spider=self)
        # from scrapy.shell import inspect_response
        # inspect_response(response)
        hxs = HtmlXPathSelector(response)
        product_id = response.meta['id']

        # yield product details
        name = hxs.select('//body//span[@id="btAsinTitle"]/text()').extract() or \
               hxs.select('//body//div[@id="title_feature_div"]//h1/text()').extract()
        if not name:
            name = hxs.select('//head/title/text()').re(r'(?:Amazon:\s+)?([^:]+)')
        price = hxs.select('//body//span[@id="actualPriceValue"]//text()').re(price_re) or \
                hxs.select('//body//div[@id="price"]//span[contains(@class, "a-color-price")]/text()').re(price_re) or \
                hxs.select('//body//div[@id="priceBlock"]//span[@class="priceLarge"]/text()').re(price_re)
        avg_stars, n_reviews = None, None
        reviews_t = hxs.select('//body//div[@id="centerCol"]//div[@id="averageCustomerReviews"]')
        if reviews_t:
            avg_stars = reviews_t.select('./span[contains(@title, "star")]/@title').re(star_rating_re)
            n_reviews = reviews_t.select('./a[contains(@href, "product-reviews")]/text()').re(number_digit_grpd)
        else:
            reviews_t = hxs.select('(//body//*[self::div[@class="buying"] or self::form[@id="handleBuy"]]//span[@class="crAvgStars"])[1]')
            if reviews_t:
                avg_stars = reviews_t.select('.//span[contains(@title, "star")]/@title').re(star_rating_re)
                n_reviews = reviews_t.select('./a[contains(@href, "product-reviews")]/text()').re(number_digit_grpd)

        sales_rank, cat, sub_cat_rank, sub_cat = [None]*4
        best_sellers_href, sub_cat_href = [], []
        parent_node = hxs.select('//body//li[@id="SalesRank"]')
        if parent_node:
            sales_rank, cat = parent_node.select('.//text()').re(sales_rank_re) or [None]*2
            best_sellers_href = parent_node.select('a[contains(lower-case(text()), "see top") and (contains(@href, "/best-sellers") or contains(@href, "/bestsellers"))]/@href').extract()
            sub_cat_node = parent_node.select('.//li[@class="zg_hrsr_item"][1]')
            if sub_cat_node:
                sub_cat_rank = sub_cat_node.select('./span[@class="zg_hrsr_rank"]/text()').re(number_digit_grpd)
                sub_cat = sub_cat_node.select('(./span[@class="zg_hrsr_ladder"]//a)[position()=last()]/text()').extract()
                sub_cat_href = sub_cat_node.select('(./span[@class="zg_hrsr_ladder"]//a)[position()=last()]/@href').extract()
        if not parent_node:
            parent_node = hxs.select('//body//div[@id="detailBullets"]//span[contains(b/text(), "Amazon Best Sellers Rank")]')
            if parent_node:
                sales_rank, cat = parent_node.select('.//text()').re(sales_rank_re) or [None]*2
                best_sellers_href = parent_node.select('a[contains(lower-case(text()), "see top") and (contains(@href, "/best-sellers") or contains(@href, "/bestsellers"))]/@href').extract()
                sub_cat_node = parent_node.select('.//li[@class="zg_hrsr_item"][1]')
                if sub_cat_node:
                    sub_cat_rank = sub_cat_node.select('./span[@class="zg_hrsr_rank"]/text()').re(number_digit_grpd)
                    sub_cat = sub_cat_node.select('(./span[@class="zg_hrsr_ladder"]//a)[position()=last()]/text()').extract()
                    sub_cat_href = sub_cat_node.select('(./span[@class="zg_hrsr_ladder"]//a)[position()=last()]/@href').extract()
        if not parent_node:
            parent_node = hxs.select('//body//tr[@id="SalesRank"]')
            if parent_node:
                sales_rank, cat = parent_node.select('.//text()').re(sales_rank_re) or [None]*2
                best_sellers_href = parent_node.select('.//a[contains(lower-case(text()), "see top") and (contains(@href, "/best-sellers") or contains(@href, "/bestsellers"))]/@href').extract()
                sub_cat_node = parent_node.select('.//li[@class="zg_hrsr_item"][1]')
                if sub_cat_node:
                    sub_cat_rank = sub_cat_node.select('./span[@class="zg_hrsr_rank"]/text()').re(number_digit_grpd)
                    sub_cat = sub_cat_node.select('(./span[@class="zg_hrsr_ladder"]//a)[position()=last()]/text()').extract()
                    sub_cat_href = sub_cat_node.select('(./span[@class="zg_hrsr_ladder"]//a)[position()=last()]/@href').extract()

        product = SingleValItemLoader(item=Product(), response=response)
        product.add_value('id', product_id)
        product.add_value('name', name)
        product.add_value('price', price)
        product.add_value('avgStars', avg_stars)
        product.add_value('nReviews', n_reviews)
        product.add_value('salesRank', sales_rank)
        product.add_value('cat', cat)
        product.add_value('subCatRank', sub_cat_rank)
        product.add_value('subCat', sub_cat)
        yield product.load_item()

        # yield same category products
        same_cat_href = only_elem_or_default(sub_cat_href or best_sellers_href)
        if same_cat_href:
            yield Request(same_cat_href, callback=self.parse_product_category_page, meta={'id': product_id, 'type': PROD_TYPE})

        #yield the product reviews page.
        yield self._rev_page_request(product_id, PROD_TYPE)

    def parse_product_rev_page(self, response):
        """
        parses a single product page and makes requests for subsequent pages
        """

        # Start parsing this page
        log.msg('Parsing product reviews: %s p%d' % (response.meta['id'], response.meta['page']), level=log.INFO, spider=self)
        # from scrapy.shell import inspect_response
        # inspect_response(response)

        hxs = HtmlXPathSelector(response)

        # yield reviews and members posting them
        product_id = response.meta['id']
        revElems = hxs.select('//table[@id="productReviews"]//td/div')
        for rev in revElems:
            # yield review info
            review = SingleValItemLoader(item=Review(), response=response)
            member_id = only_elem_or_default(rev.select('.//div[contains(text(), "By")]/following-sibling::div/a[contains(@href, "profile")]/@href').re(member_url_id_re))
            if member_id:
                member_id = str(member_id)
            star_rating_tmp = rev.select('.//span/span[contains(@class, "swSprite")]/@title').re(star_rating_re)
            if not star_rating_tmp:
                # It is probably a manufacturer response, not a review
                continue
            review.add_value('starRating', star_rating_tmp)
            review.add_value('id', rev.select('.//span[@class="tiny"]/a[contains(text(), "Permalink")]/@href').re(review_id_re))
            review.add_value('productId', product_id)
            review.add_value('memberId', member_id)
            review.add_value('helpful', rev.select('div[contains(text(), "helpful")]/text()').re(r'\d+'))
            review.add_value('title', rev.select('.//span[contains(span/@class, "swSprite")]/following-sibling::span//b/text()').extract())
            review.add_value('date', rev.select('div/span/nobr/text()').extract())
            review.add_value('verifiedPurchase', rev.select('.//span[contains(@class, "crVerifiedStripe")]'))
            review.add_value('reviewTxt', rev.select('text()').extract())
            nComments_tmp = only_elem_or_default(rev.select('.//div//div/a/text()').re(r'Comments\s+\((\d+)\)'), '0')
            review.add_value('nComments', nComments_tmp)
            review.add_value('vine', rev.select('.//span/b[contains(text(), "Customer review from the Amazon Vine Program")]'))
            yield review.load_item()

            # yield the reviewer
            yield self._item_page_request(member_id, MEMBER_TYPE)

        # request subsequent pages to be downloaded
        # Find out the number of review pages
        noPagesXPath = '(//table[@class="CMheadingBar"])[1]//span[@class="paging"]//a[following-sibling::a[1][starts-with(text(),"Next")]]/text()'
        noPages = int(only_elem_or_default(hxs.select(noPagesXPath).re(r'\d+'), default='1'))
        if response.meta['page'] < noPages:
            yield self._successor_page_request(response)

    def parse_member_profile_page(self, response):
        """
        Parses member profile page
        """

        log.msg('Parsing member info: %s' % response.meta['id'], level=log.INFO, spider=self)
        # from scrapy.shell import inspect_response
        # inspect_response(response)
        member_id = response.meta['id']
        hxs = HtmlXPathSelector(response)

        # Abort if member reviews are not available
        if hxs.select('//body//b[@class="h1" and contains(text(), "this customer\'s list of reviews is currently not available")]'):
            log.msg(r'Aborting unavailable member page: %s' % response.url, level=log.INFO, spider=self)
            return

        # yield the reviewer info
        member = SingleValItemLoader(item=Member(), response=response)
        member.add_value('id', member_id)
        member.add_value('fullname', hxs.select('//body//div[@id="profileHeader"]//h1//b/text()').extract())
        member.add_value('badges', hxs.select('//body//div[@id="profileHeader"]//div[@class="badges"]//a/img/@alt').re(r'\((.+)\)'))
        # For top reviewers the ranking is inside an anchor while for lower rank people it's part of a text()
        ranking = hxs.select('//body//div[@id="reviewsStripe"]/div[@class="stripeContent"]/div//text()[contains(., "Reviewer Ranking")]/following-sibling::a[1]//text()').re(r'\d+(?:,\d+)?')
        ranking = ranking or hxs.select('//body//div[@id="reviewsStripe"]/div[@class="stripeContent"]/div//text()').re(r'Top Reviewer Ranking:\s+(%s)' % number_digit_grpd)
        member.add_value('ranking', ranking)
        member.add_value('helpfulStat', hxs.select('//body//div[@id="reviewsStripe"]/div[@class="stripeContent"]/div//text()[contains(., "Helpful Votes")]').re(r'\d+'))
        member.add_value('location', hxs.select('//body//td[@id="pdpLeftColumn"]//div[contains(@class, "personalDetails")]//b[contains(text(), "Location")]/following-sibling::text()').extract())
        review_stat = hxs.select('//body//div[@id="reviewsStripe"]//div[contains(@class, "seeAll")]/a/text()').re(r'See all\s+(%s)' % number_digit_grpd)
        # for low number of reviews the "See all" link won't appear. So we count the number of reviews
        if not review_stat:
            review_stat = len(hxs.select('//body//div[@id="reviewsStripe"]//div[contains(@class, "stripeContent")]//span/img[contains(@alt, "stars")]'))
        member.add_value('reviewStat', review_stat)
        yield member.load_item()

        # yield the reviews written by the member
        yield self._rev_page_request(member_id, MEMBER_TYPE)

    def parse_member_rev_page(self, response):
        """
        Parses a member reviews page and makes requests for subsequent pages
        """

        log.msg('Parsing member reviews: %s p%d' % (response.meta['id'], response.meta['page']), level=log.INFO, spider=self)

        hxs = HtmlXPathSelector(response)
        member_id = response.meta['id']

        # yield each review
        rev_body_elems = hxs.select('//table//td[not(@width)]//table//tr[not(@valign)]/td[@class="small"]/div')
        rev_header_elems = hxs.select('//table//td[not(@width)]//table//tr[@valign]/td[@align][2]//table[@class="small"]')
        for rev_header, rev_body in zip(rev_header_elems, rev_body_elems):
            # populating review data
            review = SingleValItemLoader(item=Review(), response=response)
            product_id = only_elem_or_default(rev_header.select('.//b/a/@href').re(product_url_id_re))
            if product_id:
                product_id = str(product_id)
            star_rating_tmp = rev_body.select('.//span/img[contains(@title, "stars")]/@title').re(star_rating_re)
            if not star_rating_tmp:
                # The review is probably a manufacturer response and not an actual review
                continue
            review.add_value('starRating', star_rating_tmp)
            review.add_value('productId', product_id)
            review.add_value('memberId', member_id)
            review.add_value('id', rev_body.select('div/a[contains(text(), "Permalink")]/@href').re(review_id_re))
            review.add_value('helpful', rev_body.select('div[contains(text(), "helpful")]/text()').re(r'\d+'))
            review.add_value('title', rev_body.select('div/span[contains(img/@alt, "stars")]/following-sibling::b[1]/text()').extract())
            review.add_value('date', rev_body.select('div/nobr/text()').extract())
            review.add_value('verifiedPurchase', rev_body.select('.//span[contains(@class, "crVerifiedStripe")]'))
            review.add_value('reviewTxt', rev_body.select('text()').extract())
            nComments_tmp = only_elem_or_default(rev_body.select('.//div/a/text()').re(r'Comments\s+\((\d+)\)'), '0')
            review.add_value('nComments', nComments_tmp)
            review.add_value('vine', rev_body.select('.//span/b[contains(text(), "Customer review from the Amazon Vine Program")]'))

            yield review.load_item()

            # yield the product
            yield self._item_page_request(product_id, PROD_TYPE)

        #make request for subsequent pages
        if hxs.select('//table//table//td[@class="small"]/b/text()').re(r'(\d+)\s+\|'):
            yield self._successor_page_request(response)

    def start_requests(self):
        settings = self.crawler.settings
        reqs = []
        with open(settings['SPIDER_SEED_FILENAME'], 'r') as read_file:
            reader = csv.DictReader(read_file)
            for seed in reader:
                # forming the url for each seed ID
                itm_type = seed['Type']
                itm_id = seed['ID']
                try:
                    req = self._item_page_request(itm_id, itm_type)
                except KeyError:
                    raise ValueError("The type of seed with ID %s was %s. Expected 'p' or 'm'" % (itm_id, itm_type))
                reqs += arg_to_iter(req)
        return reqs
