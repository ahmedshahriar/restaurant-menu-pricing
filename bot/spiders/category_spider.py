import re
import time
import urllib
from json import JSONDecodeError
from urllib.parse import urljoin

import chompjs
import scrapy

from core import settings


class CategorySpider(scrapy.Spider):
    name = "category_us"
    allowed_domains = ["ubereats.com"]
    handle_httpstatus_list = [403, 429]
    # You can scrape other country/location pages by changing the URL below
    # Either crawl from the main location page or directly from a specific location page
    # e.g., Canada: "https://www.ubereats.com/ca/location"
    # specific city: "https://www.ubereats.com/region/ny" # New York
    start_urls = ["https://www.ubereats.com/location"]  # US location page
    BASE_URL = "https://www.ubereats.com"

    # custom settings for feed export
    # export URLs to CSV file in data directory
    custom_settings = {
        "FEED_URI": settings.CRAWLED_TASK_DATA_PATH,
        "FEED_FORMAT": "csv",
        "FEED_EXPORTERS": {
            "csv": "scrapy.exporters.CsvItemExporter",
        },
        "FEED_EXPORT_ENCODING": "utf-8",
        "ITEM_PIPELINES": {},  # disable item pipelines for this spider
    }

    def clean_text(self, raw_html):
        """
        :param raw_html: this will take raw html code
        :return: text without html tags
        """
        cleaner = re.compile("<.*?>|&([a-z0-9]+|#[0-9]{1,6}|#x[0-9a-f]{1,6});")
        return re.sub(cleaner, "", raw_html)

    def parse(self, response, **kwargs):
        if response.status in self.handle_httpstatus_list:
            self.logger.info("Force-retrying request")
            re_request = response.request.copy()
            re_request.dont_filter = True
            yield re_request
            return None

        """ parse listings """
        raw_data = re.compile(r'<script type="application/json" id="__REACT_QUERY_STATE__">\n(.*)\n').search(
            response.text
        )

        json_data_uber = dict()

        try:
            # option 1: using unicodedata to decode unicode escape characters
            # json_data_uber = chompjs.parse_js_object(
            #     unicodedata.normalize('NFKD', raw_data.group(1).strip().encode().decode('unicode_escape'))
            #                           # replace('\\u0022', ''))
            #     #  unicode_escape=True,
            #     #  jsonlines=True
            # )
            # option 2: using chompjs
            json_data_uber = chompjs.parse_js_object(
                raw_data.group(1)
                .strip()
                .replace("\\u0022", '"')
                .replace("\\u003C", "<")
                .replace("\\u003E", ">")
                .replace("\\u0026", "&")
                .replace("%5C", "\\")
            )
        except JSONDecodeError as je:
            self.logger.error(je)
        except AttributeError as ae:
            self.logger.error(ae)

        region_city_dict = json_data_uber.get("queries")[2].get("state").get("data").get("regionCityLinks").get("links")

        all_location_links, all_location_names = [], []
        for link_data in region_city_dict:
            all_location_names.extend(li.get("title").strip() for li in link_data.get("links"))
            all_location_links.extend(
                [
                    # response.request.headers['origin'].decode('utf-8').strip() +
                    urljoin(self.BASE_URL, urllib.parse.quote(li.get("href").replace("city", "category").strip()))
                    for li in link_data.get("links")
                ]
            )
        loc_dict = dict(zip(all_location_links, all_location_names, strict=True))
        yield from response.follow_all(all_location_links, self.parse_category, meta={"loc_dict": loc_dict})

    def parse_category(self, response, **kwargs):
        """parse listings"""
        loc_dict = response.request.meta["loc_dict"]
        categories_data = response.xpath('//*[@id="main-content"]/div[2]/div[3]/a')
        for category_data in categories_data:
            self.logger.info("response url", response.url)
            location = loc_dict.get(response.url)
            # url = response.request.headers['origin'].decode('utf-8') + category_data.css('a ::attr("href")').get()
            url = urljoin(self.BASE_URL, category_data.css('a ::attr("href")').get())

            category = self.clean_text(category_data.css("div:last-child").get())
            yield {
                "location": location,
                "category": category,
                "url": url,
                "started_at": time.strftime("%d/%m/%Y %H:%M", time.localtime()),
            }
