import json
import re
import time
from json import JSONDecodeError
from urllib.parse import urljoin

import chompjs
import pandas as pd
import scrapy
from bs4 import BeautifulSoup

from core import settings


class RestaurantSpiderUS(scrapy.Spider):
    name = "restaurant_us"
    allowed_domains = ["ubereats.com"]
    handle_httpstatus_list = [403, 429]
    BASE_URL = "https://www.ubereats.com"

    # either use pipelines to store data in MongoDB or use feed export to store data in a json file
    # example for jsonlines export:
    # custom_settings = {
    #     'FEED_URI': pathlib.Path('data') / 'restaurant_data.json',
    #     'FORMAT': 'jsonlines',
    #     'FEED_EXPORTERS': {
    #         # https://docs.scrapy.org/en/latest/topics/exporters.html#jsonitemexporter
    #         # consider using jsonlines exporter to export large json files
    #         'json': 'scrapy.exporters.JsonLinesItemExporter',  # JsonLinesItemExporter or JsonItemExporter
    #
    #     },
    #     'FEED_EXPORT_ENCODING': 'utf-8',
    # }

    # using pipelines to store data in MongoDB
    custom_settings = {
        # this will store the data in MongoDB as per the pipeline settings
        "ITEM_PIPELINES": {
            "bot.pipelines.BotPipeline": 300,
        },
    }

    def start_requests(self):
        """
        :return: requests
        """
        # read csv file for restaurant urls (https://www.ubereats.com/category/<location>/<category>)
        #
        df = pd.read_csv(settings.CRAWLED_TASK_DATA_PATH)
        # df = df.head(1) # test with 1 row
        urls = df.url.values
        for url in urls:
            yield scrapy.Request(url=url, callback=self.parse, meta={"task_id": int(df[df.url == url].id.values[0])})

    def clean_text(self, raw_html):
        """
        :param raw_html: this will take raw html code
        :return: text without html tags
        """
        cleaner = re.compile("<.*?>|&([a-z0-9]+|#[0-9]{1,6}|#x[0-9a-f]{1,6});")
        return re.sub(cleaner, "", raw_html)

    def parse(self, response, **kwargs):
        task_id = response.request.meta["task_id"]

        if response.status in self.handle_httpstatus_list:
            self.logger.info("Force-retrying request")
            re_request = response.request.copy()
            re_request.dont_filter = True
            yield re_request
            return None

        """ parse listings """
        restaurant_links = [
            urljoin(self.BASE_URL, url)
            for url in response.xpath('//*[@id="main-content"]/div[5]/div/div/div[1]//a/@href').getall()
        ]
        restaurant_pos_dict = {link: position for position, link in enumerate(restaurant_links, 1)}
        yield from response.follow_all(
            restaurant_links,
            self.parse_restaurant,
            meta={"restaurant_pos_dict": restaurant_pos_dict, "task_id": task_id},
        )

    def parse_restaurant(self, response, **kwargs):
        """parse listings"""
        # url
        # position
        # name
        # score
        # ratings
        # category
        # price_range
        # full_address
        # zip_code
        # lat
        # lng
        # phone
        # image_url

        restaurant_pos_dict = response.request.meta["restaurant_pos_dict"]

        json_data = response.xpath('//*[@id="main-content"]/script').get()
        json_normalized = dict()
        # option #1
        try:
            json_normalized = chompjs.parse_js_object(json_data)
        except JSONDecodeError:
            self.logger.error("JSONDecodeError")
        except Exception as e:
            self.logger.error("Exception: ", e)

        # option #2
        # try:
        #     json_data = json_data.replace('<script type="application/ld+json">', '').replace('</script>', '')
        # except Exception as e:
        #     self.logger.error(e)
        # json_normalized = dict()
        # try:
        #     json_normalized = unicodedata.normalize("NFKD", json_data.encode().decode('unicode_escape'))
        #     json_normalized = json.loads(json_normalized.replace('\r', '').replace('\n', ''))
        # except JSONDecodeError as e:
        #     self.logger.error(e)

        restaurant_dict = dict()

        try:
            restaurant_dict["task_id"] = response.request.meta["task_id"]
        except Exception as e:
            self.logger.error(e)

        restaurant_dict["url"] = response.url

        try:
            restaurant_dict["position"] = restaurant_pos_dict.get(response.url)
        except Exception as e:
            self.logger.error(e)

        try:
            restaurant_dict["name"] = json_normalized.get("name")
        except AttributeError:
            restaurant_dict["name"] = None
        try:
            restaurant_dict["score"] = json_normalized.get("aggregateRating").get("ratingValue")
        except AttributeError:
            restaurant_dict["score"] = None
        try:
            restaurant_dict["ratings"] = json_normalized.get("aggregateRating").get("reviewCount")
        except AttributeError:
            restaurant_dict["ratings"] = None
        try:
            restaurant_dict["category"] = ", ".join(json_normalized.get("servesCuisine"))
        except AttributeError:
            restaurant_dict["category"] = None
        try:
            restaurant_dict["price_range"] = json_normalized.get("priceRange")
        except AttributeError:
            restaurant_dict["price_range"] = None

        try:
            street = json_normalized.get("address").get("streetAddress")
            locality = json_normalized.get("address").get("addressLocality")
            region = json_normalized.get("address").get("addressRegion")
            postal_code = json_normalized.get("address").get("postalCode")
            restaurant_dict["full_address"] = f"{street}, {locality}, {region}, {postal_code}"
        except AttributeError:
            restaurant_dict["full_address"] = None
        try:
            restaurant_dict["zip_code"] = json_normalized.get("address").get("postalCode")
        except AttributeError:
            restaurant_dict["zip_code"] = None
        try:
            restaurant_dict["lat"] = json_normalized.get("geo").get("latitude")
        except AttributeError:
            restaurant_dict["lat"] = None
        try:
            restaurant_dict["lng"] = json_normalized.get("geo").get("longitude")
        except AttributeError:
            restaurant_dict["lng"] = None
        try:
            restaurant_dict["phone"] = json_normalized.get("telephone")
        except AttributeError:
            restaurant_dict["phone"] = None
        try:
            restaurant_dict["image_url"] = json_normalized.get("image")[0]
        except AttributeError:
            restaurant_dict["image_url"] = None

        if json_normalized.get("hasMenu"):
            try:
                menu_items = []
                for menu_item in json_normalized.get("hasMenu").get("hasMenuSection"):
                    if menu_item.get("hasMenuItem"):
                        for item in menu_item.get("hasMenuItem"):
                            try:
                                menu_category = menu_item.get("name")
                            except AttributeError:
                                menu_category = None
                            try:
                                menu_item_name = item.get("name")
                            except AttributeError:
                                menu_item_name = None
                            try:
                                menu_item_description = item.get("description")
                            except AttributeError:
                                menu_item_description = None
                            try:
                                menu_item_price = (
                                    str(item.get("offers").get("price") / 100)
                                    + " "
                                    + item.get("offers").get("priceCurrency")
                                )
                            except AttributeError:
                                menu_item_price = None

                            d = {
                                "category": menu_category,
                                "name": menu_item_name,
                                "description": menu_item_description,
                                "price": menu_item_price,
                            }
                            menu_items.append(d)

                restaurant_dict["menu_items"] = menu_items
            except Exception as e:
                self.logger.error("No menu items", e)
        else:
            try:
                element, json_data_backup, data_dict = None, None, None
                try:
                    soup = BeautifulSoup(response.text.encode("utf-8"), "html.parser")
                    scraped_js = str(soup.select_one('script[id="__REDUX_STATE__"]').text).strip()
                    element = (
                        scraped_js.replace("\\u0022", '"')
                        .replace("\\u003C", "<")
                        .replace("\\u003E", ">")
                        .replace("\\u0026", "&")
                        .replace("%5C", "\\")
                    )
                    del scraped_js
                except Exception as e:
                    self.logger.error("Failed to scrape items", e)
                try:
                    json_data_backup = json.loads(element)
                    del element
                except Exception as e:
                    self.logger.error("Failed to normalize json", e)

                try:
                    data_dict = json_data_backup.get("stores")
                except Exception as e:
                    self.logger.error("Failed to fetch json", e)

                if len(data_dict.keys()) == 1:
                    uuid = list(data_dict.keys())[0]
                    menu_list = data_dict.get(uuid).get("data").get("catalogSectionsMap").get(uuid)
                    menu_items = []
                    for menu in menu_list:
                        try:
                            menu_category = menu.get("payload").get("standardItemsPayload").get("title").get("text")
                        except AttributeError:
                            menu_category = None

                        # menu item details
                        menu_item_details_list = menu.get("payload").get("standardItemsPayload").get("catalogItems")
                        for menu_item_details in menu_item_details_list:
                            try:
                                menu_name = menu_item_details.get("title")
                            except AttributeError:
                                menu_name = None

                            try:
                                menu_description = menu_item_details.get("itemDescription")
                            except AttributeError:
                                menu_description = None

                            try:
                                menu_price = str(menu_item_details.get("price") / 100) + " " + "USD"
                            except AttributeError:
                                menu_price = None
                            d = {
                                "category": menu_category,
                                "name": menu_name,
                                "description": menu_description,
                                "price": menu_price,
                            }
                            menu_items.append(d)
                    restaurant_dict["menu_items"] = menu_items
            except Exception as e:
                self.logger.error("No menu items", e)
        restaurant_dict["ended_at"] = time.strftime("%d/%m/%Y %H:%M", time.localtime())

        yield restaurant_dict
