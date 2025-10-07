# Scrapy settings for bot project
#
# For simplicity, this file contains only settings considered important or
# commonly used. You can find more settings consulting the documentation:
#
#     https://docs.scrapy.org/en/latest/topics/settings.html
#     https://docs.scrapy.org/en/latest/topics/downloader-middleware.html
#     https://docs.scrapy.org/en/latest/topics/spider-middleware.html
from core.settings import settings

BOT_NAME = "bot"

SPIDER_MODULES = ["bot.spiders"]
NEWSPIDER_MODULE = "bot.spiders"

MONGO_URI = settings.DATABASE_HOST  # Replace with your MongoDB connection string
MONGO_DATABASE = settings.DATABASE_NAME  # Replace with your MongoDB database name
MONGO_COLLECTION = settings.DATABASE_COLLECTION

ADDONS = {}


# Crawl responsibly by identifying yourself (and your website) on the user-agent
# USER_AGENT = "bot (+http://www.yourdomain.com)"

# Obey robots.txt rules
ROBOTSTXT_OBEY = False

# Concurrency and throttling settings
# CONCURRENT_REQUESTS = 16
CONCURRENT_REQUESTS_PER_DOMAIN = 1
DOWNLOAD_DELAY = 0.50
RANDOMIZE_DOWNLOAD_DELAY = True

# Disable cookies (enabled by default)
# COOKIES_ENABLED = False

# Disable Telnet Console (enabled by default)
TELNETCONSOLE_ENABLED = False

# Override the default request headers:
DEFAULT_REQUEST_HEADERS = {
    "authority": "www.ubereats.com",
    "accept": "*/*",
    "accept-language": "en-US,en;q=0.9,bn;q=0.8,hi;q=0.7",
    "cache-control": "no-cache",
    # Already added when you pass json=
    # 'content-type': 'application/json',
    # Requests sorts cookies = alphabetically
    # 'cookie': f"uev2.id.xp=ddf22734-0a57-4fec-973d-cb9af299145b; dId=6b50bc32-04db-4f8c-aa6b-fbe4751723f3; marketing_vistor_id=c353021d-5d8d-4a34-a8f8-5763c1031b56; uev2.gg=true; utm_medium=undefined; CONSENTMGR=c1:1%7Cc2:1%7Cc3:1%7Cc4:1%7Cc5:1%7Cc6:1%7Cc7:1%7Cc8:1%7Cc9:1%7Cc10:1%7Cc11:1%7Cc12:1%7Cc13:1%7Cc14:1%7Cc15:1%7Cts:1652650958456%7Cconsent:true; _gcl_au=1.1.945356044.1652650972; mcd_restaurant=Le 24; uev2.id.session=dfdee5ed-3567-49a1-9be6-96875049fed6; uev2.ts.session=1652777672091; jwt-session=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpYXQiOjE2NTI3Nzc2NzIsImV4cCI6MTY1Mjg2NDA3Mn0.qK2-wc_1YROMuqJZorCQX9i20y6q5pIg8S7_tRgVpho; _userUuid=; utag_main=v_id:0180c9ac5e6c0055834f38496fd005073003906b00bd0{_sn:2$_se:10$_ss:0$_st:1652779824249$ses_id:1652777676689%3Bexp-session$_pn:4%3Bexp-session;} uev2.diningMode=DELIVERY",
    "dnt": "1",
    "origin": "https://www.ubereats.com",
    "pragma": "no-cache",
    "sec-ch-ua": '" Not A;Brand";v="99", "Chromium";v="101", "Google Chrome";v="101"',
    "sec-ch-ua-mobile": "?0",
    "sec-ch-ua-platform": '"Windows"',
    "sec-fetch-dest": "empty",
    "sec-fetch-mode": "cors",
    "sec-fetch-site": "same-origin",
    "user-agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/101.0.4951.67 Safari/537.36",
}

# Enable or disable spider middlewares
# See https://docs.scrapy.org/en/latest/topics/spider-middleware.html
# SPIDER_MIDDLEWARES = {
#    "bot.middlewares.BotSpiderMiddleware": 543,
# }

# Enable or disable downloader middlewares
# See https://docs.scrapy.org/en/latest/topics/downloader-middleware.html
# DOWNLOADER_MIDDLEWARES = {
#    "bot.middlewares.BotDownloaderMiddleware": 543,
# }

# Enable or disable extensions
# See https://docs.scrapy.org/en/latest/topics/extensions.html
# EXTENSIONS = {
#    "scrapy.extensions.telnet.TelnetConsole": None,
# }

# Configure item pipelines
# See https://docs.scrapy.org/en/latest/topics/item-pipeline.html
ITEM_PIPELINES = {
    "bot.pipelines.BotPipeline": 300,
}

# Enable and configure the AutoThrottle extension (disabled by default)
# See https://docs.scrapy.org/en/latest/topics/autothrottle.html
AUTOTHROTTLE_ENABLED = True
# The initial download delay
AUTOTHROTTLE_START_DELAY = 5
# The maximum download delay to be set in case of high latencies
AUTOTHROTTLE_MAX_DELAY = 60
# The average number of requests Scrapy should be sending in parallel to
# each remote server
AUTOTHROTTLE_TARGET_CONCURRENCY = 1.0
# Enable showing throttling stats for every response received:
# AUTOTHROTTLE_DEBUG = False

# Enable and configure HTTP caching (disabled by default)
# See https://docs.scrapy.org/en/latest/topics/downloader-middleware.html#httpcache-middleware-settings
# HTTPCACHE_ENABLED = True
# HTTPCACHE_EXPIRATION_SECS = 0
# HTTPCACHE_DIR = "httpcache"
# HTTPCACHE_IGNORE_HTTP_CODES = []
# HTTPCACHE_STORAGE = "scrapy.extensions.httpcache.FilesystemCacheStorage"

# Set settings whose default value is deprecated to a future-proof value
FEED_EXPORT_ENCODING = "utf-8"
