from scrapy.utils.project import get_project_settings
from w3lib.http import basic_auth_header


class ProxyMiddleware:
    def process_request(self, request, spider):
        settings = get_project_settings()
        request.meta["proxy"] = settings.get("PROXY_HOST") + ":" + settings.get("PROXY_PORT")
        request.headers["Proxy-Authorization"] = basic_auth_header(
            settings.get("PROXY_USER"), settings.get("PROXY_PASSWORD")
        )
        spider.log(f"Proxy : {request.meta['proxy']}")
