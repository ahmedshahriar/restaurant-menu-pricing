from scrapy.utils.project import get_project_settings
from w3lib.http import basic_auth_header


class ProxyMiddleware:
    def process_request(self, request, spider):
        settings = get_project_settings()
        proxy_host = settings.get("PROXY_HOST")
        proxy_port = settings.get("PROXY_PORT")
        if proxy_host is None or proxy_port is None:
            raise ValueError("PROXY_HOST and PROXY_PORT must be set in settings.")
        request.meta["proxy"] = f"{proxy_host}:{proxy_port}"
        request.headers["Proxy-Authorization"] = basic_auth_header(
            settings.get("PROXY_USER"), settings.get("PROXY_PASSWORD")
        )
        spider.log(f"Proxy: {request.meta['proxy']}")
