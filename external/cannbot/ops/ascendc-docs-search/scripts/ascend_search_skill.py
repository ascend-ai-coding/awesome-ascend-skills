"""Ascend社区文档搜索技能 - 提供Ascend社区文档搜索功能"""
import base64
import json
import logging
from urllib.parse import urljoin, quote

import requests

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AscendSearchSkill:
    """Ascend社区搜索技能主类"""

    def __init__(self, base_url="https://www.hiascend.com"):
        self.base_url = base_url
        self.search_endpoint = "/ascendgateway/ascendservice/content/search"
        self.headers = {
            "x-request-type": "machine",
            "Content-Type": "application/json",
            "Referer": base_url,
            "User-Agent": "Mozilla/5.0 (compatible; AscendSearchSkill/1.0)"
        }

    def search_documents(self,
                         keyword,
                         lang="zh",
                         doc_type="DOC",
                         page_num=1,
                         page_size=10,
                         sort=1,
                         ignore_correction=False,
                         search_type=True):
        """搜索Ascend社区文档"""
        min_page_num = 1
        max_page_num = 100
        min_page_size = 1
        max_page_size = 10

        is_keyword_empty = not keyword or not keyword.strip()
        if is_keyword_empty:
            return self._error_response("关键词参数是必需的")

        is_page_num_invalid = page_num < min_page_num or page_num > max_page_num
        if is_page_num_invalid:
            return self._error_response("页码必须在1到100之间")

        is_page_size_invalid = page_size < min_page_size or page_size > max_page_size
        if is_page_size_invalid:
            return self._error_response("页面大小必须在1到10之间")

        params = {
            "keyword": quote(base64.b64encode(keyword.strip().encode('utf-8')).decode('utf-8')),
            "lang": lang,
            "type": doc_type,
            "pageNum": page_num,
            "pageSize": page_size,
            "sort": sort,
            "ignoreCorrection": str(ignore_correction).lower(),
            "searchType": str(search_type).lower()
        }

        try:
            full_url = urljoin(self.base_url, self.search_endpoint)
            response = requests.get(full_url,
                                    params=params,
                                    headers=self.headers,
                                    timeout=10)
            response.raise_for_status()

            result_data = response.json()
            is_success = result_data.get("success")
            has_data = "data" in result_data

            if is_success and has_data:
                data_content = result_data["data"]
                is_data_dict = isinstance(data_content, dict)
                has_nested_data = "data" in data_content

                if is_data_dict and has_nested_data:
                    document_data = data_content["data"]
                    if isinstance(document_data, list):
                        formatted_data = []
                        for item in document_data:
                            formatted_item = {
                                "title": item.get("docTitle", ""),
                                "summary": item.get("docContent", ""),
                                "url": item.get("docUrl", ""),
                                "version": item.get("version", ""),
                                "publishTime": item.get("publishTime", "")
                            }
                            formatted_data.append(formatted_item)

                        formatted_data = self._transform_urls(formatted_data)
                        return {
                            "success": True,
                            "message": result_data.get("msg", "success"),
                            "data": formatted_data
                        }

            return result_data

        except requests.exceptions.RequestException as e:
            logger.error("API请求失败", exc_info=True)
            return self._error_response(f"API请求失败: {str(e)}")
        except json.JSONDecodeError as e:
            logger.error("JSON响应解析失败", exc_info=True)
            return self._error_response(f"无效的JSON响应: {str(e)}")
        except Exception as e:
            logger.error("未知错误", exc_info=True)
            return self._error_response(f"未知错误: {str(e)}")

    def _transform_urls(self, data):
        """将内部URL转换为公共URL"""
        transformed_data = []
        source_prefix = "/source/"
        document_prefix = "/document/detail/"

        for item in data:
            has_url = "url" in item and item["url"]
            if has_url:
                transformed_url = item["url"].replace(source_prefix,
                                                      document_prefix)
                is_relative_url = transformed_url.startswith("/")
                if is_relative_url:
                    transformed_url = urljoin(self.base_url, transformed_url)
                item["url"] = transformed_url

            transformed_data.append(item)

        return transformed_data

    def _error_response(self, message):
        """创建标准错误响应"""
        return {"success": False, "message": message, "data": []}
    
    