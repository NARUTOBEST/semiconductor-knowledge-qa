# -*- coding: utf-8 -*-
"""上下文管理测试:工具结果截断逻辑。"""
import json
from context_management import truncate_tool_result


class TestTruncateToolResult:
    def test_error_dict_passthrough(self):
        err = {"error": "检索服务连接失败"}
        assert truncate_tool_result(err) == json.dumps(err, ensure_ascii=False)

    def test_long_content_truncated_metadata_preserved(self):
        long_text = "ALD" * 200
        result = [{"chunk_id": "c1", "source_stem": "manual",
                   "page_num": 12, "score": 0.9, "content": long_text}]
        parsed = json.loads(truncate_tool_result(result, max_chars=100))
        assert "已截断" in parsed[0]["content"]
        assert len(parsed[0]["content"]) < 200
        # 短元数据原样保留
        assert parsed[0]["chunk_id"] == "c1"
        assert parsed[0]["source_stem"] == "manual"
        assert parsed[0]["score"] == 0.9

    def test_short_content_not_truncated(self):
        short = "ALD 是一种薄膜沉积技术。"
        result = [{"chunk_id": "c1", "content": short}]
        parsed = json.loads(truncate_tool_result(result))
        assert parsed[0]["content"] == short
        assert "已截断" not in parsed[0]["content"]

    def test_short_fields_never_truncated(self):
        """chunk_id 等短字段即使异常长也不截断(SHORT_FIELDS 白名单)。"""
        long_id = "chunk_" + "x" * 200
        result = [{"chunk_id": long_id, "content": "short"}]
        parsed = json.loads(truncate_tool_result(result, max_chars=10))
        assert parsed[0]["chunk_id"] == long_id

    def test_table_html_truncated(self):
        result = [{"chunk_id": "c1", "table_html": "<table>" + "r" * 200 + "</table>"}]
        parsed = json.loads(truncate_tool_result(result, max_chars=50))
        assert "已截断" in parsed[0]["table_html"]

    def test_unknown_long_string_field_fallback_truncated(self):
        result = [{"chunk_id": "c1", "custom_field": "x" * 200}]
        parsed = json.loads(truncate_tool_result(result, max_chars=30))
        assert "已截断" in parsed[0]["custom_field"]

    def test_single_chunk_dict(self):
        result = {"chunk_id": "c1", "content": "ALD" * 200}
        parsed = json.loads(truncate_tool_result(result, max_chars=80))
        assert "已截断" in parsed["content"]
        assert parsed["chunk_id"] == "c1"

    def test_image_descriptions_list_joined_and_truncated(self):
        descs = ["这是一张图" * 50, "另一段描述" * 50]
        result = [{"chunk_id": "i1", "image_descriptions": descs}]
        parsed = json.loads(truncate_tool_result(result, max_chars=60))
        assert "已截断" in parsed[0]["image_descriptions"]
