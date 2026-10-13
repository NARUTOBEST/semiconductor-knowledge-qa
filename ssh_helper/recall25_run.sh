#!/bin/bash
OUT=/tmp/recall25_out.tsv; : > $OUT
echo "CASE dev_041" >> $OUT
curl -s -m100 "http://127.0.0.1:6333/collections/ald_text/points/scroll" -H "Content-Type: application/json" --data-raw '{"filter": {"should": [{"key": "content", "match": {"text": "end effector"}}]}, "limit": 8, "with_payload": true}' >> $OUT
echo "" >> $OUT
echo "CASE dev_070" >> $OUT
curl -s -m100 "http://127.0.0.1:6333/collections/ald_text/points/scroll" -H "Content-Type: application/json" --data-raw '{"filter": {"should": [{"key": "content", "match": {"text": "5 nm"}}, {"key": "content", "match": {"text": "4 nm"}}]}, "limit": 8, "with_payload": true}' >> $OUT
echo "" >> $OUT
echo "CASE dev_130" >> $OUT
curl -s -m100 "http://127.0.0.1:6333/collections/ald_text/points/scroll" -H "Content-Type: application/json" --data-raw '{"filter": {"should": [{"key": "content", "match": {"text": "Heat Seal Dwell"}}]}, "limit": 8, "with_payload": true}' >> $OUT
echo "" >> $OUT
echo "CASE dev_111" >> $OUT
curl -s -m100 "http://127.0.0.1:6333/collections/ald_text/points/scroll" -H "Content-Type: application/json" --data-raw '{"filter": {"should": [{"key": "content", "match": {"text": "65°C"}}]}, "limit": 8, "with_payload": true}' >> $OUT
echo "" >> $OUT
echo "CASE dev_036" >> $OUT
curl -s -m100 "http://127.0.0.1:6333/collections/ald_text/points/scroll" -H "Content-Type: application/json" --data-raw '{"filter": {"should": [{"key": "content", "match": {"text": "aqueous concentrated hydrofluoric acid"}}]}, "limit": 8, "with_payload": true}' >> $OUT
echo "" >> $OUT
echo "CASE dev_180" >> $OUT
curl -s -m100 "http://127.0.0.1:6333/collections/ald_text/points/scroll" -H "Content-Type: application/json" --data-raw '{"filter": {"should": [{"key": "content", "match": {"text": "Fe on HF-last"}}, {"key": "content", "match": {"text": "Table II"}}, {"key": "content", "match": {"text": "SC1 only"}}]}, "limit": 8, "with_payload": true}' >> $OUT
echo "" >> $OUT
echo "CASE dev_003" >> $OUT
curl -s -m100 "http://127.0.0.1:6333/collections/ald_text/points/scroll" -H "Content-Type: application/json" --data-raw '{"filter": {"should": [{"key": "content", "match": {"text": "Water Supply"}}]}, "limit": 8, "with_payload": true}' >> $OUT
echo "" >> $OUT
echo "CASE dev_043" >> $OUT
curl -s -m100 "http://127.0.0.1:6333/collections/ald_text/points/scroll" -H "Content-Type: application/json" --data-raw '{"filter": {"should": [{"key": "content", "match": {"text": "mm TIR"}}]}, "limit": 8, "with_payload": true}' >> $OUT
echo "" >> $OUT
echo "CASE dev_083" >> $OUT
curl -s -m100 "http://127.0.0.1:6333/collections/ald_text/points/scroll" -H "Content-Type: application/json" --data-raw '{"filter": {"should": [{"key": "content", "match": {"text": "Paddle position recognition"}}, {"key": "content", "match": {"text": "Recognition Setting"}}, {"key": "content", "match": {"text": "manual recognition"}}]}, "limit": 8, "with_payload": true}' >> $OUT
echo "" >> $OUT
echo "CASE dev_124" >> $OUT
curl -s -m100 "http://127.0.0.1:6333/collections/ald_text/points/scroll" -H "Content-Type: application/json" --data-raw '{"filter": {"should": [{"key": "content", "match": {"text": "alternative materials/Force 模式"}}, {"key": "content", "match": {"text": "AlTiC and alternative materials"}}, {"key": "content", "match": {"text": "alternative materials"}}, {"key": "content", "match": {"text": "ATC Grinder"}}]}, "limit": 8, "with_payload": true}' >> $OUT
echo "" >> $OUT
echo "CASE dev_184" >> $OUT
curl -s -m100 "http://127.0.0.1:6333/collections/ald_text/points/scroll" -H "Content-Type: application/json" --data-raw '{"filter": {"should": [{"key": "content", "match": {"text": "right probe and reference data"}}, {"key": "content", "match": {"text": "change the offending wheel"}}, {"key": "content", "match": {"text": "Notify Maintenance"}}]}, "limit": 8, "with_payload": true}' >> $OUT
echo "" >> $OUT
echo "CASE dev_107" >> $OUT
curl -s -m100 "http://127.0.0.1:6333/collections/ald_text/points/scroll" -H "Content-Type: application/json" --data-raw '{"filter": {"should": [{"key": "content", "match": {"text": "50°C"}}, {"key": "content", "match": {"text": "56°C"}}]}, "limit": 8, "with_payload": true}' >> $OUT
echo "" >> $OUT
echo "CASE dev_127" >> $OUT
curl -s -m100 "http://127.0.0.1:6333/collections/ald_text/points/scroll" -H "Content-Type: application/json" --data-raw '{"filter": {"should": [{"key": "content", "match": {"text": "405nm"}}]}, "limit": 8, "with_payload": true}' >> $OUT
echo "" >> $OUT
echo "CASE dev_028" >> $OUT
curl -s -m100 "http://127.0.0.1:6333/collections/ald_text/points/scroll" -H "Content-Type: application/json" --data-raw '{"filter": {"should": [{"key": "content", "match": {"text": "more than"}}]}, "limit": 8, "with_payload": true}' >> $OUT
echo "" >> $OUT
echo "CASE dev_169" >> $OUT
curl -s -m100 "http://127.0.0.1:6333/collections/ald_text/points/scroll" -H "Content-Type: application/json" --data-raw '{"filter": {"should": [{"key": "content", "match": {"text": "sensor position"}}, {"key": "content", "match": {"text": "tool offset"}}]}, "limit": 8, "with_payload": true}' >> $OUT
echo "" >> $OUT
echo ALLDONE >> $OUT