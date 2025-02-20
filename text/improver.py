import re
import os
import json
from openai import OpenAI
from ..utils.logger import logger


class TextImprover:
    MAX_TOKENS = 2048  # DeepSeek reasoner token limit

    def __init__(self):
        api_key = os.getenv("DASHSCOPE_API_KEY")
        if not api_key:
            raise ValueError("DASHSCOPE_API_KEY environment variable is not set")

        self.client = OpenAI(
            api_key=api_key,
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        )
        logger.info("DeepSeek client initialized")
        logger.debug(f"Using DeepSeek API URL: {self.client.base_url}")

    def _parse_srt(self, srt_content):
        logger.debug(f"Parsing SRT content of length: {len(srt_content)}")
        segments = []
        pattern = r"(\d+:\d+:\d+) --> (\d+:\d+:\d+)\n(.*?)\n\n"
        matches = re.finditer(pattern, srt_content + "\n\n", re.DOTALL)

        try:
            for match in matches:
                segments.append(
                    {
                        "timeline": f"{match.group(1)} --> {match.group(2)}",
                        "text": match.group(3).strip(),
                    }
                )
            logger.debug(f"Successfully parsed {len(segments)} segments")
            return segments
        except Exception as e:
            logger.error(f"Error parsing SRT: {str(e)}")
            raise

    def _estimate_tokens(self, text):
        # Rough estimation: 1 token ≈ 4 characters
        return len(text) // 4

    def _batch_segments(self, segments):
        logger.debug(f"Batching {len(segments)} segments")
        batches = []
        current_batch = []
        current_tokens = 0

        for segment in segments:
            segment_tokens = self._estimate_tokens(segment["text"])

            if current_tokens + segment_tokens > self.MAX_TOKENS:
                batches.append(current_batch)
                current_batch = [segment]
                current_tokens = segment_tokens
            else:
                current_batch.append(segment)
                current_tokens += segment_tokens

        if current_batch:
            batches.append(current_batch)
        logger.debug(f"Created {len(batches)} batches")
        return batches

    def _improve_batch(self, batch):
        try:
            batch_data = [[seg["timeline"], seg["text"]] for seg in batch]
            combined_text = json.dumps(batch_data, ensure_ascii=False)
            logger.debug(f"Improving batch with {len(batch)} segments")

            messages = [
                {
                    "role": "system",
                    "content": (
                        f"对视频字幕进行微调（更正术语，允许重复），格式为[时间轴, 文本]。"
                        f"遵循以下规则：1. 调整后的字幕，在时间线上与原字幕对应。"
                        f"2. 输出必须使用Markdown代码块，采用JSON格式的数组。"
                        f"示例输出：```json\n[\"00:00:01 --> 00:00:05\", \"改进文本\"]```"
                    ),
                },
                {"role": "user", "content": combined_text},
            ]

            logger.debug(f"Sending message to DeepSeek: {messages}")
            completion = self.client.chat.completions.create(
                model="deepseek-r1",
                messages=messages, # type: ignore
                stream=True,
                stream_options={
                    "include_usage": True
                }
            )
            
            content = ""
            reasoning_content = ""
            is_answering = False
            
            for chunk in completion:
                if not chunk.choices:
                    logger.debug(f"Usage info received: {chunk.usage}")
                    continue
                    
                delta = chunk.choices[0].delta
                
                # Track reasoning content
                if hasattr(delta, 'reasoning_content') and delta.reasoning_content is not None:
                    reasoning_content += delta.reasoning_content
                
                # Track answer content
                if delta.content:
                    if not is_answering:
                        logger.debug("Starting to receive answer content")
                        is_answering = True
                    content += delta.content

            logger.debug(f"Reasoning process: {reasoning_content}")
            logger.debug(f"Final content: {content}")

            # Extract JSON from markdown code block
            json_match = re.search(r'```json\s*(\[.*?\])\s*```', content, re.DOTALL)
            
            if not json_match:
                logger.error("No JSON found in markdown response")
                return [seg["text"] for seg in batch]
                
            improved_texts = json.loads(json_match.group(1))
            logger.debug(f"Received {len(improved_texts)} improved texts")

            return improved_texts

        except Exception as e:
            logger.error(f"Error in _improve_batch: {str(e)}")
            return [seg["text"] for seg in batch]

    def improve_text(self, input_path, output_path):
        logger.info(f"Starting improvement of {input_path}")
        try:
            if not os.path.exists(input_path):
                logger.error(f"Input file not found: {input_path}")
                return False

            with open(input_path, "r", encoding="utf-8") as f:
                content = f.read()
                logger.debug(f"Read input file of size: {len(content)}")

            segments = self._parse_srt(content)
            logger.info(f"Parsed {len(segments)} segments from SRT")

            batches = self._batch_segments(segments)
            logger.info(f"Created {len(batches)} batches for processing")

            improved_segments = []
            for i, batch in enumerate(batches, 1):
                logger.info(f"Processing batch {i} of {len(batches)}")
                try:
                    improved_texts = self._improve_batch(batch)
                    for [timeline, improved_text] in improved_texts:
                        improved_segments.append(
                            {
                                "timeline": timeline,
                                "text": improved_text.strip(),
                            }
                        )
                except Exception as e:
                    logger.error(f"Error processing batch {i}: {str(e)}")
                    raise

            os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
            with open(output_path, "w", encoding="utf-8") as f:
                for i, segment in enumerate(improved_segments, 1):
                    f.write(f"{segment['timeline']}\n")
                    f.write(f"{segment['text']}\n\n")

            logger.info(f"Successfully saved improved SRT to: {output_path}")
            return True

        except Exception as e:
            logger.error(f"Error improving text: {str(e)}")
            logger.error(f"Stack trace:", exc_info=True)
            return False
