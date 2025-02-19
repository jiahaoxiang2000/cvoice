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
        pattern = r"(\d+:\d+:\d+,\d+) --> (\d+:\d+:\d+,\d+)\n(.*?)\n\n"
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
            combined_text = "\n---\n".join(seg["text"] for seg in batch)
            logger.debug(f"Improving batch with {len(batch)} segments")
            logger.debug(f"Combined text length: {len(combined_text)}")

            messages = [
                {
                    "role": "system",
                    "content": (
                        f"Convert exactly {len(batch)} subtitle sections separated by '---' into improved text. "
                        "Output must contain exactly the same number of sections as input. "
                        "Output should be in JSON format with an array of improved texts. "
                        "Make them clear and natural while preserving meaning.\n\n"
                        "EXAMPLE INPUT:\nText1\n---\nText2\n\n"
                        '{"improved_texts": ["Improved Text1", "Improved Text2"]}\n\n'
                        f"YOUR OUTPUT MUST CONTAIN EXACTLY {len(batch)} IMPROVED TEXTS."
                    ),
                },
                {"role": "user", "content": combined_text},
            ]

            response = self.client.chat.completions.create(
                model="deepseek-r1",
                messages=messages,
            )

            content = json.loads(response.choices[0].message.content)
            logger.debug(f"improved texts: {content}")

            improved_texts = content.get("improved_texts", [])

            # Strict validation of output count
            if len(improved_texts) != len(batch):
                logger.error(
                    f"API returned wrong number of segments: {len(improved_texts)} instead of {len(batch)}"
                )
                # Fall back to original texts if counts don't match
                improved_texts = [seg["text"] for seg in batch]

            return improved_texts

        except Exception as e:
            logger.error(f"Error in _improve_batch: {str(e)}")
            # Fall back to original texts on error
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
                    for original_seg, improved_text in zip(batch, improved_texts):
                        improved_segments.append(
                            {
                                "timeline": original_seg["timeline"],
                                "text": improved_text.strip(),
                            }
                        )
                except Exception as e:
                    logger.error(f"Error processing batch {i}: {str(e)}")
                    raise

            os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
            with open(output_path, "w", encoding="utf-8") as f:
                for i, segment in enumerate(improved_segments, 1):
                    f.write(f"{i}\n")
                    f.write(f"{segment['timeline']}\n")
                    f.write(f"{segment['text']}\n\n")

            logger.info(f"Successfully saved improved SRT to: {output_path}")
            return True

        except Exception as e:
            logger.error(f"Error improving text: {str(e)}")
            logger.error(f"Stack trace:", exc_info=True)
            return False
