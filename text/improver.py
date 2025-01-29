import re
from openai import OpenAI
from utils.logger import logger

class TextImprover:
    MAX_TOKENS = 2048  # DeepSeek reasoner token limit
    
    def __init__(self, api_key):
        self.client = OpenAI(
            api_key=api_key,
            base_url="https://api.deepseek.com"
        )

    def _parse_srt(self, srt_content):
        segments = []
        pattern = r'(\d+:\d+:\d+,\d+) --> (\d+:\d+:\d+,\d+)\n(.*?)\n\n'
        matches = re.finditer(pattern, srt_content + '\n\n', re.DOTALL)
        
        for match in matches:
            segments.append({
                'timeline': f"{match.group(1)} --> {match.group(2)}",
                'text': match.group(3).strip()
            })
        return segments

    def _estimate_tokens(self, text):
        # Rough estimation: 1 token ≈ 4 characters
        return len(text) // 4

    def _batch_segments(self, segments):
        batches = []
        current_batch = []
        current_tokens = 0

        for segment in segments:
            segment_tokens = self._estimate_tokens(segment['text'])
            
            if current_tokens + segment_tokens > self.MAX_TOKENS:
                batches.append(current_batch)
                current_batch = [segment]
                current_tokens = segment_tokens
            else:
                current_batch.append(segment)
                current_tokens += segment_tokens

        if current_batch:
            batches.append(current_batch)
        return batches

    def _improve_batch(self, batch):
        combined_text = "\n---\n".join(seg['text'] for seg in batch)
        messages = [{
            "role": "system",
            "content": "Improve each subtitle section separated by '---'. "
                      "Keep the same number of sections and maintain similar length for each. "
                      "Make them clear and natural while preserving meaning."
        }, {
            "role": "user",
            "content": combined_text
        }]

        response = self.client.chat.completions.create(
            model="deepseek-reasoner",
            messages=messages
        )
        
        improved_texts = response.choices[0].message.content.split("\n---\n")
        return improved_texts

    def improve_text(self, input_path, output_path):
        try:
            # Read and parse SRT
            with open(input_path, 'r', encoding='utf-8') as f:
                content = f.read()
            segments = self._parse_srt(content)

            # Batch and improve segments
            batches = self._batch_segments(segments)
            improved_segments = []
            
            for batch in batches:
                improved_texts = self._improve_batch(batch)
                for original_seg, improved_text in zip(batch, improved_texts):
                    improved_segments.append({
                        'timeline': original_seg['timeline'],
                        'text': improved_text.strip()
                    })

            # Write improved SRT
            with open(output_path, 'w', encoding='utf-8') as f:
                for i, segment in enumerate(improved_segments, 1):
                    f.write(f"{i}\n")
                    f.write(f"{segment['timeline']}\n")
                    f.write(f"{segment['text']}\n\n")

            logger.info(f"Improved SRT file saved to: {output_path}")
            return True

        except Exception as e:
            logger.error(f"Error improving text: {str(e)}")
            return False
