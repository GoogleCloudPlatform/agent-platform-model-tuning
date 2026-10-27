# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


from google.cloud.aiplatform_v1.types import Content, GenerateContentRequest


def extract_text_from_content(content: Content) -> str:
    texts: list[str] = []
    for part in content.parts:
        text = part.text
        if text:
            texts.append(text)
    return "\n".join(texts).strip()

def extract_system_instruction_text(prompt: GenerateContentRequest) -> str:
    return extract_text_from_content(prompt.system_instruction)
