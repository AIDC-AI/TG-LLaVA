#    Copyright (C) 2024 AIDC-AI
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.


import os
from .clip_encoder import CLIPVisionTower, CLIPVisionTowerS2, CLIPTextTower


def build_vision_tower(vision_tower_cfg, **kwargs):
    vision_tower = getattr(vision_tower_cfg, 'mm_vision_tower', getattr(vision_tower_cfg, 'vision_tower', None))
    is_absolute_path_exists = os.path.exists(vision_tower)
    use_s2 = getattr(vision_tower_cfg, 's2', False)
    if is_absolute_path_exists or vision_tower.startswith("openai") or vision_tower.startswith("laion") or "ShareGPT4V" in vision_tower:
        if use_s2:
            return CLIPVisionTowerS2(vision_tower, args=vision_tower_cfg, **kwargs)
        else:
            return CLIPVisionTower(vision_tower, args=vision_tower_cfg, **kwargs)

    raise ValueError(f'Unknown vision tower: {vision_tower}')

def build_text_tower(text_tower_cfg, **kwargs):
    text_tower = getattr(text_tower_cfg, 'mm_text_tower', getattr(text_tower_cfg, 'text_tower', None))   # ?????
    is_absolute_path_exists = os.path.exists(text_tower)

    if is_absolute_path_exists or text_tower.startswith("openai") or text_tower.startswith("laion") or "ShareGPT4V" in text_tower:
        return CLIPTextTower(text_tower, args=text_tower_cfg, **kwargs)

    raise ValueError(f'Unknown text tower: {text_tower}')