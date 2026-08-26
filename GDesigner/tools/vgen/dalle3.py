# This code is adapted from https://github.com/abi/screenshot-to-code/blob/5e3a174203dd6a59603c2fa944b14c7b398bfade/backend/image_generation.py
#!/usr/bin/env python
# -*- coding: utf-8 -*-

import asyncio
import re

from bs4 import BeautifulSoup
from openai import AsyncOpenAI


async def process_tasks(prompts, api_key):
    tasks = [generate_image(prompt, api_key) for prompt in prompts]
    results = await asyncio.gather(*tasks, return_exceptions=True)
    processed_results = []
    for result in results:
        if isinstance(result, Exception):
            print(f"An exeception occured: {result}")
            processed_results.append(None)
        else:
            processed_results.append(result)
    return processed_results


async def generate_image(prompt, api_key):
    client = AsyncOpenAI(api_key=api_key)
    res = await client.images.generate(
        model="dall-e-3",
        quality="standard",
        style="natural",
        n=1,
        size="1024x1024",
        prompt=prompt,
    )
    return res.data[0].url


def extract_dimensions(url):
    matches = re.findall(r"(\d+)x(\d+)", url)
    if matches:
        width, height = matches[0]
        return int(width), int(height)
    return 100, 100


def create_alt_url_mapping(code):
    soup = BeautifulSoup(code, "html.parser")
    return {
        image["alt"]: image["src"]
        for image in soup.find_all("img")
        if not image["src"].startswith("https://placehold.co")
    }


async def generate_images(code, api_key, image_cache):
    soup = BeautifulSoup(code, "html.parser")
    images = soup.find_all("img")
    alts = [
        img.get("alt")
        for img in images
        if img["src"].startswith("https://placehold.co")
        and image_cache.get(img.get("alt")) is None
        and img.get("alt") is not None
    ]
    prompts = list(set(alts))
    if not prompts:
        return code

    results = await process_tasks(prompts, api_key)
    mapped_image_urls = {**dict(zip(prompts, results)), **image_cache}
    for img in images:
        if not img["src"].startswith("https://placehold.co"):
            continue
        new_url = mapped_image_urls[img.get("alt")]
        if new_url:
            width, height = extract_dimensions(img["src"])
            img["width"] = width
            img["height"] = height
            img["src"] = new_url
        else:
            print("Image generation failed for alt text:" + img.get("alt"))
    return soup.prettify()
